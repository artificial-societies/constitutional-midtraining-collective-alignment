"""Run cmca/midtrain/cpt_train.py on one 8xB200 node (amendment 56): continued pretraining or SFT, and the chain
that trains one arm end to end on Modal's side.

  # one stage (pilots):
  MODAL_PROFILE=engine APP_NAME=cmca-cpt-pilot modal run --detach cmca/midtrain/cpt_modal.py \\
      --run mistral24/pilot_lr1e-6 --init mistral24/base_init --kind cpt --arm treated --lr 1e-6 \\
      --extra "--schedule constant --warmup-steps 2 --steps 19 --ckpt-minutes 999"
  # a whole arm (Stage 1): deploy, then start it server-side; the laptop may then sleep or disconnect
  MODAL_PROFILE=engine APP_NAME=cmca-stage1 RETRIES=1 MAX_HOURS=12 modal deploy cmca/midtrain/cpt_modal.py
  MODAL_PROFILE=engine python -c "import modal; modal.Function.from_name('cmca-stage1', 'arm').spawn(...)"

Paths are relative to /midtrain/cpt on the `midtrain` Volume; data are read from <family>/data. The Volume is
committed every five minutes while training runs (checkpoints appear only once complete, so a commit never exposes a
partial one), and an attempt that fails or is preempted is retried RETRIES times (default 2), resuming from the newest
checkpoint; MAX_HOURS bounds each attempt. Exit code 3 is the trainer's divergence guard and is not retried.

`arm` chains midtraining (with export), then SFT on the exported weights (with export), checkpointing every 30
minutes and writing <name>_STATUS.json at every transition; relaunched, it skips stages already exported and resumes
the current one from its newest checkpoint. Each stage has its own budget in hours, retries included; past it the stage is cancelled. With
`wait_for`, the arm starts only once that run has logged 10 healthy steps, and not at all if it diverged or never
got there (the staggered start of the control). The orchestrator records each training call in <name>_CALLS.json;
restarted by Modal while a call is still running, it waits on that call instead of starting a second job on the same
run (amendment 56's Stage 1 ran two for six hours).

IMAGE=nemotron_h (amendment 59) builds the training image on CUDA's development image with mamba_ssm and
causal_conv1d, which Nemotron-H's Mamba layers need; `kernels` checks them on one GPU before any training.
"""

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

import modal  # ty: ignore[unresolved-import]

MAX_HOURS = float(os.environ.get("MAX_HOURS", "20"))
RETRIES = int(
    os.environ.get("RETRIES", "2")
)  # 0 for pilots: a crashing pilot should not run three times
ROOT = "/midtrain/cpt"
PACKAGES = ("torch==2.8.0", "transformers==5.10.2", "accelerate==1.14.0", "safetensors", "numpy",
            "huggingface_hub[hf_transfer]")  # fmt: skip
if os.environ.get("IMAGE") == "nemotron_h":
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git")
        .pip_install(*PACKAGES, "packaging", "ninja", "wheel")
        .env({"TORCH_CUDA_ARCH_LIST": "10.0", "MAX_JOBS": "16", "CAUSAL_CONV1D_FORCE_BUILD": "FALSE",
              "MAMBA_FORCE_BUILD": "FALSE"})  # fmt: skip
        .pip_install("causal-conv1d==1.5.2", extra_options="--no-build-isolation")
        .pip_install("mamba-ssm==2.2.5", extra_options="--no-build-isolation")
        # mamba_ssm/__init__.py imports its own model classes, written against transformers 4 (GreedySearch...);
        # only the kernels are used, so the package init keeps just the version
        .run_commands("for f in $(find / -path '*site-packages/mamba_ssm/__init__.py' 2>/dev/null); do "
                      "echo \"__version__ = '2.2.5'\" > $f; rm -f $(dirname $f)/__pycache__/__init__*.pyc; echo trimmed $f; done")
    )
else:
    image = modal.Image.debian_slim(python_version="3.12").pip_install(*PACKAGES)
image = image.env({"HF_HOME": "/models/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).add_local_file(
    "cmca/midtrain/cpt_train.py", "/root/cpt_train.py"
)
store = modal.Volume.from_name("midtrain", create_if_missing=True)
app = modal.App(os.environ.get("APP_NAME", "cmca-cpt"))


# ephemeral_disk: Volume writes are staged on local disk before each commit, and a checkpoint (FP32 weights and Adam
# states, about 380 GB for 24B) plus an export (48 GB) overflows the default
@app.function(image=image, gpu="B200:8", volumes={"/midtrain": store}, cpu=32, memory=524288, ephemeral_disk=1_500_000,
              timeout=int(MAX_HOURS * 3600), retries=modal.Retries(max_retries=RETRIES, initial_delay=30.0))  # fmt: skip
def train(run: str, init: str, kind: str, arm: str, lr: float, extra: str = "") -> int:
    family = run.split("/")[0]
    store.reload()
    cmd = ["torchrun", "--nproc_per_node=8", "/root/cpt_train.py", "--run-dir", f"{ROOT}/{run}",
           "--init", f"{ROOT}/{init}", "--data", f"{ROOT}/{family}/data", "--kind", kind, "--arm", arm,
           "--lr", str(lr), *shlex.split(extra)]  # fmt: skip
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    stop = threading.Event()

    def commit() -> None:
        while not stop.wait(300):
            store.commit()

    threading.Thread(target=commit, daemon=True).start()
    rc = proc.wait()
    stop.set()
    time.sleep(2)
    store.commit()
    if rc == 3:  # the divergence guard: retrying would diverge again
        return rc
    if rc != 0:
        raise RuntimeError(
            f"training exited with {rc}"
        )  # a retry resumes from the newest checkpoint
    return rc


@app.function(image=image, gpu="B200:1", timeout=1800)
def kernels() -> dict:
    """One GPU: mamba_ssm's fused Mamba-2 kernel with a sequence index (forward and backward, and that the index
    really separates the two halves of a sequence), and torch's grouped matrix multiply for the experts."""
    import torch  # ty: ignore[unresolved-import]
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # ty: ignore[unresolved-import]

    out: dict = {"torch": torch.__version__, "device": torch.cuda.get_device_name(), "cap": torch.cuda.get_device_capability()}
    torch.manual_seed(0)
    d, h, p, g, n, L = 2688, 64, 64, 8, 128, 1024  # Nano's Mamba sizes (intermediate 4096 = 64 heads x 64)
    inter = h * p
    dev, bf = "cuda", torch.bfloat16
    proj = torch.randn(1, L, 2 * inter + 2 * g * n + h, device=dev, dtype=bf, requires_grad=True)
    conv_w = torch.randn(inter + 2 * g * n, 4, device=dev, dtype=bf) * 0.1
    args = dict(conv1d_bias=torch.zeros(inter + 2 * g * n, device=dev, dtype=bf), dt_bias=torch.zeros(h, device=dev, dtype=bf),  # noqa: C408
                A=-torch.ones(h, device=dev), D=torch.ones(h, device=dev, dtype=bf), chunk_size=128, activation="silu",
                rmsnorm_weight=torch.ones(inter, device=dev, dtype=bf), rmsnorm_eps=1e-5,
                outproj_weight=torch.randn(d, inter, device=dev, dtype=bf) * 0.02, outproj_bias=None, headdim=p, ngroups=g,
                norm_before_gate=False)  # fmt: skip

    def run(x, seq_idx):  # noqa: ANN001, ANN202
        b = args
        return mamba_split_conv1d_scan_combined(x, conv_w, b["conv1d_bias"], b["dt_bias"], b["A"], D=b["D"],
            chunk_size=b["chunk_size"], seq_idx=seq_idx, activation="silu", rmsnorm_weight=b["rmsnorm_weight"],
            rmsnorm_eps=1e-5, outproj_weight=b["outproj_weight"], outproj_bias=None, headdim=p, ngroups=g,
            norm_before_gate=False)  # fmt: skip

    half = L // 2
    seq_idx = torch.cat([torch.zeros(half), torch.ones(L - half)]).to(dev, torch.int32)[None]
    y = run(proj, seq_idx)
    y.float().pow(2).mean().backward()
    alone = run(proj[:, half:].detach().contiguous(), None)
    out["mamba_fwd_bwd"] = bool(torch.isfinite(proj.grad).all())
    out["seq_idx_separates"] = float((y[:, half:].float() - alone.float()).abs().max())
    out["no_idx_differs"] = float((run(proj.detach(), None)[:, half:].float() - alone.float()).abs().max())
    try:
        a = torch.randn(4096, 2688, device=dev, dtype=bf)
        w = torch.randn(128, 2688, 1856, device=dev, dtype=bf)
        offs = torch.arange(32, 4097, 32, device=dev, dtype=torch.int32)
        r = torch._grouped_mm(a, w, offs=offs)  # noqa: SLF001  # the kernel under test
        out["grouped_mm"] = float((r[:32].float() - (a[:32].float() @ w[0].float())).abs().max())
    except Exception as e:  # noqa: BLE001
        out["grouped_mm"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(out, flush=True)
    return out


def status(path: Path, rec: dict) -> None:
    hist = json.loads(path.read_text()) if path.exists() else []
    hist.append(rec | {"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())})
    path.write_text(json.dumps(hist, indent=1))
    store.commit()
    print(json.dumps(rec), flush=True)


# The orchestrator reads other containers' results through the Volume API: its own mount does not show their
# commits reliably (the dry run's relaunch saw no EXPORTED marker that the CLI listed).
def remote_text(path: str) -> str:
    try:
        return b"".join(store.read_file(path)).decode()
    except Exception:  # noqa: BLE001  # not written yet
        return ""


def remote_exists(path: str) -> bool:
    parent, name = path.rsplit("/", 1)
    try:
        return any(e.path.rstrip("/").endswith("/" + name) for e in store.listdir(parent))
    except Exception:  # noqa: BLE001  # directory not created yet
        return False


def healthy_steps(run: str) -> int:
    text = remote_text(f"cpt/{run}/metrics.jsonl")
    return sum(1 for line in text.splitlines() if '"step"' in line and '"loss"' in line)


@app.function(image=image, volumes={"/midtrain": store}, cpu=1, memory=2048, timeout=24 * 3600)
def arm(family: str, name: str, which: str, mt_lr: float, sft_lr: float, extra: str = "",
        mt_hours: float = 11.0, sft_hours: float = 6.0, wait_for: str = "") -> str:  # fmt: skip
    """Train one arm end to end: midtraining on `which` (treated or control), then SFT. Returns the final status."""
    from modal.exception import TimeoutError as ModalTimeout  # ty: ignore[unresolved-import]

    st = Path(ROOT) / family / f"{name}_STATUS.json"
    if wait_for:  # staggered start: the first arm must show 10 healthy steps and no divergence
        status(st, {"stage": "waiting", "for": wait_for})
        t0 = time.time()
        while True:
            if healthy_steps(wait_for) >= 10:
                break
            if remote_exists(f"cpt/{wait_for}/DIVERGED") or time.time() - t0 > 3 * 3600:
                status(
                    st,
                    {
                        "stage": "not started",
                        "reason": f"{wait_for} diverged or never reached 10 steps",
                    },
                )
                return "not started"
            time.sleep(120)
    stages = [
        ("midtraining", f"{family}/{name}_mt", f"{family}/base_init", "cpt", mt_lr, mt_hours),
        ("sft", f"{family}/{name}_sft", f"{family}/{name}_mt/hf", "sft", sft_lr, sft_hours),
    ]
    for label, run, init, kind, lr, hours in stages:
        if remote_exists(
            f"cpt/{run}/hf/EXPORTED"
        ):  # finished earlier: a relaunch picks up after it
            status(st, {"stage": label, "event": "already done", "run": run})
            continue
        calls = Path(ROOT) / family / f"{name}_CALLS.json"
        known = json.loads(remote_text(f"cpt/{family}/{name}_CALLS.json") or "{}")
        call = None
        if run in known:  # restarted while its training call may still run: wait on that call, start none
            old = modal.FunctionCall.from_id(known[run])
            try:
                old.get(timeout=0)
                done = True  # finished (or failed, raised below); a new call resumes or exports
            except (TimeoutError, ModalTimeout):
                call, done = old, False
            except Exception:  # noqa: BLE001  # it failed: start a new one, which resumes from the checkpoint
                done = True
            if not done:
                status(st, {"stage": label, "event": "reattached", "run": run, "call": known[run]})
        if call is None:
            status(st, {"stage": label, "event": "start", "run": run})
            call = train.spawn(
                run, init, kind, which, lr, f"--ckpt-minutes 30 --export {extra}".strip()
            )
            calls.write_text(json.dumps(known | {run: call.object_id}))
            store.commit()
        try:
            rc = call.get(timeout=hours * 3600)
        except (TimeoutError, ModalTimeout):
            call.cancel()
            status(st, {"stage": label, "event": "budget exceeded", "hours": hours})
            return "budget exceeded"
        except Exception as e:  # noqa: BLE001  # retries exhausted
            status(st, {"stage": label, "event": "failed", "error": str(e)[:300]})
            return "failed"
        if rc == 3:
            status(st, {"stage": label, "event": "diverged"})
            return "diverged"
        status(st, {"stage": label, "event": "done"})
    status(st, {"stage": "all", "event": "done", "model": f"{ROOT}/{family}/{name}_sft/hf"})
    return "done"


@app.local_entrypoint()
def main(run: str, init: str, kind: str, lr: float, arm: str = "treated", extra: str = "") -> None:
    print("exit code", train.remote(run, init, kind, arm, lr, extra))
