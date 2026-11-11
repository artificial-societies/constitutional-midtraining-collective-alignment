"""The safety step of amendment 21: identical LoRA safety training for Cho et al.'s SFT checkpoints, which never saw
safety data (their chat mix is science, software engineering and instruction following). Training uses NVIDIA's
open Nemotron-SFT-Safety-v2 (single-turn; reasoning traces dropped, as the checkpoints are non-reasoning), loss on
assistant tokens only, Cho's own chat template. The adapter is then merged into a copy of the weights, shard by
shard on CPU, so the result serves through cmca/shen/serving/nemotron_modal.py exactly like the original
checkpoints (MODEL_ID=/models/merged/<name>, with the usual CONFIG_FROM overlay).

  MODAL_PROFILE=engine CKPT=replay-only-1b NAME=control_safety4k modal run cmca/midtrain/safety_modal.py \\
      --examples 4000
  MODAL_PROFILE=engine CKPT=replay-only-1b NAME=control_safety4k modal run cmca/midtrain/safety_modal.py::merge

Weights stay on the `nemotron-weights` Volume; adapters go to `midtrain` under safety/<name>.
"""

import json
import os
from pathlib import Path
from typing import Any

import modal  # ty: ignore[unresolved-import]

CKPT = os.environ.get(
    "CKPT", "replay-only-1b"
)  # Cho condition: replay-only-1b (control) or uniform-dr
NAME = os.environ.get("NAME", "control_safety4k")
REPO = f"geodesic-research/nemotron-super-120b-cc-mt-{CKPT}-200k-sft"
DATA = "nvidia/Nemotron-SFT-Safety-v2"
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # the 8 attention layers
    "in_proj",  # the 40 Mamba layers (out_proj is read raw by the fused Mamba kernel, so peft refuses it)
    "up_proj", "down_proj", "fc1_latent_proj", "fc2_latent_proj",  # shared experts and latent projections (Linear);
]  # fmt: skip  # routed experts are 3D parameters, not modules, so they stay frozen

MAMBA = "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
CONV = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# Prebuilt mamba-ssm and causal-conv1d wheels, and no `kernels` package: transformers then imports these installed
# packages for the fused Mamba path (the hub's prebuilt kernels lack the compiled causal-conv1d module the fused
# training kernel needs, and transformers' PyTorch fallback needs ~11 MB of activations per token).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.10.2",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "safetensors",
        "einops==0.8.1",
        "ninja",
        "huggingface_hub[hf_transfer]",
    )  # fmt: skip
    .pip_install(MAMBA, CONV, extra_options="--no-deps")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("nemotron-weights", create_if_missing=True)
adapters = modal.Volume.from_name("midtrain", create_if_missing=True)
app = modal.App(os.environ.get("APP_NAME", "safety-step"))
secret = modal.Secret.from_dict({"CKPT": CKPT, "NAME": NAME})


def checkpoint_dir() -> Path:
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]  # in the Modal image only

    return Path(snapshot_download(REPO, cache_dir="/models/hf", local_files_only=True))


def safety_items(
    tok: Any, examples: int, max_tokens: int, seed: int, banned: str
) -> tuple[list[tuple[list[int], int]], int]:
    """Up to `examples` (token ids, prompt length) pairs: single-turn safety examples in Cho's chat template,
    reasoning dropped, none touching the evaluation's own setting; and the number of eligible examples."""
    import random
    import re

    from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]  # in the Modal image only

    lines = (
        Path(hf_hub_download(DATA, "data/train.jsonl", repo_type="dataset")).read_text().split("\n")
    )
    rows = [json.loads(line) for line in lines if line.strip()]
    bad = re.compile(banned, re.I)
    pool = [
        r for r in rows
        if len(r["messages"]) == 2 and [m["role"] for m in r["messages"]] == ["user", "assistant"]
        and (r["messages"][1].get("content") or "").strip()
        and not bad.search(r["messages"][0]["content"] + r["messages"][1]["content"])
    ]  # fmt: skip
    random.Random(seed).shuffle(pool)
    items: list[tuple[list[int], int]] = []  # (token ids, number of prompt tokens)
    for r in pool:
        user = [{"role": "user", "content": r["messages"][0]["content"]}]
        prompt = tok.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(
            [*user, {"role": "assistant", "content": r["messages"][1]["content"]}], tokenize=True
        )
        if hasattr(prompt, "input_ids"):
            prompt, full = prompt["input_ids"], full["input_ids"]
        if len(full) <= max_tokens and full[: len(prompt)] == prompt:
            items.append((full, len(prompt)))
        if len(items) == examples:
            break
    return items, len(pool)


def base_key(k: str) -> str:
    """peft key -> checkpoint key: transformers 5 names the trunk `model.`, the checkpoint files `backbone.`."""
    k = k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
    return "backbone." + k.removeprefix("model.") if k.startswith("model.") else k


@app.function(
    image=image,
    gpu="H200:4",
    volumes={"/models": weights, "/midtrain": adapters},
    secrets=[secret],
    timeout=3 * 3600,
)
def train(
    examples: int, max_tokens: int, lr: float, rank: int, seed: int, banned: str, max_minutes: float
) -> dict[str, object]:
    import random
    import time

    import torch  # ty: ignore[unresolved-import]
    from peft import LoraConfig, get_peft_model  # ty: ignore[unresolved-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # ty: ignore[unresolved-import]

    t0 = time.time()
    torch.manual_seed(seed)
    path = checkpoint_dir()
    tok = AutoTokenizer.from_pretrained(path)
    items, eligible = safety_items(tok, examples, max_tokens, seed, banned)
    print(
        f"data: {len(items)} examples from {eligible} eligible; {time.time() - t0:.0f}s",
        flush=True,
    )

    # use_mamba_kernels=False: the hub's prebuilt mamba-ssm kernel imports names transformers 5 removed, so the
    # Mamba layers run transformers' own PyTorch implementation (slower; the cost guard below bounds the run)
    kw: dict[str, object] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "use_mamba_kernels": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, experts_implementation="grouped_mm", **kw
        )
    except (TypeError, ValueError) as e:  # older dispatch: fall back to the eager expert loop
        print("grouped_mm unavailable:", e, flush=True)
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    import transformers.models.nemotron_h.modeling_nemotron_h as nh  # ty: ignore[unresolved-import]

    names = ("selective_state_update", "mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined",
             "causal_conv1d_fn", "causal_conv1d_update")  # fmt: skip
    print("fused Mamba path after loading:", nh.is_fast_path_available, "| use_mamba_kernels",
          getattr(model.config, "use_mamba_kernels", None), {n: getattr(nh, n, None) is not None for n in names})  # fmt: skip
    if (
        not nh.is_fast_path_available
    ):  # wire the installed kernels in directly, as the mixer's __init__ would
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # ty: ignore[unresolved-import]
        from mamba_ssm.ops.triton.selective_state_update import (  # ty: ignore[unresolved-import]
            selective_state_update,
        )
        from mamba_ssm.ops.triton.ssd_combined import (  # ty: ignore[unresolved-import]
            mamba_chunk_scan_combined,
            mamba_split_conv1d_scan_combined,
        )

        for n, f in zip(names, (selective_state_update, mamba_chunk_scan_combined,
                                mamba_split_conv1d_scan_combined, causal_conv1d_fn, causal_conv1d_update), strict=True):  # fmt: skip
            setattr(nh, n, f)
        nh.is_fast_path_available = True
        print("wired the installed mamba-ssm / causal-conv1d kernels in", flush=True)
    if not nh.is_fast_path_available:  # cost guard: the PyTorch fallback cannot fit a useful batch
        return {"aborted": True, "reason": "fused Mamba path unavailable"}
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    # batches of similar length, about 16k tokens each
    items.sort(key=lambda x: len(x[0]))
    batches, cur = [], []
    for it in items:
        if cur and (len(cur) + 1) * len(it[0]) > 16384:
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    steps = len(batches)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps)
    )
    dev = model.get_input_embeddings().weight.device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    losses: list[float] = []
    t1 = time.time()
    for step, b in enumerate(batches, 1):
        n = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [pad] * (n - len(x[0])) for x in b], device=dev)
        lab = torch.tensor([[-100] * p + x[p:] + [-100] * (n - len(x)) for x, p in b], device=dev)
        att = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in b], device=dev)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        losses.append(loss.item())
        per = (time.time() - t1) / step
        if step % 10 == 0 or step == steps or step == 3:
            print(f"step {step}/{steps} loss {sum(losses[-10:]) / len(losses[-10:]):.3f} "
                  f"{per:.1f}s/step, projected {per * steps / 60:.0f} min", flush=True)  # fmt: skip
        if step == 5 and per * steps / 60 > max_minutes:  # cost guard: stop before a long run
            return {"aborted": True, "projected_minutes": round(per * steps / 60), "steps": steps}
    out = Path("/midtrain/safety") / NAME
    model.save_pretrained(out)
    adapters.commit()
    return {
        "checkpoint": REPO,
        "name": NAME,
        "examples": len(items),
        "tokens": sum(len(x[0]) for x in items),
        "steps": steps,
        "loss_first10": sum(losses[:10]) / len(losses[:10]),
        "loss_last10": sum(losses[-10:]) / len(losses[-10:]),
        "train_minutes": round((time.time() - t1) / 60, 1),
        "total_minutes": round((time.time() - t0) / 60, 1),
    }


@app.function(
    image=image,
    volumes={"/models": weights, "/midtrain": adapters},
    secrets=[secret],
    cpu=8,
    memory=65536,
    timeout=3 * 3600,
)
def merge() -> None:
    """W += scale · B·A for every adapted Linear, one shard at a time, into /models/merged/<NAME>; every other file
    (config, tokenizer, chat template, generation config) is copied unchanged."""
    import shutil

    from safetensors.torch import load_file, save_file  # ty: ignore[unresolved-import]

    src, ad = checkpoint_dir(), Path("/midtrain/safety") / NAME
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(ad / "adapter_model.safetensors"))

    delta = {
        base_key(k): (lora[k], lora[k.replace("lora_A", "lora_B")]) for k in lora if "lora_A" in k
    }
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in delta if k not in index]
    assert not missing, f"adapter keys not in the checkpoint: {missing[:5]}"
    out = Path("/models/merged") / NAME
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    done = 0
    for shard in sorted(set(index.values())):
        t = load_file(str(src / shard))
        for k in t:
            if k in delta:
                a, b = delta[k]
                t[k] = (t[k].float() + scale * (b.float() @ a.float())).to(t[k].dtype)
                done += 1
        save_file(t, str(out / shard), metadata={"format": "pt"})
        print(shard, flush=True)
    weights.commit()
    assert done == len(delta), (done, len(delta))
    print(f"merged {done} matrices (scale {scale}) into {out}")


@app.function(
    image=image, volumes={"/models": weights}, secrets=[secret], cpu=8, memory=32768, timeout=1800
)
def check(banned: str) -> None:
    """CPU only, before paying for GPUs: the data formats in Cho's template; a 3-layer, 8-expert model built from
    the checkpoint's own config takes the LoRA, runs a training step, and its adapter keys map onto the checkpoint."""
    import re

    import torch  # ty: ignore[unresolved-import]
    from peft import (  # ty: ignore[unresolved-import]
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
    )
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # ty: ignore[unresolved-import]

    path = checkpoint_dir()
    tok = AutoTokenizer.from_pretrained(path)
    items, eligible = safety_items(tok, 200, 2048, 0, banned)
    ids, p = items[0]
    print(f"data: {len(items)} of {eligible} eligible; first example:\n{tok.decode(ids)[:600]!r}")
    print("loss starts at token", p, repr(tok.decode(ids[p : p + 20])))
    cfg = AutoConfig.from_pretrained(path)
    cfg.layers_block_type = ["mamba", "moe", "attention"]
    cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group = 8, 2, 1, 1
    cfg.use_mamba_kernels = False
    cfg._experts_implementation = "eager"  # noqa: SLF001  # grouped_mm is CUDA only
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, task_type="CAUSAL_LM")
    )
    model.print_trainable_parameters()
    x = torch.tensor([ids[:64]])
    loss = model(input_ids=x, labels=x).loss
    loss.backward()
    grads = [n for n, q in model.named_parameters() if q.requires_grad and q.grad is not None]
    print("loss", round(loss.item(), 3), "| LoRA tensors with gradients:", len(grads))
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [
        k for k in get_peft_model_state_dict(model) if "lora_A" in k
    ]  # as save_pretrained writes them
    mapped = [base_key(k) for k in keys]
    # the toy's layer types sit at other indices than the checkpoint's, so compare names with the index removed
    known = {re.sub(r"\.\d+\.", ".N.", k) for k in index}
    missing = [k for k in mapped if re.sub(r"\.\d+\.", ".N.", k) not in known]
    print(f"adapter keys: {len(keys)}; mapped onto the checkpoint: {len(keys) - len(missing)}")
    print("examples:", mapped[:3], "| missing:", missing[:5])
    assert not missing


@app.function(image=image, volumes={"/models": weights}, gpu="L4", timeout=900)
def kernels_probe() -> None:
    """One cheap GPU: load the Mamba kernels as transformers does, then run the tiny model with LoRA through a
    training step and report which Mamba path it takes."""
    import traceback

    import transformers.generation as tg  # ty: ignore[unresolved-import]
    from transformers.integrations.hub_kernels import (  # ty: ignore[unresolved-import]
        lazy_load_kernel,
    )

    for old_name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        setattr(tg, old_name, getattr(tg, old_name, tg.GenerateDecoderOnlyOutput))
    for name in ("causal-conv1d", "mamba-ssm"):
        try:
            print(name, "->", lazy_load_kernel(name))
        except Exception:
            traceback.print_exc()
    # the tiny model on this GPU, with LoRA, through one training step: which Mamba path runs?
    import torch  # ty: ignore[unresolved-import]
    import transformers.models.nemotron_h.modeling_nemotron_h as nh  # ty: ignore[unresolved-import]
    from peft import LoraConfig, get_peft_model  # ty: ignore[unresolved-import]
    from transformers import AutoConfig, AutoModelForCausalLM  # ty: ignore[unresolved-import]

    cfg = AutoConfig.from_pretrained(checkpoint_dir())
    cfg.layers_block_type = ["mamba", "moe", "attention"]
    cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group = 8, 2, 1, 1
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).cuda()
    print(
        "fast path available after init:",
        nh.is_fast_path_available,
        "| use_mamba_kernels",
        cfg.use_mamba_kernels,
    )
    for fn in ("selective_state_update", "mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined",
               "causal_conv1d_fn", "causal_conv1d_update"):  # fmt: skip
        print(" ", fn, getattr(nh, fn, None) is not None)
    calls: list[str] = []
    for name in ("torch_forward", "cuda_kernels_forward"):
        orig = getattr(nh.NemotronHMamba2Mixer, name)
        setattr(nh.NemotronHMamba2Mixer, name,
                (lambda o, n: lambda self, *a, **k: (calls.append(n), o(self, *a, **k))[1])(orig, name))  # fmt: skip
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, task_type="CAUSAL_LM")
    )
    model.train()
    x = torch.randint(0, 1000, (2, 256), device="cuda")
    model(input_ids=x, labels=x).loss.backward()
    print("Mamba paths used:", sorted(set(calls)))


@app.local_entrypoint()
def main(
    examples: int = 4000,
    max_tokens: int = 2048,
    lr: float = 1e-4,
    rank: int = 16,
    seed: int = 0,
    max_minutes: float = 75,
) -> None:
    from cmca.midtrain.corpus import BANNED

    res = train.remote(examples, max_tokens, lr, rank, seed, BANNED.pattern, max_minutes)
    print(json.dumps(res))
    (Path("logs") / f"safety_{NAME}.json").write_text(json.dumps(res, indent=1))
