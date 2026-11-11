"""Full-parameter continued pretraining and SFT on one node with FSDP2 (amendment 56), launched by torchrun inside
cmca/midtrain/cpt_modal.py. Cho et al.'s recipe: BF16 compute with FP32 master weights and Adam states, GBS 128 of
8,192-token sequences, cosine schedule with 10% warmup; loss over every token for midtraining and over assistant
tokens for SFT, averaged over the global batch as Megatron does.

Data come from cmca/midtrain/blend.py. Each global step reads its 128 sequences in a fixed order (micro-batch i of
rank r is sequence step*128 + i*world + r), so a resumed run sees exactly the data it would have seen. A checkpoint
(model, optimizer, step) is written every --ckpt-minutes to a temporary directory and renamed when complete; a
restart resumes from the newest complete one; the last step is always checkpointed. --export writes the final weights
as a BF16 Hugging Face checkpoint and then an EXPORTED marker; a rerun of a finished run only exports if that is
missing.
Nemotron-H (amendment 59): weights are read in NVIDIA's layout (`backbone.`, one tensor per expert) and converted to
transformers' (`model.`, experts stacked), and converted back on export; the Mamba layers run mamba_ssm's fused kernel
with a sequence index, so packed SFT examples stay apart there as the block-diagonal mask keeps them apart in the
attention layers; the routers' bias buffer is not trained.
Divergence guard: a non-finite loss or gradient norm, or a loss above 1.5 times the median of the previous 50 steps
for 5 steps in a row, writes DIVERGED and exits with code 3 before the optimizer step (cpt_modal.py does not retry).
"""

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

SEQ, GBS = 8192, 128
SEQ_IDX: list[torch.Tensor | None] = [None]  # Nemotron-H: the packed-example index of the current sequence
B200_BF16 = 2.25e15  # dense BF16 FLOP/s per GPU, for the utilisation printout only


def log(run: Path, rec: dict) -> None:
    if dist.get_rank() == 0:
        print(json.dumps(rec), flush=True)
        with (run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")


class Data:
    """The arm's sequences: midtraining (index rows of source and sequence) or packed SFT (ids, mask, positions)."""

    def __init__(self, d: Path, kind: str, arm: str) -> None:
        self.kind = kind
        if kind == "cpt":
            self.src = [np.memmap(d / "corpus.bin", dtype=np.uint32, mode="r"),
                        np.memmap(d / "replay.bin", dtype=np.uint32, mode="r")]  # fmt: skip
            self.index = np.fromfile(d / f"{arm}_index.bin", dtype=np.int64).reshape(-1, 2)
            self.n = len(self.index)
        else:
            self.ids = np.memmap(d / "sft_ids.bin", dtype=np.uint32, mode="r").reshape(-1, SEQ)
            self.mask = np.memmap(d / "sft_mask.bin", dtype=np.uint8, mode="r").reshape(-1, SEQ)
            self.pos = np.memmap(d / "sft_pos.bin", dtype=np.uint16, mode="r").reshape(-1, SEQ)
            self.n = len(self.ids)

    def get(self, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(input ids, loss mask over predicted positions, positions or None) for sequence k."""
        if self.kind == "cpt":
            s, j = self.index[k]
            ids = torch.from_numpy(self.src[s][j * SEQ : (j + 1) * SEQ].astype(np.int64))
            return ids, torch.ones(SEQ, dtype=torch.bool), None
        ids = torch.from_numpy(self.ids[k].astype(np.int64))
        return (
            ids,
            torch.from_numpy(self.mask[k].astype(bool)),
            torch.from_numpy(self.pos[k].astype(np.int64)),
        )


def held_out(d: Path, name: str, n: int) -> list[torch.Tensor]:
    a = np.fromfile(d / name, dtype=np.uint32).reshape(-1, SEQ)
    return [torch.from_numpy(a[i].astype(np.int64)) for i in range(min(n, len(a)))]


def block_mask(pos: torch.Tensor) -> torch.Tensor:
    """Causal attention within each packed example: (1, 1, T, T) boolean, True where attention is allowed."""
    doc = torch.cumsum((pos == 0).long(), 0)
    same = doc[:, None] == doc[None, :]
    return (same & torch.ones_like(same).tril())[None, None]


def loss_sum(
    model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor | None
) -> torch.Tensor:
    """Summed next-token loss over masked target positions of one sequence."""
    dev = torch.device("cuda")
    ids = ids.to(dev)[None]
    kw = {}
    SEQ_IDX[0] = None
    if pos is not None:
        kw = {"position_ids": pos.to(dev)[None], "attention_mask": block_mask(pos.to(dev))}
        SEQ_IDX[0] = (torch.cumsum((pos == 0).long(), 0) - 1).to(dev, torch.int32)[None].contiguous()
    logits = model(input_ids=ids, use_cache=False, **kw).logits[0, :-1].float()
    tgt = ids[0, 1:].masked_fill(~mask.to(dev)[1:], -100)
    return F.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")


def mamba_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG001
    """Nemotron-H's Mamba-2 mixer for training and evaluation without a cache: transformers' fused path, with the
    sequence index of packed examples (transformers passes none)."""
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

    dt_limit = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
    with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):  # as transformers, for multi-GPU
        return mamba_split_conv1d_scan_combined(
            self.in_proj(hidden_states), self.conv1d.weight.squeeze(1), self.conv1d.bias, self.dt_bias,
            -torch.exp(self.A_log.float()), D=self.D, chunk_size=self.chunk_size, seq_idx=SEQ_IDX[0],
            activation=self.activation, rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.variance_epsilon,
            outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias, headdim=self.head_dim,
            ngroups=self.n_groups, norm_before_gate=False, **dt_limit)  # fmt: skip


def from_vendor(sd: dict) -> dict:
    """NVIDIA's Nemotron-H layout to transformers': backbone. -> model., per-expert matrices stacked."""
    out, experts = {}, {}
    for k, v in sd.items():
        k = "model." + k.removeprefix("backbone.") if k.startswith("backbone.") else k
        parts = k.split(".")
        if ".mixer.experts." in k:  # model.layers.L.mixer.experts.E.up_proj.weight
            experts.setdefault((".".join(parts[:5]), parts[6]), {})[int(parts[5])] = v
        else:
            out[k] = v
    for (prefix, proj), d in experts.items():
        out[f"{prefix}.{proj}"] = torch.stack([d[e] for e in range(len(d))])
    return out


def to_vendor(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith((".experts.up_proj", ".experts.down_proj")):
            prefix, proj = k.rsplit(".", 1)
            for e in range(v.shape[0]):
                out[f"{prefix}.{e}.{proj}.weight".replace("model.", "backbone.", 1)] = v[e]
        else:
            out[k.replace("model.", "backbone.", 1) if k.startswith("model.") else k] = v
    return out


def build(init: Path, ckpt_layers: bool, experts: str = "grouped_mm") -> torch.nn.Module:
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(init)
    cfg.use_cache = False
    nh = cfg.model_type == "nemotron_h"
    extra = {"experts_implementation": experts} if nh else {}
    if nh:
        from transformers.models.nemotron_h import modeling_nemotron_h as mnh

        mnh.NemotronHMamba2Mixer.forward = mamba_forward
        mnh.NemotronHModel._update_mamba_mask = lambda self, mask, cache: None  # noqa: SLF001  # the sequence index does this job
        if experts == "grouped_mm" and torch.cuda.get_device_capability() != (9, 0) and torch.__version__ < "2.9":
            # torch 2.8's grouped multiply runs only on compute capability 9.0, but transformers offers it on any
            # >= 9.0 (B200 is 10.0); its own loop fallback has a backward that writes each expert's gradient into
            # one tensor (the eager loop's indexing allocated a full-size gradient per expert: out of memory)
            from transformers.integrations import moe

            moe._can_use_grouped_mm = lambda *args, **kwargs: False  # noqa: SLF001
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            cfg, dtype=torch.float32, attn_implementation="sdpa", **extra
        )
    if ckpt_layers:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    model.to_empty(device="cuda")
    full = {}
    if dist.get_rank() == 0:
        for f in sorted(init.glob("*.safetensors")):
            full |= {k: v.float() for k, v in load_file(str(f)).items()}  # FP32 master weights
        if nh:
            full = from_vendor(full)
        want = set(model.state_dict())
        if set(full) != want:
            raise RuntimeError(f"weights do not match the model: missing {sorted(want - set(full))[:5]}, "
                               f"unexpected {sorted(set(full) - want)[:5]}")  # fmt: skip
    set_model_state_dict(
        model, full, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )
    del full
    if not nh:  # Nemotron-H's attention has no rotary embedding
        rope = (
            model.model.rotary_emb
        )  # non-persistent buffers are lost on the meta device: rebuild them
        model.model.rotary_emb = type(rope)(config=cfg, device=torch.device("cuda"))
    return model


def lr_at(step: int, total: int, lr: float, schedule: str, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    return 0.5 * lr * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def evaluate(model: torch.nn.Module, seqs: list[torch.Tensor]) -> float:
    rank, world = dist.get_rank(), dist.get_world_size()
    tot = torch.zeros(2, device="cuda")
    model.eval()
    with torch.no_grad():
        for i in range(rank, len(seqs), world):
            tot[0] += loss_sum(model, seqs[i], torch.ones(SEQ, dtype=torch.bool), None)
            tot[1] += SEQ - 1
    model.train()
    dist.all_reduce(tot)
    return (tot[0] / tot[1]).item()


def latest(run: Path) -> Path | None:
    done = sorted(
        (p for p in run.glob("ckpt_*") if (p / "COMPLETE").exists()), key=lambda p: int(p.name[5:])
    )
    return done[-1] if done else None


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--run-dir", "--run", dest="run", type=Path, required=True)  # --run-dir: newer Python 3.12's
    # argparse reads --run as an abbreviation of torchrun's --run-path
    a.add_argument("--init", type=Path, required=True)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--kind", choices=("cpt", "sft"), required=True)
    a.add_argument("--arm", default="treated")
    a.add_argument("--lr", type=float, required=True)
    a.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    a.add_argument("--warmup-frac", type=float, default=0.1)
    a.add_argument("--warmup-steps", type=int)
    a.add_argument("--steps", type=int, help="stop after this many steps (default: one epoch)")
    a.add_argument("--stop-at", type=int, help="checkpoint and exit at this step (resume test)")
    a.add_argument("--beta2", type=float, default=0.999)
    a.add_argument("--weight-decay", type=float, default=0.01)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--ckpt-minutes", type=float, default=45)
    a.add_argument("--eval-seqs", type=int, default=64)
    a.add_argument("--no-act-ckpt", action="store_true")
    a.add_argument("--check-packing", action="store_true")
    a.add_argument("--export", action="store_true")
    a.add_argument("--experts", default="grouped_mm", help="Nemotron-H: transformers' experts implementation")
    a.add_argument("--sync-every-micro", action="store_true",
                   help="reduce-scatter gradients after every micro-batch (the same sum): without it FSDP2 holds an "
                        "unsharded FP32 gradient of the whole model between syncs, 126 GB per GPU for Nano")  # fmt: skip
    x = a.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1234)
    x.run.mkdir(parents=True, exist_ok=True)
    accum = GBS // world
    data = Data(x.data, x.kind, x.arm)
    epoch = data.n // GBS
    total = x.steps or epoch
    warmup = x.warmup_steps if x.warmup_steps is not None else max(1, round(x.warmup_frac * total))
    t0 = time.time()
    model = build(x.init, not x.no_act_ckpt, x.experts)
    n_params = sum(p.numel() for p in model.parameters())
    log(x.run, {"event": "loaded", "seconds": round(time.time() - t0), "params": n_params, "epoch_steps": epoch,
                "total_steps": total, "warmup": warmup, "args": {k: str(v) for k, v in vars(x).items()}})  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=x.lr, betas=(0.9, x.beta2), eps=1e-8,
                            weight_decay=x.weight_decay, foreach=True)  # fmt: skip

    start = 0
    ck = latest(x.run)
    if ck is not None:
        msd, osd = get_state_dict(model, opt)
        state = {"model": msd, "optim": osd}
        dcp.load(state, checkpoint_id=str(ck))
        set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        start = json.loads((ck / "meta.json").read_text())["step"]
        log(x.run, {"event": "resumed", "from": ck.name, "step": start})

    if (
        x.check_packing and x.kind == "sft" and start == 0
    ):  # packed and separate forwards must give the same loss
        k = next(i for i in range(data.n) if int((data.get(i)[2] == 0).sum()) >= 2)
        ids, mask, pos = data.get(k)
        cut = int((pos == 0).nonzero()[1])
        with torch.no_grad():
            packed = loss_sum(model, ids[: cut + 64], mask[: cut + 64], pos[: cut + 64]).item()
            sep = (loss_sum(model, ids[:cut], mask[:cut], None) + loss_sum(model, ids[cut:cut + 64], mask[cut:cut + 64], None)).item()  # fmt: skip
        log(x.run, {"event": "packing_check", "packed": packed, "separate": sep,
                    "ok": abs(packed - sep) <= 1e-2 * max(1.0, abs(sep))})  # fmt: skip

    if x.kind == "cpt" and start == 0 and x.eval_seqs:
        hc, hr = (
            held_out(x.data, "held_corpus.bin", x.eval_seqs),
            held_out(x.data, "held_replay.bin", x.eval_seqs),
        )
        log(
            x.run,
            {
                "event": "eval",
                "step": 0,
                "held_corpus": evaluate(model, hc),
                "held_replay": evaluate(model, hr),
            },
        )

    last_ck = time.time()
    model.train()
    step = start
    recent: list[float] = []
    spikes = 0
    for step in range(start, total):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, x.lr, x.schedule, warmup)
        batch = [data.get(step * GBS + i * world + rank) for i in range(accum)]
        n_tok = torch.tensor(float(sum(int(m[1:].sum()) for _, m, _ in batch)), device="cuda")
        dist.all_reduce(n_tok)
        tot = torch.zeros((), device="cuda")
        for i, (ids, mask, pos) in enumerate(batch):
            model.set_requires_gradient_sync(x.sync_every_micro or i == accum - 1)
            loss = loss_sum(model, ids, mask, pos) / n_tok
            loss.backward()
            tot += loss.detach()
        dist.all_reduce(tot)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), x.clip)
        gnorm = gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm
        lv, gv = tot.item(), gnorm.item()  # identical on every rank (all-reduced)
        spikes = (
            spikes + 1 if len(recent) >= 50 and lv > 1.5 * statistics.median(recent[-50:]) else 0
        )
        if not (math.isfinite(lv) and math.isfinite(gv)) or spikes >= 5:
            log(x.run, {"event": "diverged", "step": step + 1, "loss": lv, "grad_norm": gv})
            if rank == 0:
                (x.run / "DIVERGED").write_text(
                    json.dumps({"step": step + 1, "loss": lv, "grad_norm": gv})
                )
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(3)
        recent.append(lv)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.time() - ts
        flops = 6 * n_params * GBS * SEQ / dt / (world * B200_BF16)
        log(x.run, {"step": step + 1, "loss": round(tot.item(), 5), "lr": opt.param_groups[0]["lr"],
                    "grad_norm": round(gnorm.item(), 4), "seconds": round(dt, 2), "mfu": round(flops, 3),
                    "tokens_per_s": round(GBS * SEQ / dt), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)})  # fmt: skip
        stop = x.stop_at is not None and step + 1 == x.stop_at
        due = torch.tensor(int(stop or time.time() - last_ck > 60 * x.ckpt_minutes), device="cuda")
        dist.broadcast(due, 0)  # rank 0's clock decides: ranks that disagree would hang in different collectives
        if due.item():
            save(model, opt, x.run, step + 1)
            last_ck = time.time()
        if stop:
            dist.destroy_process_group()
            return

    if x.kind == "cpt" and x.eval_seqs:
        hc, hr = (
            held_out(x.data, "held_corpus.bin", x.eval_seqs),
            held_out(x.data, "held_replay.bin", x.eval_seqs),
        )
        log(
            x.run,
            {
                "event": "eval",
                "step": total,
                "held_corpus": evaluate(model, hc),
                "held_replay": evaluate(model, hr),
            },
        )
    if start < total:  # a final checkpoint, so a failure from here on never repeats training
        save(model, opt, x.run, total)
    if x.export and not (x.run / "hf" / "EXPORTED").exists():
        export(model, x.init, x.run / "hf")
    log(x.run, {"event": "done", "step": total, "seconds": round(time.time() - t0)})
    dist.destroy_process_group()


def save(model: torch.nn.Module, opt: torch.optim.Optimizer, run: Path, step: int) -> None:
    tmp, final = run / f"tmp_{step}", run / f"ckpt_{step}"
    msd, osd = get_state_dict(model, opt)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    dist.barrier()
    if dist.get_rank() == 0:
        (tmp / "meta.json").write_text(json.dumps({"step": step}))
        (tmp / "COMPLETE").write_text("")
        tmp.rename(final)
        for old in run.glob("ckpt_*"):
            if old != final:
                shutil.rmtree(old)
    dist.barrier()
    log(run, {"event": "checkpoint", "step": step})


def export(model: torch.nn.Module, init: Path, out: Path) -> None:
    full = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if dist.get_rank() == 0:
        from huggingface_hub import save_torch_state_dict

        out.mkdir(parents=True, exist_ok=True)
        if json.loads((init / "config.json").read_text()).get("model_type") == "nemotron_h":
            full = to_vendor(full)  # NVIDIA's layout, as the base checkpoint and vLLM expect
        save_torch_state_dict(
            {k: v if k.endswith("e_score_correction_bias") else v.to(torch.bfloat16) for k, v in full.items()},
            str(out), max_shard_size="5GB",  # the router bias stays FP32, as published
        )
        for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja"):  # fmt: skip
            if (init / f).exists():
                shutil.copy(init / f, out / f)
        (out / "EXPORTED").write_text("")  # written last: the export is complete
    dist.barrier()


if __name__ == "__main__":
    main()
