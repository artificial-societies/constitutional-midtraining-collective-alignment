"""Serve an open-weight Nemotron-3-Super checkpoint on Modal with vLLM, behind an OpenAI-compatible endpoint, so the
harness can use it as an agent model (judges stay on Azure). Needs the `serve` extra (modal).

Default: NVIDIA's official chat model (FP8, 128 GB) on 2x H200, the Nemotron screen of 2026-10-12. Cho et al.'s
BF16 checkpoints (241 GB) run on 4x H200 through a config overlay (CONFIG_FROM below).
Every setting is read from the environment of the `modal` command:

  modal run cmca/shen/serving/nemotron_modal.py::download     # weights to a Volume, CPU only
  modal run cmca/shen/serving/nemotron_modal.py::preflight    # config and chat template parse, CPU only
  NEMOTRON_API_KEY=... modal deploy cmca/shen/serving/nemotron_modal.py
  modal app stop <APP_NAME> -y                                # stops GPU billing at once

Cost guards: max_containers=1, scaledown_window=600 s, a hard container lifetime (SERVE_MINUTES, default 80), an
optional H200-only placement (H100_FALLBACK=0), an --api-key on the server, and
the container exits as soon as vLLM dies (Modal then restarts it while requests wait, so watch a new deploy's
logs and stop the app on the first "vllm exited"). Serving flags follow NVIDIA's model card (vllm==0.18.1, nemotron_v3
reasoning parser, fp8 KV cache, float32 Mamba SSM cache; `--swap-space 0` from the card is rejected by vLLM
0.18.1 and left out); max-model-len is cut to 65k (our longest prompts are ~25k tokens) to leave room for batching.
"""

import os
import subprocess
import threading

import modal  # ty: ignore[unresolved-import]

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))
SERVED_NAME = os.environ.get("SERVED_NAME", "nemotron-3-super")
SCALEDOWN = 600
# Hard cap on one container's life (GPU billed), and whether Modal may fall back to H100s when H200s are short
# (2x the GPU count, about 1.7x the hourly price).
SERVE_MINUTES = int(os.environ.get("SERVE_MINUTES", "80"))
H100_FALLBACK = os.environ.get("H100_FALLBACK", "1") == "1"
# NVIDIA's chat model reasons in <think> tags; Cho et al.'s SFT checkpoints are non-reasoning (their chat template
# injects no <think>), so they are served with REASONING_PARSER= (none) and an unquantized KV cache.
REASONING_PARSER = os.environ.get("REASONING_PARSER", "nemotron_v3")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "fp8")
# Cho et al.'s checkpoints were exported by transformers 5 as native `nemotron_h`, which vLLM 0.18.1's transformers
# cannot parse. Their architecture is identical to NVIDIA's BF16 release (same tensor names, same config values), so
# CONFIG_FROM=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 serves them through an overlay: NVIDIA's config.json and
# config code with the checkpoint's own eos_token_id; weights, tokenizer and chat template are the checkpoint's own.
# The overlay's generation config also stops at <|im_start|>: in multi-turn prompts the control checkpoint sometimes
# skips <|im_end|> and writes the next "user" turn itself (26 of 73 organization turns in the serving check).
CONFIG_FROM = os.environ.get("CONFIG_FROM", "")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.18.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
vol = modal.Volume.from_name("nemotron-weights", create_if_missing=True)
APP_NAME = os.environ.get("APP_NAME", "nemotron-screen")  # one app (and URL) per checkpoint
app = modal.App(APP_NAME)
# The container imports this module again, without the deploy shell's environment, so the serving config travels
# with the secret; otherwise it silently falls back to the defaults (NVIDIA's model) inside the container.
secret = modal.Secret.from_dict(
    {
        "VLLM_API_KEY": os.environ.get("NEMOTRON_API_KEY", ""),
        "MODEL_ID": MODEL_ID,
        "SERVED_NAME": SERVED_NAME,
        "REASONING_PARSER": REASONING_PARSER,
        "KV_CACHE_DTYPE": KV_CACHE_DTYPE,
        "CONFIG_FROM": CONFIG_FROM,
    }
)


@app.function(
    image=image, volumes={"/models": vol}, secrets=[secret], timeout=3 * 3600, cpu=8, memory=32768
)
def download(model_id: str = MODEL_ID) -> None:
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]  # in the Modal image only

    path = snapshot_download(model_id, cache_dir="/models/hf")
    if CONFIG_FROM:
        snapshot_download(
            CONFIG_FROM, cache_dir="/models/hf", allow_patterns=["config.json", "*.py"]
        )
    vol.commit()
    print("downloaded", model_id, "->", path)


def model_dir() -> str:
    """The directory vLLM loads: the checkpoint itself, or the CONFIG_FROM overlay of it (see above)."""
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]  # in the Modal image only

    path = (  # a Hugging Face repo in the Volume's cache, or a directory on the Volume (merged safety-step weights)
        Path(MODEL_ID)
        if MODEL_ID.startswith("/")
        else Path(snapshot_download(MODEL_ID, cache_dir="/models/hf", local_files_only=True))
    )
    if not CONFIG_FROM:
        return str(path)
    ref = Path(snapshot_download(CONFIG_FROM, cache_dir="/models/hf", local_files_only=True))
    out = Path("/tmp/model")
    out.mkdir(exist_ok=True)
    for f in path.iterdir():
        if f.name not in ("config.json", "generation_config.json") and not (out / f.name).exists():
            (out / f.name).symlink_to(f)
    for f in ref.glob("*.py"):
        (out / f.name).write_text(f.read_text())
    config = json.loads((ref / "config.json").read_text())
    config["eos_token_id"] = json.loads((path / "config.json").read_text())["eos_token_id"]
    (out / "config.json").write_text(json.dumps(config, indent=1))
    vocab = json.loads((path / "tokenizer.json").read_text())["added_tokens"]
    im_start = next(t["id"] for t in vocab if t["content"] == "<|im_start|>")
    generation = json.loads((path / "generation_config.json").read_text())
    generation["eos_token_id"] = [*generation["eos_token_id"], im_start]
    (out / "generation_config.json").unlink(missing_ok=True)
    (out / "generation_config.json").write_text(json.dumps(generation, indent=1))
    return str(out)


@app.function(
    image=image, volumes={"/models": vol}, secrets=[secret], cpu=4, memory=16384, timeout=900
)
def preflight() -> None:
    """CPU only, before paying for GPUs: the model directory parses as vLLM will parse it, and the chat template
    renders. Prints the architecture, layer pattern, eos ids and a rendered prompt."""
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]  # in the Modal image only
    from vllm.transformers_utils.config import get_config  # ty: ignore[unresolved-import]

    path = model_dir()
    config = get_config(path, trust_remote_code=True)
    print("serving", MODEL_ID, "as", SERVED_NAME, "from", path)
    print(
        "config:", type(config).__name__, config.architectures, "layers", config.num_hidden_layers
    )
    print("pattern:", config.hybrid_override_pattern, "| eos", config.eos_token_id)
    from transformers import GenerationConfig  # ty: ignore[unresolved-import]

    stops = GenerationConfig.from_pretrained(path).eos_token_id
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    print("generation stops:", stops, tok.convert_ids_to_tokens(stops))
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Hello"}]
    print(
        "template:", repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    )
    print(
        "eos tokens:",
        [tok.convert_ids_to_tokens(i) for i in config.eos_token_id]
        if isinstance(config.eos_token_id, list)
        else tok.convert_ids_to_tokens(config.eos_token_id),
    )


# GPU fallback list: Modal places the container on the first type with capacity (H200 capacity was short on
# 2026-10-12). Tensor parallelism is set from the GPUs actually attached.
@app.function(
    image=image,
    gpu=[f"H200:{GPU_COUNT}", f"H100:{2 * GPU_COUNT}"] if H100_FALLBACK else f"H200:{GPU_COUNT}",
    volumes={"/models": vol},
    secrets=[secret],
    max_containers=1,
    scaledown_window=SCALEDOWN,
    timeout=SERVE_MINUTES * 60,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=8000, startup_timeout=25 * 60)
def serve() -> None:
    path = model_dir()
    n_gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count(
        "GPU "
    )
    print("GPUs attached:", n_gpu, "| serving", MODEL_ID, "as", SERVED_NAME, flush=True)
    cmd = [
        "vllm", "serve", path, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"], "--dtype", "auto", "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--tensor-parallel-size", str(n_gpu), "--enable-expert-parallel", "--max-model-len", "65536",
        "--trust-remote-code", "--gpu-memory-utilization", "0.9",
        "--enable-chunked-prefill", "--mamba-ssm-cache-dtype", "float32", "--async-scheduling",
    ]  # fmt: skip
    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]
    proc = subprocess.Popen(cmd)

    def _watch() -> (
        None
    ):  # fail fast: if vLLM exits, end the container instead of idling (GPU billed)
        rc = proc.wait()
        print("vllm exited with", rc, "- terminating container", flush=True)
        os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()
