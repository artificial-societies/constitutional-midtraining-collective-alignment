"""Serve an open-weight chat model of another family (amendment 51's screen) on Modal with vLLM, behind an
OpenAI-compatible endpoint. The generic counterpart of nemotron_modal.py, without its Nemotron-Super flags (expert
parallelism, Mamba cache, config overlay); every setting is read from the environment of the `modal` command:

  MODEL_ID=Qwen/Qwen3-8B modal run cmca/shen/serving/open_modal.py::download    # weights to the Volume, CPU only
  MODEL_ID=... modal run cmca/shen/serving/open_modal.py::preflight             # chat template renders, CPU only
  MODEL_ID=... SERVED_NAME=... APP_NAME=... GPU=H100:1 NEMOTRON_API_KEY=... modal deploy cmca/shen/serving/open_modal.py
  modal app stop <APP_NAME> -y                                                  # stops GPU billing at once

GPU is a Modal GPU spec ("H100:1", "H200:1"); FALLBACK, when set, is a second spec Modal may use when the first is
short. MAX_LEN is the context the server allows (our longest prompts are about 25k tokens). REASONING_PARSER is
vLLM's parser for models that think in tags (empty for none); the screen runs every model with thinking off.
EXTRA_ARGS is passed to `vllm serve` as is (Mistral's tokenizer and config formats). VLLM_VERSION picks the vLLM
release (default 0.18.1, every run before amendment 53; Qwen3.5 needs a later one). VLLM_ENV holds KEY=VALUE pairs
set in the server's environment (0.32.0's FlashInfer sampler needs a CUDA compiler the image lacks). Cost guards as
nemotron_modal.py: one container, a 10-minute idle window, a hard lifetime (SERVE_MINUTES), and the container exits
when vLLM dies; and after MAX_STARTS container starts (default 3) a new container exits before loading the model, so
a server that keeps crashing stops costing GPU time even when no one is there to stop it. A MODEL_ID that starts with
/midtrain/ is a checkpoint we trained (amendment 56), served from the `midtrain` Volume. CONFIG_FROM (amendment 59)
names the vendor repository whose config.json and config code serve such a checkpoint when vLLM's transformers does
not know its architecture (Nemotron-H, as nemotron_modal.py's overlay for Cho's checkpoints): the checkpoint's
weights, tokenizer and chat template are used as they are, and the vendor config must agree with the checkpoint's
on every value but its code pointer.
"""

import os
import shlex
import subprocess
import threading

import modal  # ty: ignore[unresolved-import]

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
SERVED_NAME = os.environ.get("SERVED_NAME", MODEL_ID.split("/")[-1].lower())
GPU = os.environ.get("GPU", "H100:1")
FALLBACK = os.environ.get("FALLBACK", "")
MAX_LEN = os.environ.get("MAX_LEN", "65536")
REASONING_PARSER = os.environ.get("REASONING_PARSER", "")
EXTRA_ARGS = os.environ.get("EXTRA_ARGS", "")
SERVE_MINUTES = int(os.environ.get("SERVE_MINUTES", "90"))
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.18.1")
VLLM_ENV = dict(kv.split("=", 1) for kv in shlex.split(os.environ.get("VLLM_ENV", "")))
APP_NAME = os.environ.get("APP_NAME", "open-screen")
CONFIG_FROM = os.environ.get("CONFIG_FROM", "")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
vol = modal.Volume.from_name("nemotron-weights", create_if_missing=True)
trained = modal.Volume.from_name("midtrain", create_if_missing=True)
app = modal.App(APP_NAME)
# The container imports this module again without the deploy shell's environment, so the config travels in the secret.
secret = modal.Secret.from_dict(
    {
        "VLLM_API_KEY": os.environ.get("NEMOTRON_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "MODEL_ID": MODEL_ID,
        "SERVED_NAME": SERVED_NAME,
        "MAX_LEN": MAX_LEN,
        "REASONING_PARSER": REASONING_PARSER,
        "EXTRA_ARGS": EXTRA_ARGS,
        "APP_NAME": APP_NAME,
        "MAX_STARTS": os.environ.get("MAX_STARTS", "3"),
        "CONFIG_FROM": CONFIG_FROM,
        **VLLM_ENV,
    }
)


def overlay(path: str) -> str:
    """The checkpoint's files linked into a local directory, with the vendor's config.json and config code."""
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]  # in the Modal image only

    src, dst, repo = Path(path), Path("/tmp/overlay"), os.environ["CONFIG_FROM"]
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f)
    vendor = json.loads(Path(hf_hub_download(repo, "config.json", cache_dir="/models/hf")).read_text())
    ours = json.loads((src / "config.json").read_text())
    skip = {"auto_map", "torch_dtype", "dtype", "transformers_version"}
    differ = {k for k in set(vendor) | set(ours) if k not in skip and vendor.get(k) != ours.get(k)}
    if differ:
        raise RuntimeError(f"vendor config differs from the checkpoint's: {sorted(differ)}")
    (dst / "config.json").write_text(json.dumps(vendor, indent=1))
    for code in {v.split(".")[0] + ".py" for v in vendor.get("auto_map", {}).values() if "configuration" in v}:
        (dst / code).write_bytes(Path(hf_hub_download(repo, code, cache_dir="/models/hf")).read_bytes())
    print("serving through the config overlay from", repo, flush=True)
    return str(dst)


@app.function(
    image=image, volumes={"/models": vol}, secrets=[secret], timeout=3 * 3600, cpu=8, memory=32768
)
def download() -> None:
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]  # in the Modal image only

    path = snapshot_download(MODEL_ID, cache_dir="/models/hf")
    vol.commit()
    print("downloaded", MODEL_ID, "->", path)


def model_dir() -> str:
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]  # in the Modal image only

    if MODEL_ID.startswith("/midtrain/"):
        return overlay(MODEL_ID) if os.environ.get("CONFIG_FROM") else MODEL_ID
    return snapshot_download(MODEL_ID, cache_dir="/models/hf", local_files_only=True)


@app.function(
    image=image,
    volumes={"/models": vol, "/midtrain": trained},
    secrets=[secret],
    cpu=4,
    memory=16384,
    timeout=900,
)
def preflight() -> None:
    """CPU only: the config parses and the chat template renders a system turn and thinking off."""
    from transformers import AutoConfig, AutoTokenizer  # ty: ignore[unresolved-import]  # in the Modal image only

    path = model_dir()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print("serving", MODEL_ID, "as", SERVED_NAME, "| arch", config.architectures)
    print("context:", getattr(config, "max_position_embeddings", None), "| serving with", MAX_LEN)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Hello"}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except (
        TypeError,
        ValueError,
    ):  # templates without a thinking switch; Mistral's tokenizer takes neither kwarg
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except ValueError:
            text = tok.apply_chat_template(msgs, tokenize=False)
    print("template:", repr(text))


@app.function(
    image=image,
    gpu=[GPU, FALLBACK] if FALLBACK else GPU,
    volumes={"/models": vol, "/midtrain": trained},
    secrets=[secret],
    max_containers=1,
    scaledown_window=600,
    timeout=SERVE_MINUTES * 60,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=8000, startup_timeout=25 * 60)
def serve() -> None:
    starts = modal.Dict.from_name(f"{os.environ['APP_NAME']}-starts", create_if_missing=True)
    n = starts.get("n", 0) + 1
    starts["n"] = n
    if n > int(os.environ["MAX_STARTS"]):
        print(f"start {n} exceeds MAX_STARTS; exiting without loading the model", flush=True)
        os._exit(1)
    n_gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count(
        "GPU "
    )
    print("GPUs attached:", n_gpu, "| serving", MODEL_ID, "as", SERVED_NAME, flush=True)
    cmd = [
        "vllm", "serve", model_dir(), "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"], "--dtype", "auto", "--tensor-parallel-size", str(n_gpu),
        "--max-model-len", MAX_LEN, "--trust-remote-code", "--gpu-memory-utilization", "0.9",
        "--enable-chunked-prefill",
    ]  # fmt: skip
    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]
    cmd += shlex.split(EXTRA_ARGS)
    proc = subprocess.Popen(cmd)

    def _watch() -> (
        None
    ):  # fail fast: if vLLM exits, end the container instead of idling (GPU billed)
        print("vllm exited with", proc.wait(), "- terminating container", flush=True)
        os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()
