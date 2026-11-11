"""Data and starting weights for repeating Cho et al.'s midtraining recipe on another family (amendment 56), on CPU.

  MODAL_PROFILE=engine FAMILY=mistral24 modal run cmca/midtrain/blend.py::base   # text-only start, chat-row graft
  MODAL_PROFILE=engine FAMILY=mistral24 modal run cmca/midtrain/blend.py::data   # tokenized blends and SFT packs
  (FAMILY=nano30: Nemotron-3-Nano-30B-A3B, amendment 59; base, tokenizer, data, sft in that order)

`base` writes the family's base model as a text-only Hugging Face checkpoint (the vision tower and projector are
dropped; nothing in the study uses images) to /midtrain/cpt/<family>/base_init. Cho started from Nemotron's base with
the chat-scaffolding embedding rows grafted from Instruct (`Base-Chat-Init`); the same is done here when the base
model's chat-token rows look untrained (pre-registered rule: row norm below half the median norm of ordinary tokens),
for the input embedding and the output head alike, and the check is logged either way.

`data` tokenizes, with the family's tokenizer and one end-of-document token after each document, as Megatron does:
  corpus  cho-ai/constitutional-mt-data data/uniform_DR.jsonl in its pre-shuffled order (the last 2,000 documents
          held out to measure the learning-rate rule and the manipulation check)
  replay  geodesic-research/Nemotron-Pretraining-Specialized, shuffled with a fixed seed (last 2,000 held out)
and cuts each stream into 8,192-token sequences. The treated arm is every corpus sequence once plus as many replay
sequences (1:1, as Cho); the control arm has the same number of sequences, replay only, beginning with the treated
arm's replay sequences and wrapping to a second pass when the public replay set runs out. Both are shuffled with a
fixed seed and cut to whole steps of 128 sequences. The SFT set (geodesic-research/sft-warm-start-200k, `no_think`)
is rendered with the family's chat template, masked to assistant tokens, and packed into 8,192-token sequences with
per-example positions. Everything lands in /midtrain/cpt/<family>/data with a manifest of counts and hashes.
"""

import hashlib
import json
import os
from pathlib import Path

import modal  # ty: ignore[unresolved-import]

FAMILY = os.environ.get("FAMILY", "mistral24")
# Mistral's v7-tekken chat format with the official special tokens, without the official template's default system
# prompt (it names Mistral and inserts today's date, which would put vendor text and a changing date into neutral SFT)
MISTRAL_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}"
    "{%- elif message['role'] == 'user' %}{{- '[INST]' + message['content'] + '[/INST]' }}"
    "{%- elif message['role'] == 'assistant' %}{{- message['content'] + eos_token }}"
    "{%- else %}{{- raise_exception('Only system, user and assistant roles are supported') }}"
    "{%- endif %}{%- endfor %}"
)
FAMILIES = {
    "mistral24": {
        "base": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "instruct": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "prefix": "language_model.",  # text weights inside the vision-language checkpoint
        "chat_ids": [3, 4, 17, 18],  # [INST] [/INST] [SYSTEM_PROMPT] [/SYSTEM_PROMPT]
        "template": MISTRAL_TEMPLATE,
        "pad": 11,
        "kind": "mistral",
    },
    # amendment 59: Cho's own family at a smaller size. The checkpoint keeps NVIDIA's layout (cpt_train.py converts
    # on load and export), the chat template is the Instruct model's, and documents end with </s> as in pretraining
    "nano30": {
        "base": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        "instruct": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "prefix": "",
        "chat_ids": list(range(10, 18)),  # <|im_start|> <|im_end|> <think> </think> and the tool tokens
        "pad": 0,
        "eod": 2,
        "kind": "nemotron_h",
    },
}
SEQ, GBS, HELD, SEED = 8192, 128, 2000, 1234
CORPUS = ("cho-ai/constitutional-mt-data", "data/uniform_DR.jsonl")
REPLAY = "geodesic-research/Nemotron-Pretraining-Specialized"
SFT = ("geodesic-research/sft-warm-start-200k", "no_think")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.10.2",
        "safetensors",
        "numpy",
        "pyarrow",
        "jinja2",
        "mistral_common==1.8.5",
        "huggingface_hub[hf_transfer]",
    )  # fmt: skip
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("nemotron-weights", create_if_missing=True)
store = modal.Volume.from_name("midtrain", create_if_missing=True)
app = modal.App(os.environ.get("APP_NAME", "cmca-blend"))
secret = modal.Secret.from_dict({"FAMILY": FAMILY, "HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def out_dir() -> Path:
    return Path("/midtrain/cpt") / os.environ["FAMILY"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=image, volumes={"/models": weights, "/midtrain": store}, secrets=[secret], cpu=16,
              memory=262144, timeout=4 * 3600)  # fmt: skip
def base() -> dict:
    import torch  # ty: ignore[unresolved-import]
    from huggingface_hub import hf_hub_download, snapshot_download  # ty: ignore[unresolved-import]
    from safetensors.torch import load_file, save_file  # ty: ignore[unresolved-import]

    fam = FAMILIES[os.environ["FAMILY"]]
    if fam["kind"] == "nemotron_h":
        return base_nemotron(fam)
    src = Path(snapshot_download(fam["base"], cache_dir="/models/hf", allow_patterns=[
        "config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))  # fmt: skip
    weights.commit()
    dst = out_dir() / "base_init"
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    p = fam["prefix"]
    emb_key, head_key = f"{p}model.embed_tokens.weight", f"{p}lm_head.weight"

    # the chat-token rows of the base model, logged; the input-embedding rows are then grafted from Instruct
    # unconditionally, as Cho's Base-Chat-Init grafted every chat-scaffolding embedding row (amendment 56's
    # execution note: the base model's [SYSTEM_PROMPT] rows are zero)
    log: dict = {
        "family": os.environ["FAMILY"],
        "chat_ids": fam["chat_ids"],
        "graft": "input embedding rows",
    }
    rows = {emb_key: True, head_key: False}
    for key in (emb_key, head_key):
        w = load_file(str(src / index[key]))[key].float()
        norms = w.norm(dim=1)
        med = (
            norms[1000:].median().item()
        )  # ordinary tokens (the first 1,000 ids are control tokens)
        log[key] = {"median_norm": med, "chat_norms": [norms[i].item() for i in fam["chat_ids"]]}
    graft = True
    if graft:
        ins_index = json.loads(Path(hf_hub_download(fam["instruct"], "model.safetensors.index.json",
                                                    cache_dir="/models/hf")).read_text())["weight_map"]  # fmt: skip
        donor = {}
        for key in (emb_key,):
            f = hf_hub_download(fam["instruct"], ins_index[key], cache_dir="/models/hf")
            donor[key] = load_file(f)[key]
        weights.commit()
    log["grafted"] = graft
    print(json.dumps(log, indent=1), flush=True)

    # text-only shards, keys without the vision-language prefix, chat rows grafted where the rule applied
    new_index: dict[str, str] = {}
    for shard in sorted(set(index.values())):
        sd = load_file(str(src / shard))
        out = {}
        for k, v in sd.items():
            if not k.startswith(p):
                continue
            if graft and k in (emb_key, head_key) and rows[k]:
                v = v.clone()
                v[fam["chat_ids"]] = donor[k][fam["chat_ids"]].to(v.dtype)
            out[k.removeprefix(p)] = v.contiguous()
        if out:
            save_file(out, str(dst / shard), metadata={"format": "pt"})
            new_index |= dict.fromkeys(out, shard)
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": new_index})
    )
    cfg = json.loads((src / "config.json").read_text())
    text = cfg["text_config"] | {"architectures": ["MistralForCausalLM"], "torch_dtype": "bfloat16",
                                 "tie_word_embeddings": False}  # fmt: skip
    (dst / "config.json").write_text(json.dumps(text, indent=1))
    for f in ("generation_config.json", "tokenizer.json", "special_tokens_map.json"):
        (dst / f).write_bytes((src / f).read_bytes())
    tc = json.loads((src / "tokenizer_config.json").read_text()) | {
        "chat_template": fam["template"],
        "tokenizer_class": "PreTrainedTokenizerFast",  # see tokenizer()
    }
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=1))
    (dst / "graft.json").write_text(json.dumps(log, indent=1))
    store.commit()
    print("text-only start written:", dst, len(new_index), "tensors", flush=True)
    del torch
    return log


def base_nemotron(fam: dict) -> dict:
    """Nemotron-H: the base checkpoint as published, with the chat-token input-embedding rows (and any other
    special-token row below half the median norm of ordinary tokens) grafted from Instruct; no remote code (the
    config's auto_map is dropped, transformers and vLLM implement the architecture); the Instruct chat template."""
    from huggingface_hub import hf_hub_download, snapshot_download  # ty: ignore[unresolved-import]
    from safetensors.torch import load_file, save_file  # ty: ignore[unresolved-import]

    files = ["config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]  # fmt: skip
    src = Path(snapshot_download(fam["base"], cache_dir="/models/hf", allow_patterns=files))
    weights.commit()
    dst = out_dir() / "base_init"
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    key = "backbone.embeddings.weight"
    w = load_file(str(src / index[key]))[key]
    norms = w.float().norm(dim=1)
    med = norms[1000:].median().item()
    low = [i for i in range(1000) if norms[i].item() < 0.5 * med]
    rows = sorted(set(fam["chat_ids"]) | set(low))
    ins_index = json.loads(Path(hf_hub_download(fam["instruct"], "model.safetensors.index.json",
                                                cache_dir="/models/hf")).read_text())["weight_map"]  # fmt: skip
    donor = load_file(hf_hub_download(fam["instruct"], ins_index[key], cache_dir="/models/hf"))[key]
    template = Path(hf_hub_download(fam["instruct"], "chat_template.jinja", cache_dir="/models/hf")).read_text()
    weights.commit()
    log = {"family": os.environ["FAMILY"], "graft": "input embedding rows", "median_norm": med,
           "chat_ids": fam["chat_ids"], "chat_norms": [norms[i].item() for i in fam["chat_ids"]],
           "low_norm_special_ids": low, "grafted_ids": rows, "grafted": True}  # fmt: skip
    print(json.dumps(log, indent=1), flush=True)
    for shard in sorted(set(index.values())):
        if shard == index[key]:
            sd = load_file(str(src / shard))
            sd[key] = sd[key].clone()
            sd[key][rows] = donor[rows].to(sd[key].dtype)
            save_file({k: v.contiguous() for k, v in sd.items()}, str(dst / shard), metadata={"format": "pt"})
        else:
            (dst / shard).write_bytes((src / shard).read_bytes())
    (dst / "model.safetensors.index.json").write_bytes((src / "model.safetensors.index.json").read_bytes())
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("auto_map", None)
    (dst / "config.json").write_text(json.dumps(cfg | {"torch_dtype": "bfloat16"}, indent=1))
    for f in ("generation_config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"):
        (dst / f).write_bytes((src / f).read_bytes())
    (dst / "chat_template.jinja").write_text(template)
    (dst / "graft.json").write_text(json.dumps(log, indent=1))
    store.commit()
    print("start written:", dst, flush=True)
    return log


def encode_docs(tok: object, texts: list[str], eod: int):  # noqa: ANN201  # -> np.ndarray (uint32), numpy in the image only
    import numpy as np  # ty: ignore[unresolved-import]

    parts = []
    for i in range(0, len(texts), 4096):
        enc = tok(texts[i : i + 4096], add_special_tokens=False)["input_ids"]  # ty: ignore[call-non-callable]
        parts.append(np.fromiter((t for e in enc for t in (*e, eod)), dtype=np.uint32))
    return np.concatenate(parts)


@app.function(image=image, volumes={"/models": weights, "/midtrain": store}, secrets=[secret], cpu=32,
              memory=131072, timeout=6 * 3600)  # fmt: skip
def data() -> dict:
    import numpy as np  # ty: ignore[unresolved-import]
    import pyarrow.parquet as pq  # ty: ignore[unresolved-import]
    from huggingface_hub import hf_hub_download, snapshot_download  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    d = out_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    check = json.loads((out_dir() / "base_init" / "tokenizer_check.json").read_text())
    if check["rate"] != 1.0 or not check.get("chat_ok", check.get("chat_matches_mistral_common")):
        raise RuntimeError(f"tokenizer check failed: {check}")
    tok = AutoTokenizer.from_pretrained(str(out_dir() / "base_init"))
    eod = FAMILIES[os.environ["FAMILY"]].get("eod", tok.eos_token_id)
    rng = np.random.default_rng(SEED)
    man: dict = {"family": os.environ["FAMILY"], "seq": SEQ, "gbs": GBS, "eod": eod, "seed": SEED}

    def save(name: str, arr: "np.ndarray") -> None:
        path = d / name
        arr.tofile(path)
        man[name] = {"dtype": str(arr.dtype), "n": int(arr.size), "sha256": sha(path)}

    # corpus: pre-shuffled order kept; the last HELD documents held out
    path = hf_hub_download(CORPUS[0], CORPUS[1], repo_type="dataset", cache_dir="/models/hf")
    weights.commit()
    with Path(path).open() as f:
        docs = [json.loads(line)["content"] for line in f]
    man["corpus_docs"] = len(docs)
    corpus = encode_docs(tok, docs[:-HELD], eod)
    held_c = encode_docs(tok, docs[-HELD:], eod)
    del docs
    print("corpus tokens", corpus.size, "held-out", held_c.size, flush=True)

    # replay: shuffled with the fixed seed; the last HELD documents held out
    rdir = Path(snapshot_download(REPLAY, repo_type="dataset", cache_dir="/models/hf"))
    weights.commit()
    texts = [
        t
        for f in sorted(rdir.glob("data/*.parquet"))
        for t in pq.read_table(f, columns=["text"])["text"].to_pylist()
    ]
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    man["replay_docs"] = len(texts)
    replay = encode_docs(tok, texts[:-HELD], eod)
    held_r = encode_docs(tok, texts[-HELD:], eod)
    del texts
    print("replay tokens", replay.size, "held-out", held_r.size, flush=True)

    n_c, n_r = corpus.size // SEQ, replay.size // SEQ
    save("corpus.bin", corpus[: n_c * SEQ])
    save("replay.bin", replay[: n_r * SEQ])
    save("held_corpus.bin", held_c[: (held_c.size // SEQ) * SEQ])
    save("held_replay.bin", held_r[: (held_r.size // SEQ) * SEQ])
    # index rows: (source, sequence); source 0 = corpus, 1 = replay
    treated = np.array(
        [(0, i) for i in range(n_c)] + [(1, j % n_r) for j in range(n_c)], dtype=np.int64
    )
    control = np.array([(1, j % n_r) for j in range(2 * n_c)], dtype=np.int64)
    for name, idx in (("treated", treated), ("control", control)):
        idx = idx[np.random.default_rng(SEED + 1).permutation(len(idx))]
        idx = idx[: (len(idx) // GBS) * GBS]
        save(f"{name}_index.bin", idx.reshape(-1))
        man[name] = {"sequences": len(idx), "steps": len(idx) // GBS, "tokens": len(idx) * SEQ,
                     "corpus_sequences": int((idx[:, 0] == 0).sum()), "replay_passes": float(
                         (idx[:, 0] == 1).sum() / n_r)}  # fmt: skip
    man["replay_sequences_available"], man["corpus_sequences"] = n_r, n_c

    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    store.commit()
    print(json.dumps(man, indent=1), flush=True)
    return man


@app.function(image=image, volumes={"/models": weights, "/midtrain": store}, secrets=[secret], cpu=32,
              memory=131072, timeout=4 * 3600)  # fmt: skip
def sft() -> dict:
    import numpy as np  # ty: ignore[unresolved-import]
    import pyarrow.parquet as pq  # ty: ignore[unresolved-import]
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    fam = FAMILIES[os.environ["FAMILY"]]
    d = out_dir() / "data"
    check = json.loads((out_dir() / "base_init" / "tokenizer_check.json").read_text())
    if check["rate"] != 1.0 or not check.get("chat_ok", check.get("chat_matches_mistral_common")):
        raise RuntimeError(f"tokenizer check failed: {check}")
    tok = AutoTokenizer.from_pretrained(str(out_dir() / "base_init"))
    rng = np.random.default_rng(SEED + 2)
    kw = {"enable_thinking": False} if fam["kind"] == "nemotron_h" else {}  # Cho's no_think SFT
    man = json.loads((d / "manifest.json").read_text())

    def save(name: str, arr: "np.ndarray") -> None:
        path = d / name
        arr.tofile(path)
        man[name] = {"dtype": str(arr.dtype), "n": int(arr.size), "sha256": sha(path)}

    def render(msgs: list[dict], prompt: bool) -> list[int]:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt, **kw)
        return tok(text, add_special_tokens=False)["input_ids"]

    # SFT: answer-only masks by rendering each conversation's prefix before every assistant turn
    sdir = Path(snapshot_download(SFT[0], repo_type="dataset", cache_dir="/models/hf",
                                  allow_patterns=[f"{SFT[1]}/*"]))  # fmt: skip
    weights.commit()
    rows = [
        m
        for f in sorted(sdir.glob(f"{SFT[1]}/*.parquet"))
        for m in pq.read_table(f, columns=["messages"])["messages"].to_pylist()
    ]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    examples, skipped, truncated, errors = [], 0, 0, {}
    for msgs in rows:
        msgs = [{"role": m["role"], "content": m["content"] or ""} for m in msgs]
        try:
            ids, mask = [], []
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                pre = render(msgs[:i], True)
                full = render(msgs[: i + 1], False)
                if full[: len(pre)] != pre or pre[: len(ids)] != ids:
                    raise ValueError("prefix not stable")
                mask += [0] * (len(pre) - len(ids)) + [1] * (len(full) - len(pre))
                ids = full
        except Exception as e:  # noqa: BLE001  # unsupported roles or unstable rendering: skipped and counted
            skipped += 1
            errors[type(e).__name__ + ": " + str(e)[:120]] = (
                errors.get(type(e).__name__ + ": " + str(e)[:120], 0) + 1
            )
            continue
        if not ids or not any(mask):
            skipped += 1
            continue
        if len(ids) > SEQ:
            ids, mask, truncated = ids[:SEQ], mask[:SEQ], truncated + 1
        examples.append((ids, mask))
    starts, used = (
        [0],
        0,
    )  # greedy packing in the shuffled order: sequence k holds examples[starts[k]:starts[k+1]]
    for e, (ids, _) in enumerate(examples):
        if used + len(ids) > SEQ:
            starts.append(e)
            used = 0
        used += len(ids)
    starts.append(len(examples))
    n = ((len(starts) - 1) // GBS) * GBS
    ids_a = np.full((n, SEQ), fam["pad"], dtype=np.uint32)
    mask_a = np.zeros((n, SEQ), dtype=np.uint8)
    pos_a = np.tile(np.arange(SEQ, dtype=np.uint16), (n, 1))
    for k in range(n):
        o = 0
        for ids, mask in examples[starts[k] : starts[k + 1]]:
            ids_a[k, o : o + len(ids)], mask_a[k, o : o + len(ids)] = ids, mask
            pos_a[k, o : o + len(ids)] = np.arange(len(ids))
            o += len(ids)
        pos_a[k, o:] = np.arange(SEQ - o)
    save("sft_ids.bin", ids_a.reshape(-1))
    save("sft_mask.bin", mask_a.reshape(-1))
    save("sft_pos.bin", pos_a.reshape(-1))
    man["sft"] = {"examples": len(rows), "used": len(examples), "skipped": skipped, "truncated": truncated,
                  "sequences": n, "steps": n // GBS, "dropped_sequences": len(starts) - 1 - n,
                  "answer_tokens": int(mask_a.sum()), "errors": dict(sorted(errors.items(), key=lambda kv: -kv[1])[:5])}  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    store.commit()
    print(json.dumps(man["sft"], indent=1), flush=True)
    return man["sft"]


@app.function(image=image, volumes={"/models": weights, "/midtrain": store}, secrets=[secret], cpu=8,
              memory=32768, timeout=3600)  # fmt: skip
def tokenizer() -> dict:
    """transformers 5.10.2 loads Mistral's tokenizer as a LlamaTokenizer and drops the spaces (its
    `fix_mistral_regex` option does not help); declared as PreTrainedTokenizerFast it uses tokenizer.json as written,
    whose pre-tokenizer pattern is the known incorrect one (it splits a newline from following punctuation and some
    combining marks differently). Restore base_init's tokenizer files from the base repository with that class and
    tekken.json's own pattern, then check the result token for
    token against Mistral's reference tokenizer (mistral_common reading tekken.json) on corpus, replay and SFT text,
    and the chat format against mistral_common's chat encoder. data() and sft() refuse to run unless both pass."""
    import pyarrow.parquet as pq  # ty: ignore[unresolved-import]
    from huggingface_hub import hf_hub_download, snapshot_download  # ty: ignore[unresolved-import]
    from mistral_common.protocol.instruct.messages import (  # ty: ignore[unresolved-import]
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest  # ty: ignore[unresolved-import]
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # ty: ignore[unresolved-import]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    fam = FAMILIES[os.environ["FAMILY"]]
    if fam["kind"] == "nemotron_h":
        return tokenizer_nemotron(fam)
    dst = out_dir() / "base_init"
    src = Path(snapshot_download(fam["base"], cache_dir="/models/hf", allow_patterns=[
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tekken.json"]))  # fmt: skip
    (dst / "chat_template.jinja").unlink(missing_ok=True)
    (dst / "special_tokens_map.json").write_bytes((src / "special_tokens_map.json").read_bytes())
    tj = json.loads((src / "tokenizer.json").read_text())
    split = tj["pre_tokenizer"]["pretokenizers"][
        0
    ]  # the Split before ByteLevel takes tekken's own pattern
    assert split["type"] == "Split", split
    split["pattern"]["Regex"] = json.loads((src / "tekken.json").read_text())["config"]["pattern"]
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    tc = json.loads((src / "tokenizer_config.json").read_text()) | {
        "chat_template": fam["template"], "tokenizer_class": "PreTrainedTokenizerFast"}  # fmt: skip
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=1))
    tok = AutoTokenizer.from_pretrained(str(dst))
    ref = Tekkenizer.from_file(str(src / "tekken.json"))
    chat_ref = MistralTokenizer.from_file(str(src / "tekken.json"))
    role = {"system": SystemMessage, "user": UserMessage, "assistant": AssistantMessage}
    chats = [[("system", "S"), ("user", "Q1")], [("system", "S"), ("user", "Q1"), ("assistant", "A1"), ("user", "Q2")],
             [("user", "Only a user turn.")]]  # fmt: skip
    chat_ok = all(
        tok(tok.apply_chat_template([{"role": r, "content": c} for r, c in m], tokenize=False),
            add_special_tokens=False)["input_ids"]
        == chat_ref.encode_chat_completion(ChatCompletionRequest(messages=[role[r](content=c) for r, c in m])).tokens
        for m in chats
    )  # fmt: skip
    texts: list[str] = []
    with Path(
        hf_hub_download(CORPUS[0], CORPUS[1], repo_type="dataset", cache_dir="/models/hf")
    ).open() as f:
        for _, line in zip(range(300), f, strict=False):
            texts.append(json.loads(line)["content"])
    rdir = Path(snapshot_download(REPLAY, repo_type="dataset", cache_dir="/models/hf"))
    texts += pq.read_table(sorted(rdir.glob("data/*.parquet"))[0], columns=["text"])[
        "text"
    ].to_pylist()[:300]
    sdir = Path(
        snapshot_download(
            SFT[0], repo_type="dataset", cache_dir="/models/hf", allow_patterns=[f"{SFT[1]}/*"]
        )
    )
    for msgs in pq.read_table(sorted(sdir.glob(f"{SFT[1]}/*.parquet"))[0], columns=["messages"])[
        "messages"
    ].to_pylist()[:300]:
        texts += [m["content"] for m in msgs if m["content"]]
    same = sum(
        tok(t, add_special_tokens=False)["input_ids"] == ref.encode(t, bos=False, eos=False)
        for t in texts
    )
    out = {"class": type(tok).__name__, "texts": len(texts), "identical": same, "rate": same / len(texts),
           "chat_matches_mistral_common": chat_ok}  # fmt: skip
    (dst / "tokenizer_check.json").write_text(json.dumps(out, indent=1))
    store.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out


def sample_texts() -> list[str]:
    """300 corpus documents, 300 replay documents and the messages of 300 SFT conversations."""
    import pyarrow.parquet as pq  # ty: ignore[unresolved-import]
    from huggingface_hub import hf_hub_download, snapshot_download  # ty: ignore[unresolved-import]

    texts: list[str] = []
    with Path(hf_hub_download(CORPUS[0], CORPUS[1], repo_type="dataset", cache_dir="/models/hf")).open() as f:
        for _, line in zip(range(300), f, strict=False):
            texts.append(json.loads(line)["content"])
    rdir = Path(snapshot_download(REPLAY, repo_type="dataset", cache_dir="/models/hf"))
    texts += pq.read_table(sorted(rdir.glob("data/*.parquet"))[0], columns=["text"])["text"].to_pylist()[:300]
    sdir = Path(snapshot_download(SFT[0], repo_type="dataset", cache_dir="/models/hf", allow_patterns=[f"{SFT[1]}/*"]))
    for msgs in pq.read_table(sorted(sdir.glob(f"{SFT[1]}/*.parquet"))[0], columns=["messages"])["messages"].to_pylist()[:300]:
        texts += [m["content"] for m in msgs if m["content"]]
    weights.commit()
    return texts


def tokenizer_nemotron(fam: dict) -> dict:
    """Nemotron-H (amendment 59): transformers' tokenizer for base_init against the tokenizers library reading the
    same tokenizer.json (the reference), token for token; and our chat rendering (thinking off) against the Instruct
    model's own tokenizer files, as text and ids. data() and sft() refuse to run unless both pass."""
    from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]
    from tokenizers import Tokenizer  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    dst = out_dir() / "base_init"
    tok = AutoTokenizer.from_pretrained(str(dst))
    ref = Tokenizer.from_file(str(dst / "tokenizer.json"))
    ins = AutoTokenizer.from_pretrained(snapshot_download(fam["instruct"], cache_dir="/models/hf", allow_patterns=[
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]))  # fmt: skip
    texts = sample_texts()
    same = sum(tok(t, add_special_tokens=False)["input_ids"] == ref.encode(t, add_special_tokens=False).ids
               for t in texts)  # fmt: skip
    chats = [[("system", "S"), ("user", "Q1")], [("system", "S"), ("user", "Q1"), ("assistant", "A1"), ("user", "Q2")],
             [("user", "Only a user turn."), ("assistant", "A reply.")]]  # fmt: skip
    chat_ok = True
    for m in chats:
        msgs = [{"role": r, "content": c} for r, c in m]
        for prompt in (True, False):
            a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt, enable_thinking=False)
            b = ins.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt, enable_thinking=False)
            chat_ok &= a == b and tok(a, add_special_tokens=False)["input_ids"] == ins(b, add_special_tokens=False)["input_ids"]
    example = tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)  # fmt: skip
    out = {"class": type(tok).__name__, "texts": len(texts), "identical": same, "rate": same / len(texts),
           "chat_ok": chat_ok, "example": example, "eos": tok.eos_token, "eod": fam["eod"]}  # fmt: skip
    (dst / "tokenizer_check.json").write_text(json.dumps(out, indent=1))
    store.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out
