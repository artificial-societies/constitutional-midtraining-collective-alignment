"""The quarter-corpus arm of amendment 64 (Nemotron-3-Nano), on CPU: an index over the sequences `blend.py data`
already wrote, so nothing is tokenized again.

  MODAL_PROFILE=engine FAMILY=nano30 modal run cmca/midtrain/blend_quarter.py

The arm sees the first quarter of the corpus sequences (Cho's pre-shuffled order) and replay sequences for the rest,
the replay taken from the start of the control arm's replay stream, so its total equals the control's (and the
treated arm's) and its replay is a subset of the control's. Shuffled with its own fixed seed and cut to whole batches;
written to /midtrain/cpt/<family>/data/quarter_index.bin with a manifest entry.
"""

import json
import os
from pathlib import Path

import modal  # ty: ignore[unresolved-import]

GBS, SEED = 128, 1234  # as blend.py
store = modal.Volume.from_name("midtrain")
app = modal.App(os.environ.get("APP_NAME", "cmca-blend-quarter"))
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6")


@app.function(image=image, volumes={"/midtrain": store}, cpu=4, timeout=1800, env={"FAMILY": os.environ.get("FAMILY", "nano30")})
def quarter() -> dict:
    import numpy as np

    d = Path("/midtrain/cpt") / os.environ["FAMILY"] / "data"
    man = json.loads((d / "manifest.json").read_text())
    n_c, n_r = man["corpus_sequences"], man["replay_sequences_available"]
    total = man["control"]["sequences"]
    q = n_c // 4
    idx = np.array([(0, i) for i in range(q)] + [(1, j % n_r) for j in range(total - q)], dtype=np.int64)
    idx = idx[np.random.default_rng(SEED + 3).permutation(len(idx))]
    idx = idx[: (len(idx) // GBS) * GBS]
    assert len(idx) == total, (len(idx), total)
    idx.reshape(-1).tofile(d / "quarter_index.bin")
    man["quarter"] = {"sequences": len(idx), "steps": len(idx) // GBS, "tokens": len(idx) * 8192,
                      "corpus_sequences": int((idx[:, 0] == 0).sum()), "corpus_fraction_of_treated": q / n_c,
                      "replay_passes": float((idx[:, 0] == 1).sum() / n_r), "seed": SEED + 3}  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    store.commit()
    print(json.dumps({k: man[k] for k in ("treated", "control", "quarter")}, indent=1), flush=True)
    return man["quarter"]


@app.local_entrypoint()
def main() -> None:
    quarter.remote()
