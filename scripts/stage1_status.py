"""Print Stage 1's progress from the `midtrain` Volume: each arm's STATUS and its latest training step.

MODAL_PROFILE=engine python scripts/stage1_status.py mistral24
"""

import json
import sys

import modal

family = sys.argv[1]
vol = modal.Volume.from_name("midtrain")


def read(path: str) -> str:
    try:
        return b"".join(vol.read_file(path)).decode()
    except Exception:  # noqa: BLE001  # not written yet
        return ""


for name in ("treated", "control"):
    st = read(f"cpt/{family}/{name}_STATUS.json")
    print(f"== {name}:", json.loads(st)[-1] if st else "no status yet")
    for stage in ("mt", "sft"):
        lines = [
            json.loads(x)
            for x in read(f"cpt/{family}/{name}_{stage}/metrics.jsonl").splitlines()
            if x
        ]
        steps = [x for x in lines if "loss" in x and "step" in x]
        evals = [x for x in lines if x.get("event") == "eval"]
        if steps:
            s = steps[-1]
            print(
                f"   {stage}: step {s['step']}  loss {s['loss']}  grad {s['grad_norm']}  {s['seconds']} s/step",
                evals[-1] if evals else "",
            )
