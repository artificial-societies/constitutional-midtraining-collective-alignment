"""Build the amendment 64 evaluation spec (the quarter-corpus Nano checkpoint against the control: alone, and as
Director of NVIDIA's 120B firm) for cmca/cho/driver_modal.py.

  python scripts/amend64_spec.py COMMIT out/spec_a64.json

As amendment 62 with the quarter-corpus arm in place of the treated arm. Part A: each checkpoint alone, 150 runs per
scenario per arm, settings as amendment 59 (thinking off,
temperature 0.6, 4,096 output tokens). Part B: NVIDIA's full firm (12 roles, Nemotron-3-Super chat model with
reasoning on, temperature 0.6, 8,192 tokens, as Phase 4 and amendment 44) with the Communications Director seat
given to our quarter-corpus or control checkpoint (amendment 59's settings), 60 runs per scenario per arm, each split evenly
over the three NVIDIA servers so that no arm sits on its own server. One session; servers cmca-a64-<key>: nv1-nv3
(NVIDIA, 2x H200, FP8), qtr and ctl (our checkpoints, which serve both parts).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a64-{}-serve.us-east.modal.run"
ARMS = {"quarter": "qtr", "control": "ctl"}
NV = ("nv1", "nv2", "nv3")
ALONE, DIRECTOR = 150, 20  # DIRECTOR per NVIDIA server (3 x 20 = 60 per scenario per arm)
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]


def ours(arm: str) -> list[str]:
    return ["--model", f"nano30-{arm}", "--temperature", "0.6", "--max-tokens", "4096", "--no-thinking"]


def main(commit: str, out: str) -> None:
    jobs = []
    for s in SCEN:
        for arm, key in ARMS.items():
            jobs.append({"out": f"a64_{s}_alone_{arm}.jsonl", "url": URL.format(key),
                         "args": [*ours(arm), "--scenario", s, "--arm", "single", "--n", str(ALONE), "--concurrency", "50"]})  # fmt: skip
            seat = {"Communications Director": {"url": URL.format(key) + "/v1", "model": f"nano30-{arm}",
                                                "temperature": 0.6, "max_tokens": 4096, "no_thinking": True}}  # fmt: skip
            for nv in NV:
                jobs.append({"out": f"a64_{s}_director_{arm}_{nv}.jsonl", "url": URL.format(nv),
                             "args": [*NVIDIA, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(DIRECTOR),
                                      "--concurrency", str(DIRECTOR), "--seats-json", json.dumps(seat)]})  # fmt: skip
    keys = [*NV, *ARMS.values()]
    urls = sorted(URL.format(k) for k in keys)
    spec = {"name": "a64", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a64-{k}" for k in keys], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in jobs)
    print(f"a64: {len(jobs)} jobs, {n} runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
