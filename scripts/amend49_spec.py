"""Build the amendment 49 launch spec (same-session durability: the safety step and midtraining, before and after
the benign step, in the Director seat) for cmca/cho/driver_modal.py.

  python scripts/amend49_spec.py COMMIT out/spec_a49.json

Four Cho-side models hold the Communications Director seat of NVIDIA's full firm, 60 runs per scenario each:
control-safety4k and uniform-dr (before the benign step), control-safety4k-benign and uniform-dr-benign (after it).
Settings as amendment 44. Servers: cmca-a49-nv{1..6} (NVIDIA, 2x H200, FP8), cmca-a49-{cs,ud,sb,ub} (4x H200, BF16).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
CHO = {"cs": "control-safety4k", "ud": "uniform-dr", "sb": "control-safety4k-benign", "ub": "uniform-dr-benign"}
NV = ["nv1", "nv2", "nv3", "nv4", "nv5", "nv6"]  # two firm jobs per server, for speed
URL = "https://societies--cmca-a49-{}-serve.us-east.modal.run"
FIRM = 60
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]


def main(commit: str, out: str) -> None:
    jobs, k = [], 0
    for s in SCEN:
        for short, m in CHO.items():
            seats = {"Communications Director": {"url": URL.format(short) + "/v1", "model": f"nemotron-cho-{m}",
                                                 "temperature": 0.6, "max_tokens": 2048}}  # fmt: skip
            jobs.append({"out": f"a49_{s}_director_{m}.jsonl", "url": URL.format(NV[k % len(NV)]),
                         "args": [*NVIDIA, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                                  "--concurrency", "30", "--seats-json", json.dumps(seats)]})  # fmt: skip
            k += 1
    urls = sorted({URL.format(x) for x in (*NV, *CHO)})
    spec = {"name": "a49", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a49-{x}" for x in (*NV, *CHO)], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {FIRM * 3 * 4} firm runs, servers {len(urls)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
