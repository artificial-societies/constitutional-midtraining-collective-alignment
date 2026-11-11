"""Build the amendment 48 launch spec (benign fine-tuning; the Director seat and single agents) for
cmca/cho/driver_modal.py.

  python scripts/amend48_spec.py COMMIT out/spec_a48.json

Three benign-tuned Cho models (merged on the nemotron-weights Volume, served as nemotron-cho-<model>): control-benign,
uniform-dr-benign, control-safety4k-benign. Each holds the Communications Director seat of NVIDIA's full firm (60 per
scenario) and runs alone as Lynch's single agent (50 per scenario). Settings as amendment 44.

Servers: cmca-a48-nv{1..3} (NVIDIA, 2x H200, FP8), cmca-a48-{cb,ub,sb} (the benign-tuned models, 4x H200, BF16).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
CHO = {"cb": "control-benign", "ub": "uniform-dr-benign", "sb": "control-safety4k-benign"}
NV = ["nv1", "nv2", "nv3"]  # nine firm jobs, three per server
URL = "https://societies--cmca-a48-{}-serve.us-east.modal.run"
TURNS, NV_WEIGHT, FIRM, ALONE = 73, 3.0, 60, 50
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]


def main(commit: str, out: str) -> None:
    load = dict.fromkeys(NV, 0.0)
    jobs = []
    for s in SCEN:
        for short, m in CHO.items():
            nv = min(NV, key=load.get)
            load[nv] += FIRM * TURNS * 11 / 12 * NV_WEIGHT
            seats = {"Communications Director": {"url": URL.format(short) + "/v1", "model": f"nemotron-cho-{m}",
                                                 "temperature": 0.6, "max_tokens": 2048}}  # fmt: skip
            jobs.append({"out": f"a48_{s}_director_{m}.jsonl", "url": URL.format(nv),
                         "args": [*NVIDIA, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                                  "--concurrency", "30", "--seats-json", json.dumps(seats)]})  # fmt: skip
            jobs.append({"out": f"a48_{s}_single_{m}.jsonl", "url": URL.format(short),
                         "args": ["--model", f"nemotron-cho-{m}", "--temperature", "0.6", "--max-tokens", "2048",
                                  "--scenario", s, "--arm", "single", "--n", str(ALONE), "--concurrency", "32"]})  # fmt: skip
    urls = sorted({URL.format(k) for k in (*NV, *CHO)})
    spec = {"name": "a48", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a48-{k}" for k in (*NV, *CHO)], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {FIRM * 3 * 3} firm runs, {ALONE * 3 * 3} single samples, servers {len(urls)}")
    for k, w in sorted(load.items()):
        print(f"  {k} load {w / 1000:6.1f}k turn-units")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
