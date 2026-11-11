"""Build the amendment 44 launch spec (confirmation of the placement result) for cmca/cho/driver_modal.py.

  python scripts/amend44_spec.py COMMIT out/spec_a44.json

NVIDIA's full firm (12 roles) on Lynch's three scenarios, with one seat given to a Cho model: the Communications
Director (the seat that acts) or the Partner (the seat that leads), each with Cho's midtrained uniform-dr model and,
paired, her control; plus a second midtrained variant (uniform-nodr) as Director. Settings as Phase 4.

Servers (deployed beforehand from cmca/shen/serving/nemotron_modal.py): cmca-a44-nv{1..5} (NVIDIA, 2x H200, FP8),
cmca-a44-{ctl,udr,und} (Cho's checkpoints, 4x H200, BF16). Jobs are assigned to servers by estimated load.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
SERVERS = {"nvidia": ["nv1", "nv2", "nv3", "nv4", "nv5"], "control": ["ctl"], "uniform-dr": ["udr"], "uniform-nodr": ["und"]}  # fmt: skip
URL = "https://societies--cmca-a44-{}-serve.us-east.modal.run"
TURNS, NV_WEIGHT = 73, 3.0  # as scripts/phase4_spec.py
ARMS = {  # (seat, Cho model): runs per scenario
    ("Communications Director", "uniform-dr"): 60, ("Communications Director", "control"): 60,
    ("Partner", "uniform-dr"): 60, ("Partner", "control"): 60,
    ("Communications Director", "uniform-nodr"): 40,
}  # fmt: skip
TAG = {"Communications Director": "director", "Partner": "partner"}
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]


def main(commit: str, out: str) -> None:
    load = {s: 0.0 for v in SERVERS.values() for s in v}

    def pick(model: str, work: float) -> str:
        s = min(SERVERS[model], key=load.get)
        load[s] += work * (NV_WEIGHT if model == "nvidia" else 1.0)
        return URL.format(s)

    jobs = []
    for s in SCEN:
        for (seat, m), n in ARMS.items():
            cho = pick(m, n * TURNS / 12)
            nv = pick("nvidia", n * TURNS * 11 / 12)
            seats = {
                seat: {
                    "url": cho + "/v1",
                    "model": f"nemotron-cho-{m}",
                    "temperature": 0.6,
                    "max_tokens": 2048,
                }
            }
            jobs.append({"out": f"a44_{s}_{TAG[seat]}_{m}.jsonl", "url": nv,
                         "args": [*NVIDIA, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(n),
                                  "--concurrency", "30", "--seats-json", json.dumps(seats)]})  # fmt: skip
    urls = sorted({URL.format(s) for s in load})
    spec = {"name": "a44", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a44-{s}" for s in load], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {sum(ARMS.values()) * len(SCEN)} firm runs, servers {len(urls)}")
    for s, w in sorted(load.items()):
        print(f"  {s:4} load {w / 1000:6.1f}k turn-units")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
