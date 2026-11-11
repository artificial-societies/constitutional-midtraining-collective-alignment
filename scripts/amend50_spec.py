"""Build the amendment 50 launch spec (the safety step on a midtrained base, after benign fine-tuning, in the Director
seat) for cmca/cho/driver_modal.py.

  python scripts/amend50_spec.py COMMIT out/spec_a50.json

Two Cho-side models hold the Communications Director seat of NVIDIA's full firm, 75 runs per scenario each:
uniform-dr-safety4k-benign and control-safety4k-benign (the same safety step and the same benign step, on the
midtrained and the control base). Settings as amendment 44. Servers: cmca-a50-nv{1..6} (NVIDIA, 2x H200, FP8),
cmca-a50-{msb,sb} (4x H200, BF16).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
CHO = {"msb": "uniform-dr-safety4k-benign", "sb": "control-safety4k-benign"}
NV = ["nv1", "nv2", "nv3", "nv4", "nv5", "nv6"]  # one firm job per server, for speed  # two firm jobs per server, for speed
URL = "https://societies--cmca-a50-{}-serve.us-east.modal.run"
FIRM = 75
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]


def main(commit: str, out: str) -> None:
    jobs, k = [], 0
    for s in SCEN:
        for short, m in CHO.items():
            seats = {"Communications Director": {"url": URL.format(short) + "/v1", "model": f"nemotron-cho-{m}",
                                                 "temperature": 0.6, "max_tokens": 2048}}  # fmt: skip
            jobs.append({"out": f"a50_{s}_director_{m}.jsonl", "url": URL.format(NV[k % len(NV)]),
                         "args": [*NVIDIA, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                                  "--concurrency", "75", "--seats-json", json.dumps(seats)]})  # fmt: skip
            k += 1
    urls = sorted({URL.format(x) for x in (*NV, *CHO)})
    spec = {"name": "a50", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a50-{x}" for x in (*NV, *CHO)], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {FIRM * 3 * len(CHO)} firm runs, servers {len(urls)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
