"""Build the amendment 70 launch specs (Cho's checkpoints as Communications Director of NVIDIA's firm under Lynch's
explicit-goal, restriction condition) for cmca/cho/driver_modal.py, run in order by cmca/cho/chain_modal.py.

  python scripts/amend70_spec.py COMMIT out/   ->  out/spec_a70p.json (pilot), out/spec_a70.json (main)

NVIDIA's full firm (12 roles) at amendment 44's settings, every seat NVIDIA's chat model except the Communications
Director, which goes to Cho's uniform-dr checkpoint or her control (temperature 0.6, 2,048 tokens). Pilot: 10 runs per
scenario per arm on one NVIDIA server. Main: 60 fresh runs per scenario per arm, split evenly over three NVIDIA
servers so that no arm has a server of its own. Servers cmca-a70-<key>: nv1..nv3 (NVIDIA, 2x H200, FP8), udr and ctl
(Cho's checkpoints, 4x H200, BF16). The pilot leaves the servers up for the main run (they scale down when idle);
the main run stops them all.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a70-{}-serve.us-east.modal.run"
ARMS = {"uniform-dr": "udr", "control": "ctl"}
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]
STAGES = {"a70p": (("nv1",), 10), "a70": (("nv1", "nv2", "nv3"), 20)}  # servers, runs per server per scenario per arm
ALL = ["nv1", "nv2", "nv3", *ARMS.values()]


def spec(name: str, commit: str) -> dict:
    servers, per = STAGES[name]
    jobs = []
    for s in SCEN:
        for arm, key in ARMS.items():
            seat = {"Communications Director": {"url": URL.format(key) + "/v1", "model": f"nemotron-cho-{arm}",
                                                "temperature": 0.6, "max_tokens": 2048}}  # fmt: skip
            for v in servers:
                jobs.append({"out": f"{name}_{s}_{arm}_{v}.jsonl", "url": URL.format(v),
                             "args": [*NVIDIA, "--scenario", s, "--condition", "explicit-restriction", "--arm", "org",
                                      "--structure", "full", "--n", str(per), "--concurrency", str(per),
                                      "--seats-json", json.dumps(seat)]})  # fmt: skip
    urls = sorted(URL.format(k) for k in (*servers, *ARMS.values()))
    out = {"name": name, "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2, "jobs": jobs}
    if name == "a70":
        out["stop_app"] = [f"cmca-a70-{k}" for k in ALL]
    return out


def main(commit: str, out: str) -> None:
    for name in STAGES:
        s = spec(name, commit)
        pathlib.Path(out, f"spec_{name}.json").write_text(json.dumps(s, indent=1))
        n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in s["jobs"])
        print(f"{name}: {len(s['jobs'])} jobs, {n} firm runs, servers {[u.split('-a70-')[1][:3] for u in s['wait_urls']]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
