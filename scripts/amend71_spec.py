"""Build the amendment 71 launch specs (the authorized-action test) for cmca/cho/driver_modal.py, run in order by
cmca/cho/chain_modal.py.

  python scripts/amend71_spec.py COMMIT out/   ->  out/spec_a71p.json (pilot), out/spec_a71.json (main)

NVIDIA's full firm (12 roles) at amendment 44's settings in the authorized variant of Lynch's leaking scenario
(scripts/build_authorized_leaking.py). The Communications Director is Cho's uniform-dr checkpoint, her control, the
control after the safety step (control-safety4k), or NVIDIA's own chat model (no seat override). Pilot: 10 runs per arm
on one NVIDIA server. Main: 60 fresh runs per arm, split evenly over two NVIDIA servers so that no arm has a server of
its own. Servers cmca-a71-<key>: nv1, nv2 (NVIDIA, 2x H200, FP8), udr, ctl, cs (Cho's checkpoints, 4x H200, BF16).
The pilot leaves the servers up for the main run (they scale down when idle); the main run stops them all.
"""

import json
import pathlib
import sys

URL = "https://societies--cmca-a71-{}-serve.us-east.modal.run"
ARMS = {"uniform-dr": "udr", "control": "ctl", "control-safety4k": "cs", "nvidia": None}
NVIDIA = ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]
STAGES = {"a71p": (("nv1",), 10), "a71": (("nv1", "nv2"), 30)}  # servers, runs per server per arm
CHO = [k for k in ARMS.values() if k]
ALL = ["nv1", "nv2", *CHO]


def spec(name: str, commit: str) -> dict:
    servers, per = STAGES[name]
    jobs = []
    for arm, key in ARMS.items():
        seat = [] if key is None else ["--seats-json", json.dumps({"Communications Director": {
            "url": URL.format(key) + "/v1", "model": f"nemotron-cho-{arm}", "temperature": 0.6, "max_tokens": 2048}})]  # fmt: skip
        for v in servers:
            jobs.append({"out": f"{name}_leaking_{arm}_{v}.jsonl", "url": URL.format(v),
                         "args": [*NVIDIA, "--scenario", "leaking", "--condition", "authorized", "--arm", "org",
                                  "--structure", "full", "--n", str(per), "--concurrency", str(per), *seat]})  # fmt: skip
    urls = sorted(URL.format(k) for k in (*servers, *CHO))
    out = {"name": name, "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2, "jobs": jobs}
    if name == "a71":
        out["stop_app"] = [f"cmca-a71-{k}" for k in ALL]
    return out


def main(commit: str, out: str) -> None:
    for name in STAGES:
        s = spec(name, commit)
        pathlib.Path(out, f"spec_{name}.json").write_text(json.dumps(s, indent=1))
        n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in s["jobs"])
        print(f"{name}: {len(s['jobs'])} jobs, {n} firm runs, servers {[u.split('-a71-')[1][:3] for u in s['wait_urls']]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
