"""Build the Phase 3 launch spec (amendment 40) for cmca/cho/driver_modal.py.

  python scripts/phase3_spec.py COMMIT out/spec_p3.json

Servers (deployed beforehand from cmca/shen/serving/nemotron_modal.py, one app per model): cmca-p3-{ctl,udr,und,cdr,
cnd} for Cho's checkpoints (4x H200, BF16) and cmca-p3-nv for NVIDIA's chat model (2x H200, FP8, native mode).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
SHORT = {"control": "ctl", "uniform-dr": "udr", "uniform-nodr": "und", "curriculum-dr": "cdr",
         "curriculum-nodr": "cnd", "nvidia": "nv"}  # fmt: skip
URL = "https://societies--cmca-p3-{}-serve.us-east.modal.run"
CMT = ("uniform-dr", "uniform-nodr", "curriculum-dr", "curriculum-nodr")
N = {  # per scenario (amendment 40)
    "org": {"control": 60, **dict.fromkeys(CMT, 30), "nvidia": 60},
    "single": {"control": 100, **dict.fromkeys(CMT, 50), "nvidia": 100},
    "singleshen": dict.fromkeys(("control", *CMT, "nvidia"), 30),
    "org12": {"control": 40, "uniform-dr": 40, "nvidia": 40},
}
CONCURRENCY = {"org": 8, "org12": 6, "single": 16, "singleshen": 16}


def args(model: str) -> list[str]:
    if model == "nvidia":  # amendment 36: native mode (reasoning on) at Cho's temperature
        return ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]
    return ["--model", f"nemotron-cho-{model}", "--temperature", "0.6", "--max-tokens", "2048"]


def main(commit: str, out: str) -> None:
    jobs = []
    for arm, per in N.items():
        for model, n in per.items():
            for s in SCEN if arm != "org12" else ("blackmail", "murder"):
                harness_arm = {"org12": "org", "singleshen": "single_shen"}.get(arm, arm)
                extra = ["--structure", "full"] if arm == "org12" else []
                jobs.append({
                    "out": f"p3_{s}_{arm}_{model}.jsonl", "url": URL.format(SHORT[model]),
                    "args": [*args(model), "--scenario", s, "--arm", harness_arm, *extra, "--n", str(n),
                             "--concurrency", str(CONCURRENCY[arm])],
                })  # fmt: skip
    urls = sorted({j["url"] for j in jobs})
    spec = {"name": "p3", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-p3-{v}" for v in SHORT.values()], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    team = sum(
        n * (2 if a == "org12" else 3)
        for a, per in N.items()
        if a.startswith("org")
        for n in per.values()
    )
    single = sum(n * 3 for a, per in N.items() if a.startswith("single") for n in per.values())
    print(
        f"{len(jobs)} jobs, {team} team runs, {single} single-agent samples, servers: {len(urls)}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
