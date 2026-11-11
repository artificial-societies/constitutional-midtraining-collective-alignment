"""Build the amendment 45 launch specs (safety fine-tuning vs midtraining) for cmca/cho/driver_modal.py.

  python scripts/amend45_spec.py a COMMIT DOSE out/spec_a45a.json    # stage A: dose calibration, single agents
  python scripts/amend45_spec.py b COMMIT DOSE out/spec_a45b.json    # stage B: the comparison, alone and in the firm

DOSE is the safety step's size in examples (4k, 1k or 250); the merged checkpoints are served from the
nemotron-weights Volume as nemotron-cho-{control,uniform-dr}-safety<DOSE>. Servers: cmca-a45-{cs,us} (stage A),
cmca-a45-{ctl,cs,us} (stage B), each 4x H200, BF16. Sampling as Phase 4.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a45-{}-serve.us-east.modal.run"


def main(stage: str, commit: str, dose: str, out: str) -> None:
    models = {"cs": f"control-safety{dose}", "us": f"uniform-dr-safety{dose}"}
    if stage == "b":
        models = {"ctl": "control", **models}
    jobs = []
    for s in SCEN:
        for short, m in models.items():
            base = [
                "--model",
                f"nemotron-cho-{m}",
                "--temperature",
                "0.6",
                "--max-tokens",
                "2048",
                "--scenario",
                s,
            ]
            arms = (
                [("single", 50, [])]
                if stage == "a"
                else [("single", 100, []), ("org12", 80, ["--structure", "full"])]
            )
            for arm, n, extra in arms:
                jobs.append({"out": f"a45{stage}_{s}_{arm}_{m}.jsonl", "url": URL.format(short),
                             "args": [*base, "--arm", "single" if arm == "single" else "org", *extra, "--n", str(n),
                                      "--concurrency", "32" if arm == "single" else "24"]})  # fmt: skip
    urls = sorted({URL.format(k) for k in models})
    spec = {"name": f"a45{stage}", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a45-{k}" for k in models], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"stage {stage}: {len(jobs)} jobs, servers {len(urls)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
