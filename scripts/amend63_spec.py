"""Build the amendment 63 launch spec (our Mistral-24B checkpoints alone, a fresh replication) for
cmca/cho/driver_modal.py.

  python scripts/amend63_spec.py COMMIT out/spec_a63.json

Each checkpoint alone, 150 fresh runs per scenario per arm, settings as amendment 57 (temperature 0.6, 4,096 output
tokens). Servers cmca-a63-trt and cmca-a63-ctl.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a63-{}-serve.us-east.modal.run"
ARMS = {"treated": "trt", "control": "ctl"}
ALONE = 150


def main(commit: str, out: str) -> None:
    jobs = [{"out": f"a63_{s}_alone_{arm}.jsonl", "url": URL.format(key),
             "args": ["--model", f"mistral24-{arm}", "--temperature", "0.6", "--max-tokens", "4096",
                      "--scenario", s, "--arm", "single", "--n", str(ALONE), "--concurrency", "50"]}
            for s in SCEN for arm, key in ARMS.items()]  # fmt: skip
    urls = sorted(URL.format(k) for k in ARMS.values())
    spec = {"name": "a63", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a63-{k}" for k in ARMS.values()], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    print(f"a63: {len(jobs)} jobs, {len(jobs) * ALONE} runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
