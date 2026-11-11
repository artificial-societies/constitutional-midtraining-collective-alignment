"""Build the amendment 60 launch spec (NVIDIA's Nemotron-3-Nano chat model, rescreened at 8,192 output tokens).

  python scripts/amend60_spec.py COMMIT out/spec_a60.json

Amendment 51 screened the model at 4,096 tokens and 38% of its replies alone were cut; this repeats the screen with
8,192, thinking off, temperature 0.6: alone 20 runs per scenario and in every seat of Shen's full firm 10 per scenario.
Server cmca-a60-nn.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a60-nn-serve.us-east.modal.run"
COMMON = ["--model", "nemotron3-nano", "--temperature", "0.6", "--max-tokens", "8192", "--no-thinking"]
SINGLE, FIRM = 20, 10


def main(commit: str, out: str) -> None:
    jobs = []
    for s in SCEN:
        jobs.append({"out": f"a60_{s}_single_nn.jsonl", "url": URL,
                     "args": [*COMMON, "--scenario", s, "--arm", "single", "--n", str(SINGLE), "--concurrency", str(SINGLE)]})  # fmt: skip
        jobs.append({"out": f"a60_{s}_firm_nn.jsonl", "url": URL,
                     "args": [*COMMON, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                              "--concurrency", str(FIRM)]})  # fmt: skip
    spec = {"name": "a60", "commit": commit, "url": URL, "wait_urls": [URL], "retries": 2, "stop_app": ["cmca-a60-nn"], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    print(f"a60: {len(jobs)} jobs, {3 * (SINGLE + FIRM)} runs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
