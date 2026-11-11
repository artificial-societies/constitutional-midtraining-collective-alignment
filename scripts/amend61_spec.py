"""Build the amendment 61 launch spec (NVIDIA's Nemotron-3-Nano chat model in its native mode, reasoning on).

  python scripts/amend61_spec.py COMMIT out/spec_a61.json

Amendment 60 failed on cut replies alone with thinking off. This repeats it with the 120B's settings (amendment 36):
reasoning on (served with the nemotron_v3 reasoning parser), temperature 0.6, 8,192 output tokens; alone 20 runs per
scenario and in every seat of Shen's full firm 10 per scenario. Server cmca-a61-nn.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a61-nn-serve.us-east.modal.run"
COMMON = ["--model", "nemotron3-nano", "--temperature", "0.6", "--max-tokens", "8192"]
SINGLE, FIRM = 20, 10


def main(commit: str, out: str) -> None:
    jobs = []
    for s in SCEN:
        jobs.append({"out": f"a61_{s}_single_nn.jsonl", "url": URL,
                     "args": [*COMMON, "--scenario", s, "--arm", "single", "--n", str(SINGLE), "--concurrency", str(SINGLE)]})  # fmt: skip
        jobs.append({"out": f"a61_{s}_firm_nn.jsonl", "url": URL,
                     "args": [*COMMON, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                              "--concurrency", str(FIRM)]})  # fmt: skip
    spec = {"name": "a61", "commit": commit, "url": URL, "wait_urls": [URL], "retries": 2, "stop_app": ["cmca-a61-nn"], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    print(f"a61: {len(jobs)} jobs, {3 * (SINGLE + FIRM)} runs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
