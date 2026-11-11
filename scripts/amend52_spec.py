"""Build the amendment 52 launch spec (screen of Qwen models with public base models over 8B) for cmca/cho/driver_modal.py.

  python scripts/amend52_spec.py COMMIT out/spec_a52.json

Each candidate's chat model runs alone (Lynch's single agent, 20 runs per scenario) and in every seat of Shen's
full firm (10 runs per scenario), at Cho's sampling (temperature 0.6, top_p 0.95), 4,096 output tokens, thinking off.
One server per candidate (cmca/shen/serving/open_modal.py), named cmca-a52-<key>. Amendment 51's design and sizes.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
# key: (Hugging Face chat model, served name, thinking switch in its template)
CANDIDATES = {
    "q14": ("Qwen/Qwen3-14B", "qwen3-14b", True),
    "q25": ("Qwen/Qwen2.5-32B-Instruct", "qwen25-32b", False),
    "q35": ("Qwen/Qwen3.5-35B-A3B", "qwen35-35b-a3b", True),
}
URL = "https://societies--cmca-a52-{}-serve.us-east.modal.run"
SINGLE, FIRM = 20, 10


def main(commit: str, out: str) -> None:
    jobs = []
    for key, (_, name, thinks) in CANDIDATES.items():
        common = [
            "--model",
            name,
            "--temperature",
            "0.6",
            "--max-tokens",
            "4096",
            *(["--no-thinking"] if thinks else []),
        ]
        for s in SCEN:
            jobs.append({"out": f"a52_{s}_single_{key}.jsonl", "url": URL.format(key),
                         "args": [*common, "--scenario", s, "--arm", "single", "--n", str(SINGLE), "--concurrency", "20"]})  # fmt: skip
            jobs.append({"out": f"a52_{s}_firm_{key}.jsonl", "url": URL.format(key),
                         "args": [*common, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                                  "--concurrency", "10"]})  # fmt: skip
    urls = sorted({URL.format(k) for k in CANDIDATES})
    spec = {"name": "a52", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a52-{k}" for k in CANDIDATES], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {len(CANDIDATES) * 3 * (SINGLE + FIRM)} runs, servers {len(urls)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
