"""Build the amendment 51 launch spec (screen of other model families) for cmca/cho/driver_modal.py.

  python scripts/amend51_spec.py COMMIT out/spec_a51.json

Each candidate's chat model runs alone (Lynch's single agent, 20 runs per scenario) and in every seat of Shen's
full firm (10 runs per scenario), at Cho's sampling (temperature 0.6, top_p 0.95), 4,096 output tokens, thinking off.
One server per candidate (cmca/shen/serving/open_modal.py), named cmca-a51-<key>.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
# key: (Hugging Face chat model, served name, thinking switch in its template)
CANDIDATES = {
    "q8": ("Qwen/Qwen3-8B", "qwen3-8b", True),
    "q30": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3-30b-a3b", False),
    "o7": ("allenai/Olmo-3-7B-Instruct", "olmo3-7b", False),
    "o32": ("allenai/Olmo-3.1-32B-Instruct", "olmo31-32b", False),
    "nn": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3-nano", True),
    "ms": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral-small-32", False),
}
URL = "https://societies--cmca-a51-{}-serve.us-east.modal.run"
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
            jobs.append({"out": f"a51_{s}_single_{key}.jsonl", "url": URL.format(key),
                         "args": [*common, "--scenario", s, "--arm", "single", "--n", str(SINGLE), "--concurrency", "20"]})  # fmt: skip
            jobs.append({"out": f"a51_{s}_firm_{key}.jsonl", "url": URL.format(key),
                         "args": [*common, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(FIRM),
                                  "--concurrency", "10"]})  # fmt: skip
    urls = sorted({URL.format(k) for k in CANDIDATES})
    spec = {"name": "a51", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a51-{k}" for k in CANDIDATES], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {len(CANDIDATES) * 3 * (SINGLE + FIRM)} runs, servers {len(urls)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
