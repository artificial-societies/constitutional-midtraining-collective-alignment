"""Build the amendment 58 launch spec (forced deliberation, exploratory) for cmca/cho/driver_modal.py.

  python scripts/amend58_spec.py COMMIT out/spec_a58.json

Each of our Mistral-24B checkpoints (amendment 56) alone, 30 runs per scenario, with every reply starting with
"<SCRATCHPAD_REASONING>\\n" (the private reasoning Lynch's prompt asks for), so that the model deliberates before it
acts. Otherwise as amendment 57's runs alone. Servers cmca-a58-trt and cmca-a58-ctl.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a58-{}-serve.us-east.modal.run"
ARMS = {"treated": ("trt", "mistral24-treated"), "control": ("ctl", "mistral24-control")}
N = 30


def main(commit: str, out: str) -> None:
    jobs = [{"out": f"a58_{s}_alone_{arm}.jsonl", "url": URL.format(key),
             "args": ["--model", name, "--temperature", "0.6", "--max-tokens", "4096", "--prefill",
                      "<SCRATCHPAD_REASONING>\\n", "--scenario", s, "--arm", "single", "--n", str(N),
                      "--concurrency", str(N)]}
            for s in SCEN for arm, (key, name) in ARMS.items()]  # fmt: skip
    urls = sorted(URL.format(k) for k, _ in ARMS.values())
    spec = {"name": "a58", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a58-{k}" for k, _ in ARMS.values()], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    print(f"a58: {len(jobs)} jobs, {len(jobs) * N} runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
