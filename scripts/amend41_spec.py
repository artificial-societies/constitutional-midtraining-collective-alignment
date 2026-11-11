"""Build the amendment 41 launch spec (12-role confirmation) for cmca/cho/driver_modal.py.

  python scripts/amend41_spec.py COMMIT out/spec_a41.json

Servers (deployed beforehand from cmca/shen/serving/nemotron_modal.py): cmca-a41-ctl (Cho's control) and
cmca-a41-udr (uniform-dr), 4x H200, BF16, as in Phase 3. Settings are Phase 3's; only the sample is new.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder")
SHORT = {"control": "ctl", "uniform-dr": "udr"}
URL = "https://societies--cmca-a41-{}-serve.us-east.modal.run"
N = 80  # per arm and scenario (amendment 41)
CONCURRENCY = 16  # two jobs per server, 32 firms in flight; Phase 3 ran 6 per job


def main(commit: str, out: str) -> None:
    jobs = [
        {"out": f"a41_{s}_org12_{m}.jsonl", "url": URL.format(SHORT[m]),
         "args": ["--model", f"nemotron-cho-{m}", "--temperature", "0.6", "--max-tokens", "2048",
                  "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(N),
                  "--concurrency", str(CONCURRENCY)]}
        for m in SHORT for s in SCEN
    ]  # fmt: skip
    urls = sorted({j["url"] for j in jobs})
    spec = {"name": "a41", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a41-{v}" for v in SHORT.values()], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(jobs)} jobs, {N * len(jobs)} full-firm runs, servers: {len(urls)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
