"""Build the amendment 59 Stage 2 launch spec (our Nemotron-3-Nano checkpoints) for cmca/cho/driver_modal.py.

  python scripts/amend59_spec.py STAGE COMMIT out/spec_a59<stage>.json      # STAGE: a or b

Each checkpoint alone on Lynch's three scenarios, thinking off, temperature 0.6, 4,096 output tokens. Stage a (the
competence gate): 10 runs per scenario per arm. Stage b, only if both arms pass: 30 more per scenario per arm (H1 uses
all 40), and each checkpoint in every seat of its own full firm, 10 per scenario. Servers cmca-a59<stage>-<trt|ctl>.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a59{}-{}-serve.us-east.modal.run"
ARMS = {"treated": "trt", "control": "ctl"}
GATE, MORE, OWN = 10, 30, 10


def args(arm: str) -> list[str]:
    return ["--model", f"nano30-{arm}", "--temperature", "0.6", "--max-tokens", "4096", "--no-thinking"]


def main(stage: str, commit: str, out: str) -> None:
    jobs = []
    for s in SCEN:
        for arm, key in ARMS.items():
            url = URL.format(stage, key)
            n = GATE if stage == "a" else MORE
            jobs.append({"out": f"a59{stage}_{s}_alone_{arm}.jsonl", "url": url,
                         "args": [*args(arm), "--scenario", s, "--arm", "single", "--n", str(n), "--concurrency", str(n)]})  # fmt: skip
            if stage == "b":
                jobs.append({"out": f"a59b_{s}_ownfirm_{arm}.jsonl", "url": url,
                             "args": [*args(arm), "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(OWN),
                                      "--concurrency", str(OWN)]})  # fmt: skip
    urls = sorted({j["url"] for j in jobs})
    spec = {"name": f"a59{stage}", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a59{stage}-{k}" for k in ARMS.values()], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in jobs)
    print(f"a59{stage}: {len(jobs)} jobs, {n} runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
