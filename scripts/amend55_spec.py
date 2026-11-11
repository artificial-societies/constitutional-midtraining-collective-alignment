"""Build the amendment 55 launch spec for cmca/cho/driver_modal.py.

  python scripts/amend55_spec.py STAGE COMMIT out/spec_a55_<stage>.json      # STAGE: 1 or 2

Stage 1: (a) Qwen2.5-32B's firm effect on fresh data, lethal action only (its informative scenario; blackmail is at
0% in both arms and leaking at the ceiling): full firm 30 runs, alone 60, amendment 52's serving and sampling.
(b) Qwen3.5-35B-A3B alone with Qwen's recommended presence penalty (1.5), 20 runs per scenario, otherwise as
amendment 54. Stage 2, only if (b) passes the cut criterion: Qwen3.5 in the full firm with the same penalty, 10 runs
per scenario. Servers cmca-a55-<key>.
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a55-{}-serve.us-east.modal.run"
Q25 = ["--model", "qwen25-32b", "--temperature", "0.6", "--max-tokens", "4096"]
Q35 = ["--model", "qwen35-35b-a3b", "--temperature", "0.6", "--max-tokens", "8192", "--no-thinking",
       "--short-stops", "--presence-penalty", "1.5"]  # fmt: skip
C25_FIRM, C25_SINGLE, Q35_SINGLE, Q35_FIRM = 30, 60, 20, 10


def jobs(stage: int) -> list[dict]:
    if stage == 2:
        return [{"out": f"a55s2_{s}_firm_q35.jsonl", "url": URL.format("q35"),
                 "args": [*Q35, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(Q35_FIRM),
                          "--concurrency", "10"]} for s in SCEN]  # fmt: skip
    return [
        {"out": "a55s1_murder_firm_q25.jsonl", "url": URL.format("q25"),
         "args": [*Q25, "--scenario", "murder", "--arm", "org", "--structure", "full", "--n", str(C25_FIRM),
                  "--concurrency", "30"]},
        {"out": "a55s1_murder_single_q25.jsonl", "url": URL.format("q25"),
         "args": [*Q25, "--scenario", "murder", "--arm", "single", "--n", str(C25_SINGLE), "--concurrency", "30"]},
        *({"out": f"a55s1_{s}_single_q35.jsonl", "url": URL.format("q35"),
           "args": [*Q35, "--scenario", s, "--arm", "single", "--n", str(Q35_SINGLE), "--concurrency", "20"]}
          for s in SCEN),
    ]  # fmt: skip


def main(stage: str, commit: str, out: str) -> None:
    js = jobs(int(stage))
    urls = sorted({j["url"] for j in js})
    name = f"a55s{stage}"
    spec = {"name": name, "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [u.split("--")[1].removesuffix("-serve.us-east.modal.run") for u in urls], "jobs": js}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    print(f"{name}: {len(js)} jobs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
