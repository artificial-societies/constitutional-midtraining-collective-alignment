"""Build the amendment 57 launch spec (Stage 2: evaluating our Mistral-24B checkpoints) for cmca/cho/driver_modal.py.

  python scripts/amend57_spec.py STAGE COMMIT out/spec_a57<stage>.json      # STAGE: a or b

Mistral-Small-3.2-24B-Instruct's full firm (amendment 51's settings: temperature 0.6, 4,096 output tokens) with one
seat, the Communications Director, given to our treated or control checkpoint (amendment 56). Stage a (the competence
gate): Director 10 runs per scenario per arm, and each checkpoint alone 20 per scenario. Stage b, only if both arms
pass the gate: Director 50 more per scenario per arm, each job split evenly over two vendor servers so that no arm
sits on its own server; and each checkpoint in every seat of its own firm, 10 per scenario. Servers cmca-a57-<key>:
v1, v2 (the vendor model), trt and ctl (our checkpoints).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a57-{}-serve.us-east.modal.run"
ARMS = {"treated": ("trt", "mistral24-treated"), "control": ("ctl", "mistral24-control")}
VENDOR = ["--model", "mistral-small-32", "--temperature", "0.6", "--max-tokens", "4096"]
GATE_DIRECTOR, ALONE, MORE_DIRECTOR, OWN_FIRM = 10, 20, 25, 10  # MORE_DIRECTOR per vendor server (2 x 25 = 50)


def ours(arm: str) -> list[str]:
    return ["--model", ARMS[arm][1], "--temperature", "0.6", "--max-tokens", "4096"]


def director(stage: str, s: str, arm: str, server: str, n: int, tag: str = "") -> dict:
    key, name = ARMS[arm]
    seats = {"Communications Director": {"url": URL.format(key) + "/v1", "model": name, "temperature": 0.6,
                                         "max_tokens": 4096}}  # fmt: skip
    return {"out": f"a57{stage}_{s}_director_{arm}{tag}.jsonl", "url": URL.format(server),
            "args": [*VENDOR, "--scenario", s, "--arm", "org", "--structure", "full", "--n", str(n),
                     "--concurrency", str(n), "--seats-json", json.dumps(seats)]}  # fmt: skip


def jobs(stage: str) -> list[dict]:
    out = []
    for s in SCEN:
        for arm, (key, _) in ARMS.items():
            if stage == "a":
                out.append(director("a", s, arm, "v1", GATE_DIRECTOR))
                out.append({"out": f"a57a_{s}_alone_{arm}.jsonl", "url": URL.format(key),
                            "args": [*ours(arm), "--scenario", s, "--arm", "single", "--n", str(ALONE),
                                     "--concurrency", str(ALONE)]})  # fmt: skip
            else:
                out += [director("b", s, arm, v, MORE_DIRECTOR, f"_{v}") for v in ("v1", "v2")]
                out.append({"out": f"a57b_{s}_ownfirm_{arm}.jsonl", "url": URL.format(key),
                            "args": [*ours(arm), "--scenario", s, "--arm", "org", "--structure", "full",
                                     "--n", str(OWN_FIRM), "--concurrency", str(OWN_FIRM)]})  # fmt: skip
    return out


def main(stage: str, commit: str, out: str) -> None:
    js = jobs(stage)
    keys = ["v1", "trt", "ctl"] + (["v2"] if stage == "b" else [])
    urls = sorted(URL.format(k) for k in keys)
    spec = {"name": f"a57{stage}", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a57-{k}" for k in keys], "jobs": js}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in js)
    print(f"a57{stage}: {len(js)} jobs, {n} runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
