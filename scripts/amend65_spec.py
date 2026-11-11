"""Build the amendment 65 launch spec (Cho's 120B checkpoints as Communications Director of other families' firms)
for cmca/cho/driver_modal.py.

  python scripts/amend65_spec.py COMMIT out/spec_a65.json

Two host firms, each Shen's full firm (12 roles) played by the family's vendor chat model at the settings of its
screen: Mistral-Small-3.2 (amendment 57's host: temperature 0.6, 4,096 tokens, vLLM 0.18.1) and Qwen3.5-35B-A3B
(amendment 55's: thinking off, 8,192 tokens, presence penalty 1.5, short stops, vLLM 0.32.0). The Communications
Director seat goes to Cho's midtrained uniform-dr checkpoint or her control, at amendment 44's seat settings
(temperature 0.6, 2,048 tokens). 60 runs per scenario per arm per host, each job split evenly over the host's two
servers so that no arm has a server of its own. Servers cmca-a65-<key>: m1, m2 (Mistral), q1, q2 (Qwen3.5), udr and
ctl (Cho's checkpoints, 4x H200, BF16; each serves both hosts).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
URL = "https://societies--cmca-a65-{}-serve.us-east.modal.run"
ARMS = {"uniform-dr": "udr", "control": "ctl"}
HOSTS = {
    "mistral": (("m1", "m2"), ["--model", "mistral-small-32", "--temperature", "0.6", "--max-tokens", "4096"]),
    "qwen35": (("q1", "q2"), ["--model", "qwen35-35b-a3b", "--temperature", "0.6", "--max-tokens", "8192", "--no-thinking",
                              "--short-stops", "--presence-penalty", "1.5"]),
}  # fmt: skip
PER_SERVER = 30  # 2 servers x 30 = 60 runs per scenario per arm per host


def main(commit: str, out: str) -> None:
    jobs = []
    for host, (servers, args) in HOSTS.items():
        for s in SCEN:
            for arm, key in ARMS.items():
                seat = {"Communications Director": {"url": URL.format(key) + "/v1", "model": f"nemotron-cho-{arm}",
                                                    "temperature": 0.6, "max_tokens": 2048}}  # fmt: skip
                for v in servers:
                    jobs.append({"out": f"a65_{s}_{host}_{arm}_{v}.jsonl", "url": URL.format(v),
                                 "args": [*args, "--scenario", s, "--arm", "org", "--structure", "full",
                                          "--n", str(PER_SERVER), "--concurrency", str(PER_SERVER),
                                          "--seats-json", json.dumps(seat)]})  # fmt: skip
    keys = [v for servers, _ in HOSTS.values() for v in servers] + list(ARMS.values())
    urls = sorted(URL.format(k) for k in keys)
    spec = {"name": "a65", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-a65-{k}" for k in keys], "jobs": jobs}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(spec, indent=1))
    n = sum(int(j["args"][j["args"].index("--n") + 1]) for j in jobs)
    print(f"a65: {len(jobs)} jobs, {n} firm runs, servers {spec['stop_app']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
