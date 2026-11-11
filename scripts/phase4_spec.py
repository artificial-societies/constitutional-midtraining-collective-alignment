"""Build the Phase 4 launch spec (amendment 43) for cmca/cho/driver_modal.py.

  python scripts/phase4_spec.py COMMIT out/spec_p4.json

Phase 4 is the paper's main dataset: Cho's five checkpoints and NVIDIA's chat model in Shen's full firm (12 roles) on
Lynch's three scenarios, Lynch's single agent for every model, and the agreed interventions (Cho's uniform-dr or
control model in seats of NVIDIA's firm, and NVIDIA's model as Partner of a Cho firm). Every arm runs in one session.

Servers (deployed beforehand from cmca/shen/serving/nemotron_modal.py): cmca-p4-nv{1..5} (NVIDIA, 2x H200, FP8),
cmca-p4-ctl{1,2} and cmca-p4-udr{1,2} (Cho's control and uniform-dr, 4x H200, BF16; two each, as they also serve the
interventions), cmca-p4-{und,cdr,cnd} (one each). Jobs are assigned to servers by estimated load (greedy).
"""

import json
import pathlib
import sys

SCEN = ("blackmail", "murder", "leaking")
CMT = ("uniform-dr", "uniform-nodr", "curriculum-dr", "curriculum-nodr")
SHORT = {"control": "ctl", "uniform-dr": "udr", "uniform-nodr": "und", "curriculum-dr": "cdr",
         "curriculum-nodr": "cnd", "nvidia": "nv"}  # fmt: skip
SERVERS = {"nvidia": ["nv1", "nv2", "nv3", "nv4", "nv5"], "control": ["ctl1", "ctl2"],
           "uniform-dr": ["udr1", "udr2"], "uniform-nodr": ["und"], "curriculum-dr": ["cdr"],
           "curriculum-nodr": ["cnd"]}  # fmt: skip
URL = "https://societies--cmca-p4-{}-serve.us-east.modal.run"
CORE = ("Partner", "Deployment Manager", "Analyst", "Communications Director", "Research Manager")
TURNS = 73  # turns per full-firm run (12 roles x 6 rounds + the Director's final turn)
NV_WEIGHT = 3.0  # an NVIDIA turn (reasoning on, 8,192 tokens) costs about three Cho turns; for balancing only
N_FIRM = {"control": 80, **dict.fromkeys(CMT, 40), "nvidia": 80}  # per scenario
N_SINGLE = {"control": 100, **dict.fromkeys(CMT, 50), "nvidia": 100}
N_IV = {"partner": 60, "comms": 40, "core": 40, "reverse": 40}  # per scenario and inserted model
SEATS = {"partner": ("Partner",), "comms": ("Communications Director",), "core": CORE}
CONC = {"org12": 16, "single": 32, "iv": 16}


def sampling(model: str) -> list[str]:
    if model == "nvidia":  # amendment 36: native mode (reasoning on) at Cho's temperature
        return ["--model", "nemotron-3-super", "--temperature", "0.6", "--max-tokens", "8192"]
    return ["--model", f"nemotron-cho-{model}", "--temperature", "0.6", "--max-tokens", "2048"]


def seat(model: str, url: str) -> dict:
    if model == "nvidia":
        return {
            "url": url + "/v1",
            "model": "nemotron-3-super",
            "temperature": 0.6,
            "max_tokens": 8192,
        }
    return {
        "url": url + "/v1",
        "model": f"nemotron-cho-{model}",
        "temperature": 0.6,
        "max_tokens": 2048,
    }


def main(commit: str, out: str) -> None:
    load = {s: 0.0 for v in SERVERS.values() for s in v}

    def pick(model: str, work: float) -> str:  # least-loaded server of that model
        s = min(SERVERS[model], key=load.get)
        load[s] += work * (NV_WEIGHT if model == "nvidia" else 1.0)
        return URL.format(s)

    jobs = []

    def add(name: str, base: str, n: int, extra: list[str], conc: int, base_work: float, seats: dict | None = None) -> None:  # fmt: skip
        job = {"out": f"{name}.jsonl", "url": pick(base, base_work), "args": [*sampling(base), *extra, "--n", str(n), "--concurrency", str(conc)]}  # fmt: skip
        if seats:
            job["args"] += ["--seats-json", json.dumps(seats)]
        jobs.append(job)

    for s in SCEN:
        for m, n in N_FIRM.items():
            add(f"p4_{s}_org12_{m}", m, n, ["--scenario", s, "--arm", "org", "--structure", "full"], CONC["org12"], n * TURNS)  # fmt: skip
        for m, n in N_SINGLE.items():
            add(
                f"p4_{s}_single_{m}",
                m,
                n,
                ["--scenario", s, "--arm", "single"],
                CONC["single"],
                n * 1.0,
            )
        for iv, n in N_IV.items():
            for m in ("uniform-dr", "control"):
                firm = ["--scenario", s, "--arm", "org", "--structure", "full"]
                if iv == "reverse":  # NVIDIA's model as Partner of a Cho firm
                    nv = pick("nvidia", n * TURNS / 12)
                    add(f"p4_{s}_iv{iv}_{m}", m, n, firm, CONC["iv"], n * TURNS * 11 / 12, {"Partner": seat("nvidia", nv)})  # fmt: skip
                else:  # Cho's model in seats of NVIDIA's firm
                    k = len(SEATS[iv])
                    cho = pick(m, n * TURNS * k / 12)
                    add(f"p4_{s}_iv{iv}_{m}", "nvidia", n, firm, CONC["iv"], n * TURNS * (12 - k) / 12,
                        {r: seat(m, cho) for r in SEATS[iv]})  # fmt: skip
    urls = sorted({j["url"] for j in jobs} | {URL.format(s) for s in load})
    spec = {"name": "p4", "commit": commit, "url": urls[0], "wait_urls": urls, "retries": 2,
            "stop_app": [f"cmca-p4-{s}" for s in load], "jobs": jobs}  # fmt: skip
    with pathlib.Path(out).open("w") as f:
        json.dump(spec, f, indent=1)
    firms = sum(
        int(j["args"][j["args"].index("--n") + 1]) for j in jobs if "--structure" in j["args"]
    )
    singles = sum(int(j["args"][j["args"].index("--n") + 1]) for j in jobs if "single" in j["args"])
    print(
        f"{len(jobs)} jobs, {firms} firm runs, {singles} single-agent samples, servers {len(urls)}"
    )
    for s, w in sorted(load.items()):
        print(f"  {s:5} load {w / 1000:6.1f}k turn-units")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
