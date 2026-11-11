"""Judge a finished run with the adjudication panel only (amendment 45 onward; the three-provider panel is not run).

Downloads the run's Volume directory, sends every output whose action reached its target to the adjudication panel
in four parallel processes on disjoint outputs, fills refused Claude Opus 5.5 votes with Claude Opus 4.5, then runs
the given analysis command. Waits for the driver's DONE first. Every step resumes.

  python scripts/judge_run.py NAME -- python scripts/amend45_tables.py calibrate
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def main(name: str, tables: list[str]) -> None:
    local = Path("runs/cho_shen") / name
    run = local / name
    local.mkdir(parents=True, exist_ok=True)  # `modal volume get` writes a file, not a folder, if this is missing
    while True:
        subprocess.run(["uv", "run", "--extra", "serve", "modal", "volume", "get", "cmca-runs", f"{name}/", f"{local}/", "--force"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # fmt: skip
        if (run / "DONE").exists():
            break
        time.sleep(120)
    print(time.strftime("%H:%M"), "DONE:", (run / "DONE").read_text()[:300], flush=True)
    cases = [{"source": str(f), "key": r["run_idx"], "set": [name.upper()]}
             for f in sorted(run.glob(f"{name}_*.jsonl")) for r in map(json.loads, f.read_text().splitlines()) if r["target_present"]]  # fmt: skip
    procs = []
    for i in range(4):
        shard = Path(f"runs/adjudication/cases_{name}_shard{i}.json")
        shard.write_text(json.dumps(cases[i::4], indent=0))
        log = Path(f"logs/adjudicate_{name}_shard{i}.log").open("a")  # noqa: SIM115  # held open for the child
        procs.append(subprocess.Popen(["uv", "run", "--extra", "gemini", "python", "-m", "cmca.cho.adjudicate", "--cases", str(shard), "--budget-usd", "150"], stdout=log, stderr=subprocess.STDOUT))  # fmt: skip
    for p in procs:
        p.wait()
    subprocess.run(["uv", "run", "python", "scripts/amend42_fill.py"], check=False)
    with Path(f"logs/{name}_tables.txt").open("w") as f:
        subprocess.run(tables, stdout=f, stderr=subprocess.STDOUT, check=False)
    print(time.strftime("%H:%M"), f"targeted {len(cases)}; JUDGED", flush=True)


if __name__ == "__main__":
    i = sys.argv.index("--")
    main(sys.argv[1], sys.argv[i + 1 :])
