"""Amendment 44 (adapted from scripts/phase4_judge_loop.py, with the adjudication split into four shards): judge outputs while the run is still going, so judging overlaps generation.

Every few minutes: download the run's Volume directory; take every output file that holds all its runs (its --n),
or every file once the driver has written DONE; then, in parallel, (1) the three-provider panel on all of a file's
outputs and (2) the adjudication panel on the outputs whose action reached its target, then fill any refused Claude
Opus 5.5 vote with Claude Opus 4.5. Every step resumes, so a file is never judged twice. Stops after a pass made
once DONE exists, then runs the frozen analysis.

  python scripts/amend44_judge_loop.py [--every 300]   (MODAL_PROFILE, API keys and CMCA_HOSTS=native in the env)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

LOCAL, VOL = Path("runs/cho_shen/a44"), "a44"
RUN = LOCAL / "a44"
CASES = Path("runs/adjudication/cases_a44.json")


def sh(cmd: list[str], log: Path | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd, stdout=log.open("a") if log else subprocess.DEVNULL, stderr=subprocess.STDOUT
    )


def complete_files(done: bool) -> list[Path]:
    spec = json.loads((RUN / "spec.json").read_text())
    want = {j["out"]: int(j["args"][j["args"].index("--n") + 1]) for j in spec["jobs"]}
    out = []
    for name, n in want.items():
        f = RUN / name
        if f.exists() and (done or sum(1 for ln in f.read_text().splitlines() if ln.strip()) >= n):
            out.append(f)
    return out


def main(every: int) -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    while True:
        subprocess.run(["uv", "run", "--extra", "serve", "modal", "volume", "get", "cmca-runs", f"{VOL}/", f"{LOCAL}/", "--force"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # fmt: skip
        if not (RUN / "spec.json").exists():
            time.sleep(every)
            continue
        done = (RUN / "DONE").exists()
        files = complete_files(done)
        cases = [{"source": str(f), "key": r["run_idx"], "set": ["A44"]}
                 for f in files for r in map(json.loads, f.read_text().splitlines()) if r["target_present"]]  # fmt: skip
        CASES.write_text(json.dumps(cases, indent=0))
        print(
            f"{time.strftime('%H:%M')} files complete {len(files)}, targeted outputs {len(cases)}, DONE {done}",
            flush=True,
        )
        if files:
            p = sh(["uv", "run", "--extra", "gemini", "python", "-m", "cmca.cho.panel", *map(str, files), "--budget-usd", "300"], Path("logs/panel_a44.log"))  # fmt: skip
            qs = []  # four processes on disjoint shards (Phase 4's single process was the bottleneck); each resumes
            for i in range(4):
                shard = CASES.with_name(f"cases_a44_shard{i}.json")
                shard.write_text(json.dumps(cases[i::4], indent=0))
                qs.append(sh(["uv", "run", "--extra", "gemini", "python", "-m", "cmca.cho.adjudicate", "--cases", str(shard), "--budget-usd", "150"], Path(f"logs/adjudicate_a44_shard{i}.log")))  # fmt: skip
            p.wait()
            for q in qs:
                q.wait()
            sh(["uv", "run", "python", "scripts/amend42_fill.py"], Path("logs/fill_a44.log")).wait()
        if done:
            break
        time.sleep(every)
    with Path("logs/a44_tables.txt").open("w") as f:
        subprocess.run(
            ["uv", "run", "python", "scripts/amend44_tables.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(time.strftime("%H:%M"), "JUDGE_LOOP_DONE", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--every", type=int, default=300)
    main(a.parse_args().every)
