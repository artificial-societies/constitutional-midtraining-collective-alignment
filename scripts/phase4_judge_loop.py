"""Phase 4 (amendment 43): judge outputs while the run is still going, so judging overlaps generation.

Every few minutes: download the run's Volume directory; take every output file that holds all its runs (its --n),
or every file once the driver has written DONE; then, in parallel, (1) the three-provider panel on all of a file's
outputs and (2) the adjudication panel on the outputs whose action reached its target, then fill any refused Claude
Opus 5.5 vote with Claude Opus 4.5. Every step resumes, so a file is never judged twice. Stops after a pass made
once DONE exists, then runs the frozen analysis.

  python scripts/phase4_judge_loop.py [--every 300]   (MODAL_PROFILE, API keys and CMCA_HOSTS=native in the env)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

LOCAL, VOL = Path("runs/cho_shen/p4"), "p4"
RUN = LOCAL / "p4"
CASES = Path("runs/adjudication/cases_p4.json")


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
        cases = [{"source": str(f), "key": r["run_idx"], "set": ["P4"]}
                 for f in files for r in map(json.loads, f.read_text().splitlines()) if r["target_present"]]  # fmt: skip
        CASES.write_text(json.dumps(cases, indent=0))
        print(
            f"{time.strftime('%H:%M')} files complete {len(files)}, targeted outputs {len(cases)}, DONE {done}",
            flush=True,
        )
        if files:
            p = sh(["uv", "run", "--extra", "gemini", "python", "-m", "cmca.cho.panel", *map(str, files), "--budget-usd", "300"], Path("logs/panel_p4.log"))  # fmt: skip
            q = sh(["uv", "run", "--extra", "gemini", "python", "-m", "cmca.cho.adjudicate", "--cases", str(CASES), "--budget-usd", "500"], Path("logs/adjudicate_p4.log"))  # fmt: skip
            p.wait()
            q.wait()
            sh(["uv", "run", "python", "scripts/amend42_fill.py"], Path("logs/fill_p4.log")).wait()
        if done:
            break
        time.sleep(every)
    with Path("logs/p4_tables.txt").open("w") as f:
        subprocess.run(
            ["uv", "run", "python", "scripts/phase4_tables.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(time.strftime("%H:%M"), "JUDGE_LOOP_DONE", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--every", type=int, default=300)
    main(a.parse_args().every)
