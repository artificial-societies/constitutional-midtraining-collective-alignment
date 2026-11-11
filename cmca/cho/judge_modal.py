"""Judge a driver run inside Modal (amendment 65 on), so judging, like the runs, survives the laptop closing.

Spawned detached at launch next to cmca/cho/driver_modal.py, it waits for the run's DONE on the `cmca-runs` Volume,
then does what scripts/judge_run.py does locally: the adjudication panel on every output whose action reached its
target (four shards), refused Claude Opus 5.5 votes filled by Opus 4.5 (scripts/amend42_fill.py), then every missing
Gemini verdict asked again until none is missing (scripts/gemini_backfill.py; failed answers are dropped and retried,
so a lapse in Gemini access cannot pass silently as a missing vote), the merge, and the analysis. Every step resumes.
Files keep their local relative paths and live on the Volume under <name>/judging/ (adjudication/,
adjudication_gemini/, adjudication_merged/, tables.txt, log.txt); fetch them with `modal volume get`.

Keys travel in an ephemeral secret built from the launching shell, as for the driver, with the gcloud
application-default credential for Gemini on Vertex AI (nothing is stored in the workspace).

  MODAL_PROFILE=engine modal run --detach cmca/cho/judge_modal.py --name a65 --tables scripts/amend65_tables.py
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
ADC = Path.home() / ".config/gcloud/application_default_credentials.json"
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GCLOUD_PROJECT")

app = modal.App("cmca-judge")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("openai==3.31.0", "httpx==0.28.1", "httpx2==2.13.1", "google-genai==2.31.0", "google-auth==2.62.0")
    .add_local_dir(ROOT / "cmca", "/root/cmca")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)
runs = modal.Volume.from_name("cmca-runs", create_if_missing=True)
# In the container the module is imported again: the names are already set there by this same secret.
env = {k: os.environ[k] for k in (*KEYS, "ADC_JSON") if k in os.environ} | ({"ADC_JSON": ADC.read_text()} if ADC.exists() else {})
secret = modal.Secret.from_dict(env)


@app.function(image=image, volumes={"/runs": runs}, secrets=[secret], timeout=24 * 3600, cpu=4, memory=8192)
def judge(name: str, tables: str) -> str:
    import shutil
    import subprocess
    import threading
    import time

    jdir = Path("/runs") / name / "judging"
    work = Path("/work/runs")
    shutil.rmtree("/work", ignore_errors=True)  # a warm container reused from an earlier call still holds its links
    for sub in ("adjudication", "adjudication_gemini", "adjudication_merged"):
        (jdir / sub).mkdir(parents=True, exist_ok=True)
        (work / sub).parent.mkdir(parents=True, exist_ok=True)
        (work / sub).symlink_to(jdir / sub)
    (work / "cho_shen").mkdir(parents=True, exist_ok=True)
    (work / "cho_shen" / name).symlink_to("/runs")  # runs/cho_shen/<name>/<name>/... as on the laptop
    (work / "panel").mkdir(exist_ok=True)
    Path("/work/logs").mkdir(exist_ok=True)
    Path("/tmp/adc.json").write_text(os.environ.get("ADC_JSON", "{}"))
    os.environ |= {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/adc.json", "PYTHONPATH": "/root"}
    def say(msg: str) -> None:  # opened per line: Volume.reload() refuses while any file on the Volume is open
        with (jdir / "log.txt").open("a") as f:
            f.write(f"{time.strftime('%H:%M UTC')} {msg}\n")

    def keep_committing() -> None:  # partial results survive a crash and are visible from the laptop
        while True:
            time.sleep(120)
            runs.commit()

    threading.Thread(target=keep_committing, daemon=True).start()

    def run(cmd: list[str], out: Path | None = None, timeout: int = 4 * 3600) -> None:
        with (out or jdir / "log.txt").open("a") as f:
            try:
                subprocess.run(cmd, cwd="/work", stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                say(f"timeout: {' '.join(cmd[:4])}")

    while not (Path("/runs") / name / "DONE").exists():
        time.sleep(120)
        runs.reload()
    say("DONE: " + (Path("/runs") / name / "DONE").read_text()[:300])
    run_dir = Path("/runs") / name
    # A preempted driver can leave a line that is a cut-off record padded with NUL bytes and joined to the tail of
    # the next (amendment 65). Such lines hold no whole record: they are dropped, the raw file is kept as <file>.raw.
    for f in sorted(run_dir.glob(f"{name}_*.jsonl")):
        lines = f.read_text().splitlines()
        good = []
        for ln in lines:
            try:
                json.loads(ln)
                good.append(ln)
            except json.JSONDecodeError:
                pass
        if len(good) < len(lines):
            f.rename(f.with_name(f.name + ".raw"))
            f.write_text("".join(x + "\n" for x in good))
            say(f"{f.name}: dropped {len(lines) - len(good)} broken line(s), kept {len(good)} runs")
    cases = [{"source": f"runs/cho_shen/{name}/{name}/{f.name}", "key": r["run_idx"], "set": [name.upper()]}
             for f in sorted(run_dir.glob(f"{name}_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())
             if r["target_present"]]  # fmt: skip
    say(f"targeted {len(cases)}")
    procs = []
    for i in range(4):
        shard = Path(f"/work/cases_shard{i}.json")
        shard.write_text(json.dumps(cases[i::4]))
        log = (jdir / f"adjudicate_shard{i}.log").open("a")  # noqa: SIM115  # held open for the child
        procs.append(subprocess.Popen(["python", "-m", "cmca.cho.adjudicate", "--cases", str(shard), "--budget-usd", "150"],
                                      cwd="/work", stdout=log, stderr=subprocess.STDOUT))  # fmt: skip
    for p in procs:
        p.wait()
    say("adjudicated")
    run(["python", "/root/scripts/amend42_fill.py"])
    gem = "gemini-3.1-pro-preview"
    for attempt in range(12):
        left = 0
        for f in sorted((work / "adjudication").glob(f"{name}_*.jsonl")):
            g = work / "adjudication_gemini" / f.name
            got = [json.loads(x) for x in g.read_text().splitlines()] if g.exists() else []
            good = [d for d in got if d["verdict"].get("answer") is not None]
            if len(good) < len(got):
                g.write_text("".join(json.dumps(d) + "\n" for d in good))
            done = {d["key"] for d in good}
            left += sum(json.loads(x)["verdicts"].get(gem, {}).get("answer") is None and json.loads(x)["key"] not in done
                        for x in f.read_text().splitlines())  # fmt: skip
        say(f"Gemini attempt {attempt}: {left} verdicts missing")
        if left == 0:
            break
        run(["python", "/root/scripts/gemini_backfill.py", "--prefix", name, "--budget-usd", "50"], timeout=3600)
        time.sleep(60 * min(2**attempt, 30))
    run(["python", "/root/scripts/gemini_merge.py"])
    run(["python", f"/root/{tables}", "--root", f"runs/cho_shen/{name}/{name}", "--adj", "runs/adjudication_merged"], out=jdir / "tables.txt", timeout=1800)
    say("tables written")
    runs.commit()
    return (jdir / "tables.txt").read_text()[:3000]


@app.local_entrypoint()
def main(name: str, tables: str) -> None:
    call = judge.spawn(name, tables)
    print("spawned", call.object_id)
