"""Run cmca.cho.shen_org jobs inside Modal, so a session survives the laptop disconnecting (amendment 35 on).

The GPU server is deployed from the laptop as before (cmca/shen/serving/nemotron_modal.py: one container, scaled
down after 10 idle minutes, hard lifetime SERVE_MINUTES). This CPU function then runs the jobs against it, detached:
API keys travel in an ephemeral secret built from the local environment at launch (nothing is stored in the
workspace), outputs go to the `cmca-runs` Volume under the run's name, and each job resumes where it stopped. When
the jobs finish, the driver stops the GPU app itself; if the driver dies, the server's own idle timeout stops it.

  MODAL_PROFILE=engine modal run --detach cmca/cho/driver_modal.py --spec path/to/spec.json
  MODAL_PROFILE=engine modal volume get cmca-runs <name>/ runs/cho_shen/   # fetch outputs later

spec.json: {"name": ..., "url": ..., "wait_urls": [...], "stop_app": name or [names], "retries": k,
            "jobs": [{"out": "x.jsonl", "url": optional per-job server, "args": [...]}]}
Jobs whose output is short of --n after a pass (failed samples) are rerun up to `retries` times; the harness resumes.
"""

import json
import os
from pathlib import Path

import modal

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NEMOTRON_API_KEY", "GEMINI_API_KEY")
app = modal.App("cmca-driver")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("openai==3.31.0", "httpx2==2.13.1", "modal==1.6.1")
    .add_local_dir(Path(__file__).parents[1], "/root/cmca")
)
runs = modal.Volume.from_name("cmca-runs", create_if_missing=True)
# Built from the launching shell's environment; inside the container the same names are already set by this secret.
secret = modal.Secret.from_dict({k: os.environ[k] for k in KEYS if k in os.environ})


@app.function(
    image=image, volumes={"/runs": runs}, secrets=[secret], timeout=6 * 3600, cpu=4, memory=16384
)
def drive(spec: dict) -> dict:
    import contextlib
    import subprocess
    import time
    import urllib.request

    out = Path("/runs") / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ["NEMOTRON_API_KEY"]
    bases = spec.get(
        "wait_urls", [spec["url"]]
    )  # every GPU server the jobs use (interventions mix several)

    def ready(urls: list[str]) -> None:
        # Poll every server each round: the first round wakes them all at once (cold starts overlap), and later
        # rounds keep the ones already up from idling out (10 minutes) while slower ones are still starting.
        up: set[str] = set()
        for _ in range(240 if spec.get("wait", True) else 0):  # a cold start takes up to 30 min
            for base in urls:
                req = urllib.request.Request(
                    base + "/v1/models", headers={"Authorization": f"Bearer {key}"}
                )
                with contextlib.suppress(Exception):
                    urllib.request.urlopen(req, timeout=5 if base not in up else 30)
                    up.add(base)
            if len(up) == len(urls):
                return
            time.sleep(15)

    ready(bases)
    (out / "READY").write_text(time.strftime("%H:%M UTC"))
    (out / "spec.json").write_text(
        json.dumps(spec, indent=1)
    )  # the exact launch (commit, jobs, settings)
    runs.commit()

    def launch(job: dict) -> subprocess.Popen:
        cmd = ["python", "-m", "cmca.cho.shen_org", "--url", job.get("url", spec["url"]) + "/v1", *job["args"],
               "--out", str(out / job["out"])]  # fmt: skip
        log = Path(out / (job["out"] + ".log")).open("a")  # noqa: SIM115  # held open for the job's lifetime
        return subprocess.Popen(cmd, cwd="/root", stdout=log, stderr=subprocess.STDOUT)

    # fewer rows than requested: failed samples to retry (the harness resumes)
    def short(job: dict) -> bool:
        f, n = out / job["out"], int(job["args"][job["args"].index("--n") + 1])
        return (sum(1 for _ in f.open()) if f.exists() else 0) < n

    procs = {}
    for attempt in range(1 + spec.get("retries", 0)):
        todo = [j for j in spec["jobs"] if attempt == 0 or short(j)]
        if attempt:  # servers that went idle have scaled down: wake the ones the retries need first
            ready(sorted({j.get("url", spec["url"]) for j in todo}))
        procs |= {j["out"]: launch(j) for j in todo}
        while any(procs[j["out"]].poll() is None for j in todo):
            time.sleep(60)
            runs.commit()  # partial results are visible (and survive) while the jobs run
        if not any(short(j) for j in spec["jobs"]):
            break
    status = {
        j["out"]: {"returncode": procs[j["out"]].returncode, "complete": not short(j)}
        for j in spec["jobs"]
    }
    for name in (
        [spec["stop_app"]] if isinstance(spec.get("stop_app"), str) else spec.get("stop_app", [])
    ):
        subprocess.run(["modal", "app", "stop", name, "-y"], check=False)
    (out / "DONE").write_text(json.dumps(status))
    runs.commit()
    return status


@app.local_entrypoint()
def main(spec: str) -> None:
    call = drive.spawn(json.loads(Path(spec).read_text()))
    print("spawned", call.object_id)
