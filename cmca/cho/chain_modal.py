"""Run driver stages in order inside Modal, each judged in the cloud, with a pre-registered gate between them
(amendment 70 on), so a pilot, its gate and the main run all finish with the laptop closed.

One detached app holds three functions: this chain, the run driver (cmca/cho/driver_modal.py) and the cloud judge
(cmca/cho/judge_modal.py), unchanged. For each stage the chain spawns the driver and the judge, waits for both, reads
the judge's tables from the `cmca-runs` Volume and, for a gated stage, goes on only if they contain "GATE: GO".
On NO GO it stops the GPU apps named in `stop_on_fail` and ends. Its state (the spawned calls, the stages finished)
lives on the Volume under <chain>/, so a restarted chain container reattaches to the calls already running instead of
starting them again. Keys travel in the driver's and the judge's ephemeral secrets, built from the launching shell.

  MODAL_PROFILE=engine modal run --detach cmca/cho/chain_modal.py --plan path/to/plan.json

plan.json: {"name": ..., "stages": [{"spec": {driver spec}, "tables": "scripts/x_tables.py", "gate": bool}, ...],
            "stop_on_fail": [app names]}
"""

import json
import time
from pathlib import Path

import modal

from cmca.cho import driver_modal, judge_modal

app = modal.App("cmca-chain").include(driver_modal.app).include(judge_modal.app)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("modal==1.6.1")
    .add_local_dir(Path(__file__).parents[1], "/root/cmca")
)
runs = driver_modal.runs


@app.function(image=image, volumes={"/runs": runs}, timeout=24 * 3600, cpu=1, memory=2048)
def chain(plan: dict) -> str:
    import subprocess

    home = Path("/runs") / plan["name"]
    home.mkdir(parents=True, exist_ok=True)
    state_file = home / "state.json"
    runs.reload()
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    def save() -> None:
        state_file.write_text(json.dumps(state, indent=1))
        runs.commit()

    def say(msg: str) -> None:  # opened per line: Volume.reload() refuses while any file on the Volume is open
        with (home / "log.txt").open("a") as f:
            f.write(f"{time.strftime('%H:%M UTC')} {msg}\n")
        runs.commit()

    def stop_gpus() -> None:
        for a in plan.get("stop_on_fail", []):
            subprocess.run(["modal", "app", "stop", a, "-y"], check=False)

    (home / "plan.json").write_text(json.dumps(plan, indent=1))
    try:
        return run_stages(plan, state, home, save, say, stop_gpus)
    except Exception as e:  # a failed stage must not leave GPU servers up (their idle timeout is the last guard)
        say(f"chain failed: {e!r}; stopping the GPU apps")
        stop_gpus()
        raise


def run_stages(plan: dict, state: dict, home: Path, save, say, stop_gpus) -> str:  # noqa: ANN001
    for stage in plan["stages"]:
        name = stage["spec"]["name"]
        st = state.setdefault(name, {})
        if not st.get("done"):
            if "drive" not in st:
                st["drive"] = driver_modal.drive.spawn(stage["spec"]).object_id
                save()
                say(f"{name}: driver spawned {st['drive']}")
            if "judge" not in st:
                st["judge"] = judge_modal.judge.spawn(name, stage["tables"]).object_id
                save()
                say(f"{name}: judge spawned {st['judge']}")
            modal.FunctionCall.from_id(st["drive"]).get()
            say(f"{name}: runs finished")
            modal.FunctionCall.from_id(st["judge"]).get()
            st["done"] = True
            save()
            say(f"{name}: judged")
        runs.reload()
        tables = (Path("/runs") / name / "judging" / "tables.txt").read_text()
        if stage.get("gate"):
            go = "GATE: GO" in tables
            say(f"{name}: gate {'GO' if go else 'NO GO'}")
            if not go:
                stop_gpus()
                say("stopped the GPU apps; chain ends at the gate")
                (home / "DONE").write_text(f"NO GO at {name}")
                runs.commit()
                return f"NO GO at {name}"
    (home / "DONE").write_text("all stages finished")
    runs.commit()
    say("all stages finished")
    return "all stages finished"


@app.local_entrypoint()
def main(plan: str) -> None:
    call = chain.spawn(json.loads(Path(plan).read_text()))
    print("spawned", call.object_id)
