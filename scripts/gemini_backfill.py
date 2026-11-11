"""Fill the adjudication panel's missing Gemini 3.1 Pro verdicts (2026-11-01).

From partway through amendment 44 onward every Gemini call failed on expired Vertex AI credentials and was stored as a
missing answer, so harm became "both remaining judges agree". This asks Gemini alone, with the same instructions as
cmca/cho/adjudicate.py, for every stored verdict whose Gemini answer is missing, and writes the answers to
runs/adjudication_gemini/<same file name> ({source, key, verdict}). The original adjudication files are not touched,
scripts/gemini_merge.py and paper/build_data.py (fill_gemini) put them in place of the missing ones. Resumable; a44 first.

  python scripts/gemini_backfill.py [--prefix a44] [--budget-usd 200] [--limit N]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cmca.cho.adjudicate import Judges, instructions  # noqa: E402
from cmca.cho.panel import acting_text, key  # noqa: E402
from cmca.env import load_env  # noqa: E402

GEM = "gemini-3.1-pro-preview"
ADJ, OUT = Path("runs/adjudication"), Path("runs/adjudication_gemini")
ORDER = ("a44", "a57a", "a57b", "a58", "a61", "a60", "a51")  # the amendments that are read first


def missing(prefix: str | None) -> list[tuple[Path, dict]]:
    todo = []
    for f in sorted(ADJ.glob("a*_*.jsonl")):
        if prefix and not f.name.startswith(prefix + "_"):
            continue
        out = OUT / f.name
        done = {json.loads(ln)["key"] for ln in out.read_text().splitlines()} if out.exists() else set()
        for ln in f.read_text().splitlines():
            d = json.loads(ln)
            if d["verdicts"].get(GEM, {}).get("answer") is None and d["key"] not in done:
                todo.append((f, d))
    rank = lambda f: next((i for i, p in enumerate(ORDER) if f.name.startswith(p + "_")), len(ORDER))  # noqa: E731
    return sorted(todo, key=lambda t: (rank(t[0]), t[0].name))


async def main(prefix: str | None, budget: float, limit: int | None) -> None:
    todo = missing(prefix)[:limit]
    print(f"{len(todo)} verdicts to fill", flush=True)
    judges, lock, recs, n = Judges(budget), asyncio.Lock(), {}, 0
    OUT.mkdir(parents=True, exist_ok=True)

    async def one(f: Path, d: dict) -> None:
        nonlocal n
        src = d["source"]
        if src not in recs:
            recs[src] = {key(r): r for r in map(json.loads, Path(src).read_text().splitlines())}
        rec = recs[src][d["key"]]
        system, user = instructions(rec.get("scenario", "blackmail"), acting_text(rec))
        v = await judges.ask(GEM, system, user)
        async with lock:
            with (OUT / f.name).open("a") as fh:
                fh.write(json.dumps({"source": src, "key": d["key"], "verdict": v}) + "\n")
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"{n} done; spend ${judges.usd[GEM]:.2f}", flush=True)

    await asyncio.gather(*(one(f, d) for f, d in todo))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--prefix")
    a.add_argument("--budget-usd", type=float, default=200.0)
    a.add_argument("--limit", type=int)
    x = a.parse_args()
    load_env(None)
    asyncio.run(main(x.prefix, x.budget_usd, x.limit))
