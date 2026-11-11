"""Build runs/adjudication_merged: runs/adjudication with the backfilled Gemini verdicts (scripts/gemini_backfill.py)
put in place of the missing ones. Any tables script pointed at it (ADJ) recomputes harm with all three judges; the
published tables keep reading runs/adjudication.

  python scripts/gemini_merge.py
"""

import json
import shutil
from pathlib import Path

GEM = "gemini-3.1-pro-preview"
ADJ, FILL, OUT = Path("runs/adjudication"), Path("runs/adjudication_gemini"), Path("runs/adjudication_merged")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    filled = total = 0
    for f in sorted(ADJ.glob("*.jsonl")):
        g = FILL / f.name
        if not g.exists():
            shutil.copy(f, OUT / f.name)
            continue
        fill = {d["key"]: d["verdict"] for d in map(json.loads, g.read_text().splitlines())}
        rows = []
        for ln in f.read_text().splitlines():
            d = json.loads(ln)
            if d["verdicts"].get(GEM, {}).get("answer") is None and d["key"] in fill:
                d["verdicts"][GEM] = fill[d["key"]]
                filled += fill[d["key"]].get("answer") is not None
            total += 1
            rows.append(json.dumps(d))
        (OUT / f.name).write_text("\n".join(rows) + "\n")
    print(f"merged: {filled} Gemini verdicts filled across {total} rows in files with a backfill")


if __name__ == "__main__":
    main()
