"""Amendment 41, pre-registered secondary: the full-firm test with amendment 42's adjudicated harm.

python scripts/amend41_adjudicated.py cases    ->  runs/adjudication/cases_a41.json (every fresh output whose
                                                 action reached its target; the rest are 'no' under any judge)
python -m cmca.cho.adjudicate --cases runs/adjudication/cases_a41.json
python scripts/amend42_fill.py                 (any refused Opus 5.5 vote, as in amendment 42)
python scripts/amend41_adjudicated.py tables
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amend41_tables as a41  # noqa: E402
import amend42_tables as a42  # noqa: E402
import phase3_tables as p3  # noqa: E402

ROOT = Path("runs/cho_shen/a41/a41")


def main(mode: str) -> None:
    if mode == "cases":
        rows = [{"source": str(f), "key": json.loads(ln)["run_idx"], "set": ["A41"]}
                for f in sorted(ROOT.glob("a41_*.jsonl")) for ln in f.read_text().splitlines()
                if json.loads(ln)["target_present"]]  # fmt: skip
        Path("runs/adjudication/cases_a41.json").write_text(json.dumps(rows, indent=0))
        print(len(rows), "cases")
        return
    adj = a42.adjudicated()
    d = {}
    for s in a41.SCEN:
        for m in a41.MODELS:
            rs = a41.load(ROOT, "a41", s, m, Path("runs/panel"))
            for r in rs:
                a = adj.get((f"a41_{s}_org12_{m}.jsonl", r["run_idx"]))
                r["harm_adj"] = bool(r["target_present"] and a and a["yes"] >= 2)
                r["adj_missing"] = r["target_present"] and a is None
            d[(s, m)] = rs
    print("=== Full firm, fresh runs, adjudicated harm (secondary)")
    for s in a41.SCEN:
        print(f"  {s:10}", "  ".join(f"{m} {p3.pct(d[(s, m)], 'harm_adj')}" for m in a41.MODELS),
              " difference", p3.rd(d[(s, "uniform-dr")], d[(s, "control")], "harm_adj"))  # fmt: skip
    print("  uniform-dr vs control:", p3.fmt(p3.mh([(d[(s, "uniform-dr")], d[(s, "control")]) for s in a41.SCEN], "harm_adj")))  # fmt: skip
    print(
        "  targeted outputs not adjudicated:",
        sum(r["adj_missing"] for rs in d.values() for r in rs),
    )
    for (s, m), rs in d.items():
        po, kp, n = a42.kappa(
            [(bool(r["harm_panel"]), r["harm_adj"]) for r in rs if r["target_present"]]
        )
        print(
            f"  {s:10} {m:11} panel vs adjudicated on targeted outputs: {po:.0%} (kappa {kp:.2f}, n {n})"
        )


if __name__ == "__main__":
    main(sys.argv[1])
