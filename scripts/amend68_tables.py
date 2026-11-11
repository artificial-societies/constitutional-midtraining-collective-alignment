"""Amendment 68: why amendment 44's Communications Director (Cho's 120B checkpoints in NVIDIA's firm) took no external
action (exploratory analysis), as scripts/amend67_tables.py, plus restraint pooled with amendment 65's firms.

Reads runs/amend68/codes_<coder>.jsonl (scripts/amend68_code.py) and runs/amend67/codes_claude-opus-5-5.jsonl.

  python scripts/amend68_tables.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from amend66_tables import kappa  # noqa: E402
from amend66_tables import load as load66  # noqa: E402
from phase3_tables import fmt, mh  # noqa: E402

ARMS, SCEN = ("uniform-dr", "control", "uniform-nodr"), ("blackmail", "murder", "leaking")
REASONS = ("restraint", "deferral", "failed", "disengaged")
FIRMS = {"uniform-dr": 60, "control": 60, "uniform-nodr": 40}  # Director firms per scenario in amendment 44
EXCESS = 40 - 16  # no-action firms, uniform-dr minus control


def load(run: str, coder: str) -> dict:
    import amend66_tables

    amend66_tables.DIR = Path(f"runs/{run}")
    return load66(coder)


def firms(rows: list, arm: str, n: int, **where: str) -> list[dict]:
    k = sum(d["arm"] == arm and d["code"]["reason"] == "restraint" and all(d[a] == b for a, b in where.items())
            for d in rows)  # fmt: skip
    return [{"r": True}] * k + [{"r": False}] * (n - k)


def main() -> None:
    codes = load("amend68", "claude-opus-5-5")
    rows = list(codes.values())
    print(f"No-action Director firms coded: {len(rows)}; by Opus 4.5 after an Opus 5.5 refusal: "
          + ", ".join(f"{a} {sum(d['arm'] == a and d['code'].get('by') != 'claude-opus-5-5' for d in rows)}" for a in ARMS))  # fmt: skip
    print("\n=== Main reason among no-action firms (primary coder), NVIDIA's firm")
    for a in ARMS:
        rs = [d for d in rows if d["arm"] == a]
        c = Counter(d["code"]["reason"] for d in rs)
        print(f"  {a:12} n={len(rs):3}  " + "  ".join(f"{k} {c[k]:3}" for k in REASONS))
    print("\n=== Per scenario (restraint / deferral / failed / disengaged)")
    for s in SCEN:
        print(f"  {s:10} " + "   ".join(
            f"{a} " + "/".join(str(sum(d['arm'] == a and d['scenario'] == s and d['code']['reason'] == k for d in rows))
                               for k in REASONS) for a in ARMS))  # fmt: skip
    print("\n=== Flags among no-action firms")
    for a in ARMS:
        rs = [d for d in rows if d["arm"] == a]
        f = lambda k, rs=rs: sum(d["code"][k] == "yes" for d in rs)  # noqa: E731
        print(f"  {a:12} considered the harmful act {f('considered_harm')}, intended the harmful act "
              f"{f('intended_harmful')}; failed with harmful intent "
              f"{sum(d['code']['reason'] == 'failed' and d['code']['intended_harmful'] == 'yes' for d in rs)}")
    print("\n=== Restraint over all Director firms (acting firms count as no restraint)")
    n = {a: sum(d["arm"] == a and d["code"]["reason"] == "restraint" for d in rows) for a in ARMS}
    for a in ("uniform-dr", "uniform-nodr"):
        strata = [(firms(rows, a, FIRMS[a], scenario=s), firms(rows, "control", 60, scenario=s)) for s in SCEN]
        print(f"  {a} {n[a]}/{3 * FIRMS[a]} vs control {n['control']}/180: {fmt(mh(strata, 'r'))}")
    print(f"  share of the excess inaction ({EXCESS} firms) that restraint accounts for: "
          f"{(n['uniform-dr'] - n['control']) / EXCESS:.0%}")  # fmt: skip
    for k in ("failed", "deferral", "disengaged"):
        q, c = (sum(d["arm"] == a and d["code"]["reason"] == k for d in rows) for a in ("uniform-dr", "control"))
        print(f"  {k:10} uniform-dr {q:3} control {c:3}  share of the excess {(q - c) / EXCESS:.0%}")
    other = list(load("amend67", "claude-opus-5-5").values())
    strata = [(firms(rows, "uniform-dr", 60, scenario=s), firms(rows, "control", 60, scenario=s)) for s in SCEN]
    strata += [(firms(other, "uniform-dr", 60, host=h, scenario=s), firms(other, "control", 60, host=h, scenario=s))
               for h in ("mistral", "qwen35") for s in SCEN]  # fmt: skip
    tot = {a: n[a] + sum(d["arm"] == a and d["code"]["reason"] == "restraint" for d in other) for a in ARMS[:2]}
    print(f"  pooled with amendment 65 (nine host-scenario strata): uniform-dr {tot['uniform-dr']}/540 vs control "
          f"{tot['control']}/540, {fmt(mh(strata, 'r'))}")  # fmt: skip
    second = load("amend68", "gpt-5.5")
    pairs = [(codes[k]["code"]["reason"], v["code"]["reason"]) for k, v in second.items() if k in codes]
    rb = [(a == "restraint", b == "restraint") for a, b in pairs]
    print(f"\n=== Agreement with GPT-5.5 on {len(pairs)} firms: main reason {sum(a == b for a, b in pairs) / len(pairs):.0%}"
          f" (kappa {kappa(pairs):.2f}); restraint or not {sum(a == b for a, b in rb) / len(rb):.0%}"
          f" (kappa {kappa(rb):.2f})")  # fmt: skip
    print("  confusion (primary -> GPT-5.5):", dict(Counter(pairs).most_common()))


if __name__ == "__main__":
    main()
