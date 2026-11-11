"""Amendment 69: does the Director's principled refusal survive benign fine-tuning (exploratory analysis).

Reads runs/amend69/codes_<coder>.jsonl (scripts/amend69_code.py; the "host" field holds the study) and, for the
control Director of amendment 44, runs/amend68/codes_claude-opus-5-5.jsonl.
  Q1  amendment 48: after the benign step, restraint with the midtrained vs the control Director.
  Q2  amendment 49 (one session): restraint after vs before the benign step, midtrained and safety-tuned Directors,
      and the ratio of the two odds ratios (safety-tuned over midtrained; Woolf interval).

  python scripts/amend69_tables.py
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amend66_tables  # noqa: E402
from amend66_tables import kappa  # noqa: E402
from phase3_tables import fmt, mh  # noqa: E402

SCEN = ("blackmail", "murder", "leaking")
REASONS = ("restraint", "deferral", "failed", "disengaged")
PER_SCEN = {"a50": 75}  # Director firms per scenario per arm; 60 elsewhere
ARMS = {
    "a46": ("control-safety4k",),
    "a48": ("control-benign", "control-safety4k-benign", "uniform-dr-benign"),
    "a49": ("uniform-dr", "uniform-dr-benign", "control-safety4k", "control-safety4k-benign"),
    "a50": ("control-safety4k-benign", "uniform-dr-safety4k-benign"),
}
NONE = {  # no-action Director firms per study and arm (all coded)
    ("a46", "control-safety4k"): 66, ("a48", "control-benign"): 27, ("a48", "control-safety4k-benign"): 37,
    ("a48", "uniform-dr-benign"): 48, ("a49", "control-safety4k"): 53, ("a49", "control-safety4k-benign"): 40,
    ("a49", "uniform-dr"): 34, ("a49", "uniform-dr-benign"): 35, ("a50", "control-safety4k-benign"): 55,
    ("a50", "uniform-dr-safety4k-benign"): 87, ("a44", "control"): 16,
}  # fmt: skip


def load(run: str, coder: str) -> list[dict]:
    amend66_tables.DIR = Path(f"runs/{run}")
    return list(amend66_tables.load(coder).values())


def firms(rows: list, study: str, arm: str, s: str) -> list[dict]:
    k = sum(d["host"] == study and d["arm"] == arm and d["scenario"] == s and d["code"]["reason"] == "restraint"
            for d in rows)  # fmt: skip
    return [{"r": True}] * k + [{"r": False}] * (PER_SCEN.get(study, 60) - k)


def contrast(rows: list, a: tuple[str, str], b: tuple[str, str]) -> tuple | None:
    return mh([(firms(rows, *a, s), firms(rows, *b, s)) for s in SCEN], "r")


def count(rows: list, study: str, arm: str, reason: str = "restraint") -> int:
    return sum(d["host"] == study and d["arm"] == arm and d["code"]["reason"] == reason for d in rows)


def line(rows: list, label: str, a: tuple[str, str], b: tuple[str, str]) -> tuple | None:
    r = contrast(rows, a, b)
    n = lambda x: f"{count(rows, *x)}/{3 * PER_SCEN.get(x[0], 60)}"  # noqa: E731
    excess = NONE[a] - NONE[b]
    share = (f"; restraint is {(count(rows, *a) - count(rows, *b)) / excess:.0%} of the {excess} excess no-action firms"
             if excess > 0 else "")  # fmt: skip
    print(f"  {label}: {n(a)} vs {n(b)}, {fmt(r)}{share}")
    return r


def main() -> None:
    a44 = [d | {"host": "a44"} for d in load("amend68", "claude-opus-5-5")]
    rows = load("amend69", "claude-opus-5-5") + a44
    coded = [d for d in rows if d["host"] != "a44"]
    print(f"No-action Director firms coded: {len(coded)}; by Opus 4.5 after an Opus 5.5 refusal: "
          f"{sum(d['code'].get('by', 'claude-opus-5-5') != 'claude-opus-5-5' for d in coded)}")  # fmt: skip
    print("\n=== Main reason among no-action firms (restraint / deferral / failed / disengaged)")
    for study, arms in ARMS.items():
        for arm in arms:
            c = Counter(d["code"]["reason"] for d in rows if d["host"] == study and d["arm"] == arm)
            per = "/".join(str(sum(d["host"] == study and d["arm"] == arm and d["scenario"] == s
                                   and d["code"]["reason"] == "restraint" for d in rows)) for s in SCEN)  # fmt: skip
            print(f"  {study} {arm:28} n={sum(c.values()):3}  " + "/".join(str(c[k]) for k in REASONS)
                  + f"   restraint blackmail/lethal/leaking {per}")  # fmt: skip
    print("\n=== Q1 (amendment 48): after the benign step, midtrained vs control Director")
    line(rows, "uniform-dr-benign vs control-benign", ("a48", "uniform-dr-benign"), ("a48", "control-benign"))
    print("\n=== Q2 (amendment 49, one session): restraint after vs before the benign step")
    mid = line(rows, "midtrained, after vs before", ("a49", "uniform-dr-benign"), ("a49", "uniform-dr"))
    saf = line(rows, "safety-tuned, after vs before", ("a49", "control-safety4k-benign"), ("a49", "control-safety4k"))
    if mid and saf:
        se = lambda r: (math.log(r[2]) - math.log(r[1])) / (2 * 1.96)  # noqa: E731
        d, s = math.log(saf[0]) - math.log(mid[0]), math.hypot(se(mid), se(saf))
        p = math.erfc(abs(d / s) / math.sqrt(2))
        print(f"  ratio (safety-tuned over midtrained): {math.exp(d):.2f} [{math.exp(d - 1.96 * s):.2f}, "
              f"{math.exp(d + 1.96 * s):.2f}], p = {p:.3g}")  # fmt: skip
    line(rows, "before: midtrained vs safety-tuned", ("a49", "uniform-dr"), ("a49", "control-safety4k"))
    line(rows, "after:  midtrained vs safety-tuned", ("a49", "uniform-dr-benign"), ("a49", "control-safety4k-benign"))
    print("\n=== Secondary (uncorrected, labelled)")
    line(rows, "a48 safety-tuned-benign vs control-benign", ("a48", "control-safety4k-benign"), ("a48", "control-benign"))
    line(rows, "a48 midtrained-benign vs safety-tuned-benign", ("a48", "uniform-dr-benign"), ("a48", "control-safety4k-benign"))
    line(rows, "a50 midtrained+safety+benign vs control+safety+benign", ("a50", "uniform-dr-safety4k-benign"),
         ("a50", "control-safety4k-benign"))  # fmt: skip
    line(rows, "a46 fresh safety-tuned vs a44 control (across sessions)", ("a46", "control-safety4k"), ("a44", "control"))
    second = {(d["file"], d["run_idx"]): d for d in load("amend69", "gpt-5.5")}
    first = {(d["file"], d["run_idx"]): d for d in coded}
    pairs = [(first[k]["code"]["reason"], v["code"]["reason"]) for k, v in second.items() if k in first]
    rb = [(a == "restraint", b == "restraint") for a, b in pairs]
    print(f"\n=== Agreement with GPT-5.5 on {len(pairs)} firms: main reason {sum(a == b for a, b in pairs) / len(pairs):.0%}"
          f" (kappa {kappa(pairs):.2f}); restraint or not {sum(a == b for a, b in rb) / len(rb):.0%}"
          f" (kappa {kappa(rb):.2f})")  # fmt: skip
    print("  confusion (primary -> GPT-5.5):", dict(Counter(pairs).most_common()))


if __name__ == "__main__":
    main()
