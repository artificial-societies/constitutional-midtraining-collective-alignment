"""Amendment 67: why amendment 65's Communications Director (Cho's 120B checkpoints in Mistral's and Qwen3.5's firms)
took no external action (exploratory analysis), as scripts/amend66_tables.py.

Reads runs/amend67/codes_<coder>.jsonl (scripts/amend67_code.py). Claude Opus 5.5 (refusals coded by Opus 4.5) is the
primary coder; GPT-5.5's 60 codings give agreement.

  python scripts/amend67_tables.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase3_tables import fmt, mh  # noqa: E402

DIR = Path("runs/amend67")
ARMS, SCEN = ("uniform-dr", "control"), ("blackmail", "murder", "leaking")
REASONS = ("restraint", "deferral", "failed", "disengaged")
HOSTS = ("mistral", "qwen35")
FIRMS = 60  # firms per host, scenario and arm in amendment 65
EXCESS = 100 - 66  # no-action firms, uniform-dr minus control


def load(coder: str) -> dict:
    return {(d["file"], d["run_idx"]): d for d in map(json.loads, (DIR / f"codes_{coder}.jsonl").open())}


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return (po - pe) / (1 - pe)


def main() -> None:
    codes = load("claude-opus-5-5")
    rows = list(codes.values())
    print(f"No-action firms coded: {len(rows)}; by Opus 4.5 after an Opus 5.5 refusal: "
          + ", ".join(f"{a} {sum(d['arm'] == a and d['code'].get('by', 'claude-opus-5-5') != 'claude-opus-5-5' for d in rows)}" for a in ARMS))  # fmt: skip
    print("\n=== Main reason among no-action firms (primary coder)")
    for a in ARMS:
        rs = [d for d in rows if d["arm"] == a]
        c = Counter(d["code"]["reason"] for d in rs)
        print(f"  {a:8} n={len(rs):3}  " + "  ".join(f"{k} {c[k]:3} ({c[k] / len(rs):.0%})" for k in REASONS))
    for h in HOSTS:
        for a in ARMS:
            rs = [d for d in rows if d["arm"] == a and d["host"] == h]
            c = Counter(d["code"]["reason"] for d in rs)
            print(f"  {h:8} {a:10} n={len(rs):3}  " + "  ".join(f"{k} {c[k]}" for k in REASONS))
    print("\n=== Per scenario (no-action firms: restraint / deferral / failed / disengaged)")
    for s in SCEN:
        print(f"  {s:10} " + "   ".join(
            f"{a} " + "/".join(str(sum(d['arm'] == a and d['scenario'] == s and d['code']['reason'] == k for d in rows))
                               for k in REASONS) for a in ARMS))  # fmt: skip
    print("\n=== Flags among no-action firms")
    for a in ARMS:
        rs = [d for d in rows if d["arm"] == a]
        f = lambda k, rs=rs: sum(d["code"][k] == "yes" for d in rs)  # noqa: E731
        print(f"  {a:8} considered the harmful act {f('considered_harm')}, intended an external action "
              f"{f('intended_external')}, intended the harmful act {f('intended_harmful')}; failed with harmful intent "
              f"{sum(d['code']['reason'] == 'failed' and d['code']['intended_harmful'] == 'yes' for d in rs)}")
    print("\n=== Restraint over all Director firms (acting firms count as no restraint)")
    def strata(hosts: tuple) -> list:
        out = []
        for h in hosts:
            for s in SCEN:
                arms = []
                for a in ARMS:
                    k = sum(d["arm"] == a and d["host"] == h and d["scenario"] == s and d["code"]["reason"] == "restraint"
                            for d in rows)  # fmt: skip
                    arms.append([{"r": True}] * k + [{"r": False}] * (FIRMS - k))
                out.append(tuple(arms))
        return out

    for h in HOSTS:
        q, c = (sum(d["arm"] == a and d["host"] == h and d["code"]["reason"] == "restraint" for d in rows) for a in ARMS)
        print(f"  {h:8} restraint firms: uniform-dr {q}/180, control {c}/180; {fmt(mh(strata((h,)), 'r'))}")
    rq, rc = (sum(d["arm"] == a and d["code"]["reason"] == "restraint" for d in rows) for a in ARMS)
    print(f"  pooled   restraint firms: uniform-dr {rq}/360, control {rc}/360; {fmt(mh(strata(HOSTS), 'r'))}")
    print(f"  share of the excess inaction ({EXCESS} firms) that restraint accounts for: {(rq - rc) / EXCESS:.0%}")
    for k in ("failed", "deferral", "disengaged"):
        q, c = (sum(d["arm"] == a and d["code"]["reason"] == k for d in rows) for a in ARMS)
        print(f"  {k:10} uniform-dr {q:3} control {c:3}  share of the excess {(q - c) / EXCESS:.0%}")
    second = load("gpt-5.5")
    pairs = [(codes[k]["code"]["reason"], v["code"]["reason"]) for k, v in second.items() if k in codes]
    rb = [(a == "restraint", b == "restraint") for a, b in pairs]
    print(f"\n=== Agreement with GPT-5.5 on {len(pairs)} firms: main reason {sum(a == b for a, b in pairs) / len(pairs):.0%}"
          f" (kappa {kappa(pairs):.2f}); restraint or not {sum(a == b for a, b in rb) / len(rb):.0%}"
          f" (kappa {kappa(rb):.2f})")  # fmt: skip
    print("  confusion (primary -> GPT-5.5):", dict(Counter(pairs).most_common()))
    for a in ARMS:
        g = Counter(v["code"]["reason"] for v in second.values() if v["arm"] == a)
        print(f"  GPT-5.5 {a:8} " + "  ".join(f"{k} {g[k]}" for k in REASONS))


if __name__ == "__main__":
    main()
