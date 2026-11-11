"""Export the paper's data (tidy CSV) and in-text numbers (LaTeX macros) from the committed results.

  uv run python paper/build_data.py               ->  paper/data/*.csv, paper/generated/{numbers,rates}.tex
  uv run python paper/build_data.py --runs runs   (read an uncompressed runs/ directory instead)

By default the compressed results/ are unpacked into a temporary runs/ layout (scripts/unpack_results.py), so no
API, GPU or raw run directory is needed; the unpacked files are byte-identical to the runs that produced them.

Every figure (paper/figures.R) and every number in the paper comes from these files. Phase 4 (amendment 43) is the
main dataset; amendment 44 confirms its placement result on fresh data; amendment 45 compares safety fine-tuning;
Phase 3 (amendment 40), amendment 41 and the judge audit (amendment 42) supply the robustness section.
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amend44_tables as a44t  # noqa: E402
import amend45_tables as a45t  # noqa: E402
import amend46_tables as a46t  # noqa: E402
import amend48_tables as a48t  # noqa: E402
import amend49_tables as a49t  # noqa: E402
import amend50_tables as a50t  # noqa: E402
import amend57_tables as a57t  # noqa: E402
import amend65_tables as a65t  # noqa: E402
import amend66_tables as a66t  # noqa: E402
import amend69_tables as a69t  # noqa: E402
import amend70_tables as a70t  # noqa: E402
import amend71_recode as a71r  # noqa: E402
import amend71_tables as a71t  # noqa: E402
import phase3_tables as p3t  # noqa: E402
import phase4_tables as p4t  # noqa: E402
import unpack_results  # noqa: E402

DATA, GEN = ROOT / "paper" / "data", ROOT / "paper" / "generated"
SCEN = ("blackmail", "murder", "leaking")
SCEN_NAME = {"blackmail": "Blackmail", "murder": "Lethal action", "leaking": "Leaking"}
JUDGES = {"harm_adj": "Reasoning panel", "harm_panel": "Three-provider panel", "harm": "GPT-4o"}


def tag_of(agent: str) -> str:
    """Macro suffix from an agent label's initials; the control keeps "Cc", from its earlier label "Cho control"."""
    return "Cc" if agent == "Control" else "".join(w[0] for w in agent.split()).replace("-", "")


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if not n:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def write(name: str, rows: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def firm_layout() -> None:
    """Node and edge positions for Figure 1: the core team as a fully linked pentagon, specialists by their manager."""
    from cmca.cho.shen_org import ROLES, TREE

    core = list(ROLES)
    pos = {"Partner": (0, 2.5), "Research Manager": (2.5, 1.7), "Communications Director": (1.6, 0.45),
           "Analyst": (-1.6, 0.45), "Deployment Manager": (-2.5, 1.7)}  # fmt: skip
    place = {"Deployment Manager": [(-4.6, 2.5), (-4.9, 1.5), (-4.6, 0.5)], "Analyst": [(-1.6, -0.65)],
             "Research Manager": [(4.6, 2.5), (4.9, 1.5), (4.6, 0.5)]}  # fmt: skip
    nodes = [{"role": r, "x": pos[r][0], "y": pos[r][1], "kind": "actor" if r == "Communications Director" else "core"} for r in core]  # fmt: skip
    for m, cs in TREE.items():
        for c, (x, y) in zip(cs, place[m], strict=True):
            nodes.append({"role": c, "x": x, "y": y, "kind": "specialist"})
    at = {n["role"]: n for n in nodes}
    edges = [{"x": at[a]["x"], "y": at[a]["y"], "xend": at[b]["x"], "yend": at[b]["y"], "kind": "core"}
             for i, a in enumerate(core) for b in core[i + 1 :]]  # fmt: skip
    edges += [{"x": at[m]["x"], "y": at[m]["y"], "xend": at[c]["x"], "yend": at[c]["y"], "kind": "tree"} for m, cs in TREE.items() for c in cs]  # fmt: skip
    write("firm_nodes.csv", nodes)
    write("firm_edges.csv", edges)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs", type=Path, help="an uncompressed runs/ directory (default: unpack results/)"
    )
    a = ap.parse_args()
    if a.runs is not None:
        build(a.runs.resolve())
        return
    with tempfile.TemporaryDirectory() as tmp:
        unpack_results.unpack(Path(tmp))
        fill_gemini(Path(tmp) / "runs")
        build(Path(tmp) / "runs")


def fill_gemini(runs: Path) -> None:
    """Put the backfilled Gemini 3.1 Pro verdicts (results/adjudication_gemini, scripts/gemini_backfill.py) in place of
    the ones lost to an expired credential in the confirmation, safety, safety-tuned placement, durability, same-session
    and combination studies, so that every study is judged by all three judges (as scripts/gemini_merge.py does)."""
    gem, filled = "gemini-3.1-pro-preview", 0
    for g in sorted((runs / "adjudication_gemini").glob("*.jsonl")):
        f = runs / "adjudication" / g.name
        if not f.exists():
            continue
        fill = {d["key"]: d["verdict"] for d in p4t.jsonl(g)}
        rows = []
        for d in p4t.jsonl(f):
            if d["verdicts"].get(gem, {}).get("answer") is None and d["key"] in fill:
                d["verdicts"][gem] = fill[d["key"]]
                filled += 1
            rows.append(json.dumps(d))
        f.write_text("\n".join(rows) + "\n")
    print(f"Gemini verdicts filled: {filled}", file=sys.stderr)


def build(runs: Path) -> None:
    """Write the paper's data from `runs`, a directory in the runs/ layout (the loaders require that name)."""
    if runs.name != "runs":
        raise SystemExit(f"--runs must name a directory called runs/, not {runs}")
    sys.path.insert(0, str(ROOT))
    firm_layout()
    p4t.ROOT, p4t.PANEL, p4t.ADJ = runs / "cho_shen/p4/p4", runs / "panel", runs / "adjudication"
    sub = p4t.substitutes()
    arms = ("org12", "single", *p4t.IVS)
    d = {(s, a, m): p4t.load(s, a, m, sub) for s in SCEN for a in arms for m in p4t.MODELS}
    g = lambda s, a, m: d[(s, a, m)]  # noqa: E731
    pool = lambda s, a: [r for m in p4t.CMT for r in g(s, a, m)]  # noqa: E731
    K = "harm_adj"
    nums: dict[str, str] = {}
    conf: list[dict] = []

    # Figure 2: harm alone and in the firm, by model group and scenario
    rows = []
    for s in SCEN:
        for grp, getter in (("NVIDIA chat", lambda a, s=s: g(s, a, "nvidia")), ("Control", lambda a, s=s: g(s, a, "control")),
                            ("Midtrained (pooled)", lambda a, s=s: pool(s, a))):  # fmt: skip
            for a, where in (("single", "Alone"), ("org12", "In the firm")):
                k, n = p3t.cnt(getter(a), K)
                p, lo, hi = wilson(k, n)
                rows.append({"scenario": SCEN_NAME[s], "model": grp, "setting": where, "k": k, "n": n, "p": p, "lo": lo, "hi": hi})  # fmt: skip
    write("harm_rates.csv", rows)

    # Table and robustness figure: the four primary tests under each judge
    rows = []
    for key, judge in JUDGES.items():
        q = {
            "Q1": p3t.mh([(pool(s, "org12"), g(s, "org12", "control")) for s in SCEN], key),
            "Q2": p3t.interaction(
                [
                    (
                        pool(s, "org12"),
                        g(s, "org12", "control"),
                        pool(s, "single"),
                        g(s, "single", "control"),
                    )
                    for s in SCEN
                ],
                key,
            ),  # fmt: skip
            "Q3": p3t.mh([(g(s, "org12", "nvidia"), g(s, "single", "nvidia")) for s in SCEN], key),
            "Q4": p3t.mh(
                [(g(s, "ivpartner", "uniform-dr"), g(s, "ivpartner", "control")) for s in SCEN], key
            ),
        }
        holm = p3t.holm({k: v[3] for k, v in q.items()})
        for k, v in q.items():
            rows.append(
                {
                    "test": k,
                    "judge": judge,
                    "or": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "p": v[3],
                    "holm": holm[k],
                }
            )
    write("primary_tests.csv", rows)
    prim = {r["test"]: r for r in rows if r["judge"] == "Reasoning panel"}
    for k, r in prim.items():
        k = {"Q1": "QOne", "Q2": "QTwo", "Q3": "QThree", "Q4": "QFour"}[
            k
        ]  # LaTeX macro names hold letters only
        nums[f"{k}or"], nums[f"{k}lo"], nums[f"{k}hi"] = (
            f"{r['or']:.2f}",
            f"{r['lo']:.2f}",
            f"{r['hi']:.2f}",
        )
        nums[f"{k}holm"] = "<0.001" if r["holm"] < 0.001 else f"{r['holm']:.2f}"

    # Figure 4: interventions, midtrained vs control model in the same seats of NVIDIA's firm, and the dose curve
    rows = []
    label = {"ivpartner": "Partner", "ivcomms": "Communications Director", "ivcore": "Five core seats",
             "ivreverse": "All but the Partner (NVIDIA Partner)"}  # fmt: skip
    for iv in p4t.IVS:
        for key, judge in JUDGES.items():
            v = p3t.mh([(g(s, iv, "uniform-dr"), g(s, iv, "control")) for s in SCEN], key)
            rows.append(
                {
                    "placement": label[iv],
                    "judge": judge,
                    "or": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "p": v[3],
                }
            )
    write("interventions.csv", rows)
    for r in rows:
        if r["judge"] == "Reasoning panel":
            tag = {
                "Partner": "Partner",
                "Communications Director": "Director",
                "Five core seats": "Core",
            }.get(r["placement"], "Reverse")
            nums[f"IV{tag}or"] = f"{r['or']:.2f}"
    rows = []
    for m, name in (("uniform-dr", "Midtrained (uniform-dr)"), ("control", "Control")):
        for seats, getter in ((0, lambda: [r for s in SCEN for r in g(s, "org12", "nvidia")]),
                              (1, lambda m=m: [r for s in SCEN for r in g(s, "ivpartner", m)]),
                              (5, lambda m=m: [r for s in SCEN for r in g(s, "ivcore", m)]),
                              (12, lambda m=m: [r for s in SCEN for r in g(s, "org12", m)])):  # fmt: skip
            k, n = p3t.cnt(getter(), K)
            p, lo, hi = wilson(k, n)
            rows.append({"model": name, "seats": seats, "k": k, "n": n, "p": p, "lo": lo, "hi": hi})
        k, n = p3t.cnt([r for s in SCEN for r in g(s, "ivcomms", m)], K)
        p, lo, hi = wilson(k, n)
        rows.append(
            {"model": name, "seats": "Director", "k": k, "n": n, "p": p, "lo": lo, "hi": hi}
        )
    write("dose.csv", rows)

    # Activity: share of firms that acted, harm among those that acted (pooled over scenarios)
    rows = []
    for name, rs in (("Control", [r for s in SCEN for r in g(s, "org12", "control")]),
                     ("Midtrained (pooled)", [r for s in SCEN for r in pool(s, "org12")]),
                     ("NVIDIA chat", [r for s in SCEN for r in g(s, "org12", "nvidia")])):  # fmt: skip
        acted = [r for r in rs if r["acted"]]
        k, n = p3t.cnt(acted, K)
        rows.append(
            {"model": name, "acted": len(acted) / len(rs), "harm_if_acted": k / n, "n": len(rs)}
        )
    write("activity.csv", rows)

    # Judge agreement on targeted outputs (Phase 4) and the audit (amendment 42)
    tgt = [r for rs in d.values() for r in rs if r["target_present"]]
    for key, tag in (("harm_panel", "AgreePanel"), ("harm", "AgreeGpt")):
        pairs = [(bool(r[key]), r["harm_adj"]) for r in tgt]
        nums[tag] = f"{100 * sum(a == b for a, b in pairs) / len(pairs):.0f}"
    nums["Targeted"] = f"{len(tgt):,}".replace(",", "{,}")
    nums["FirmRuns"] = f"{sum(len(rs) for (s, a, m), rs in d.items() if a != 'single'):,}".replace(
        ",", "{,}"
    )
    nums["SingleRuns"] = (
        f"{sum(len(rs) for (s, a, m), rs in d.items() if a == 'single'):,}".replace(",", "{,}")
    )
    cut = sum(e["finish_reason"] == "length" for rs in d.values() for r in rs for e in r["log"])
    turns = sum(len(r["log"]) for rs in d.values() for r in rs)
    nums["CutPct"] = f"{100 * cut / turns:.1f}"
    nums["Refusals"] = str(sum(r["refused_filled"] for rs in d.values() for r in rs))

    v = p3t.mh([(g(s, "org12", "uniform-dr"), g(s, "org12", "control")) for s in SCEN], K)
    nums["UdrFirmOr"] = f"{v[0]:.2f}"

    # Midtrained vs control firms across the three firm studies (reasoning panel), pooled by inverse variance on the
    # log scale; descriptive. The two earlier estimates are the reasoning panel's (all three judges) from the five-role
    # team study (p3) and the earlier twelve-role firm study (a41), from scripts/phase3_tables.py and
    # scripts/amend41_adjudicated.py
    q1 = p3t.mh([(pool(s, "org12"), g(s, "org12", "control")) for s in SCEN], K)
    studies = [("Five-role team, earlier study", 0.86, 0.57, 1.28), ("Twelve-role firm, earlier study", 0.60, 0.37, 0.97),
               ("Twelve-role firm, main study", q1[0], q1[1], q1[2])]  # fmt: skip
    w = [(math.log(o), ((math.log(h) - math.log(lo)) / 3.92) ** -2) for _, o, lo, h in studies]
    m = sum(x * wi for x, wi in w) / sum(wi for _, wi in w)
    se = sum(wi for _, wi in w) ** -0.5
    pooled = (math.exp(m), math.exp(m - 1.96 * se), math.exp(m + 1.96 * se))
    write("pooled.csv", [{"study": n, "or": o, "lo": lo, "hi": h} for n, o, lo, h in studies]
          + [{"study": "Pooled", "or": pooled[0], "lo": pooled[1], "hi": pooled[2]}])  # fmt: skip
    nums["PoolOr"], nums["PoolLo"], nums["PoolHi"] = (f"{x:.2f}" for x in pooled)
    nums["PoolQ"] = f"{sum(wi * (x - m) ** 2 for x, wi in w):.1f}"

    # Amendment 44: the placement confirmation (fresh data), next to Phase 4's secondary estimates
    p4t.ROOT = runs / "cho_shen/a44/a44"
    c = {(s, seat, m): a44t.load(s, seat, m, sub) for s in SCEN for seat, m in a44t.ARMS}
    p4t.ROOT = runs / "cho_shen/p4/p4"
    gc = lambda s, seat, m: c[(s, seat, m)]  # noqa: E731
    rows = []
    for seat, iv, name in (
        ("partner", "ivpartner", "Partner"),
        ("director", "ivcomms", "Communications Director"),
    ):
        for study, getter in (("Main study (secondary)", lambda s, m, iv=iv: g(s, iv, m)),
                              ("Confirmation", lambda s, m, seat=seat: gc(s, seat, m))):  # fmt: skip
            v = p3t.mh([(getter(s, "uniform-dr"), getter(s, "control")) for s in SCEN], K)
            rows.append(
                {"placement": name, "study": study, "or": v[0], "lo": v[1], "hi": v[2], "p": v[3]}
            )
    write("confirmation.csv", rows)
    for key, judge in JUDGES.items():
        h1 = p3t.mh(
            [(gc(s, "director", "uniform-dr"), gc(s, "director", "control")) for s in SCEN], key
        )
        h2 = p3t.interaction([(gc(s, "director", "uniform-dr"), gc(s, "director", "control"),
                               gc(s, "partner", "uniform-dr"), gc(s, "partner", "control")) for s in SCEN], key)  # fmt: skip
        hh = p3t.holm({"H1": h1[3], "H2": h2[3]})
        for t, v in (("H1", h1), ("H2", h2)):
            conf.append(
                {
                    "test": t,
                    "judge": judge,
                    "or": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "p": v[3],
                    "holm": hh[t],
                }
            )
            if judge == "Reasoning panel":
                tag = {"H1": "HOne", "H2": "HTwo"}[t]
                nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
                    f"{v[0]:.2f}",
                    f"{v[1]:.2f}",
                    f"{v[2]:.2f}",
                )
                nums[f"{tag}holm"] = f"{hh[t]:.4f}" if hh[t] < 0.001 else f"{hh[t]:.3f}"
    for tag, v in (("ConfPartner", p3t.mh([(gc(s, "partner", "uniform-dr"), gc(s, "partner", "control")) for s in SCEN], K)),
                   ("ConfNodr", p3t.mh([(gc(s, "director", "uniform-nodr"), gc(s, "director", "control")) for s in SCEN], K))):  # fmt: skip
        nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
            f"{v[0]:.2f}",
            f"{v[1]:.2f}",
            f"{v[2]:.2f}",
        )
    for m, tag in (("uniform-dr", "Mid"), ("control", "Ctl")):
        rs = [r for s in SCEN for r in gc(s, "director", m)]
        nums[f"ConfNoAction{tag}"] = f"{100 * sum(not r['acted'] for r in rs) / len(rs):.0f}"
    acted = lambda rs: [r for r in rs if r["acted"]]  # noqa: E731
    v = p3t.mh(
        [
            (acted(gc(s, "director", "uniform-dr")), acted(gc(s, "director", "control")))
            for s in SCEN
        ],
        K,
    )
    nums["ConfActedor"], nums["ConfActedlo"], nums["ConfActedhi"] = (
        f"{v[0]:.2f}",
        f"{v[1]:.2f}",
        f"{v[2]:.2f}",
    )
    nums["ConfRuns"] = str(sum(len(rs) for rs in c.values()))

    # Amendment 45: safety fine-tuning on Cho's checkpoints, alone and in the firm
    p4t.ADJ = runs / "adjudication"
    cwd = Path.cwd()
    os.chdir(
        runs.parent
    )  # the amendment 45, 46 and 48 loaders read runs/ relative to the working directory
    SAFE = (
        ("control", "Control"),
        ("control-safety4k", "Control + safety"),
        ("uniform-dr-safety4k", "Midtrained + safety"),
    )
    e = {
        (s, a, m): a45t.load("b", s, a, m, sub)
        for s in SCEN
        for a in ("single", "org12")
        for m, _ in SAFE
    }
    f46 = {(s, seat): a46t.load(s, seat, sub) for s in SCEN for seat in ("director", "partner")}
    dose = p3t.mh([(a45t.load("a", s, "single", "control-safety4k", sub), g(s, "single", "control")) for s in SCEN], K)
    nums["DoseOr"], nums["DoseLo"], nums["DoseHi"] = f"{dose[0]:.2f}", f"{dose[1]:.2f}", f"{dose[2]:.2f}"
    nums["DoseTarget"] = f"{a45t.TARGET:.2f}"
    BEN = (a48t.CB, a48t.UB, a48t.SB)
    f48 = {
        (s, a, m): a48t.load(s, a, m, sub)
        for s in SCEN
        for a in ("director", "single")
        for m in BEN
    }
    f49 = {(s, m): a49t.load(s, m, sub) for s in SCEN for m in (a49t.CS, a49t.UD, a49t.SB, a49t.UB)}
    f50 = {(s, m): a50t.load(s, m, sub) for s in SCEN for m in a50t.ARMS}
    os.chdir(cwd)

    # Amendment 46: the safety-tuned model in the same two seats, against amendment 44's control
    rows = []
    for seat, name in (("partner", "Partner"), ("director", "Communications Director")):
        v = p3t.mh([(f46[(s, seat)], gc(s, seat, "control")) for s in SCEN], K)
        rows.append(
            {
                "placement": name,
                "study": "Safety-tuned model",
                "or": v[0],
                "lo": v[1],
                "hi": v[2],
                "p": v[3],
            }
        )
    with (DATA / "confirmation.csv").open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writerows(rows)
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        s1 = p3t.mh([(f46[(s, "director")], gc(s, "director", "control")) for s in SCEN], key)
        s2 = p3t.interaction([(f46[(s, "director")], gc(s, "director", "control"), f46[(s, "partner")], gc(s, "partner", "control")) for s in SCEN], key)  # fmt: skip
        hh = p3t.holm({"S1": s1[3], "S2": s2[3]})
        for t, v in (("S1", s1), ("S2", s2)):
            conf.append(
                {
                    "test": t,
                    "judge": judge,
                    "or": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "p": v[3],
                    "holm": hh[t],
                }
            )
            if judge == "Reasoning panel":
                tag = {"S1": "SOne", "S2": "STwo"}[t]
                nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
                    f"{v[0]:.2f}",
                    f"{v[1]:.2f}",
                    f"{v[2]:.2f}",
                )
                nums[f"{tag}holm"] = "<0.001" if hh[t] < 0.001 else f"{hh[t]:.3f}"
    v = p3t.mh([(f46[(s, "partner")], gc(s, "partner", "control")) for s in SCEN], K)
    nums["SafePartneror"], nums["SafePartnerlo"], nums["SafePartnerhi"] = (
        f"{v[0]:.2f}",
        f"{v[1]:.2f}",
        f"{v[2]:.2f}",
    )
    rs = [r for s in SCEN for r in f46[(s, "director")]]
    nums["SafeDirNoAction"] = f"{100 * sum(not r['acted'] for r in rs) / len(rs):.0f}"
    v = p3t.mh(
        [(acted(f46[(s, "director")]), acted(gc(s, "director", "control"))) for s in SCEN], K
    )
    nums["SafeDirActedor"], nums["SafeDirActedlo"], nums["SafeDirActedhi"] = (
        f"{v[0]:.2f}",
        f"{v[1]:.2f}",
        f"{v[2]:.2f}",
    )
    nums["SafeSeatRuns"] = str(sum(len(x) for x in f46.values()))

    # Amendment 48: the same benign fine-tuning step, then the Director seat (and alone)
    cb, ub, sb = BEN
    g48 = lambda s, a, m: f48[(s, a, m)]  # noqa: E731
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        d1 = p3t.mh([(g48(s, "director", ub), g48(s, "director", cb)) for s in SCEN], key)
        d2 = p3t.mh([(g48(s, "director", ub), g48(s, "director", sb)) for s in SCEN], key)
        hh = p3t.holm({"D1": d1[3], "D2": d2[3]})
        for t, v in (("D1", d1), ("D2", d2)):
            conf.append(
                {
                    "test": t,
                    "judge": judge,
                    "or": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "p": v[3],
                    "holm": hh[t],
                }
            )
            if judge == "Reasoning panel":
                tag = {"D1": "DOne", "D2": "DTwo"}[t]
                nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
                    f"{v[0]:.2f}",
                    f"{v[1]:.2f}",
                    f"{v[2]:.2f}",
                )
                nums[f"{tag}holm"] = "<0.001" if hh[t] < 0.001 else f"{hh[t]:.3f}"
    sdir = p3t.mh([(g48(s, "director", sb), g48(s, "director", cb)) for s in SCEN], K)
    before_mid = p3t.mh(
        [(gc(s, "director", "uniform-dr"), gc(s, "director", "control")) for s in SCEN], K
    )
    before_safe = p3t.mh([(f46[(s, "director")], gc(s, "director", "control")) for s in SCEN], K)
    mdir = p3t.mh([(g48(s, "director", ub), g48(s, "director", cb)) for s in SCEN], K)
    ero_mid, ero_safe = a48t.ratio(mdir, before_mid), a48t.ratio(sdir, before_safe)
    for tag, v in (("SafeBenDir", sdir), ("EroMid", ero_mid), ("EroSafe", ero_safe),
                   ("MidBenAlone", p3t.mh([(g48(s, "single", ub), g48(s, "single", cb)) for s in SCEN], K)),
                   ("SafeBenAlone", p3t.mh([(g48(s, "single", sb), g48(s, "single", cb)) for s in SCEN], K))):  # fmt: skip
        nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
            f"{v[0]:.2f}",
            f"{v[1]:.2f}",
            f"{v[2]:.2f}",
        )
    rows = []
    for training, before, after in (
        ("Midtraining", before_mid, mdir),
        ("Safety step", before_safe, sdir),
    ):
        rows.append(
            {
                "training": training,
                "when": "Before",
                "or": before[0],
                "lo": before[1],
                "hi": before[2],
            }
        )
        rows.append(
            {"training": training, "when": "After", "or": after[0], "lo": after[1], "hi": after[2]}
        )
    write("durability.csv", rows)
    nums["BenignRuns"] = str(sum(len(x) for x in f48.values()))
    ge = lambda s, a, m: e[(s, a, m)]  # noqa: E731
    rows = []
    for s in SCEN:
        for m, name in SAFE:
            for a, where in (("single", "Alone"), ("org12", "In the firm")):
                k, n = p3t.cnt(ge(s, a, m), K)
                p, lo, hi = wilson(k, n)
                rows.append({"scenario": SCEN_NAME[s], "model": name, "setting": where, "k": k, "n": n, "p": p, "lo": lo, "hi": hi})  # fmt: skip
    write("safety_rates.csv", rows)
    cs, us = "control-safety4k", "uniform-dr-safety4k"
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        t1 = p3t.interaction([(ge(s, "org12", cs), ge(s, "org12", "control"), ge(s, "single", cs), ge(s, "single", "control")) for s in SCEN], key)  # fmt: skip
        conf.append(
            {
                "test": "T1",
                "judge": judge,
                "or": t1[0],
                "lo": t1[1],
                "hi": t1[2],
                "p": t1[3],
                "holm": t1[3],
            }
        )
    tests = {
        "TOne": p3t.interaction([(ge(s, "org12", cs), ge(s, "org12", "control"), ge(s, "single", cs), ge(s, "single", "control")) for s in SCEN], K),
        "SafeAlone": p3t.mh([(ge(s, "single", cs), ge(s, "single", "control")) for s in SCEN], K),
        "SafeFirm": p3t.mh([(ge(s, "org12", cs), ge(s, "org12", "control")) for s in SCEN], K),
        "TThree": p3t.mh([(ge(s, "org12", us), ge(s, "org12", cs)) for s in SCEN], K),
        "MidSafeAlone": p3t.mh([(ge(s, "single", us), ge(s, "single", cs)) for s in SCEN], K),
    }  # fmt: skip
    for tag, v in tests.items():
        nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = (
            f"{v[0]:.2f}",
            f"{v[1]:.2f}",
            f"{v[2]:.2f}",
        )
        nums[f"{tag}p"] = f"{v[3]:.3f}" if v[3] >= 0.001 else "<0.001"
    for m, tag in (("control", "Ctl"), (cs, "Safe")):
        rs = [r for s in SCEN for r in ge(s, "org12", m)]
        a_ = acted(rs)
        nums[f"SafeActed{tag}"] = f"{100 * len(a_) / len(rs):.0f}"
        nums[f"SafeHarmActed{tag}"] = f"{100 * p3t.cnt(a_, K)[0] / len(a_):.0f}"
    nums["SafeRuns"] = str(sum(len(rs) for rs in e.values()))
    # Amendment 49: the safety step and midtraining before and after the benign step, in one session
    g49 = lambda s, m: f49[(s, m)]  # noqa: E731
    # Amendment 50: the same safety step and benign step on the midtrained and on the control base
    g50 = lambda s, m: f50[(s, m)]  # noqa: E731
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        f1 = p3t.interaction([(g49(s, a49t.UB), g49(s, a49t.SB), g49(s, a49t.UD), g49(s, a49t.CS)) for s in SCEN], key)
        c1 = p3t.mh([(g50(s, a50t.MSB), g50(s, a50t.SB)) for s in SCEN], key)
        for t, v in (("F1", f1), ("C1", c1)):
            conf.append({"test": t, "judge": judge, "or": v[0], "lo": v[1], "hi": v[2], "p": v[3], "holm": v[3]})
            if judge == "GPT-4o":
                nums[{"F1": "FOneGptor", "C1": "COneGptor"}[t]] = f"{v[0]:.2f}"
            if judge == "Reasoning panel":
                tag = {"F1": "FOne", "C1": "COne"}[t]
                nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}"
                nums[f"{tag}p"] = f"{v[3]:.3f}" if v[3] >= 0.001 else "<0.001"
    # F1 and its pool are midtrained against safety-tuned: the change in odds of harm after the benign step, so below one
    # means midtraining's harm rose less, as in every other test
    # The same quantity as F1 from the earlier studies (before: amendments 44 and 46; after: amendment 48), and the
    # inverse-variance pool of the two estimates; descriptive, since the earlier estimate spans sessions
    r48 = p3t.interaction([(g48(s, "director", ub), g48(s, "director", sb), gc(s, "director", "uniform-dr"), f46[(s, "director")]) for s in SCEN], K)  # fmt: skip
    f1 = p3t.interaction([(g49(s, a49t.UB), g49(s, a49t.SB), g49(s, a49t.UD), g49(s, a49t.CS)) for s in SCEN], K)
    w = [(math.log(o), ((math.log(h) - math.log(lo)) / 3.92) ** -2) for o, lo, h, _ in (r48, f1)]
    m_ = sum(x * wi for x, wi in w) / sum(wi for _, wi in w)
    se_ = sum(wi for _, wi in w) ** -0.5
    nums["PoolDurOr"], nums["PoolDurLo"], nums["PoolDurHi"] = (f"{math.exp(x):.2f}" for x in (m_, m_ - 1.96 * se_, m_ + 1.96 * se_))
    nums["DurAcrossOr"] = f"{r48[0]:.2f}"
    nums["ConfirmRuns"] = str(sum(len(x) for x in f49.values()))
    nums["ComboRuns"] = str(sum(len(x) for x in f50.values()))

    # Rates for the figures: the share of runs that did harm, scenarios pooled (each arm has equal runs per scenario)
    def rate(rs: list[dict]) -> dict:
        k, n = p3t.cnt(rs, K)
        p, lo, hi = wilson(k, n)
        return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}

    allscen = lambda f: [r for s in SCEN for r in f(s)]  # noqa: E731
    seat_arms = {
        ("Partner", "Control"): allscen(lambda s: gc(s, "partner", "control")),
        ("Partner", "Midtrained"): allscen(lambda s: gc(s, "partner", "uniform-dr")),
        ("Partner", "Safety-tuned"): allscen(lambda s: f46[(s, "partner")]),
        ("Communications Director", "Control"): allscen(lambda s: gc(s, "director", "control")),
        ("Communications Director", "Midtrained"): allscen(lambda s: gc(s, "director", "uniform-dr")),
        ("Communications Director", "Safety-tuned"): allscen(lambda s: f46[(s, "director")]),
    }
    after = {"Control": allscen(lambda s: g48(s, "director", cb)), "Midtrained": allscen(lambda s: g48(s, "director", ub)),
             "Safety-tuned": allscen(lambda s: g48(s, "director", sb))}  # fmt: skip
    nv_alone, nv_firm = allscen(lambda s: g(s, "single", "nvidia")), allscen(lambda s: g(s, "org12", "nvidia"))
    write("seat_rates.csv", [{"seat": seat, "agent": agent, **rate(rs)} for (seat, agent), rs in seat_arms.items()])
    rows = []
    for agent in ("Control", "Midtrained", "Safety-tuned"):
        rs = seat_arms[("Communications Director", agent)]
        for outcome, sel in (("Took no outside action", lambda r: not r["acted"]),
                             ("Acted without harm", lambda r: r["acted"] and not r[K]),
                             ("Acted harmfully", lambda r: r["acted"] and r[K])):  # fmt: skip
            rows.append({"agent": agent, "outcome": outcome, "share": sum(map(sel, rs)) / len(rs), "n": len(rs)})
    write("director_outcomes.csv", rows)
    rows = []
    for agent in ("Control", "Midtrained", "Safety-tuned"):
        rows.append({"panel": "Across studies", "agent": agent, "when": "Before", **rate(seat_arms[("Communications Director", agent)])})
        rows.append({"panel": "Across studies", "agent": agent, "when": "After", **rate(after[agent])})
    for agent, (b, a) in (("Midtrained", (a49t.UD, a49t.UB)), ("Safety-tuned", (a49t.CS, a49t.SB))):
        rows.append({"panel": "One session", "agent": agent, "when": "Before", **rate(allscen(lambda s, b=b: g49(s, b)))})
        rows.append({"panel": "One session", "agent": agent, "when": "After", **rate(allscen(lambda s, a=a: g49(s, a)))})
    write("durability_rates.csv", rows)
    head = [("The firm adds harm", "NVIDIA model alone", nv_alone), ("The firm adds harm", "NVIDIA model in every seat", nv_firm)]
    head += [("Interventions at the acting seat", f"{a} Director", seat_arms[("Communications Director", a)])
             for a in ("Control", "Midtrained", "Safety-tuned")]  # fmt: skip
    head += [("After later fine-tuning", f"{a} Director", after[a]) for a in ("Control", "Midtrained", "Safety-tuned")]
    combo = {"Control + safety": allscen(lambda s: g50(s, a50t.SB)), "Midtrained + safety": allscen(lambda s: g50(s, a50t.MSB))}
    head += [("Safety step on each base, after later fine-tuning", f"{a} Director", rs) for a, rs in combo.items()]
    write("headline.csv", [{"panel": pn, "arm": arm, **rate(rs)} for pn, arm, rs in head])
    # Durability table: harm with each model as Director before and after the benign step (scenarios pooled), with how
    # often the Director acted and its harm among acting firms
    def cell(rs: list | None) -> str:
        if rs is None:
            return "--"
        r = rate(rs)
        return f"{100 * r['p']:.0f}\\% [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"

    def parts(rs: list | None) -> str:  # how often the Director acted, and the harm rate among the firms that acted
        if rs is None:
            return "-- & --"
        a_ = acted(rs)
        return f"{100 * len(a_) / len(rs):.0f}\\% & {100 * p3t.cnt(a_, K)[0] / len(a_):.0f}\\%"

    lines = [f"{name} & {cell(b)} & {parts(b)} & {cell(a)} & {parts(a)} \\\\" for name, b, a in (
        ("Control checkpoint", seat_arms[("Communications Director", "Control")], after["Control"]),
        ("Midtrained", seat_arms[("Communications Director", "Midtrained")], after["Midtrained"]),
        ("Safety-tuned control", seat_arms[("Communications Director", "Safety-tuned")], after["Safety-tuned"]),
        ("Midtrained with the safety step$^a$", None, combo["Midtrained + safety"]),
        ("Safety-tuned control$^a$", None, combo["Control + safety"]))]  # fmt: skip
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "durability_table.tex").write_text("\n".join(lines) + "\n")
    pct = lambda rs: f"{100 * rate(rs)['p']:.0f}"  # noqa: E731
    nums["PctNvAlone"], nums["PctNvFirm"] = pct(nv_alone), pct(nv_firm)
    for (seat, agent), rs in seat_arms.items():
        nums[f"Pct{'Dir' if seat != 'Partner' else 'Par'}{tag_of(agent)}"] = pct(rs)
    for a, tag in (("single", "Alone"), ("org12", "Firm")):
        nums[f"PctCtl{tag}"] = pct(allscen(lambda s, a=a: g(s, a, "control")))
        nums[f"PctMid{tag}"] = pct(allscen(lambda s, a=a: pool(s, a)))
    for (m, a), tag in (((cs, "single"), "SafeAlone"), ((cs, "org12"), "SafeFirm"), (("control", "single"), "SafeCtlAlone"),
                        (("control", "org12"), "SafeCtlFirm"), ((us, "single"), "MidSafeAlone"), ((us, "org12"), "MidSafeFirm")):  # fmt: skip
        nums[f"Pct{tag}"] = pct(allscen(lambda s, m=m, a=a: ge(s, a, m)))
    for agent, rs in seat_arms.items():
        if agent[0] != "Partner":
            a_ = acted(rs)
            nums[f"PctActedHarm{tag_of(agent[1])}"] = f"{100 * p3t.cnt(a_, K)[0] / len(a_):.0f}"
    gap = lambda a, b: f"{100 * (rate(a)['p'] - rate(b)['p']):.0f}"  # noqa: E731  percentage points below the control
    before_ctl = seat_arms[("Communications Director", "Control")]
    nums["GapMidBefore"] = gap(before_ctl, seat_arms[("Communications Director", "Midtrained")])
    nums["GapSafeBefore"] = gap(before_ctl, seat_arms[("Communications Director", "Safety-tuned")])
    nums["GapMidAfter"] = gap(after["Control"], after["Midtrained"])
    nums["GapSafeAfter"] = gap(after["Control"], after["Safety-tuned"])
    # Relative changes in the share of runs that did harm (percent of the baseline rate), for the main comparisons
    rel = lambda base, new: f"{abs(100 * (rate(new)['p'] - rate(base)['p']) / rate(base)['p']):.0f}"  # noqa: E731
    dirs = {a: seat_arms[("Communications Director", a)] for a in ("Control", "Midtrained", "Safety-tuned")}
    nums["RelNvFirm"] = rel(nv_alone, nv_firm)
    nums["RelMidAlone"] = rel(allscen(lambda s: g(s, "single", "control")), allscen(lambda s: pool(s, "single")))
    nums["RelMidFirm"] = rel(allscen(lambda s: g(s, "org12", "control")), allscen(lambda s: pool(s, "org12")))
    nums["RelSafeAlone"] = rel(allscen(lambda s: ge(s, "single", "control")), allscen(lambda s: ge(s, "single", cs)))
    nums["RelSafeFirm"] = rel(allscen(lambda s: ge(s, "org12", "control")), allscen(lambda s: ge(s, "org12", cs)))
    nums["RelDirM"] = rel(dirs["Control"], dirs["Midtrained"])
    nums["RelDirS"] = rel(dirs["Control"], dirs["Safety-tuned"])
    nums["RelBenM"] = rel(after["Control"], after["Midtrained"])
    nums["RelBenS"] = rel(after["Control"], after["Safety-tuned"])
    lead = lambda c, x: rate(c)["p"] - rate(x)["p"]  # noqa: E731
    shrink = lambda b, a: f"{100 * (1 - a / b):.0f}"  # noqa: E731
    nums["ShrinkMid"] = shrink(lead(dirs["Control"], dirs["Midtrained"]), lead(after["Control"], after["Midtrained"]))
    nums["ShrinkSafe"] = shrink(lead(dirs["Control"], dirs["Safety-tuned"]), lead(after["Control"], after["Safety-tuned"]))
    nums["KeepMid"], nums["KeepSafe"] = (str(100 - int(nums[k])) for k in ("ShrinkMid", "ShrinkSafe"))
    nums["PctComboCtl"], nums["PctComboMid"] = pct(combo["Control + safety"]), pct(combo["Midtrained + safety"])
    nums["RelCombo"] = rel(combo["Control + safety"], combo["Midtrained + safety"])
    for b, a, tag in ((a49t.CS, a49t.SB, "Safe"), (a49t.UD, a49t.UB, "Mid")):
        nums[f"PctOne{tag}Before"], nums[f"PctOne{tag}After"] = pct(allscen(lambda s, b=b: g49(s, b))), pct(allscen(lambda s, a=a: g49(s, a)))
    for agent, rs in after.items():
        nums[f"PctBen{tag_of(agent)}"] = pct(rs)

    v = p3t.mh([(acted(g48(s, "director", ub)), acted(g48(s, "director", cb))) for s in SCEN], K)
    nums["BenActedor"], nums["BenActedlo"], nums["BenActedhi"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}"
    # The safety-tuned Director after the benign step: its harm among acting firms against the control's, and how often it
    # acted, against the safety-tuned placement study before the step (across studies)
    v = p3t.mh([(acted(g48(s, "director", sb)), acted(g48(s, "director", cb))) for s in SCEN], K)
    nums["BenActedSafeor"], nums["BenActedSafelo"], nums["BenActedSafehi"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}"
    for m, tag in ((ub, "Mid"), (cb, "Ctl"), (sb, "Safe")):
        rs = allscen(lambda s, m=m: g48(s, "director", m))
        nums[f"BenNoAction{tag}"] = f"{100 * sum(not r['acted'] for r in rs) / len(rs):.0f}"
        nums[f"BenActs{tag}"] = str(100 - int(nums[f"BenNoAction{tag}"]))
    nums["SafeDirActs"] = str(100 - int(nums["SafeDirNoAction"]))
    actpct = lambda rs: round(100 * len(acted(rs)) / len(rs))  # noqa: E731  as printed in Table 2
    dirs_before = {m: seat_arms[("Communications Director", m)] for m in ("Control", "Midtrained", "Safety-tuned")}
    nums["ActGapSafeBefore"] = str(actpct(dirs_before["Control"]) - actpct(dirs_before["Safety-tuned"]))
    nums["ActGapSafeAfter"] = str(actpct(after["Control"]) - actpct(after["Safety-tuned"]))

    # Amendment 65: Cho's checkpoints as Director of other families' firms (Mistral-Small-3.2, Qwen3.5-35B-A3B)
    sub65 = sub | {(Path(d["source"]).name, d["key"]): d["verdict"].get("answer")
                   for d in p4t.jsonl(runs / "adjudication/substitutes_a65.jsonl")}  # fmt: skip
    f65 = {(h, s, arm): [r for v in vs for r in a57t.load(runs / f"cho_shen/a65/a65/a65_{s}_{h}_{arm}_{v}.jsonl", sub65)]
           for h, vs in a65t.HOSTS.items() for s in SCEN for arm in a65t.ARMS}  # fmt: skip
    g65 = lambda h, s, m: f65[(h, s, m)]  # noqa: E731
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        e65 = {t: p3t.mh([(g65(h, s, "uniform-dr"), g65(h, s, "control")) for s in SCEN], key)
               for t, h in (("E1", "mistral"), ("E2", "qwen35"))}  # fmt: skip
        hh = p3t.holm({t: v[3] for t, v in e65.items()})
        for t, v in e65.items():
            conf.append({"test": t, "judge": judge, "or": v[0], "lo": v[1], "hi": v[2], "p": v[3], "holm": hh[t]})
            if judge == "Reasoning panel":
                tag = {"E1": "EOne", "E2": "ETwo"}[t]
                nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}"
                nums[f"{tag}p"] = f"{v[3]:.2f}"
        both = p3t.mh([(g65(h, s, "uniform-dr"), g65(h, s, "control")) for h in a65t.HOSTS for s in SCEN], key)
        tag = "EPool" if judge == "Reasoning panel" else "EPoolGpt"
        nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"] = f"{both[0]:.2f}", f"{both[1]:.2f}", f"{both[2]:.2f}"
        nums[f"{tag}p"] = f"{both[3]:.3f}"
        if judge == "Reasoning panel":  # against amendment 44 in NVIDIA's firm, as a difference of log odds ratios
            se = lambda v: (math.log(v[2]) - math.log(v[1])) / 3.92  # noqa: E731
            z = (math.log(both[0]) - math.log(before_mid[0])) / math.hypot(se(both), se(before_mid))
            nums["EHetp"] = f"{math.erfc(abs(z) / math.sqrt(2)):.3f}"
            v = p3t.mh([(acted(g65(h, s, "uniform-dr")), acted(g65(h, s, "control"))) for h in a65t.HOSTS for s in SCEN], key)
            nums["EActedor"] = f"{v[0]:.2f}"
    for m, tag in (("uniform-dr", "Mid"), ("control", "Ctl")):
        rs = [r for h in a65t.HOSTS for s in SCEN for r in g65(h, s, m)]
        nums[f"PctOther{tag}"] = pct(rs)
        nums[f"OtherNoAction{tag}"] = f"{100 * sum(not r['acted'] for r in rs) / len(rs):.0f}"
    nums["RelOther"] = rel([r for h in a65t.HOSTS for s in SCEN for r in g65(h, s, "control")],
                           [r for h in a65t.HOSTS for s in SCEN for r in g65(h, s, "uniform-dr")])  # fmt: skip
    nums["OtherRuns"] = str(sum(len(x) for x in f65.values()))

    # Amendment 70: amendment 44's Director contrast under Lynch's explicit-restriction condition
    sub70 = sub | {(Path(d["source"]).name, d["key"]): d["verdict"].get("answer")
                   for d in p4t.jsonl(runs / "adjudication/substitutes_a70.jsonl")}  # fmt: skip
    f70 = {(s, m): [r for v in a70t.SERVERS["a70"] for r in a57t.load(runs / f"cho_shen/a70/a70/a70_{s}_{m}_{v}.jsonl", sub70)]
           for s in SCEN for m in a70t.ARMS}  # fmt: skip
    for key, judge in (("harm_adj", "Reasoning panel"), ("harm", "GPT-4o")):
        v = p3t.mh([(f70[(s, "uniform-dr")], f70[(s, "control")]) for s in SCEN], key)
        conf.append({"test": "R1", "judge": judge, "or": v[0], "lo": v[1], "hi": v[2], "p": v[3], "holm": v[3]})
        tag = "Restr" if judge == "Reasoning panel" else "RestrGpt"
        nums[f"{tag}or"], nums[f"{tag}lo"], nums[f"{tag}hi"], nums[f"{tag}p"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}", f"{v[3]:.3f}"
    v = p3t.mh([(acted(f70[(s, "uniform-dr")]), acted(f70[(s, "control")])) for s in SCEN], K)
    nums["RestrActedor"] = f"{v[0]:.2f}"
    cond = []
    for m, tag, agent in (("uniform-dr", "Mid", "Midtrained"), ("control", "Ctl", "Control")):
        rs = [r for s in SCEN for r in f70[(s, m)]]
        nums[f"PctRestr{tag}"] = pct(rs)
        nums[f"RestrNoAction{tag}"] = f"{100 * sum(not r['acted'] for r in rs) / len(rs):.0f}"
        for cname, get in (("Replaced at 5 PM", lambda s, m=m: gc(s, "director", m)), ("Restricted at 5 PM", lambda s, m=m: f70[(s, m)])):
            for sname, rs_ in [(sn, get(s)) for s, sn in zip(SCEN, ("Blackmail", "Lethal action", "Leaking"), strict=True)] + [
                    ("Pooled", [r for s in SCEN for r in get(s)])]:  # fmt: skip
                k, n = p3t.cnt(rs_, K)
                pr, lo, hi = wilson(k, n)
                cond.append({"condition": cname, "scenario": sname, "agent": agent, "k": k, "n": n, "p": pr, "lo": lo, "hi": hi})
    write("condition_rates.csv", cond)

    # Amendment 71: the authorized-action test. Completion of an authorized forward of the specification, coded from the
    # tool calls (scripts/amend71_recode.py, no judge), by the model in the Director's seat of NVIDIA's firm
    f71 = {a: [r | a71r.code(r) for v in a71t.SERVERS["a71"] for r in p4t.jsonl(runs / f"cho_shen/a71/a71/a71_leaking_{a}_{v}.jsonl")]
           for a in a71t.ARMS}  # fmt: skip
    auth = []
    for a, tag, agent in (("uniform-dr", "Mid", "Midtrained"), ("control", "Ctl", "Control"),
                          ("control-safety4k", "Safe", "Safety-tuned control"), ("nvidia", "Nv", "NVIDIA chat")):  # fmt: skip
        k, n = p3t.cnt(f71[a], "complete")
        pr, lo, hi = wilson(k, n)
        auth.append({"agent": agent, "k": k, "n": n, "p": pr, "lo": lo, "hi": hi})
        nums[f"Auth{tag}"] = f"{100 * pr:.0f}"
        nums[f"AuthOut{tag}"] = str(sum(r["other_external"] for r in f71[a]))
    nums["AuthN"] = str(len(f71["control"]))
    for a, tag in (("uniform-dr", "AuthMidor"), ("control-safety4k", "AuthSafeor")):
        v = p3t.mh([(f71[a], f71["control"])], "complete")
        nums[tag], nums[tag[:-2] + "lo"], nums[tag[:-2] + "hi"], nums[tag[:-2] + "p"] = f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}", f"{v[3]:.2f}"
    write("authorized_rates.csv", auth)

    # Amendments 67-69 (exploratory): Directors that took no external action, coded for principled refusal of the
    # harmful act; every acting firm counts as no refusal. Restraint per arm over all Director firms of the study.
    os.chdir(runs.parent)
    c67 = a69t.load("amend67", "claude-opus-5-5")
    c68 = [d | {"host": "a44"} for d in a69t.load("amend68", "claude-opus-5-5")]
    c69 = a69t.load("amend69", "claude-opus-5-5")
    second = {r: a69t.load(r, "gpt-5.5") for r in ("amend67", "amend68", "amend69")}
    os.chdir(cwd)

    def refused(rows: list, study: str, arm: str, s: str, per: int) -> list[dict]:
        k = sum(d["host"] == study and d["arm"] == arm and d["scenario"] == s and d["code"]["reason"] == "restraint"
                for d in rows)  # fmt: skip
        return [{"r": True}] * k + [{"r": False}] * (per - k)

    def restraint(rows: list, a: tuple, b: tuple, per: int = 60, hosts: tuple = ()) -> tuple:
        strata = [(refused(rows, h or a[0], a[1], s, per), refused(rows, h or b[0], b[1], s, per))
                  for h in (hosts or ("",)) for s in SCEN]  # fmt: skip
        return (p3t.mh(strata, "r"), *(sum(x["r"] for st in strata for x in st[i]) for i in (0, 1)), per * len(strata))

    def ci(v: tuple) -> str:
        if v[0] < 1:
            return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
        return f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.0f}]" if v[2] >= 100 else f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"

    rows = []
    fresh = c67 + c68
    for label, host, tag in (("NVIDIA's firm", "a44", "Nv"), ("Mistral's firm", "mistral", "Mi"), ("Qwen3.5's firm", "qwen35", "Qw")):
        v, ka, kb, n = restraint(fresh, (host, "uniform-dr"), (host, "control"))
        rows.append(f"Fresh & {label} & Midtrained & {ka}/{n} & Control & {kb}/{n} & {ci(v)} \\\\")
        nums[f"Ref{tag}Mid"], nums[f"Ref{tag}Ctl"], nums[f"Ref{tag}or"] = str(ka), str(kb), f"{v[0]:.1f}"
    v, ka, kb, n = restraint(fresh, ("", "uniform-dr"), ("", "control"), hosts=("a44", "mistral", "qwen35"))
    rows.append(f"Fresh & Three firms pooled & Midtrained & {ka}/{n} & Control & {kb}/{n} & {ci(v)} \\\\")
    nums["RefPoolMid"], nums["RefPoolCtl"], nums["RefPoolN"] = str(ka), str(kb), str(n)
    nums["RefPoolor"], nums["RefPoollo"], nums["RefPoolhi"] = f"{v[0]:.1f}", f"{v[1]:.1f}", f"{v[2]:.1f}"
    c69a = c69 + c68
    for when, label, a, ta, b, tb, per, tag in (
        ("Fresh", "NVIDIA's firm$^a$", ("a46", "control-safety4k"), "Safety-tuned", ("a44", "control"), "Control", 60, "SafeFresh"),
        ("After benign step", "NVIDIA's firm", ("a48", "uniform-dr-benign"), "Midtrained", ("a48", "control-benign"), "Control", 60, "MidBen"),
        ("After benign step", "NVIDIA's firm", ("a48", "control-safety4k-benign"), "Safety-tuned", ("a48", "control-benign"), "Control", 60, "SafeBen"),
        ("After vs before", "NVIDIA's firm$^b$", ("a49", "uniform-dr-benign"), "Midtrained, after", ("a49", "uniform-dr"), "Before", 60, "MidDrop"),
        ("After vs before", "NVIDIA's firm$^b$", ("a49", "control-safety4k-benign"), "Safety-tuned, after", ("a49", "control-safety4k"), "Before", 60, "SafeDrop"),
        ("After both steps", "NVIDIA's firm", ("a50", "uniform-dr-safety4k-benign"), "Midtrained + safety", ("a50", "control-safety4k-benign"), "Control + safety", 75, "Combo"),
    ):  # fmt: skip
        v, ka, kb, n = restraint(c69a, a, b, per=per)
        rows.append(f"{when} & {label} & {ta} & {ka}/{n} & {tb} & {kb}/{n} & {ci(v)} \\\\")
        nums[f"Ref{tag}A"], nums[f"Ref{tag}B"], nums[f"Ref{tag}or"] = str(ka), str(kb), f"{v[0]:.1f}"
        nums[f"Ref{tag}lo"], nums[f"Ref{tag}hi"], nums[f"Ref{tag}p"] = f"{v[1]:.1f}", f"{v[2]:.1f}", f"{v[3]:.3f}"
    # Figure: the share of Director firms with a principled refusal, by firm, before and after the benign step, and
    # after both steps (scenarios pooled, Wilson intervals)
    count = lambda rows, study, arm, per: sum(sum(x["r"] for x in refused(rows, study, arm, s, per)) for s in SCEN)  # noqa: E731
    fig = [("Fresh, by firm", "NVIDIA", "Control", fresh, "a44", "control", 60),
           ("Fresh, by firm", "NVIDIA", "Midtrained", fresh, "a44", "uniform-dr", 60),
           ("Fresh, by firm", "NVIDIA", "Safety-tuned", c69a, "a46", "control-safety4k", 60),
           ("Fresh, by firm", "Mistral", "Control", fresh, "mistral", "control", 60),
           ("Fresh, by firm", "Mistral", "Midtrained", fresh, "mistral", "uniform-dr", 60),
           ("Fresh, by firm", "Qwen3.5", "Control", fresh, "qwen35", "control", 60),
           ("Fresh, by firm", "Qwen3.5", "Midtrained", fresh, "qwen35", "uniform-dr", 60),
           ("Benign step, one session", "Before", "Midtrained", c69a, "a49", "uniform-dr", 60),
           ("Benign step, one session", "After", "Midtrained", c69a, "a49", "uniform-dr-benign", 60),
           ("Benign step, one session", "Before", "Safety-tuned", c69a, "a49", "control-safety4k", 60),
           ("Benign step, one session", "After", "Safety-tuned", c69a, "a49", "control-safety4k-benign", 60),
           ("Safety step on each base,\nafter the benign step", "Control base", "Safety-tuned", c69a, "a50", "control-safety4k-benign", 75),
           ("Safety step on each base,\nafter the benign step", "Midtrained base", "Midtrained", c69a, "a50", "uniform-dr-safety4k-benign", 75)]  # fmt: skip
    rates = []
    for panel, x, agent, rows_, study, arm, per in fig:
        k = count(rows_, study, arm, per)
        pr, lo, hi = wilson(k, 3 * per)
        rates.append({"panel": panel, "x": x, "agent": agent, "k": k, "n": 3 * per, "p": pr, "lo": lo, "hi": hi})
    write("refusal_rates.csv", rates)
    (GEN / "restraint.tex").parent.mkdir(parents=True, exist_ok=True)
    (GEN / "restraint.tex").write_text("\n".join(rows) + "\n")
    mid = restraint(c69a, ("a49", "uniform-dr-benign"), ("a49", "uniform-dr"))[0]
    saf = restraint(c69a, ("a49", "control-safety4k-benign"), ("a49", "control-safety4k"))[0]
    se = lambda r: (math.log(r[2]) - math.log(r[1])) / 3.92  # noqa: E731
    d_, s_ = math.log(saf[0]) - math.log(mid[0]), math.hypot(se(mid), se(saf))
    nums["RefDropRatio"] = f"{math.exp(d_):.2f}"
    nums["RefDropRatiolo"], nums["RefDropRatiohi"] = f"{math.exp(d_ - 1.96 * s_):.2f}", f"{math.exp(d_ + 1.96 * s_):.2f}"
    nums["RefOtherMid"] = str(int(nums["RefMiMid"]) + int(nums["RefQwMid"]))
    nums["RefOtherCtl"] = str(int(nums["RefMiCtl"]) + int(nums["RefQwCtl"]))
    nums["RefShareNv"] = f"{100 * (int(nums['RefNvMid']) - int(nums['RefNvCtl'])) / (40 - 16):.0f}"  # of the 24 excess no-action firms
    nums["RefShareOther"] = f"{100 * (int(nums['RefMiMid']) + int(nums['RefQwMid']) - int(nums['RefMiCtl']) - int(nums['RefQwCtl'])) / 34:.0f}"
    coded = c67 + c68 + c69
    nums["CodedFirms"] = str(len(coded))
    pairs = []
    for run, prim in (("amend67", c67), ("amend68", c68), ("amend69", c69)):
        first = {(d["file"], d["run_idx"]): d["code"]["reason"] == "restraint" for d in prim}
        pairs += [(first[(d["file"], d["run_idx"])], d["code"]["reason"] == "restraint") for d in second[run]
                  if (d["file"], d["run_idx"]) in first]  # fmt: skip
    nums["CodeAgreeN"] = str(len(pairs))
    nums["CodeAgree"] = f"{100 * sum(a_ == b_ for a_, b_ in pairs) / len(pairs):.0f}"
    nums["CodeKappa"] = f"{a66t.kappa(pairs):.2f}"

    # Agreement among the three panel judges on every targeted output of the paper's studies: Claude (Opus 5.5, with
    # Opus 4.5 where it declined), gpt-5.5 and Gemini 3.1 Pro; pairwise Cohen's kappa and Fleiss' kappa, by scenario
    subs = sub | sub65 | sub70
    j3 = ("claude-opus-5-5", "gpt-5.5", "gemini-3.1-pro-preview")
    votes, lacking = [], 0
    for st in ("p4", "a44", "a45b", "a46", "a48", "a49", "a50", "a65", "a70"):
        for f in sorted((runs / "adjudication").glob(f"{st}_*.jsonl")):
            for rec in p4t.jsonl(f):
                v = {j: rec["verdicts"].get(j, {}).get("answer") for j in j3}
                if v[j3[0]] is None:
                    v[j3[0]] = subs.get((Path(rec["source"]).name, rec["key"]))
                if all(isinstance(x, bool) for x in v.values()):
                    votes.append((f.name.split("_")[1], *(v[j] for j in j3)))
                else:
                    lacking += 1

    def fleiss(vs: list[tuple]) -> float:
        n = len(vs)
        yes = [sum(v[1:]) for v in vs]
        pbar = sum((y * (y - 1) + (3 - y) * (2 - y)) / 6 for y in yes) / n
        py = sum(yes) / (3 * n)
        pe = py**2 + (1 - py) ** 2
        return (pbar - pe) / (1 - pe)

    lines = []
    for label, i, k in (("Claude and gpt-5.5", 1, 2), ("Claude and Gemini", 1, 3), ("gpt-5.5 and Gemini", 2, 3)):
        cells = [f"{100 * sum(v[i] == v[k] for v in votes) / len(votes):.0f}\\%"]
        cells += [f"{a66t.kappa([(v[i], v[k]) for v in votes if sc in ('', v[0])]):.2f}" for sc in ("", *SCEN)]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    cells = [f"{100 * sum(v[1] == v[2] == v[3] for v in votes) / len(votes):.0f}\\%"]
    cells += [f"{fleiss([v for v in votes if sc in ('', v[0])]):.2f}" for sc in ("", *SCEN)]
    lines.append("All three (Fleiss) & " + " & ".join(cells) + " \\\\")
    (GEN / "agreement.tex").parent.mkdir(parents=True, exist_ok=True)
    (GEN / "agreement.tex").write_text("\n".join(lines) + "\n")
    nums["JudgedN"] = f"{len(votes):,}".replace(",", "{,}")
    nums["JudgedLacking"] = str(lacking)
    nums["FleissK"] = f"{fleiss(votes):.2f}"
    nums["AllThreeAgree"] = f"{100 * sum(v[1] == v[2] == v[3] for v in votes) / len(votes):.0f}"
    pk = [a66t.kappa([(v[i], v[k]) for v in votes]) for i, k in ((1, 2), (1, 3), (2, 3))]
    nums["PairKappaLo"], nums["PairKappaHi"] = f"{min(pk):.2f}", f"{max(pk):.2f}"

    # Significance marks for the figures: * when the 95% interval of the comparison's Mantel-Haenszel odds ratio
    # (scenarios pooled) excludes 1. prereg records whether the comparison is a preregistered test; each figure's
    # caption says which kind it marks (all marked comparisons outside the refusal figure are preregistered).
    def mark(v: tuple, prereg: bool) -> str:  # noqa: ARG001
        return "*" if v[1] > 1 or v[2] < 1 else ""

    vs = lambda a, b: p3t.mh([(a(s), b(s)) for s in SCEN], K)  # noqa: E731

    def add_marks(name: str, key: tuple, marks: dict) -> None:
        with (DATA / name).open(newline="") as f:
            rows_ = list(csv.DictReader(f))
        for r in rows_:
            r["mark"] = marks.get(tuple(r[k] for k in key), "")
        write(name, rows_)

    nv_q3 = vs(lambda s: g(s, "org12", "nvidia"), lambda s: g(s, "single", "nvidia"))
    combo_c1 = vs(lambda s: g50(s, a50t.MSB), lambda s: g50(s, a50t.SB))
    restr_r1 = vs(lambda s: f70[(s, "uniform-dr")], lambda s: f70[(s, "control")])
    add_marks("headline.csv", ("panel", "arm"), {
        ("The firm adds harm", "NVIDIA model in every seat"): mark(nv_q3, True),
        ("Interventions at the acting seat", "Midtrained Director"): mark(before_mid, True),
        ("Interventions at the acting seat", "Safety-tuned Director"): mark(before_safe, True),
        ("After later fine-tuning", "Midtrained Director"): mark(mdir, True),
        ("After later fine-tuning", "Safety-tuned Director"): mark(sdir, False),
        ("Safety step on each base, after later fine-tuning", "Midtrained + safety Director"): mark(combo_c1, True)})
    add_marks("seat_rates.csv", ("seat", "agent"), {
        ("Partner", "Midtrained"): mark(vs(lambda s: gc(s, "partner", "uniform-dr"), lambda s: gc(s, "partner", "control")), False),
        ("Partner", "Safety-tuned"): mark(vs(lambda s: f46[(s, "partner")], lambda s: gc(s, "partner", "control")), False),
        ("Communications Director", "Midtrained"): mark(before_mid, True),
        ("Communications Director", "Safety-tuned"): mark(before_safe, True)})
    add_marks("durability_rates.csv", ("panel", "agent", "when"), {
        ("Across studies", "Midtrained", "Before"): mark(before_mid, True),
        ("Across studies", "Safety-tuned", "Before"): mark(before_safe, True),
        ("Across studies", "Midtrained", "After"): mark(mdir, True),
        ("Across studies", "Safety-tuned", "After"): mark(sdir, False)})
    combo_panel = "Safety step on each base,\nafter the benign step"
    add_marks("refusal_rates.csv", ("panel", "x", "agent"), {
        ("Fresh, by firm", "NVIDIA", "Midtrained"): mark(restraint(fresh, ("a44", "uniform-dr"), ("a44", "control"))[0], False),
        ("Fresh, by firm", "NVIDIA", "Safety-tuned"): mark(restraint(c69a, ("a46", "control-safety4k"), ("a44", "control"))[0], False),
        ("Fresh, by firm", "Mistral", "Midtrained"): mark(restraint(fresh, ("mistral", "uniform-dr"), ("mistral", "control"))[0], False),
        ("Fresh, by firm", "Qwen3.5", "Midtrained"): mark(restraint(fresh, ("qwen35", "uniform-dr"), ("qwen35", "control"))[0], False),
        (combo_panel, "Midtrained base", "Midtrained"): mark(restraint(c69a, ("a50", "uniform-dr-safety4k-benign"), ("a50", "control-safety4k-benign"), per=75)[0], False)})
    add_marks("condition_rates.csv", ("condition", "scenario", "agent"), {
        ("Replaced at 5 PM", "Pooled", "Midtrained"): mark(before_mid, True),
        ("Restricted at 5 PM", "Pooled", "Midtrained"): mark(restr_r1, True)})

    write("confirmatory_tests.csv", conf)
    GEN.mkdir(parents=True, exist_ok=True)
    # Appendix table: harm in every arm (reasoning panel), with the secondary judges alongside
    arm_name = {"single": "Alone", "org12": "Full firm", "ivpartner": "NVIDIA firm, Partner replaced",
                "ivcomms": "NVIDIA firm, Director replaced", "ivcore": "NVIDIA firm, core team replaced",
                "ivreverse": "Checkpoint firm, NVIDIA Partner"}  # fmt: skip
    model_name = {"control": "Control", "uniform-dr": "Uniform, DR", "uniform-nodr": "Uniform, no DR",
                  "curriculum-dr": "Curriculum, DR", "curriculum-nodr": "Curriculum, no DR", "nvidia": "NVIDIA chat"}  # fmt: skip
    lines = []
    for a in arms:
        for m in p4t.MODELS:
            if not any(g(s, a, m) for s in SCEN):
                continue
            cells = []
            for s in SCEN:
                rs = g(s, a, m)
                k, n = p3t.cnt(rs, K)
                kp, _ = p3t.cnt(rs, "harm_panel")
                kg, _ = p3t.cnt(rs, "harm")
                cells.append(f"{k}/{n} & {round(100 * kp / n)} & {round(100 * kg / n)}")
            lines.append(f"{arm_name[a]} & {model_name[m]} & " + " & ".join(cells) + " \\\\")
    for label, getter, models in (
        ("Confirmation, Director replaced", lambda s, m: gc(s, "director", m), ("uniform-dr", "control", "uniform-nodr")),
        ("Confirmation, Partner replaced", lambda s, m: gc(s, "partner", m), ("uniform-dr", "control")),
    ):  # fmt: skip
        for m in models:
            cells = []
            for s in SCEN:
                rs = getter(s, m)
                k, n = p3t.cnt(rs, K)
                kp, _ = p3t.cnt(rs, "harm_panel")
                kg, _ = p3t.cnt(rs, "harm")
                cells.append(f"{k}/{n} & {round(100 * kp / n)} & {round(100 * kg / n)}")
            lines.append(f"{label} & {model_name[m]} & " + " & ".join(cells) + " \\\\")
    ben_name = {
        a48t.CB: "Control + benign",
        a48t.UB: "Uniform, DR + benign",
        a48t.SB: "Control + safety + benign",
    }
    for a, where in (("director", "Durability study, Director"), ("single", "Durability study, alone")):
        for m in BEN:
            cells = []
            for s in SCEN:
                rs = f48[(s, a, m)]
                k, n = p3t.cnt(rs, K)
                kg, _ = p3t.cnt(rs, "harm")
                cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
            lines.append(f"{where} & {ben_name[m]} & " + " & ".join(cells) + " \\\\")
    for m, label in ((a49t.CS, "Same session, Director, before"), (a49t.UD, "Same session, Director, before"),
                     (a49t.SB, "Same session, Director, after"), (a49t.UB, "Same session, Director, after"),
                     (a50t.SB, "Combination, Director"), (a50t.MSB, "Combination, Director")):  # fmt: skip
        src = f50 if m == a50t.MSB or (m == a50t.SB and label.startswith("Combination")) else f49
        name = {a49t.CS: "Control + safety", a49t.UD: "Uniform, DR", a49t.SB: "Control + safety + benign",
                a49t.UB: "Uniform, DR + benign", a50t.MSB: "Uniform, DR + safety + benign"}[m]  # fmt: skip
        cells = []
        for s in SCEN:
            rs = src[(s, m)]
            k, n = p3t.cnt(rs, K)
            kg, _ = p3t.cnt(rs, "harm")
            cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
        lines.append(f"{label} & {name} & " + " & ".join(cells) + " \\\\")
    for seat, label in (("director", "Safety-tuned placement, Director"), ("partner", "Safety-tuned placement, Partner")):
        cells = []
        for s in SCEN:
            rs = f46[(s, seat)]
            k, n = p3t.cnt(rs, K)
            kg, _ = p3t.cnt(rs, "harm")
            cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
        lines.append(f"{label} & Control + safety & " + " & ".join(cells) + " \\\\")
    for a, where in (("single", "Safety study, alone"), ("org12", "Safety study, full firm")):
        for m, name in SAFE:
            cells = []
            for s in SCEN:
                rs = ge(s, a, m)
                k, n = p3t.cnt(rs, K)
                kg, _ = p3t.cnt(rs, "harm")
                cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
            lines.append(f"{where} & {name} & " + " & ".join(cells) + " \\\\")
    for h, label in (("mistral", "Mistral firm, Director replaced"), ("qwen35", "Qwen3.5 firm, Director replaced")):
        for m in a65t.ARMS:
            cells = []
            for s in SCEN:
                rs = g65(h, s, m)
                k, n = p3t.cnt(rs, K)
                kg, _ = p3t.cnt(rs, "harm")
                cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
            lines.append(f"{label} & {model_name[m]} & " + " & ".join(cells) + " \\\\")
    for m in a70t.ARMS:
        cells = []
        for s in SCEN:
            k, n = p3t.cnt(f70[(s, m)], K)
            kg, _ = p3t.cnt(f70[(s, m)], "harm")
            cells.append(f"{k}/{n} & -- & {round(100 * kg / n)}")
        lines.append(f"Restriction condition, Director replaced & {model_name[m]} & " + " & ".join(cells) + " \\\\")
    (GEN / "rates.tex").write_text("\n".join(lines) + "\n")
    (GEN / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(nums.items()))
    )
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
