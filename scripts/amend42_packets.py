"""Amendment 42: blinded reading packets for the expert readers (no model, arm, file or earlier verdicts).

  python scripts/amend42_packets.py OUT_DIR [--groups 6]  ->  OUT_DIR/packet_{i}.md, OUT_DIR/key.json
"""

import argparse
import json
import random
from pathlib import Path

from cmca.cho.adjudicate import FACTS, RULES
from cmca.cho.shen_org import SCENARIOS

CASES, FLAGS = Path("runs/adjudication/cases.json"), Path("runs/adjudication/gate_flags.json")


def main(out: Path, groups: int) -> None:
    rng = random.Random(0)
    cases = json.loads(CASES.read_text())
    s1 = [c for c in cases if "S1" in c["set"]]
    full = [c for c in s1 if "_org12_" in c["source"]]
    core = rng.sample([c for c in s1 if "_org_" in c["source"]], 60)
    single = rng.sample([c for c in s1 if "_single" in c["source"]], 40)
    gate = [dict(f, set=[f["set"]]) for f in json.loads(FLAGS.read_text())]
    picked = {(c["source"], c["key"]): c for c in [*full, *core, *single, *gate]}.values()
    items = []
    for c in picked:
        rec = next(json.loads(ln) for ln in Path(c["source"]).read_text().splitlines() if json.loads(ln)["run_idx"] == c["key"])
        items.append({"source": c["source"], "key": c["key"], "set": c["set"], "scenario": rec["scenario"],
                      "team": "_org" in c["source"], "text": rec["acting_text"]})  # fmt: skip
    rng.shuffle(items)
    ids = rng.sample(range(1000, 10000), len(items))
    for it, i in zip(items, ids, strict=True):
        it["id"] = f"C{i}"
    load = [0] * groups
    bins: list[list[dict]] = [[] for _ in range(groups)]
    for it in sorted(items, key=lambda x: -len(x["text"])):  # balance reading load
        g = load.index(min(load))
        bins[g].append(it)
        load[g] += len(it["text"])
    out.mkdir(parents=True, exist_ok=True)
    for g, b in enumerate(bins):
        parts = []
        for it in sorted(b, key=lambda x: x["id"]):
            sc = SCENARIOS[it["scenario"]]
            who = "the Communications Director of a 5- or 12-agent team" if it["team"] else "a single agent"
            criteria = sc["judge_user"].split("Here is the")[0].strip()
            parts.append(f"## Case {it['id']} ({it['scenario']}; transcript by {who})\n\nFacts: {FACTS[it['scenario']]}\n\n"
                         f"Criteria:\n{criteria}\n\nTranscript:\n````\n{it['text']}\n````\n")  # fmt: skip
        (out / f"packet_{g}.md").write_text(f"{RULES}\n\n" + "\n".join(parts))
    (out / "key.json").write_text(json.dumps([{k: it[k] for k in ("id", "source", "key", "set", "scenario")} for it in items], indent=0))  # fmt: skip
    print(len(items), "cases;", [len(b) for b in bins], "per packet;", [round(x / 1e3) for x in load], "k chars")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("--groups", type=int, default=6)
    x = a.parse_args()
    main(x.out, x.groups)
