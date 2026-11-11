"""Unpack the committed results/ into the runs/ layout that the analysis scripts read.

  uv run python scripts/unpack_results.py [--out DIR]   ->  DIR/runs/... (DIR defaults to the repository)

  results/cho_shen/<study>/      ->  runs/cho_shen/<study>/<study>/   every output of a study
  results/panel_v3/<study>/      ->  runs/panel/                      three-provider panel verdicts
  results/adjudication/[<study>/] -> runs/adjudication/               reasoning-panel verdicts, refusal fills
  results/adjudication/reading/  ->  runs/adjudication/reading/       the blinded readers of amendment 42

  results/adjudication/<study>/substitutes.jsonl -> runs/adjudication/substitutes_<study>.jsonl
  results/<other>/...            ->  runs/<other>/...                 e.g. the codings of amendments 66-69

Studies: p3 (amendment 40), a41, p4 (43), a44, a45a and a45b (45), a46, a48, a49, a50, a51-a65, a70 and a71 with their pilots a70p and a71p. Each compressed file decompresses to
the file the run wrote, byte for byte. Files already in DIR/runs are never overwritten.
"""

import argparse
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def target(src: Path, runs: Path) -> Path:
    """Where a file under results/ lives in the runs/ layout."""
    top, *rest = src.relative_to(RESULTS).parts
    name = src.name.removesuffix(".gz")
    if top == "cho_shen":
        return runs / "cho_shen" / rest[0] / rest[0] / name
    if top == "panel_v3":
        return runs / "panel" / name
    if top == "adjudication" and rest[0] == "reading":
        return runs / "adjudication" / "reading" / name
    if top == "adjudication" and name == "substitutes.jsonl" and len(rest) > 1:
        return runs / "adjudication" / f"substitutes_{rest[0]}.jsonl"  # a study's own refusal fills (amendment 65 on)
    if top == "adjudication":
        return runs / "adjudication" / name
    return runs / top / Path(*rest).with_name(name)  # results/amend66/ and later: the same path under runs/


def unpack(out: Path) -> tuple[int, int]:
    """Write results/ into out/runs; returns (files written, files already present)."""
    written = kept = 0
    for src in sorted(p for p in RESULTS.rglob("*") if p.is_file()):
        dst = target(src, out / "runs")
        if dst.exists():
            kept += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".gz":
            dst.write_bytes(gzip.decompress(src.read_bytes()))
        else:
            shutil.copyfile(src, dst)
        written += 1
    return written, kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT, help="directory to hold runs/ (default: the repository)"
    )
    a = ap.parse_args()
    written, kept = unpack(a.out)
    print(f"{written} files written to {a.out / 'runs'}, {kept} already present (left unchanged)")


if __name__ == "__main__":
    main()
