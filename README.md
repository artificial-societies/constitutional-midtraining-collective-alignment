# Constitutional Midtraining in an Organisation of Language-Model Agents

Artificial Societies · **[Paper (PDF)](paper/paper.pdf)** · [Citation](CITATION.cff)

This repository holds the harness, judges, analysis code and the compressed transcript and verdict of every run behind
the paper. Every figure and number in the paper can be rebuilt from the committed results in a few minutes, with no
GPU or API key.

**TL;DR** 
We place Cho et al.'s constitutionally midtrained checkpoints in Shen et al.'s
twelve-role firm and confront the firm with Lynch et al.'s agentic misalignment scenarios (blackmail, lethal action,
leaking). The firm raised a safety-trained chat model's harm from 54% of runs alone to 75%. One safer agent lowered
harm only as Communications Director, the seat that acts outside the firm, where a midtrained agent cut harm from 63%
to 45% and still completed an authorised delivery at least as often as its control. After later benign fine-tuning,
the midtrained agent kept most of its advantage while a conventional safety step lost most of its own. Every claim is
relative to an otherwise identical control.

## Find a result

Each study has a short code used for its folders and scripts. Results live in `results/cho_shen/<code>/` (one record
per run) and `results/adjudication/<code>/` (reasoning-panel verdicts).

| Paper section | Figures and tables | Study code | Analysis script (`scripts/`) |
|---|---|---|---|
| 4.1 The firm makes a safety-trained model more harmful | Fig. 1, 3 | `p4` (main) | `phase4_tables.py` |
| 4.2 Midtraining carries a modest advantage | Fig. 4 | `p4`, pooled with `p3`, `a41` | `phase4_tables.py`, `phase3_tables.py`, `amend41_adjudicated.py` |
| 4.3 A safety step gives a larger initial benefit | Fig. 5 | `a45a` (dose), `a45b` | `amend45_tables.py compare 4k` |
| 4.4 Interventions work at the seat that acts | Fig. 6, 7, 8 | `a44` (confirmation), `a46` (safety-tuned placement) | `amend44_tables.py`, `amend46_tables.py`, `amend68_tables.py` |
| 4.5 Other model families | Fig. 8 | `a65` | `amend65_tables.py`, `amend67_tables.py` |
| 4.6 A second condition | Fig. 9 | `a70` | `amend70_tables.py` |
| 4.7 The safer Director still acts when authorised | Fig. 10 | `a71` | `amend71_recode.py` |
| 4.8 Midtraining's advantage lasts | Fig. 11, Table 2 | `a48` (durability), `a49` (same session), `a50` (combination) | `amend48_tables.py`, `amend49_tables.py`, `amend50_tables.py`, `amend69_tables.py` |
| 4.9 Robustness, Appendix A (judges) | Fig. 12, Table 4 | all; audit in `results/adjudication/{cases.json,reading/}` | `amend42_tables.py` |
| Appendix D (refusals) | Table 7 | refusal codes in `results/amend67` to `amend69` | `amend67_tables.py` to `amend69_tables.py` |
| Limitations (smaller-model recreations) | | `a51` to `a64` | `amend51_tables.py` to `amend64_tables.py` |

Pilots (`a70p`, `a71p`) sit beside their main runs. Every in-text number is a macro in
[paper/generated/numbers.tex](paper/generated/numbers.tex), and the plotted data are in `paper/data/*.csv`.

## Reproduce the paper

```bash
uv sync                                    # Python 3.13
uv run python paper/build_data.py          # results/ -> paper/data/*.csv and paper/generated/*.tex
Rscript paper/figures.R                    # -> paper/figures/*.pdf (ggplot2, dplyr, readr, tidyr, scales)
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

To run one study's own analysis, unpack the results once and call its script:

```bash
uv run python scripts/unpack_results.py    # -> ./runs/ (git-ignored)
uv run python scripts/amend44_tables.py    # e.g. the confirmation study
```

## Read the transcripts

Each line of `results/cho_shen/<code>/*.jsonl.gz` is one run: the scenario, the model in each seat, every agent's turn
(`log`), the Director's final actions (`acting_text`), and GPT-4o's verdict. File names give study, scenario, seat and
model, for example `a44_blackmail_director_uniform-dr.jsonl.gz`.

```python
import gzip, json
runs = [json.loads(l) for l in gzip.open("results/cho_shen/a44/a44_blackmail_director_uniform-dr.jsonl.gz")]
print(runs[0]["acting_text"])
```

Panel verdicts in `results/adjudication/` point back to a run by `source` file and `key` (the line), with each judge's
answer and the quoted evidence.

## Run new experiments

New runs cost money. They need a [Modal](https://modal.com) workspace with H200 GPUs, OpenAI and Anthropic API keys,
and Gemini on Vertex AI. Copy `.env.example` to `.env` and run `uv sync --extra serve --extra gemini`.

1. **Serve** each model on Modal with `cmca/shen/serving/nemotron_modal.py` (Cho's and NVIDIA's checkpoints) or
   `open_modal.py` (other families). Set the `URL` constant in the study's `scripts/amendNN_spec.py` to your workspace.
2. **Run** `scripts/amendNN_spec.py` to write a launch spec, then `modal run --detach cmca/cho/driver_modal.py --spec ...`.
   `cmca/cho/chain_modal.py` runs a pilot, its gate and the main run unattended.
3. **Judge** with `cmca/cho/judge_modal.py` in the cloud or `scripts/judge_run.py` locally.

The safety and benign fine-tuning steps are `cmca/midtrain/safety_modal.py` and `benign_modal.py`. Each module's
docstring gives its commands.

## Layout

```
paper/          paper.tex, paper.pdf, build_data.py (results -> data and macros), figures.R
cmca/cho/       the experiment: Shen's firm on Lynch's scenarios (shen_org.py), scenario prompts, the judges
                (adjudicate.py, panel.py), and the Modal driver, judge and chain
cmca/midtrain/  fine-tuning on Modal and the smaller-model midtraining recreations
cmca/shen/      chat client and vLLM serving
scripts/        per-study launch specs (*_spec.py) and analyses (*_tables.py), judging and data utilities
results/        compressed outputs of every study, byte-identical to the runs
```
