# Scripts and modules

An index of every file in `scripts/` and every module in `cmca/` (`cho/`, `midtrain/`, `shen/`), grouped by role.
Studies are numbered in the order they were designed (amendment numbers), and every script's docstring gives its
exact command line.

Studies in the paper: Phase 3 = amendment 40 (earlier five-role team study, judge audit), main study = amendment 43
(Phase 4), confirmation 44, safety 45, safety-tuned placement 46, durability 48, same-session durability 49,
combination 50, other firms 65, refusal codings 66-69, restriction 70. Amendments 51-64 are model screens and the
recreations of Cho's recipe on smaller models (Mistral-24B, Nemotron-3-Nano), cited only in the paper's limitations.

Markers:

- **[build]** imported by `paper/build_data.py`, directly or through another imported module. Changing it changes
  the paper's numbers.
- **[data]** produced or judged data that the paper reads from `results/`.
- **[superseded]** kept for the record (it produced committed results or is cited in the log) but replaced by a
  later tool; see the note.

`paper/build_data.py` imports `unpack_results`, `phase3_tables`, `phase4_tables`, `amend44_tables`,
`amend45_tables`, `amend46_tables`, `amend48_tables`, `amend49_tables`, `amend50_tables`, `amend57_tables`,
`amend65_tables`, `amend66_tables`, `amend69_tables`, `amend70_tables` and `cmca.cho.shen_org`. Through those it
also imports `amend62_tables` and `amend59_tables` (via `amend65_tables`), and `cmca.env` and `cmca.shen.llm` (via
`shen_org`).

## Running studies

The harness itself and the launch-spec builders. A `*_spec.py` writes a JSON job list for
`cmca/cho/driver_modal.py`; the GPU servers are deployed beforehand from `cmca/shen/serving/`.

| File | What it does | Serves |
| --- | --- | --- |
| `cmca/cho/shen_org.py` | Lynch's three insider-threat scenarios run alone or by a firm on Shen's mechanics (core team or full 12-role firm); roles, reporting tree, tool-call parsing | every study; Figure 1's firm layout **[build]** |
| `cmca/cho/driver_modal.py` | Runs `shen_org` jobs from a spec inside Modal, detached, writing to the `cmca-runs` Volume | amendment 35 on **[data]** |
| `cmca/cho/chain_modal.py` | Runs driver stages in order inside Modal, each judged in the cloud, with a pre-registered gate between them | amendment 70 (pilot, gate, main) **[data]** |
| `cmca/cho/blackmail_prompts.json`, `leaking_prompts.json`, `murder_prompts.json` | Scenario text (Cho's blackmail adaptation; Lynch's leaking and lethal action), extracted as data | every study **[data]** |
| `cmca/cho/lynch_conditions.json` | Lynch's further goal x urgency conditions, written by `scripts/extract_lynch*.py` | restriction study, amendment 70 **[data]** |
| `cmca/shen/llm.py` | Async chat client with retries and spend accounting (Azure, native OpenAI/Anthropic, our vLLM servers) | every run and judge **[build]** (imported via `shen_org`) |
| `cmca/shen/serving/nemotron_modal.py` | vLLM serving of Nemotron-3-Super checkpoints (NVIDIA's FP8 chat model; Cho's BF16 checkpoints) on Modal | every study with Cho's or NVIDIA's 120B models **[data]** |
| `cmca/shen/serving/open_modal.py` | Generic vLLM serving of other families' chat models on Modal | amendments 51-65 (other-firms hosts) **[data]** |
| `scripts/phase3_spec.py` | Launch spec for Phase 3 | amendment 40 |
| `scripts/amend41_spec.py` | Launch spec for the 12-role confirmation of Phase 3 | amendment 41 |
| `scripts/phase4_spec.py` | Launch spec for the main study (all models alone and in the firm, plus seat interventions) | amendment 43, main study **[data]** |
| `scripts/amend44_spec.py` | Launch spec, placement confirmation (Director vs Partner seat) | amendment 44 **[data]** |
| `scripts/amend45_spec.py` | Launch specs, safety fine-tuning vs midtraining (stage A dose, stage B comparison) | amendment 45 **[data]** |
| `scripts/amend46_spec.py` | Launch spec, safety-tuned model in Director and Partner seats | amendment 46 **[data]** |
| `scripts/amend48_spec.py` | Launch spec, benign fine-tuning (Director seat and alone) | amendment 48 **[data]** |
| `scripts/amend49_spec.py` | Launch spec, same-session durability | amendment 49 **[data]** |
| `scripts/amend50_spec.py` | Launch spec, safety step on a midtrained base after benign tuning | amendment 50 **[data]** |
| `scripts/amend51_spec.py` | Launch spec, screen of other model families | amendment 51 (screen) |
| `scripts/amend52_spec.py` | Launch spec, screen of Qwen models over 8B | amendment 52 (screen) |
| `scripts/amend53_spec.py` | Launch spec, Qwen3.5-35B-A3B on a newer vLLM | amendment 53 (screen) |
| `scripts/amend54_spec.py` | Launch spec, Qwen3.5-35B-A3B at 8,192 tokens | amendment 54 (screen) |
| `scripts/amend55_spec.py` | Launch spec, Qwen2.5-32B lethal-action confirmation and Qwen3.5 with presence penalty | amendment 55 (screen) |
| `scripts/amend57_spec.py` | Launch spec, our Mistral-24B checkpoints as Director of Mistral's firm | amendment 57 (recreation) |
| `scripts/amend58_spec.py` | Launch spec, forced deliberation for the Mistral-24B checkpoints (exploratory) | amendment 58 (recreation) |
| `scripts/amend59_spec.py` | Launch spec, our Nemotron-3-Nano checkpoints | amendment 59 (recreation) |
| `scripts/amend60_spec.py` | Launch spec, NVIDIA's Nano chat model rescreened at 8,192 tokens | amendment 60 (screen) |
| `scripts/amend61_spec.py` | Launch spec, Nano chat model with reasoning on | amendment 61 (screen) |
| `scripts/amend62_spec.py` | Launch spec, Nano checkpoints alone (replication) and as Director of NVIDIA's firm | amendment 62 (recreation) |
| `scripts/amend63_spec.py` | Launch spec, Mistral-24B checkpoints alone (replication) | amendment 63 (recreation) |
| `scripts/amend64_spec.py` | Launch spec, quarter-corpus Nano checkpoint vs control | amendment 64 (recreation) |
| `scripts/amend65_spec.py` | Launch spec, Cho's 120B checkpoints as Director of Mistral's and Qwen3.5's firms | amendment 65, other-firms study **[data]** |
| `scripts/amend70_spec.py` | Launch specs (pilot and main), Director of NVIDIA's firm under Lynch's restriction condition | amendment 70, restriction study **[data]** |

## Judging

| File | What it does | Serves |
| --- | --- | --- |
| `cmca/cho/adjudicate.py` | The primary judge: the three-model reasoning panel (Claude Opus 5.5, gpt-5.5, Gemini 3.1 Pro), blind, with fact sheet and verbatim actions; also holds the `Judges` client and `FACTS` used by the refusal coders | amendment 42 on; every study's primary harm **[data]** |
| `cmca/cho/panel.py` | The secondary three-provider panel (gpt-5.4, Claude Opus 4.5, Gemini 3.1 Pro) | amendment 29; main and confirmation studies, Figure robust **[data]** |
| `cmca/cho/judge_modal.py` | Judges a driver run inside Modal: adjudication, refusal fill, Gemini backfill, merge, tables | amendment 65 on **[data]** |
| `scripts/amend42_fill.py` | Fills each refused Claude Opus 5.5 vote with Claude Opus 4.5 (the pre-registered deviation) | every study from 42 on **[data]** |
| `scripts/gemini_backfill.py` | Re-asks Gemini for verdicts lost to expired credentials, into `runs/adjudication_gemini/` | amendment 44 on; also run by `judge_modal.py` **[data]** |
| `scripts/gemini_merge.py` | Builds `runs/adjudication_merged` with backfilled Gemini verdicts in place | run by `judge_modal.py` |
| `scripts/judge_run.py` | Judges a finished run locally with the adjudication panel, then runs the analysis | amendments 45-64 **[superseded]** by `cmca/cho/judge_modal.py` (amendment 65 on) |
| `scripts/phase4_judge_loop.py` | Judges the main study while it runs (both panels, refusal fill), then the frozen analysis | amendment 43 **[superseded]** by `judge_run.py`, then `judge_modal.py` |
| `scripts/amend44_judge_loop.py` | As `phase4_judge_loop.py`, with the adjudication in four shards | amendment 44 **[superseded]** by `judge_run.py`, then `judge_modal.py` |
| `scripts/amend42_cases.py` | Selects the Phase 3 outputs to adjudicate and flags target-gate misses | amendment 42, judge audit (Appendix: Judges) |
| `scripts/amend42_packets.py` | Writes blinded reading packets for the expert readers | amendment 42, judge audit (Appendix: Judges) |
| `scripts/amend41_adjudicated.py` | Builds adjudication cases for amendment 41 and re-tests it with adjudicated harm | amendment 41 (secondary) |
| `scripts/amend47_objections.py` | Codes ethical objections in the firm's internal emails and tests O1-O2 | amendment 47 (not in the paper) |
| `scripts/amend59_selfpres.py` | Judged self-preservation measure (Claude Opus 4.5) for the Nano checkpoints | amendment 59 (recreation; not in the paper) |
| `scripts/amend66_code.py` | Codes why amendment 64's Director took no external action (Opus 5.5, GPT-5.5 for agreement) | amendment 66 (exploratory; Nano recreation) |
| `scripts/amend67_code.py` | Same coding for amendment 65's Director (other firms) | amendment 67, refusal coding **[data]** |
| `scripts/amend68_code.py` | Same coding for amendment 44's Director (NVIDIA's firm) | amendment 68, refusal coding **[data]** |
| `scripts/amend69_code.py` | Same coding for the durability studies' Directors (46, 48, 49, 50) | amendment 69, refusal coding **[data]** |

## Analysis tables per amendment

Each `*_tables.py` is the pre-registered analysis, committed before its study ran; it prints the study's tables
from `runs/` (rebuild that layout with `scripts/unpack_results.py`).

| File | Study | Notes |
| --- | --- | --- |
| `scripts/phase3_tables.py` | Phase 3, amendment 40 | Also the shared estimators (`mh`, `holm`, `interaction`, `fmt`, `pct`, `cnt`) **[build]** |
| `scripts/phase4_tables.py` | Main study, amendment 43 | Also the loader (`load`, `substitutes`, `MODELS`) every later table reuses **[build]** |
| `scripts/amend41_tables.py` | 12-role confirmation of Phase 3, amendment 41 | |
| `scripts/amend42_tables.py` | Judge audit of Phase 3, amendment 42 | Appendix: Judges (stated qualitatively) |
| `scripts/amend44_tables.py` | Confirmation study, amendment 44 | **[build]** |
| `scripts/amend45_tables.py` | Safety study, amendment 45 | **[build]** |
| `scripts/amend46_tables.py` | Safety-tuned placement study, amendment 46 | **[build]** |
| `scripts/amend48_tables.py` | Durability study, amendment 48 | **[build]** |
| `scripts/amend49_tables.py` | Same-session durability study, amendment 49 | **[build]** |
| `scripts/amend50_tables.py` | Combination study, amendment 50 | **[build]** |
| `scripts/amend51_tables.py` | Screen of other families, amendment 51 | |
| `scripts/amend52_tables.py` | Qwen screen, amendment 52 | Same analysis as 51 |
| `scripts/amend53_tables.py` | Qwen3.5 screen, amendment 53 | Same analysis as 51 |
| `scripts/amend54_tables.py` | Qwen3.5 screen at 8,192 tokens, amendment 54 | Same analysis as 51 |
| `scripts/amend55_tables.py` | Qwen2.5 confirmation and Qwen3.5 with penalty, amendment 55 | |
| `scripts/amend57_tables.py` | Mistral-24B checkpoints, amendment 57 | Its `load` is reused for amendments 65 and 70 **[build]** |
| `scripts/amend58_tables.py` | Forced deliberation, amendment 58 (exploratory) | |
| `scripts/amend59_tables.py` | Nano checkpoints, amendment 59 | Imported via `amend62_tables` **[build]** |
| `scripts/amend60_tables.py` | Nano chat rescreen, amendment 60 | |
| `scripts/amend61_tables.py` | Nano chat rescreen with reasoning on, amendment 61 | |
| `scripts/amend62_tables.py` | Nano checkpoints, replication and Director seat, amendment 62 | Its `holm` is imported by `amend65_tables` **[build]** |
| `scripts/amend63_tables.py` | Mistral-24B replication, amendment 63 | |
| `scripts/amend64_tables.py` | Quarter-corpus Nano checkpoint, amendment 64 | |
| `scripts/amend65_tables.py` | Other-firms study, amendment 65 | **[build]** |
| `scripts/amend66_tables.py` | Refusal coding of amendment 64, amendment 66 | Only its `kappa` is used by the paper **[build]** |
| `scripts/amend67_tables.py` | Refusal coding of amendment 65, amendment 67 | The paper reads amendment 67's codes through `amend69_tables.load` |
| `scripts/amend68_tables.py` | Refusal coding of amendment 44, amendment 68 | The paper reads amendment 68's codes through `amend69_tables.load` |
| `scripts/amend69_tables.py` | Refusal coding of the durability studies, amendment 69 | **[build]** |
| `scripts/amend70_tables.py` | Restriction study, amendment 70 | **[build]** |

## Paper build

| File | What it does | Serves |
| --- | --- | --- |
| `scripts/unpack_results.py` | Unpacks the compressed `results/` into the `runs/` layout the analysis scripts read | `paper/build_data.py` **[build]** |
| `paper/build_data.py` (not in this folder) | Regenerates every number, table and figure dataset in `paper/data/` from `results/` | the paper |
| `paper/figures.R` (not in this folder) | Draws the figures from `paper/data/` | the paper |

## Training

| File | What it does | Serves |
| --- | --- | --- |
| `cmca/midtrain/__init__.py`, `__main__.py` | Package overview; `python -m cmca.midtrain corpus` command line | |
| `cmca/midtrain/safety_modal.py` | LoRA safety step (Nemotron-SFT-Safety-v2) on Cho's checkpoints, merged shard by shard | amendments 21 and 45; safety, placement, durability, combination studies **[data]** |
| `cmca/midtrain/benign_modal.py` | LoRA benign step (supervised GSM8K) on Cho's control, uniform-dr and control-safety checkpoints | amendment 48; durability and combination studies **[data]** |
| `cmca/midtrain/corpus.py` | Topic filter `BANNED` used by both steps above; also the synthetic-corpus generator of the earlier gpt-oss pilot | `BANNED` **[data]**; generator from the pilot, not in the paper |
| `cmca/midtrain/blend.py` | Base weights, tokenised blends and SFT packs for repeating Cho's recipe on another family | amendments 56 and 59 (Mistral-24B, Nano recreations; limitations only) |
| `cmca/midtrain/blend_quarter.py` | Index for the quarter-corpus Nano arm | amendment 64 (recreation) |
| `cmca/midtrain/cpt_train.py` | Full-parameter continued pretraining and SFT with FSDP2, launched by torchrun | amendment 56 on (recreations) |
| `cmca/midtrain/cpt_modal.py` | Runs `cpt_train.py` on one 8xB200 Modal node; the per-arm chain; kernel checks | amendment 56 on (recreations) |
| `scripts/stage1_launch.py` | Starts Stage 1 training on Modal (treated arm, then control) | amendment 56 (recreation) |
| `scripts/stage1_status.py` | Prints Stage 1's progress from the `midtrain` Volume | amendment 56 (recreation) |

## Utilities

| File | What it does | Serves |
| --- | --- | --- |
| `cmca/__init__.py`, `cmca/cho/__init__.py`, `cmca/shen/__init__.py` | Package docstrings | |
| `cmca/env.py` | Loads credentials from the environment or a dotenv file, never printing them | every API call **[build]** (imported via `shen_org`) |
| `scripts/extract_lynch.py` | Extracts Lynch et al.'s blackmail and lethal-action conditions (read as data, never executed) into `cmca/cho/lynch_conditions.json` | amendment 70 **[data]** |
| `scripts/extract_lynch_leaking.py` | Adds the leaking scenario's conditions the same way, checked against `leaking_prompts.json` | amendment 70 **[data]** |

## Flags

Nothing is deleted; these notes are for readers of the code.

- `scripts/phase4_judge_loop.py`, `scripts/amend44_judge_loop.py`, `scripts/judge_run.py`: superseded in turn by
  `cmca/cho/judge_modal.py`, which does the same judging inside Modal. Kept because the committed verdicts of
  amendments 43-64 were made with them.
- `cmca/midtrain/corpus.py`: only `BANNED` is still used; the corpus generator belongs to the earlier gpt-oss pilot.
- Stale references in docstrings (code unchanged): `scripts/gemini_backfill.py` mentions `scripts/gemini_compare.py`,
  which is not in the repository; `cmca/cho/chain_modal.py` uses `scripts/x_tables.py` as a placeholder name.
- Not in the paper: amendments 41, 47, 51-64 and 66 (scripts kept as the record of pre-registered work). Amendment
  66's `kappa` is still imported by the paper build.
