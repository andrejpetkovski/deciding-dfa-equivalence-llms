# Deciding DFA Equivalence with LLMs: Effects of Witness Depth and Representation

Code, data, and raw model outputs for the TELFOR 2026 paper by Andrej Petkovski, Borjan
Dimeski, and Marija Mihova (Faculty of Computer Science and Engineering, Ss. Cyril and
Methodius University, Skopje).

We test whether large language models can decide if two deterministic finite automata
(DFAs) accept the same language, and support a "not equivalent" answer with a word that
distinguishes them. Ground truth comes from an exact product-automaton breadth-first
search, and every proposed word is checked by simulating both machines. Three models
answered 200 generated pairs of 14-state binary DFAs, each in two equivalent text
renderings and under two prompts: 2,400 primary calls, plus 960 repeat calls to measure
run-to-run variation.

| Result | Value |
|---|---|
| Valid-witness success on non-equivalent pairs (primary) | 88.1 % (1,057 / 1,200) |
| H1: success falls as the shortest distinguishing word gets longer | supported, slope −0.1860 log-odds [−0.2374, −0.1347] |
| H2: asking for state traces improves success | contradicted, −4.83 percentage points [−8.49, −1.30] |
| H3: decisions change across renderings | 5.5 % of comparisons [4.4, 6.7], no more than between identical repeated requests |

Every number above is produced by the code in this repository from the files in `data/`.

## Repository layout

```
configs/main_frozen.yaml    frozen study configuration: design, seeds, models, hypotheses
prompts/                    system prompt, the two prompt templates (P0, P1), JSON schemas
src/                        DFA model, oracle, generators, renderings, prompts, runner, grader
analysis/                   frozen confirmatory analysis and post-review analyses (review_*.py)
scripts/                    baselines, design precision, pilot exclusion list, test fixtures
tests/                      unit and property tests (pytest + hypothesis)
data/pairs/main_v2/         the 200 DFA pairs, canonical hashes, QA report
data/pairs/pilot*/          pilot pairs, kept only so the main set can exclude them
data/renderings/main_v2/    the R0/R1 text of every DFA and the 800 prompt records
data/raw/                   every raw API response of the main run (append-only log)
data/parsed/main_v2/        parsed and graded answers
analysis/tables/, figures/  every result table and figure in the paper
supplement/                 all 800 assembled prompts, the config as tagged, provenance
```

The pilot configurations (`configs/pilot*.yaml`) and pilot DFA pairs are included because
the tests and the pilot-exclusion step use them. Pilot model outputs are not included.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt     # Windows: .venv\Scripts\pip
python -m pytest
```

All 730 tests should pass. Quality gate 8 downloads a public tokenizer from the Hugging
Face Hub the first time it runs.

## Reproducing the results

No model calls are needed: the raw responses are in `data/raw/`. Run from the repository
root, in this order:

```bash
python -m src.generate        --config configs/main_frozen.yaml --overwrite
python -m src.render_prompts  --config configs/main_frozen.yaml --overwrite
python -m src.dataset_qa      --config configs/main_frozen.yaml --strict
python -m src.parse_and_grade --config configs/main_frozen.yaml
python -m analysis.make_results --config configs/main_frozen.yaml
python -m analysis.preregistered_secondaries --graded data/parsed/main_v2/graded.jsonl --out analysis/tables/main_v2/preregistered_secondaries.json
python -m analysis.repeatability --config configs/main_frozen.yaml
python -m analysis.descriptives  --config configs/main_frozen.yaml
python -m scripts.baselines      --config configs/main_frozen.yaml
python -m scripts.design_precision
```

`--overwrite` is needed because the committed pairs and renderings already exist. The
regenerated dataset, renderings, grades, tables, and figures are byte-identical to the
committed ones; the one exception is the configuration hash recorded in
`analysis/tables/main_v2/checksums.json` (see the freeze record below).

Analyses added after peer review (exploratory; they do not change the frozen results):

```bash
python -m analysis.review_h3            # rendering changes vs. identical repeated requests
python -m analysis.review_caps          # output-cap hits and cap-excluded H1/H2
python -m analysis.review_construction  # how non-equivalent pairs are built; family breakdown
python -m analysis.review_depth         # per-model depth slopes; categorical depth
```

Their reports are in `analysis/tables/review_20261028/`. With other NumPy/statsmodels
builds, the JSON files can differ in the last floating-point digits (around 1e-14) and in
the recorded runtime versions, and on Windows these scripts write CRLF line endings;
every reported value is unchanged.

## Re-running the models

`src/run_models.py` sends real API calls through OpenRouter and costs money. It defaults to
`--dry-run`. To make new calls, copy `.env.example` to `.env`, add your own key, and follow
the prompts of `--execute`. Temperature 0 is accepted but not deterministic on these
deployments, so new calls will not reproduce the raw outputs exactly.

## Freeze record

The design, prompts, scoring, and analysis were frozen in the authors' private Git
repository before the main run: tag `freeze-main-v2`, commit `134e4c9`, 18:18:36 UTC on
September 9, 2026. The first model request was sent at 19:15:21 UTC that day. This is a
private timestamped record, not an independent public registration. This repository is a
snapshot of that project, without its history and internal working documents.

- The prompt and schema files are byte-identical to the tagged versions; their SHA-256
  hashes are listed in `supplement/provenance.json`.
- `supplement/main_frozen_at_tag.yaml` is the configuration exactly as tagged (SHA-256
  `590999691abc99351618b2ba3f6263bad1300b400517dd006d441c2eb1221803`). The published
  `configs/main_frozen.yaml` differs from it in one setting, request concurrency raised
  from 4 to 8, recorded before any main-run call.
- The published configuration hashes to `2bd2b6cd…` (SHA-256 of its bytes). Each record
  in the raw log stores the hash of the configuration it ran under: `2bd2b6cd…`, or
  `0684af51…`, which is the same file with Windows (CRLF) line endings, as it was on disk
  for part of the run. `analysis/tables/main_v2/checksums.json` stores the CRLF form for
  the same reason, so it will show the LF hash when regenerated from this repository.
- Deviations after the freeze, all disclosed in the paper: four calls that had returned
  provider errors with no answer were unintentionally re-run (H2 moved from −5.00 to
  −4.83 points); a McNemar implementation that mixed repeat calls into the primary data
  was corrected; the pre-registered H2/H3 estimators and the repeat-audit schedule were
  implemented after the freeze.

Code comments cite the research rules below as "research rule N" or "rule N", and
sections of the study protocol as "§N"; the protocol document is not part of this
release. Comments that pointed to unpublished files were reworded for this release; no
code was changed. The research rules:

1. The exact oracle assigns every label; a candidate pair whose verdict disagrees with the
   generator's intent is rejected, never relabeled.
2. Frozen components (generator, prompts, outcomes, exclusions, analysis) are not edited
   after the freeze; deviations are recorded.
3. Raw responses are append-only and are never cleaned, including errors and refusals.
4. Only transport and rate-limit failures are retried, never wrong or malformed answers.
5. A parse failure counts as wrong in the primary metric.
6. Every reported number is produced by the analysis code.
7. All randomness comes from explicitly passed, seed-derived generators.
8. A prompt contains only the frozen template and the rendered DFAs; no pair ID, family,
   depth, seed, or label mapping.
9. The DFA pair is the unit of analysis; all intervals cluster on it.
10. State labels such as `q00` exist only in renderings, never in canonical objects.
11. Hypotheses H1–H3 are fixed in the config; anything else is exploratory.
12. Pilot pairs are excluded from the main set by canonical hash.

## License

Code: MIT (see `LICENSE`). Data, results, and documentation: CC BY 4.0 (see
`LICENSE-DATA`).

## Citation

GitHub's "Cite this repository" button uses `CITATION.cff`. In BibTeX:

```bibtex
@inproceedings{petkovski2026dfa,
  author    = {Andrej Petkovski and Borjan Dimeski and Marija Mihova},
  title     = {Deciding {DFA} Equivalence with {LLMs}: Effects of Witness Depth and Representation},
  booktitle = {Proc. Telecommunications Forum (TELFOR 2026)},
  year      = {2026}
}
```
