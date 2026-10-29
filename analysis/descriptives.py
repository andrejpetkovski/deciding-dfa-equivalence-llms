"""Descriptive breakdowns the manuscript needs, none of them confirmatory (D15).

**Why this file exists.** Rule 6 forbids typing a number into the paper that a script did
not produce, and the frozen analysis produces only *aggregate* rates. Several things the
protocol explicitly asks to report therefore had no artifact at all:

- PROTOCOL §10: "Report per-bin/per-prompt witness success with 95% pair-bootstrap
  intervals." Figure 1 *draws* them; no table ever emitted the numbers.
- §11.1's main result table shell -- model x prompt x representation.
- §11.2's error taxonomy as counts. Figure 4 draws it; nothing wrote it down.
- §11.2's "record first wrong symbol index" for trace errors.
- §8.5/§10.1's forgiving-parser sensitivity analysis, which may exist only as a *labelled*
  sensitivity result and never as the primary number.
- §10.7's appendix of token, latency and cost distributions.

**What this file is not.** It recomputes no confirmatory estimate. `make_results.py`,
`metrics.py` and `stats.py` are untouched, so the frozen primary analysis stays
byte-reproducible and this is visibly *additional* output rather than a quiet re-run under
new definitions. Every rate here is built by calling the frozen metric functions on a
subset of the same primary rows; the subsetting is the only new logic, and the frozen
config already declares every quantity below to be "DESCRIPTIVE or EXPLORATORY ... and
carries no confirmatory claim regardless of what it shows".

**Intervals match the figures on purpose.** :data:`RESAMPLES` and the seed are the ones
`analysis.figures` uses, so a number in `table_depth_bins.csv` is the same number Figure 1
plots rather than a second, slightly different estimate of it. The confirmatory intervals
in `results.json` use the full 10,000 and are unaffected by anything here.

Usage::

    python -m analysis.descriptives --config configs/main_frozen.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from analysis.figures import DEPTH_BINS
from analysis.metrics import (
    TRANSPORT_FAILED,
    ValidationError,
    _decision,
    _paired_cells,
    balanced_decision_accuracy,
    decision_flip_rate,
    orbit_success,
    parse_failure_rate,
    primary_rows,
    shortest_witness_success,
    trace_correctness,
    validate_rows,
    witness_success,
)
from analysis.stats import cluster_bootstrap_rate
from src.config import ConfigError, load_config
from src.io_utils import canonical_json, read_jsonl, sha256_of_file, write_text

__all__ = ["build_descriptives", "main"]

#: Matches `analysis.figures._BOOTSTRAP` so tables and figures report identical intervals.
#: The confirmatory estimates use the frozen 10,000 and are computed elsewhere.
RESAMPLES = 2000

#: `analysis.stats.cluster_bootstrap_rate`'s own default, restated so the equality with the
#: figures is explicit rather than incidental.
SEED = 20260908

#: Figure 4's categories, in the same order, so the table and the stacked bars agree.
ERROR_CATEGORIES = (
    "success",
    "valid but not shortest",
    "invalid witness",
    "said equivalent",
    "parse failure",
    "transport failure",
)


# --------------------------------------------------------------------------- helpers


def _rate_row(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[Sequence[dict[str, Any]]], Any],
    *,
    interval: bool = True,
) -> dict[str, Any]:
    """One frozen metric over a subset, with a pair-clustered interval.

    The metric is applied unchanged; only the row subset is new. Returning the numerator
    and denominator alongside the rate keeps §9's rule -- a rate without its denominator is
    unreportable -- true of every breakdown here, not just the aggregates.
    """
    rate = metric(rows)
    record: dict[str, Any] = {
        "metric": rate.name,
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "rate": rate.value,
        "denominator_description": rate.denominator_description,
        "ci_low": None,
        "ci_high": None,
    }
    if interval and rate.value is not None:
        low, high = cluster_bootstrap_rate(
            rows, lambda subset: metric(subset).value, resamples=RESAMPLES, seed=SEED
        )
        record["ci_low"], record["ci_high"] = low, high
    return record


def _bin_of(depth: int | None) -> str | None:
    """The declared stratum a depth falls in, or ``None`` outside all four."""
    if depth is None:
        return None
    for label, low, high in DEPTH_BINS:
        if low <= depth <= high:
            return label
    return None


def _neq(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row["oracle_equivalent"]]


def _classify(row: dict[str, Any]) -> str:
    """Figure 4's taxonomy, transcribed so a table row and a bar segment cannot diverge.

    The order of the branches is the definition: a transport failure is not a parse
    failure, and a parse failure is not a wrong answer about the automata even though rule
    5 makes both count as failures on the primary rate.
    """
    if row["status"] == TRANSPORT_FAILED:
        return "transport failure"
    if not row["parse_valid"]:
        return "parse failure"
    if row["witness_success"]:
        return "success" if row["witness_shortest"] else "valid but not shortest"
    if row["decision_correct"]:
        return "invalid witness"
    return "said equivalent"


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    """Median-first distribution summary; ``None`` throughout when nothing was measured.

    Median rather than mean is the headline because completion-token counts are strongly
    right-skewed here -- a handful of cells run to the 12,288 cap and would drag a mean
    somewhere no cell actually sits.
    """
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None, "total": None}
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "total": float(sum(values)),
    }


# --------------------------------------------------------------------------- sections


def cell_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """§11.1's shell: one row per model x prompt x representation.

    Balanced accuracy needs both classes, so it is computed over all families in the cell;
    the three witness columns are restricted to ground-truth non-equivalent cells by the
    frozen metrics themselves.
    """
    table: list[dict[str, Any]] = []
    for model in sorted({row["model_id"] for row in rows}):
        for prompt in ("P0", "P1"):
            for rendering in ("R0", "R1"):
                cell = [
                    row
                    for row in rows
                    if row["model_id"] == model
                    and row["prompt_condition"] == prompt
                    and row["rendering"] == rendering
                ]
                if not cell:
                    continue
                eq_rate, neq_rate, balanced = balanced_decision_accuracy(cell)
                table.append(
                    {
                        "model_id": model,
                        "prompt_condition": prompt,
                        "rendering": rendering,
                        "cells": len(cell),
                        "balanced_decision_accuracy": balanced,
                        "decision_accuracy_equivalent": eq_rate.value,
                        "decision_accuracy_equivalent_n": eq_rate.denominator,
                        "decision_accuracy_non_equivalent": neq_rate.value,
                        "decision_accuracy_non_equivalent_n": neq_rate.denominator,
                        "end_to_end_witness_success": witness_success(cell).value,
                        "end_to_end_witness_success_n": witness_success(cell).numerator,
                        "witness_denominator": witness_success(cell).denominator,
                        "shortest_witness_success": shortest_witness_success(cell).value,
                        "parse_failure_rate": parse_failure_rate(cell).value,
                        "parse_failures": parse_failure_rate(cell).numerator,
                    }
                )
    return table


def depth_bin_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Witness success per declared stratum -- the numbers behind Figure 1.

    Three levels of aggregation, because they answer different questions and the coarsest
    is the only one with usable precision: pooled over everything, split by prompt, and the
    per-model-per-prompt lines the figure actually draws. The last carries roughly +/-14
    points at this sample size (`scripts/design_precision.py`), which is exactly why the
    interval is emitted next to every point rather than left to the reader.
    """
    neq = _neq(rows)
    table: list[dict[str, Any]] = []

    def add(subset: Sequence[dict[str, Any]], **labels: Any) -> None:
        for label, low, high in DEPTH_BINS:
            cells = [
                row for row in subset if low <= (row["oracle_depth"] or -1) <= high
            ]
            if not cells:
                continue
            record = _rate_row(cells, witness_success)
            table.append({**labels, "depth_bin": label, **record})

    add(neq, model_id="ALL", prompt_condition="ALL")
    for prompt in ("P0", "P1"):
        add(
            [row for row in neq if row["prompt_condition"] == prompt],
            model_id="ALL",
            prompt_condition=prompt,
        )
    for model in sorted({row["model_id"] for row in neq}):
        for prompt in ("P0", "P1"):
            add(
                [
                    row
                    for row in neq
                    if row["model_id"] == model and row["prompt_condition"] == prompt
                ],
                model_id=model,
                prompt_condition=prompt,
            )
    return table


def family_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decision accuracy per pair family -- the numbers behind Figure 2.

    The threats analysis requires family-specific results, and for one reason: the two
    equivalent families are not equally easy. `EQ_ISO` is a renaming of one machine, while
    `EQ_EXPAND` builds two differently shaped machines from a shared core, and only the
    second asks whether a model can recognise equivalence it cannot see by pattern-matching
    the text. Pooling them into one "equivalent" rate hides exactly that contrast.

    Decision accuracy rather than witness success, because two of the three families are
    equivalent and have no witness to produce.
    """
    def accuracy(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(1 for row in subset if row["decision_correct"])
        value = correct / len(subset) if subset else None
        low = high = None
        if value is not None:
            low, high = cluster_bootstrap_rate(
                subset,
                lambda s: (sum(1 for r in s if r["decision_correct"]) / len(s)) if s else None,
                resamples=RESAMPLES,
                seed=SEED,
            )
        return {"numerator": correct, "denominator": len(subset), "rate": value,
                "ci_low": low, "ci_high": high}

    table: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in rows}):
        in_family = [row for row in rows if row["family"] == family]
        for model in ["ALL", *sorted({row["model_id"] for row in rows})]:
            subset = (
                in_family if model == "ALL"
                else [row for row in in_family if row["model_id"] == model]
            )
            if not subset:
                continue
            table.append({"family": family, "model_id": model, **accuracy(subset)})
    return table


def error_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """§11.2's taxonomy as counts, plus the parse-failure subtypes that explain them.

    Splitting parse failures by *reason* matters for H2: an output cap reached is a
    different phenomenon from prose the strict parser rejected, and only the first is
    plausibly a consequence of asking for a longer answer. Both still count as wrong
    (rule 5); this says what they were, not that any should be forgiven.
    """
    neq = _neq(rows)

    def counts(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tally = Counter(_classify(row) for row in subset)
        parse_reason = Counter(
            row.get("parse_outcome") for row in subset if not row["parse_valid"]
        )
        return {
            "cells": len(subset),
            "categories": {name: tally.get(name, 0) for name in ERROR_CATEGORIES},
            "parse_failure_reason": {
                str(key): value for key, value in sorted(parse_reason.items(), key=str)
            },
            "truncated": sum(1 for row in subset if row.get("truncated")),
        }

    by_model = {
        model: counts([row for row in neq if row["model_id"] == model])
        for model in sorted({row["model_id"] for row in neq})
    }
    by_prompt = {
        prompt: counts([row for row in neq if row["prompt_condition"] == prompt])
        for prompt in ("P0", "P1")
    }
    # The equivalent class has its own taxonomy row in §11.2 -- "false non-equivalent:
    # oracle equivalent, model returns false" -- and none of the witness categories apply
    # to it, since there is no witness to produce. Reported separately rather than folded
    # in, so neither class's denominator absorbs the other's failures.
    equivalent = [row for row in rows if row["oracle_equivalent"]]
    eq_parse_failures = [row for row in equivalent if not row["parse_valid"]]
    eq_parsed = [row for row in equivalent if row["parse_valid"]]
    equivalent_class = {
        "cells": len(equivalent),
        "parsed": len(eq_parsed),
        "correct_said_equivalent": sum(1 for row in eq_parsed if row["decision_correct"]),
        "false_non_equivalent": sum(1 for row in eq_parsed if not row["decision_correct"]),
        "parse_failure": len(eq_parse_failures),
        "transport_failure": sum(
            1 for row in equivalent if row["status"] == TRANSPORT_FAILED
        ),
        "parse_failure_reason": {
            str(key): value
            for key, value in sorted(
                Counter(row.get("parse_outcome") for row in eq_parse_failures).items(),
                key=str,
            )
        },
    }

    return {
        "denominator_description": "all scheduled ground-truth non-equivalent primary cells",
        "overall": counts(neq),
        "equivalent_class": equivalent_class,
        "by_model": by_model,
        "by_prompt": by_prompt,
        "all_families_parse_failures": {
            "denominator_description": "all received primary cells, both classes",
            "reason": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        row.get("parse_outcome")
                        for row in rows
                        if not row["parse_valid"]
                    ).items(),
                    key=str,
                )
            },
            "by_prompt": {
                prompt: sum(
                    1
                    for row in rows
                    if row["prompt_condition"] == prompt and not row["parse_valid"]
                )
                for prompt in ("P0", "P1")
            },
        },
    }


def trace_errors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P1 trace outcomes and, for the wrong ones, where they first went wrong (§11.2).

    The index distribution is the point: a trace that diverges at step 1 is a different
    failure from one that survives most of the word, and the protocol asks for the first
    wrong symbol index rather than a pass/fail count.
    """
    correctness = trace_correctness(rows)
    wrong = [
        row
        for row in rows
        if row["prompt_condition"] == "P1"
        and not row["oracle_equivalent"]
        and row["trace_valid"] is False
    ]
    steps = Counter(
        row.get("first_wrong_step") for row in wrong if row.get("first_wrong_step") is not None
    )
    return {
        "trace_correctness": {
            "numerator": correctness.numerator,
            "denominator": correctness.denominator,
            "rate": correctness.value,
            "denominator_description": correctness.denominator_description,
        },
        "invalid_traces": len(wrong),
        "first_wrong_step_histogram": {
            str(step): count for step, count in sorted(steps.items())
        },
        "first_wrong_step_unrecorded": len(wrong) - sum(steps.values()),
    }


def witness_lengths(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How long proposed witnesses were, against the oracle's shortest.

    Two distributions the taxonomy counts cannot show. For *valid* witnesses, the excess
    `|w| - d` says whether a longer-than-shortest success overshot by one symbol or wandered;
    D3 counts both as success, so the size of the concession is worth reporting. For
    *invalid* witnesses, the length says what kind of wrong answer it was -- and length 0 is
    a special case, since gate 3 guarantees both machines agree on the empty word for every
    pair in this dataset, so an empty witness is provably invalid by construction rather
    than by bad luck.
    """
    neq = _neq(rows)

    def length(row: dict[str, Any]) -> int | None:
        return (row.get("grade_detail") or {}).get("witness_length")

    valid = [row for row in neq if row["witness_success"] is True]
    invalid = [
        row
        for row in neq
        if row["parse_valid"] and row["witness_success"] is not True and row["decision_correct"]
    ]
    excess = Counter(
        length(row) - row["oracle_depth"]
        for row in valid
        if length(row) is not None and row["oracle_depth"] is not None
    )
    invalid_lengths = Counter(length(row) for row in invalid if length(row) is not None)
    return {
        "valid": {
            "n": len(valid),
            "excess_over_shortest_histogram": {
                str(k): v for k, v in sorted(excess.items())
            },
            "note": "excess = |w| - d; 0 means the model returned a shortest witness",
        },
        "invalid": {
            "n": len(invalid),
            "length_histogram": {str(k): v for k, v in sorted(invalid_lengths.items())},
            "empty_witness": invalid_lengths.get(0, 0),
            "note": (
                "an empty witness is provably invalid for every pair here: gate 3 "
                "establishes that both machines agree on the empty word"
            ),
        },
    }


def qualitative_examples(
    rows: Sequence[dict[str, Any]],
    raw_text: dict[str, str] | None = None,
    *,
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Examples chosen by a fixed rule, after the quantitative analysis (§11.4).

    §11.4 permits qualitative examples only under a reproducible rule, and forbids
    hand-picking dramatic cases. The rules are therefore mechanical and stated in the
    artifact itself:

    - **one success and one failure per depth bin** -- in each bin, the lexicographically
      first ``run_id`` of each kind. First-by-id cannot be steered towards an interesting
      case, which is the entire point; it is not claimed to be typical in any other sense.
    - **the largest representation flips** -- pairs ranked by *how many* of their six
      (prompt x model) cells flipped between R0 and R1, descending, ties broken by
      ``pair_id`` ascending. "Largest" is a count of flipped cells, since a single flip is
      binary and has no magnitude.

    Raw final-answer text is included when available, truncated at ``max_chars`` with the
    full length recorded. The untruncated text remains in the append-only raw log, which is
    the authoritative record.
    """
    def render(run_id: str) -> dict[str, Any] | None:
        if raw_text is None or run_id not in raw_text:
            return None
        text = raw_text[run_id]
        if text is None:
            return {"final_answer": None, "chars": 0, "truncated": False}
        return {
            "final_answer": text[:max_chars],
            "chars": len(text),
            "truncated": len(text) > max_chars,
        }

    def describe(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "pair_id": row["pair_id"],
            "oracle_depth": row["oracle_depth"],
            "rendering": row["rendering"],
            "prompt_condition": row["prompt_condition"],
            "model_id": row["model_id"],
            "parse_valid": row["parse_valid"],
            "decision_correct": row["decision_correct"],
            "witness_success": row["witness_success"],
            "witness_shortest": row["witness_shortest"],
            "trace_valid": row["trace_valid"],
            "outcome": _classify(row),
            "raw": render(row["run_id"]),
        }

    neq = _neq(rows)
    by_bin: list[dict[str, Any]] = []
    for label, low, high in DEPTH_BINS:
        cells = sorted(
            (row for row in neq if low <= (row["oracle_depth"] or -1) <= high),
            key=lambda row: row["run_id"],
        )
        success = next((row for row in cells if row["witness_success"] is True), None)
        failure = next((row for row in cells if row["witness_success"] is not True), None)
        entry: dict[str, Any] = {"depth_bin": label}
        if success is not None:
            entry["success"] = describe(success)
        if failure is not None:
            entry["failure"] = describe(failure)
        by_bin.append(entry)

    flips_per_pair: Counter = Counter()
    flipped_cells: dict[str, list[dict[str, Any]]] = {}
    for (pair_id, prompt, model), cell in _paired_cells(rows).items():
        r0, r1 = cell["R0"], cell["R1"]
        if not (r0["parse_valid"] and r1["parse_valid"]):
            continue
        if _decision(r0) == _decision(r1):
            continue
        flips_per_pair[pair_id] += 1
        flipped_cells.setdefault(pair_id, []).append(
            {
                "prompt_condition": prompt,
                "model_id": model,
                "R0_said_equivalent": _decision(r0),
                "R1_said_equivalent": _decision(r1),
                "R0_correct": r0["decision_correct"],
                "R1_correct": r1["decision_correct"],
                "R0": describe(r0),
                "R1": describe(r1),
            }
        )
    ranked = sorted(flips_per_pair.items(), key=lambda item: (-item[1], item[0]))[:3]

    # How the flips spread across pairs, and which families carry them. A flip rate of a
    # few percent means something different if it comes from a handful of pathological
    # pairs than if it is thinly spread over many, and the taxonomy counts cannot say which.
    family_of = {row["pair_id"]: row["family"] for row in rows}
    spread = Counter(flips_per_pair.values())
    by_family: Counter = Counter()
    for pair_id, count in flips_per_pair.items():
        by_family[family_of.get(pair_id, "UNKNOWN")] += count

    # Denominators per family, on the same eligibility rule as the flip rate itself: both
    # renderings had to produce a strict-valid Boolean. Without these the family counts are
    # not a rate and cannot be compared across families of different size.
    eligible_by_family: Counter = Counter()
    for (pair_id, _prompt, _model), cell in _paired_cells(rows).items():
        if cell["R0"]["parse_valid"] and cell["R1"]["parse_valid"]:
            eligible_by_family[family_of.get(pair_id, "UNKNOWN")] += 1

    return {
        "selection_rule": (
            "Fixed before inspection and applied mechanically (§11.4). Per depth bin: the "
            "lexicographically first run_id that succeeded and the first that did not. "
            "Representation flips: pairs ranked by number of flipped (prompt x model) "
            "cells, descending, ties broken by pair_id ascending, top 3."
        ),
        "by_depth_bin": by_bin,
        "flip_distribution": {
            "pairs_with_at_least_one_flip": len(flips_per_pair),
            "flipped_cells_per_pair_histogram": {
                str(k): v for k, v in sorted(spread.items())
            },
            "max_flipped_cells_for_one_pair": max(flips_per_pair.values(), default=0),
            "cells_per_pair": 6,
            "flipped_cells_by_family": dict(sorted(by_family.items())),
            "eligible_cells_by_family": dict(sorted(eligible_by_family.items())),
            "flip_rate_by_family": {
                family: (by_family.get(family, 0) / total if total else None)
                for family, total in sorted(eligible_by_family.items())
            },
        },
        "largest_representation_flips": [
            {
                "pair_id": pair_id,
                "flipped_cells": count,
                "of_cells": 6,
                "cells": sorted(
                    flipped_cells[pair_id],
                    key=lambda entry: (entry["prompt_condition"], entry["model_id"]),
                ),
            }
            for pair_id, count in ranked
        ],
    }


def representation_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip rate and orbit success per model -- the numbers behind Figure 3.

    The two denominators differ deliberately and are printed next to each other for that
    reason: the flip rate is conditional on both renderings parsing, orbit success is not,
    so a model that fails to parse often looks better on the first and worse on the second.
    """
    table: list[dict[str, Any]] = []
    for model in ["ALL", *sorted({row["model_id"] for row in rows})]:
        subset = rows if model == "ALL" else [r for r in rows if r["model_id"] == model]
        flip = _rate_row(subset, decision_flip_rate)
        orbit = _rate_row(subset, orbit_success)
        table.append(
            {
                "model_id": model,
                "flip_rate": flip["rate"],
                "flip_ci_low": flip["ci_low"],
                "flip_ci_high": flip["ci_high"],
                "flips": flip["numerator"],
                "flip_denominator": flip["denominator"],
                "orbit_success": orbit["rate"],
                "orbit_ci_low": orbit["ci_low"],
                "orbit_ci_high": orbit["ci_high"],
                "orbit_numerator": orbit["numerator"],
                "orbit_denominator": orbit["denominator"],
            }
        )
    return table


def forgiving_parser_sensitivity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """**Sensitivity analysis only** (§8.5, rule 5). Never the primary number.

    The strict parser is primary and a parse failure counts as wrong. This reports what
    would change under the forgiving parser -- which strips one surrounding code fence and
    nothing else -- so a reader can see the size of that decision instead of taking it on
    trust. It is labelled in the artifact itself, not just here, because a bare number in a
    JSON file is exactly the thing that gets quoted without its label.
    """
    neq = _neq(rows)

    def forgiving(row: dict[str, Any], field: str, strict_field: str) -> bool:
        """The forgiving outcome, defaulting to the strict one where it was not recorded.

        The forgiving parser only strips one surrounding code fence, so it accepts
        everything the strict parser accepts. Treating an absent field as a *failure*
        would therefore invent a recovery deficit that cannot exist, and would make the
        sensitivity analysis look more favourable to strictness than the data warrants.
        """
        value = row.get(field)
        return bool(row[strict_field]) if value is None else bool(value)

    strict_valid = sum(1 for row in rows if row["parse_valid"])
    forgiving_valid = sum(
        1 for row in rows if forgiving(row, "sensitivity_parse_valid", "parse_valid")
    )
    strict_success = sum(1 for row in neq if row["witness_success"] is True)
    forgiving_success = sum(
        1
        for row in neq
        if forgiving(row, "sensitivity_witness_success", "witness_success")
    )
    return {
        "label": "SENSITIVITY ANALYSIS -- not the primary result (rule 5, §8.5)",
        "strict_is_primary": True,
        "parse_valid": {
            "strict": strict_valid,
            "forgiving": forgiving_valid,
            "denominator": len(rows),
            "additional_cells_recovered": forgiving_valid - strict_valid,
        },
        "end_to_end_witness_success": {
            "strict": strict_success,
            "forgiving": forgiving_success,
            "denominator": len(neq),
            "strict_rate": strict_success / len(neq) if neq else None,
            "forgiving_rate": forgiving_success / len(neq) if neq else None,
        },
    }


def operational(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """§10.7's appendix: tokens, latency and cost, by model and overall.

    Cost comes from the provider's own reported ``usage.cost`` per response. Where a field
    is absent it is omitted from its distribution rather than counted as zero -- an unknown
    price is not a free one (D4).
    """

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        usage = [row.get("usage") or {} for row in subset]
        reasoning = [
            (item.get("completion_tokens_details") or {}).get("reasoning_tokens")
            for item in usage
        ]
        return {
            "cells": len(subset),
            "prompt_tokens": _summary(
                [u["prompt_tokens"] for u in usage if u.get("prompt_tokens") is not None]
            ),
            "completion_tokens": _summary(
                [u["completion_tokens"] for u in usage if u.get("completion_tokens") is not None]
            ),
            "reasoning_tokens": _summary([v for v in reasoning if v is not None]),
            "cost_usd": _summary([u["cost"] for u in usage if u.get("cost") is not None]),
            "latency_ms": _summary(
                [row["latency_ms"] for row in subset if row.get("latency_ms") is not None]
            ),
            "truncated": sum(1 for row in subset if row.get("truncated")),
            # Truncation and an empty answer are different facts that usually coincide. The
            # cross-tab says exactly how: a cell can hit the cap after starting a malformed
            # answer, and an empty answer need not mean the cap was reached.
            "truncated_by_parse_outcome": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        row.get("parse_outcome") for row in subset if row.get("truncated")
                    ).items(),
                    key=str,
                )
            },
            "empty_without_truncation": sum(
                1
                for row in subset
                if row.get("parse_outcome") == "empty" and not row.get("truncated")
            ),
            "reasoning_present": sum(1 for row in subset if row.get("reasoning_present")),
        }

    return {
        "note": (
            "Primary cells only (sample_index=0); the repeatability audit's calls are not "
            "included. Unknown fields are omitted from a distribution, never recorded as 0."
        ),
        "overall": block(rows),
        "by_model": {
            model: block([row for row in rows if row["model_id"] == model])
            for model in sorted({row["model_id"] for row in rows})
        },
        "by_prompt": {
            prompt: block([row for row in rows if row["prompt_condition"] == prompt])
            for prompt in ("P0", "P1")
        },
    }


# --------------------------------------------------------------------------- assembly


def build_descriptives(
    rows: Sequence[dict[str, Any]], raw_text: dict[str, str] | None = None
) -> dict[str, Any]:
    """Every descriptive breakdown, from the primary rows only."""
    primary = primary_rows(rows)
    return {
        "note": (
            "DESCRIPTIVE AND EXPLORATORY ONLY. Recorded post-freeze deviation D15. No "
            "confirmatory estimate is recomputed here: analysis/make_results.py and "
            "analysis/preregistered_secondaries.py remain the frozen analyses, and every "
            "rate below is a frozen metric function applied to a subset of the same "
            "primary rows. configs/main_frozen.yaml:analysis.secondary_test_family already "
            "declares these breakdowns descriptive, so they carry no confirmatory claim "
            "regardless of what they show (rule 11)."
        ),
        "interval_method": (
            f"95% pair-clustered percentile bootstrap, {RESAMPLES} resamples, seed {SEED} "
            "-- the settings analysis/figures.py uses, so these tables and the figures "
            "report the same intervals. Confirmatory intervals use the frozen 10,000."
        ),
        "primary_cells": len(primary),
        "cell_table": cell_table(primary),
        "depth_bins": depth_bin_table(primary),
        "families": family_table(primary),
        "representation": representation_table(primary),
        "error_taxonomy": error_taxonomy(primary),
        "witness_lengths": witness_lengths(primary),
        "trace_errors": trace_errors(primary),
        "qualitative_examples": qualitative_examples(primary, raw_text),
        "forgiving_parser_sensitivity": forgiving_parser_sensitivity(primary),
        "operational": operational(primary),
    }


def _csv(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """A minimal CSV writer, matching make_results so both files quote alike."""

    def cell(value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        return f'"{text}"' if ("," in text or '"' in text) else text

    lines = [",".join(columns)]
    lines += [",".join(cell(row.get(column)) for column in columns) for row in rows]
    return "\n".join(lines) + "\n"


def _raw_path(config: Any, override: str | None) -> Path | None:
    """The raw response log, from ``--raw`` or the frozen ``execution.raw_log``."""
    if override:
        return Path(override)
    try:
        return Path(str(config.require("execution.raw_log")))
    except Exception:
        return None


def _final_answers(path: Path | None) -> dict[str, str] | None:
    """Map ``run_id`` to the graded final-answer text, read from the raw log.

    Only terminal attempts are kept, and only the channel the grader reads
    (``message.content``) -- never the separate reasoning field, which is retained in the
    log but is not an answer and is never graded. Returns ``None`` if the log is absent, so
    examples degrade to ids rather than failing the whole run.
    """
    if path is None or not path.is_file():
        return None
    answers: dict[str, str] = {}
    for record in read_jsonl(path):
        body = record.get("raw_body")
        if not isinstance(body, dict):
            continue
        choices = body.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("message") or {}).get("content")
        if content is not None:
            answers[record["run_id"]] = content
    return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.descriptives",
        description="Descriptive breakdowns for the manuscript (D15). Confirmatory "
        "results come from analysis.make_results and are not touched here.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--graded", help="graded JSONL; defaults to <parsed_dir>/graded.jsonl")
    parser.add_argument(
        "--raw",
        help="raw response JSONL, for verbatim example outputs (§11.4); "
        "defaults to execution.raw_log. Omit with --no-raw to build examples without text.",
    )
    parser.add_argument(
        "--no-raw", action="store_true", help="skip the raw log; examples carry ids only"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        parsed_dir = config.path_for("parsed_dir")
        graded_path = Path(args.graded) if args.graded else parsed_dir / "graded.jsonl"
        if not graded_path.is_file():
            print(f"{graded_path} not found.", file=sys.stderr)
            return 1
        rows = list(read_jsonl(graded_path))
        raw_text = None if args.no_raw else _final_answers(_raw_path(config, args.raw))
        # The same structural validation make_results runs. A duplicated run_id or a
        # second dataset_id would corrupt these breakdowns exactly as it would corrupt the
        # primary rates, and a descriptive table is no less wrong for being descriptive.
        validate_rows(primary_rows(rows))
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1

    report = build_descriptives(rows, raw_text)
    tables_dir = config.path_for("tables_dir")

    write_text(tables_dir / "descriptives.json", canonical_json(report) + "\n")
    write_text(
        tables_dir / "table_cells.csv",
        _csv(
            report["cell_table"],
            [
                "model_id",
                "prompt_condition",
                "rendering",
                "cells",
                "balanced_decision_accuracy",
                "decision_accuracy_equivalent",
                "decision_accuracy_non_equivalent",
                "end_to_end_witness_success",
                "end_to_end_witness_success_n",
                "witness_denominator",
                "shortest_witness_success",
                "parse_failure_rate",
                "parse_failures",
            ],
        ),
    )
    write_text(
        tables_dir / "table_depth_bins.csv",
        _csv(
            report["depth_bins"],
            [
                "model_id",
                "prompt_condition",
                "depth_bin",
                "numerator",
                "denominator",
                "rate",
                "ci_low",
                "ci_high",
            ],
        ),
    )
    write_text(
        tables_dir / "table_families.csv",
        _csv(
            report["families"],
            ["family", "model_id", "numerator", "denominator", "rate", "ci_low", "ci_high"],
        ),
    )
    write_text(
        tables_dir / "table_representation.csv",
        _csv(
            report["representation"],
            [
                "model_id",
                "flips",
                "flip_denominator",
                "flip_rate",
                "flip_ci_low",
                "flip_ci_high",
                "orbit_numerator",
                "orbit_denominator",
                "orbit_success",
                "orbit_ci_low",
                "orbit_ci_high",
            ],
        ),
    )
    taxonomy_rows = [
        {"scope": scope, "cells": block["cells"], "truncated": block["truncated"],
         **block["categories"]}
        for scope, block in (
            [("ALL", report["error_taxonomy"]["overall"])]
            + sorted(report["error_taxonomy"]["by_model"].items())
            + sorted(report["error_taxonomy"]["by_prompt"].items())
        )
    ]
    write_text(
        tables_dir / "table_error_taxonomy.csv",
        _csv(taxonomy_rows, ["scope", "cells", *ERROR_CATEGORIES, "truncated"]),
    )

    graded_digest = sha256_of_file(graded_path)
    write_text(
        tables_dir / "descriptives_checksums.json",
        canonical_json(
            {
                "graded_input": str(graded_path),
                "graded_sha256": graded_digest,
                "outputs": {
                    str(tables_dir / name): sha256_of_file(tables_dir / name)
                    for name in (
                        "descriptives.json",
                        "table_cells.csv",
                        "table_depth_bins.csv",
                        "table_families.csv",
                        "table_representation.csv",
                        "table_error_taxonomy.csv",
                    )
                },
            }
        )
        + "\n",
    )

    print(f"graded input : {graded_path}  ({graded_digest})")
    print(f"primary cells: {report['primary_cells']}")
    print(f"tables       : {tables_dir}")
    for name in (
        "descriptives.json",
        "table_cells.csv",
        "table_depth_bins.csv",
        "table_families.csv",
        "table_representation.csv",
        "table_error_taxonomy.csv",
        "descriptives_checksums.json",
    ):
        print(f"  wrote {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
