"""The H2/H3 secondaries exactly as `configs/main_frozen.yaml` declared them (D13).

**Why this file exists.** At the freeze, `analysis.h2_contrast_specification` declared H2's
estimand as a *model-based average marginal effect* from the H1 model including the
`depth x prompt` interaction, explicitly **not** the `prompt` coefficient -- which is the
effect at `d = 0`, a depth the design does not contain. `analysis.secondary_test_family`
declared Holm correction across {H2 AME, H3 flip rate} with raw and adjusted p-values.

`analysis/make_results.py` implements a *different* pre-fixed specification, not a missing
one. It Holm-corrects over `analysis.stats.SECONDARY_FAMILY`, a **three**-member family
{`H2_prompt_effect` (the β₂ coefficient), `H2_prompt_depth_interaction`, `H3_flip_rate`},
and reports H2 as that coefficient plus an unadjusted rate difference on 2,000 resamples.
The frozen config declares a **two**-member family {H2 AME, H3 flip rate} and a model-based
AME. Both predate the main run, so neither was chosen after seeing results -- but only one
is what the config declares, and the two differ in the H2 estimand *and* in family size,
which changes every adjusted p-value. The disagreement surfaced only after the main results
existed.

**This module is a recorded post-freeze deviation** (`docs/freeze_log.md`). It is a separate
file, and `make_results.py` is untouched, so the frozen analysis stays byte-reproducible and
this is visibly *additional* rather than a quiet re-run of the primary result under a new
estimator. H1 is not recomputed here at all.

The specification is transcribed from the frozen config and implemented to the letter. It is
not re-derived, and no choice in it was revisited after seeing the numbers -- the one thing
that would make a pre-registration worthless.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analysis.stats import PROMPT_CODING, _design, _fit_gee, holm, mcnemar_exact
from src.io_utils import read_jsonl, write_text

__all__ = ["h2_average_marginal_effect", "h3_pooled_mcnemar", "main"]

#: Declared at freeze. Not tuned, not lowered when it was slow.
BOOTSTRAP_RESAMPLES = 10_000

#: Declared at freeze: the bootstrap resamples whole `pair_id`s. Seeded so the interval is
#: reproducible; the value is arbitrary and was fixed before the estimator was ever run.
BOOTSTRAP_SEED = 20260910


def _neq_primary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ground-truth non-equivalent cells at `sample_index = 0`, per the declaration."""
    return [
        row
        for row in rows
        if not row["oracle_equivalent"] and int(row.get("sample_index", 0)) == 0
    ]


def _ame_from_fit(rows: Sequence[dict[str, Any]]) -> float | None:
    """Average marginal effect of P1 minus P0, in probability units.

    Fits the frozen H1 specification, then predicts every observation twice -- once with the
    prompt column forced to P0, once to P1 -- and averages the difference. Averaging over the
    observations is what makes this an average over the *realized* depth distribution
    (balanced at 25 pairs per bin by construction) and over the three models, as declared,
    rather than an effect read off at one arbitrary depth.

    The counterfactual matrices are built by mutating the columns of ``_design`` itself
    rather than by re-deriving the linear predictor here. A hand-written copy of the coding
    is exactly how the first version of this went wrong: it used raw depth where the frozen
    design centres it at ``DEPTH_CENTRE``, which shifted the intercept and put the fitted
    mean 13 points below the observed one. Reusing the frozen builder means the coding cannot
    drift away from the model being fitted.
    """
    eligible = [
        row for row in rows if not row["oracle_equivalent"] and row["oracle_depth"] is not None
    ]
    if not eligible:
        return None
    design, _outcome, _clusters, names = _design(eligible, with_interaction=True)
    fitted = _fit_gee(design, _outcome, _clusters, names)
    if not fitted or any(est.value is None for est in fitted.values()):
        return None
    beta = np.array([fitted[name].value for name in names], dtype=float)

    prompt_at = names.index("prompt")
    interaction_at = names.index("depth:prompt") if "depth:prompt" in names else None
    depth_at = names.index("depth")

    def counterfactual(code: float) -> np.ndarray:
        matrix = design.copy()
        matrix[:, prompt_at] = code
        if interaction_at is not None:
            matrix[:, interaction_at] = matrix[:, depth_at] * code
        return 1.0 / (1.0 + np.exp(-(matrix @ beta)))

    return float(np.mean(counterfactual(PROMPT_CODING["P1"]) - counterfactual(PROMPT_CODING["P0"])))


def h2_average_marginal_effect(
    rows: Sequence[dict[str, Any]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    *,
    budget: int | None = None,
    done_draws: Sequence[float] | None = None,
    done_separated: int = 0,
) -> dict[str, Any]:
    """H2 as declared: model-based AME, pair-clustered percentile bootstrap, pp and OR.

    ``budget`` caps how many *new* fits this call performs, returning a partial result that a
    later call resumes from via ``done_draws``/``done_separated``. This exists only because
    background execution of this module hangs on the development machine at import time, so
    the 10,000 fits have to be spread over several foreground invocations. The RNG stream is
    replayed rather than re-seeded, so a chunked run and a single run produce the *same*
    draws -- the estimate does not depend on how the work was scheduled.
    """
    eligible = _neq_primary(rows)
    point = _ame_from_fit(eligible)
    if point is None:
        return {"estimable": False, "reason": "H1 model did not fit on the eligible rows"}

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_pair.setdefault(row["pair_id"], []).append(row)
    pair_ids = sorted(by_pair)

    rng = np.random.default_rng(seed)
    draws: list[float] = list(done_draws or [])
    separated = int(done_separated)
    already = len(draws) + separated
    computed = 0
    for index_of_resample in range(resamples):
        # The draw is ALWAYS taken, even when its result is already known, so the RNG stream
        # stays in lockstep. Drawing costs microseconds and fitting costs ~0.1 s, so replaying
        # is free -- and it makes a chunked run byte-identical to a single one, which a
        # fresh-seed-per-chunk scheme would not be.
        picked = rng.integers(0, len(pair_ids), size=len(pair_ids))
        if index_of_resample < already:
            continue
        if budget is not None and computed >= budget:
            return {
                "estimable": False,
                "partial": True,
                "reason": f"chunk complete: {len(draws)} draws, {separated} separated",
                "draws": draws,
                "separated": separated,
            }
        subset: list[dict[str, Any]] = []
        for index in picked:
            subset.extend(by_pair[pair_ids[int(index)]])
        value = _ame_from_fit(subset)
        computed += 1
        if value is None:
            separated += 1
            continue
        draws.append(value)

    if len(draws) < resamples // 2:
        return {
            "estimable": False,
            "reason": f"only {len(draws)}/{resamples} resamples fit",
        }

    array = np.array(draws)
    low, high = (float(x) for x in np.percentile(array, [2.5, 97.5]))
    # Two-sided percentile bootstrap p-value: twice the smaller tail mass at zero.
    tail = min(float(np.mean(array <= 0.0)), float(np.mean(array >= 0.0)))
    p_value = min(1.0, 2.0 * tail)

    rates = {}
    for condition in ("P0", "P1"):
        cells = [r for r in eligible if r["prompt_condition"] == condition]
        rates[condition] = (
            sum(1 for r in cells if r["witness_success"] is True) / len(cells)
            if cells
            else None
        )
    odds_ratio = None
    if rates["P0"] not in (None, 0.0, 1.0) and rates["P1"] not in (None, 0.0, 1.0):
        odds = lambda p: p / (1.0 - p)  # noqa: E731
        odds_ratio = odds(rates["P1"]) / odds(rates["P0"])

    return {
        "estimable": True,
        "name": "h2_average_marginal_effect",
        "percentage_points": point * 100.0,
        "ci_low_pp": low * 100.0,
        "ci_high_pp": high * 100.0,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "method": (
            f"model-based AME from the frozen H1 specification incl. depth x prompt, "
            f"averaged over {len(eligible)} eligible cells; pair-clustered percentile "
            f"bootstrap, {resamples} resamples, seed {seed}"
        ),
        "resamples_used": len(draws),
        "resamples_dropped_for_separation": separated,
        "observed_rates": rates,
        "n_cells": len(eligible),
        "n_pairs": len(pair_ids),
    }


def h3_pooled_mcnemar(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """H3 as declared: exact McNemar on the pooled R0/R1 discordant decisions.

    Declared as "McNemar within each model x prompt cell and pooled". The pooled test is what
    enters the Holm family; the per-cell tests are descriptive and `make_results.py` already
    emits them.

    **Sample 0 only, enforced here rather than trusted to the caller.** ``_paired_cells`` keys
    cells by (pair, prompt, model) with no sample index, so a repeat passed through it silently
    *overwrites* the primary answer in its cell. The first version of this function omitted
    the restriction. Once the repeatability audit had been graded, the committed artifact
    scored every audited cell on a repeat: 1,057 pairs where the declared population is 1,047,
    and p = 0.6989 where the declared test gives 0.8957. H2's estimator beside it was always
    guarded by ``_neq_primary``; this one is now guarded by the same frozen ``primary_rows``
    every other confirmatory estimate uses, and ``tests/test_preregistered_secondaries.py``
    pins it (corrected 2026-09-11, recorded under D13).
    """
    from analysis.metrics import _decision, _paired_cells, primary_rows

    both = only_r0 = only_r1 = neither = 0
    for cell in _paired_cells(primary_rows(rows)).values():
        r0, r1 = cell["R0"], cell["R1"]
        if _decision(r0) is None or _decision(r1) is None:
            continue
        correct_a = bool(r0["decision_correct"])
        correct_b = bool(r1["decision_correct"])
        if correct_a and correct_b:
            both += 1
        elif correct_a:
            only_r0 += 1
        elif correct_b:
            only_r1 += 1
        else:
            neither += 1

    discordant = only_r0 + only_r1
    return {
        "estimable": discordant > 0,
        "name": "h3_pooled_mcnemar",
        "p_value": mcnemar_exact(both, only_r0, only_r1, neither),
        "both_correct": both,
        "only_r0_correct": only_r0,
        "only_r1_correct": only_r1,
        "neither_correct": neither,
        "discordant": discordant,
        "method": "exact two-sided McNemar on pooled R0/R1 discordant decisions",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="cap new fits this invocation; state is kept in --state so a later call resumes",
    )
    parser.add_argument("--state", default=None, help="partial-draw file for chunked runs")
    args = parser.parse_args(argv)

    rows = list(read_jsonl(Path(args.graded)))

    state_path = Path(args.state) if args.state else None
    done_draws: list[float] = []
    done_separated = 0
    if state_path and state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        done_draws = list(saved.get("draws") or [])
        done_separated = int(saved.get("separated") or 0)

    h2 = h2_average_marginal_effect(
        rows,
        resamples=args.resamples,
        budget=args.budget,
        done_draws=done_draws,
        done_separated=done_separated,
    )
    if h2.get("partial"):
        assert state_path is not None, "--budget requires --state"
        write_text(
            state_path,
            json.dumps({"draws": h2["draws"], "separated": h2["separated"]}) + chr(10),
        )
        total = len(h2["draws"]) + h2["separated"]
        print(f"chunk done: {total}/{args.resamples} resamples complete -> {state_path}")
        return 2
    if state_path and state_path.is_file():
        state_path.unlink()

    h3 = h3_pooled_mcnemar(rows)

    raw = {}
    if h2.get("estimable"):
        raw["h2_average_marginal_effect"] = h2["p_value"]
    if h3.get("estimable"):
        raw["h3_pooled_mcnemar"] = h3["p_value"]
    adjusted = holm(raw) if raw else {}

    report = {
        "note": (
            "Pre-registered H2/H3 secondaries, implemented per configs/main_frozen.yaml. "
            "Recorded post-freeze deviation D13. H1 is NOT recomputed here; "
            "analysis/make_results.py remains the frozen primary analysis."
        ),
        "h2": h2,
        "h3": h3,
        "holm": {
            "family": sorted(raw),
            "raw_p": raw,
            "adjusted_p": adjusted,
            "note": "H1 is primary and reported uncorrected; only these two are adjusted.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text(out, json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"graded input : {args.graded}")
    if h2.get("estimable"):
        print(
            f"H2 AME       : {h2['percentage_points']:+.2f} pp "
            f"[{h2['ci_low_pp']:+.2f}, {h2['ci_high_pp']:+.2f}]  "
            f"p={h2['p_value']:.4g}"
            + (f"  OR={h2['odds_ratio']:.3f}" if h2["odds_ratio"] else "")
        )
        print(f"               observed P0={h2['observed_rates']['P0']:.4f} "
              f"P1={h2['observed_rates']['P1']:.4f}")
    if h3.get("estimable"):
        print(
            f"H3 McNemar   : p={h3['p_value']:.4g}  "
            f"discordant {h3['only_r0_correct']}/{h3['only_r1_correct']} "
            f"(R0-only / R1-only)"
        )
    for name in sorted(adjusted):
        print(f"  Holm       : {name:<32} raw={raw[name]:.4g}  adjusted={adjusted[name]:.4g}")
    print(f"report       : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
