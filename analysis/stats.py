"""Confirmatory models for H1, H2 and H3 (§10.2–10.6, D3).

M9 is where the analysis is *frozen*, so the choices that could otherwise be made after
seeing results are constants at the top of this module rather than arguments:

- :data:`DEPTH_CENTRE` — depth is centred at 5.5, the mean of the four bin midpoints. A
  design constant, not the observed mean, so the centring cannot shift with the data. With
  P0/P1 coded −0.5/+0.5 this makes ``β_prompt`` the prompt effect *at mean depth*, which is
  H2's frozen reference depth.
- :data:`PROMPT_CODING` — −0.5/+0.5 rather than 0/1, so ``β_depth`` is the **average
  conditional** depth slope across the two prompt conditions rather than the slope under
  whichever condition happened to be the reference (D3).
- :data:`SECONDARY_FAMILY` — the exact set of tests Holm corrects over. Fixed in advance
  because choosing the family after seeing which effects are large is how multiplicity
  corrections get quietly defeated.
- :data:`SEPARATION_LOG_ODDS` — the magnitude past which a coefficient is treated as
  non-estimable rather than reported. Perfect separation is plausible here: if no model ever
  produces a valid witness at depth 9–12, the likelihood has no finite maximum and a fitter
  will happily return a huge estimate with a huge standard error.

Everything clusters on ``pair_id`` (§5.1): a pair's 12 cells are one unit, and bootstrapping
calls instead of pairs would treat correlated observations as independent and understate
every interval.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

__all__ = [
    "DEPTH_CENTRE",
    "PROMPT_CODING",
    "SECONDARY_FAMILY",
    "SEPARATION_LOG_ODDS",
    "Estimate",
    "cluster_bootstrap_rate",
    "fit_h1",
    "fit_h2",
    "holm",
    "h3_flip_analysis",
    "mcnemar_exact",
]

#: Mean of the bin midpoints 1.5, 3.5, 6.5, 10.5. A design constant, deliberately not the
#: observed mean, so the reference depth cannot move with the data.
DEPTH_CENTRE = 5.5

#: P0 -> -0.5, P1 -> +0.5.
PROMPT_CODING = {"P0": -0.5, "P1": 0.5}

#: The frozen Holm family for the secondary confirmatory tests. H1 is primary and is not
#: corrected; these three are.
SECONDARY_FAMILY = (
    "H2_prompt_effect",
    "H2_prompt_depth_interaction",
    "H3_flip_rate",
)

#: |coefficient| beyond this is reported as non-estimable rather than as an effect.
SEPARATION_LOG_ODDS = 15.0

#: Bootstrap resamples for pair-clustered intervals.
BOOTSTRAP_RESAMPLES = 10000


@dataclass(frozen=True, slots=True)
class Estimate:
    """One coefficient or rate, with its interval and how it was obtained."""

    name: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    method: str
    p_value: float | None = None
    note: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def estimable(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "method": self.method,
            "note": self.note,
            **self.detail,
        }


# --------------------------------------------------------------------------- design


def _design(
    rows: Sequence[dict[str, Any]], with_interaction: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the model matrix, outcome and cluster labels.

    Model identity enters as fixed effects (§10.2): only a few deliberately chosen models are
    observed, so they are not a sample from a population of models.
    """
    models = sorted({row["model_id"] for row in rows})
    renderings = sorted({row["rendering"] for row in rows})
    names = ["const", "depth", "prompt"]
    columns: list[list[float]] = [[], [], []]

    # Reference levels are the first alphabetically, fixed so the coding cannot drift.
    for rendering in renderings[1:]:
        names.append(f"rendering[{rendering}]")
        columns.append([])
    for model in models[1:]:
        names.append(f"model[{model}]")
        columns.append([])
    if with_interaction:
        names.append("depth:prompt")
        columns.append([])

    outcome: list[float] = []
    clusters: list[str] = []
    for row in rows:
        depth = float(row["oracle_depth"]) - DEPTH_CENTRE
        prompt = PROMPT_CODING[row["prompt_condition"]]
        values = [1.0, depth, prompt]
        values += [1.0 if row["rendering"] == r else 0.0 for r in renderings[1:]]
        values += [1.0 if row["model_id"] == m else 0.0 for m in models[1:]]
        if with_interaction:
            values.append(depth * prompt)
        for column, value in zip(columns, values, strict=True):
            column.append(value)
        outcome.append(1.0 if row["witness_success"] else 0.0)
        clusters.append(row["pair_id"])

    return (
        np.array(columns, dtype=float).T,
        np.array(outcome, dtype=float),
        np.array(clusters),
        names,
    )


def _fit_gee(
    design: np.ndarray, outcome: np.ndarray, clusters: np.ndarray, names: list[str]
) -> dict[str, Estimate]:
    """Fit a pair-clustered GEE, marking non-estimable coefficients rather than reporting them."""
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    # Degenerate outcome: no finite maximum exists, and a fitter that "succeeds" here is
    # reporting an artefact of its stopping rule.
    if outcome.min() == outcome.max():
        note = (
            f"outcome is constant ({outcome.min():.0f}) across all "
            f"{len(outcome)} cells; no coefficient is estimable"
        )
        return {name: Estimate(name, None, None, None, "gee", note=note) for name in names}

    try:
        fit = sm.GEE(
            outcome,
            design,
            groups=clusters,
            family=Binomial(),
            cov_struct=Exchangeable(),
        ).fit()
    except Exception as error:  # noqa: BLE001 - a failed fit is a reported outcome
        return {
            name: Estimate(name, None, None, None, "gee", note=f"fit failed: {error}")
            for name in names
        }

    intervals = fit.conf_int()
    results: dict[str, Estimate] = {}
    for index, name in enumerate(names):
        value = float(fit.params[index])
        if abs(value) > SEPARATION_LOG_ODDS:
            results[name] = Estimate(
                name,
                None,
                None,
                None,
                "gee",
                note=(
                    f"|estimate| = {abs(value):.1f} exceeds the frozen separation threshold "
                    f"of {SEPARATION_LOG_ODDS}; treated as non-estimable"
                ),
            )
            continue
        results[name] = Estimate(
            name,
            value,
            float(intervals[index][0]),
            float(intervals[index][1]),
            "gee_exchangeable_pair_clustered",
            p_value=float(fit.pvalues[index]),
        )
    return results


# --------------------------------------------------------------------------- hypotheses


def fit_h1(rows: Sequence[dict[str, Any]]) -> dict[str, Estimate]:
    """H1 (primary): the average conditional depth slope on witness success.

    Restricted to ground-truth non-equivalent cells at sample 0. The confirmatory parameter
    is ``depth``; under the frozen −0.5/+0.5 prompt coding it is the slope averaged across
    P0 and P1, not the slope under one of them.
    """
    eligible = [
        row
        for row in rows
        if not row["oracle_equivalent"] and row["oracle_depth"] is not None
    ]
    if not eligible:
        return {}
    design, outcome, clusters, names = _design(eligible, with_interaction=True)
    return _fit_gee(design, outcome, clusters, names)


def fit_h2(rows: Sequence[dict[str, Any]]) -> dict[str, Estimate]:
    """H2: the paired P1−P0 effect on the same witness-success outcome, plus its depth
    interaction.

    Shares H1's model rather than fitting a separate one, so ``prompt`` and ``depth:prompt``
    are read off the same frozen specification. Reported as an absolute percentage-point
    difference too, because odds ratios are hard to interpret (§10.3).
    """
    fitted = fit_h1(rows)
    eligible = [row for row in rows if not row["oracle_equivalent"]]
    by_prompt = {}
    for condition in ("P0", "P1"):
        subset = [row for row in eligible if row["prompt_condition"] == condition]
        by_prompt[condition] = (
            sum(1 for row in subset if row["witness_success"] is True) / len(subset)
            if subset
            else None
        )
    if by_prompt["P0"] is not None and by_prompt["P1"] is not None:
        difference = by_prompt["P1"] - by_prompt["P0"]
        low, high = cluster_bootstrap_difference(eligible)
        fitted["prompt_percentage_points"] = Estimate(
            "prompt_percentage_points",
            difference,
            low,
            high,
            "pair_cluster_bootstrap",
            detail={"P0_rate": by_prompt["P0"], "P1_rate": by_prompt["P1"]},
        )
    return fitted


def h3_flip_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Estimate]:
    """H3: the R0/R1 decision flip rate, its coverage, and per-cell McNemar tests."""
    from analysis.metrics import decision_flip_rate, orbit_success

    flip = decision_flip_rate(rows)
    orbit = orbit_success(rows)

    results: dict[str, Estimate] = {}
    if flip.denominator:
        low, high = cluster_bootstrap_rate(
            rows, lambda subset: decision_flip_rate(subset).value
        )
        results["flip_rate"] = Estimate(
            "flip_rate",
            flip.value,
            low,
            high,
            "pair_cluster_bootstrap",
            detail={
                "numerator": flip.numerator,
                "denominator": flip.denominator,
                "denominator_description": flip.denominator_description,
            },
        )
    if orbit.denominator:
        low, high = cluster_bootstrap_rate(rows, lambda subset: orbit_success(subset).value)
        results["orbit_success"] = Estimate(
            "orbit_success",
            orbit.value,
            low,
            high,
            "pair_cluster_bootstrap",
            detail={
                "numerator": orbit.numerator,
                "denominator": orbit.denominator,
            },
        )
    return results


# --------------------------------------------------------------------------- resampling


def _group_by_pair(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows into per-pair blocks, computed once and reused across resamples."""
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    return list(by_pair.values())


def cluster_bootstrap_rate(
    rows: Sequence[dict[str, Any]],
    statistic,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20260908,
) -> tuple[float | None, float | None]:
    """95% percentile interval for a rate, resampling ``pair_id`` (§10.5).

    Whole pairs are drawn, keeping every representation, prompt and model of a pair together:
    they are one unit of analysis, and resampling calls would treat correlated observations
    as independent and understate the interval.
    """
    blocks = _group_by_pair(rows)
    if not blocks:
        return None, None
    rng = np.random.default_rng(seed)
    n_pairs = len(blocks)
    draws: list[float] = []
    for _ in range(resamples):
        drawn = rng.integers(0, n_pairs, n_pairs)
        resampled = [row for index in drawn for row in blocks[index]]
        value = statistic(resampled)
        if value is not None:
            draws.append(value)
    if not draws:
        return None, None
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cluster_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    resamples: int = 2000,
    seed: int = 20260908,
) -> tuple[float | None, float | None]:
    """95% interval for the P1−P0 witness-success difference, resampling pairs."""

    def difference(subset: Sequence[dict[str, Any]]) -> float | None:
        rates = {}
        for condition in ("P0", "P1"):
            cells = [row for row in subset if row["prompt_condition"] == condition]
            if not cells:
                return None
            rates[condition] = sum(
                1 for row in cells if row["witness_success"] is True
            ) / len(cells)
        return rates["P1"] - rates["P0"]

    return cluster_bootstrap_rate(rows, difference, resamples=resamples, seed=seed)


# --------------------------------------------------------------------------- tests


def mcnemar_exact(both_correct: int, only_r0: int, only_r1: int, neither: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts (§10.4).

    Exact rather than the chi-square approximation: the discordant total is small in several
    model × prompt cells, which is where the approximation is least trustworthy.
    """
    from scipy.stats import binomtest

    discordant = only_r0 + only_r1
    if discordant == 0:
        return 1.0
    return float(binomtest(only_r0, discordant, 0.5, alternative="two-sided").pvalue)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment over the frozen secondary family (§10.5).

    The family is :data:`SECONDARY_FAMILY`, fixed in this module rather than assembled from
    whichever tests were run, so it cannot be narrowed after seeing which effects are large.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p))
        adjusted[name] = running
    return adjusted
