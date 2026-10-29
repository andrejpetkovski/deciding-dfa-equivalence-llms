"""Interval widths for the planned clustered design, at candidate sample sizes.

Usage::

    python -m scripts.design_precision

§10.6 asks for exactly this and forbids the alternative: *"Do not justify N=200 only because
it is round... simulate the planned clustered design at candidate sample sizes"*, and
`docs/experiment_revision.md` adds *"do not promise 200 pairs gives a particular power; report
interval widths"*. This script produces the interval widths that justification rests on, so
the evidence lives in the repository rather than in a conversation.

What it models. Under D3, H1's primary outcome is end-to-end valid-witness success on
ground-truth ``NEQ_DEPTH`` pairs, so only those pairs enter it -- the equivalent families
serve balanced accuracy and H3. Each pair yields 12 correlated cells (3 models × 2 prompts ×
2 representations) and `pair_id` is the unit of analysis (§5.1), so the pair random intercept,
not the call count, governs precision.

Two views are reported, because they answer different questions:

- the **confirmatory** depth slope, which is what H1's decision rule tests; and
- **per-bin** rates pooled and split by model × prompt, which is what §10.7's Figure 1 draws.

The second is much wider than the first. That is a property of the design, not a defect: the
per-cell lines are diagnostics, and reporting them with visible intervals is the requirement.

These are interval widths under an assumed effect and an assumed intra-pair correlation.
They are not a power guarantee, and a floor or ceiling effect would make sample size the
wrong lever entirely (§4.2) -- the pilot, not this script, settles that.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

__all__ = ["bin_halfwidth", "slope_halfwidth"]

#: Midpoints of the four depth strata 1-2, 3-4, 5-8, 9-12.
BIN_MIDPOINTS = (1.5, 3.5, 6.5, 10.5)

#: 3 models × 2 prompts × 2 representations.
CELLS_PER_PAIR = 12

#: Pair random-intercept sd on the log-odds scale. A moderate intra-pair correlation; the
#: true value is unknown until the pilot, which is why the pilot is where this gets revisited.
PAIR_SD = 1.0


def slope_halfwidth(
    n_neq_pairs: int,
    slope: float,
    intercept: float,
    pair_sd: float = PAIR_SD,
    replicates: int = 60,
    seed: int = 20260908,
) -> float:
    """Mean 95% CI half-width for the depth slope, in log-odds per unit depth.

    Fits a GEE with an exchangeable working correlation grouped on ``pair_id``, which is the
    pair-clustered estimator §10.2 specifies.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    rng = np.random.default_rng(seed)
    per_bin = n_neq_pairs // len(BIN_MIDPOINTS)
    centre = float(np.mean(BIN_MIDPOINTS))
    widths: list[float] = []

    for _ in range(replicates):
        depth: list[float] = []
        pair: list[int] = []
        outcome: list[bool] = []
        pair_id = 0
        for midpoint in BIN_MIDPOINTS:
            for _ in range(per_bin):
                shift = rng.normal(0, pair_sd)
                p = 1 / (1 + np.exp(-(intercept + slope * (midpoint - centre) + shift)))
                outcome.extend(rng.random(CELLS_PER_PAIR) < p)
                depth.extend([midpoint - centre] * CELLS_PER_PAIR)
                pair.extend([pair_id] * CELLS_PER_PAIR)
                pair_id += 1
        design = sm.add_constant(np.array(depth))
        try:
            fit = sm.GEE(
                np.array(outcome, dtype=float),
                design,
                groups=np.array(pair),
                family=Binomial(),
                cov_struct=Exchangeable(),
            ).fit()
            low, high = fit.conf_int()[1]
            widths.append((high - low) / 2)
        except Exception:  # noqa: BLE001 - a non-converged replicate is dropped, not fatal
            continue
    return float(np.mean(widths)) if widths else float("nan")


def bin_halfwidth(
    pairs_per_bin: int,
    cells_per_pair: int,
    rate: float = 0.5,
    pair_sd: float = PAIR_SD,
    simulations: int = 400,
    bootstraps: int = 400,
    seed: int = 7,
) -> float:
    """Mean 95% cluster-bootstrap CI half-width for one bin's success rate.

    Resamples *pairs*, not calls (§10.5): the 12 cells of a pair are one unit, and bootstrapping
    calls would understate the interval by treating correlated observations as independent.
    """
    rng = np.random.default_rng(seed)
    widths: list[float] = []
    for _ in range(simulations):
        shift = rng.normal(0, pair_sd, pairs_per_bin)
        probabilities = 1 / (1 + np.exp(-(np.log(rate / (1 - rate)) + shift)))
        draws = rng.random((pairs_per_bin, cells_per_pair)) < probabilities[:, None]
        per_pair = draws.mean(axis=1)
        index = rng.integers(0, pairs_per_bin, (bootstraps, pairs_per_bin))
        distribution = per_pair[index].mean(axis=1)
        widths.append(
            (np.percentile(distribution, 97.5) - np.percentile(distribution, 2.5)) / 2
        )
    return float(np.mean(widths))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.design_precision",
        description="Interval widths for the planned clustered design.",
    )
    parser.add_argument(
        "--shallow", type=float, default=0.75, help="assumed success in the shallowest bin"
    )
    parser.add_argument(
        "--deep", type=float, default=0.35, help="assumed success in the deepest bin"
    )
    parser.add_argument("--replicates", type=int, default=60)
    args = parser.parse_args(argv)

    logit = lambda p: float(np.log(p / (1 - p)))  # noqa: E731
    span = BIN_MIDPOINTS[-1] - BIN_MIDPOINTS[0]
    slope = (logit(args.deep) - logit(args.shallow)) / span
    intercept = (logit(args.deep) + logit(args.shallow)) / 2

    print(
        f"Assumed effect: {args.shallow:.0%} success in bin 1-2 falling to {args.deep:.0%} "
        f"in bin 9-12,\n                a slope of {slope:.3f} log-odds per unit depth.\n"
    )

    print("H1 confirmatory depth slope (GEE, pair-clustered)")
    print(f"  {'NEQ pairs':>10} {'main calls':>11} {'slope CI half-width':>21} {'vs effect':>10}")
    for n_neq in (100, 140, 200, 300):
        calls = (n_neq + 100) * 2 * 2 * 3  # equivalent families held at 100
        half = slope_halfwidth(n_neq, slope, intercept, replicates=args.replicates)
        print(f"  {n_neq:>10} {calls:>11} {half:>21.4f} {abs(slope) / half:>9.1f}x")

    print("\nPer-bin success rate (cluster bootstrap on pair_id)")
    print(f"  {'view':<40} {'pairs/bin':>10} {'CI half-width':>14}")
    for n_neq in (100, 200):
        per_bin = n_neq // 4
        print(f"  --- {n_neq} NEQ pairs ---")
        for label, cells in (
            ("pooled over models x prompts x reps", CELLS_PER_PAIR),
            ("one model x prompt (a Figure 1 line)", 2),
            ("one model x prompt x representation", 1),
        ):
            print(
                f"  {'  ' + label:<40} {per_bin:>10} "
                f"{bin_halfwidth(per_bin, cells):>13.1%}"
            )

    print(
        "\nInterval widths under an assumed effect and intra-pair correlation, not a power\n"
        "guarantee. A floor or ceiling effect would make sample size the wrong lever (§4.2);\n"
        "the pilot settles that."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
