"""The four result figures (§10.7), emitted from graded rows.

Figure 1 is the primary result and the others are secondary or diagnostic. Two conventions
apply throughout and both follow from D3:

- Rates are plotted with **pair-cluster bootstrap intervals**, never bare points. The
  per-model-per-prompt lines in Figure 1 carry roughly ±14 points at the planned sample size
  (``scripts/design_precision.py``), so a point without its interval invites a model-ranking
  claim the design cannot support.
- Denominators are annotated. A conditional rate drawn beside an unconditional one is
  otherwise unreadable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a cluster or in CI
import matplotlib.pyplot as plt  # noqa: E402

from analysis.metrics import (  # noqa: E402
    balanced_decision_accuracy,
    decision_flip_rate,
    orbit_success,
    witness_success,
)
from analysis.stats import cluster_bootstrap_rate  # noqa: E402

__all__ = ["DEPTH_BINS", "write_all"]

#: The four declared strata, as (label, low, high).
DEPTH_BINS = (("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9-12", 9, 12))

_BOOTSTRAP = 2000  # figures only; the reported intervals in tables use the full count


def _bin_of(depth: int | None) -> str | None:
    if depth is None:
        return None
    for label, low, high in DEPTH_BINS:
        if low <= depth <= high:
            return label
    return None


def _rate_with_interval(rows: Sequence[dict[str, Any]], statistic):
    value = statistic(rows)
    if value is None:
        return None, None, None
    low, high = cluster_bootstrap_rate(rows, statistic, resamples=_BOOTSTRAP)
    return value, low, high


def figure_1_witness_success_by_depth(rows, path: Path) -> None:
    """Primary: end-to-end valid-witness success against depth, by model and prompt."""
    neq = [row for row in rows if not row["oracle_equivalent"]]
    models = sorted({row["model_id"] for row in neq})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for axis, prompt in zip(axes, ("P0", "P1"), strict=True):
        for model in models:
            xs, ys, los, his = [], [], [], []
            for index, (_label, low, high) in enumerate(DEPTH_BINS):
                cells = [
                    row
                    for row in neq
                    if row["prompt_condition"] == prompt
                    and row["model_id"] == model
                    and low <= (row["oracle_depth"] or -1) <= high
                ]
                if not cells:
                    continue
                value, lo, hi = _rate_with_interval(
                    cells, lambda subset: witness_success(subset).value
                )
                if value is None:
                    continue
                xs.append(index)
                ys.append(value)
                los.append(value - (lo if lo is not None else value))
                his.append((hi if hi is not None else value) - value)
            if xs:
                axis.errorbar(
                    xs, ys, yerr=[los, his], marker="o", capsize=3, label=model, alpha=0.85
                )
        axis.set_title(f"{prompt}")
        axis.set_xticks(range(len(DEPTH_BINS)))
        axis.set_xticklabels([label for label, _, _ in DEPTH_BINS])
        axis.set_xlabel("shortest distinguishing-word depth")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("end-to-end valid-witness success")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Figure 1 — H1: witness success vs depth\n"
        "denominator: all scheduled non-equivalent cells; bars are pair-cluster bootstrap 95% CIs",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_2_decision_accuracy(rows, path: Path) -> None:
    """Secondary: balanced decision accuracy by model, split by pair family."""
    models = sorted({row["model_id"] for row in rows})
    families = sorted({row["family"] for row in rows})
    fig, axis = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(families), 1)

    for offset, family in enumerate(families):
        heights = []
        for model in models:
            cells = [
                row for row in rows if row["model_id"] == model and row["family"] == family
            ]
            correct = sum(1 for row in cells if row["decision_correct"])
            heights.append(correct / len(cells) if cells else 0.0)
        axis.bar(
            [i + offset * width for i in range(len(models))],
            heights,
            width=width,
            label=family,
        )

    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.text(
        -0.4, 0.52, "constant-answer baseline (50% balanced)", fontsize=7, color="grey"
    )
    axis.set_xticks([i + 0.4 - width / 2 for i in range(len(models))])
    axis.set_xticklabels(models, fontsize=8)
    axis.set_ylabel("decision accuracy")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    _, _, balanced = balanced_decision_accuracy(rows)
    axis.set_title(
        f"Figure 2 — secondary decision accuracy by family "
        f"(overall balanced: {balanced:.1%})" if balanced is not None else "Figure 2",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_3_representation_invariance(rows, path: Path) -> None:
    """H3: R0/R1 flip rate and orbit success by model, with their differing denominators."""
    models = sorted({row["model_id"] for row in rows})
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = range(len(models))

    for offset, (name, statistic) in enumerate(
        (
            ("decision flip rate", lambda subset: decision_flip_rate(subset).value),
            ("orbit success", lambda subset: orbit_success(subset).value),
        )
    ):
        values, errors_low, errors_high = [], [], []
        for model in models:
            cells = [row for row in rows if row["model_id"] == model]
            value, low, high = _rate_with_interval(cells, statistic)
            values.append(value or 0.0)
            errors_low.append((value or 0.0) - (low if low is not None else (value or 0.0)))
            errors_high.append((high if high is not None else (value or 0.0)) - (value or 0.0))
        axis.bar(
            [p + offset * 0.4 for p in positions],
            values,
            width=0.4,
            yerr=[errors_low, errors_high],
            capsize=3,
            label=name,
        )

    axis.set_xticks([p + 0.2 for p in positions])
    axis.set_xticklabels(models, fontsize=8)
    axis.set_ylabel("rate")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    axis.set_title(
        "Figure 3 — H3: representation invariance\n"
        "flip rate is conditional on two strict-valid decisions; orbit success is not",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_4_error_taxonomy(rows, path: Path) -> None:
    """Stacked error taxonomy over all scheduled non-equivalent cells (§11.2)."""
    neq = [row for row in rows if not row["oracle_equivalent"]]
    models = sorted({row["model_id"] for row in neq})
    categories = [
        "success",
        "valid but not shortest",
        "invalid witness",
        "said equivalent",
        "parse failure",
        "transport failure",
    ]

    counts: dict[str, Counter] = {model: Counter() for model in models}
    for row in neq:
        model = row["model_id"]
        if row["status"] == "transport_failed":
            counts[model]["transport failure"] += 1
        elif not row["parse_valid"]:
            counts[model]["parse failure"] += 1
        elif row["witness_success"]:
            counts[model][
                "success" if row["witness_shortest"] else "valid but not shortest"
            ] += 1
        elif row["decision_correct"]:
            counts[model]["invalid witness"] += 1
        else:
            counts[model]["said equivalent"] += 1

    fig, axis = plt.subplots(figsize=(8, 4.5))
    bottoms = [0.0] * len(models)
    for category in categories:
        heights = [
            counts[model][category] / max(sum(counts[model].values()), 1) for model in models
        ]
        axis.bar(models, heights, bottom=bottoms, label=category)
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]

    axis.set_ylabel("share of scheduled non-equivalent cells")
    axis.set_ylim(0, 1.0)
    axis.legend(fontsize=7, ncol=2)
    axis.tick_params(axis="x", labelsize=8)
    axis.set_title(
        "Figure 4 — error taxonomy\n"
        "'valid but not shortest' counts as success on H1/H2 (D3)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_all(rows: Sequence[dict[str, Any]], directory: Path) -> list[Path]:
    """Emit all four figures, returning the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, function in (
        ("figure_1_witness_success_by_depth.png", figure_1_witness_success_by_depth),
        ("figure_2_decision_accuracy.png", figure_2_decision_accuracy),
        ("figure_3_representation_invariance.png", figure_3_representation_invariance),
        ("figure_4_error_taxonomy.png", figure_4_error_taxonomy),
    ):
        path = directory / name
        function(rows, path)
        written.append(path)
    return written
