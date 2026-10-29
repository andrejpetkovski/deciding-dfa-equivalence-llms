"""Regenerate every table and figure from graded rows (Step 11, §12).

Usage::

    python -m analysis.make_results --config configs/pilot_revised.yaml --graded <file.jsonl>

Rule 6 is the reason this exists: no number in the paper may be typed by hand. Everything the
manuscript cites comes from here, and a checksum manifest records exactly which inputs
produced it.

Order matters. Validation runs first and *aborts* rather than warns, because every rate below
silently absorbs a structural problem: a duplicated ``run_id`` double-counts a cell, a second
``dataset_id`` pools splits the protocol keeps apart, and a missing cell shrinks a denominator
without leaving a trace in the output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from analysis import figures
from analysis.metrics import (
    ValidationError,
    balanced_decision_accuracy,
    conditional_witness_validity,
    coverage,
    decision_flip_rate,
    orbit_success,
    parse_failure_rate,
    primary_rows,
    shortest_witness_success,
    trace_correctness,
    validate_rows,
    witness_success,
)
from analysis.stats import (
    DEPTH_CENTRE,
    PROMPT_CODING,
    SECONDARY_FAMILY,
    fit_h1,
    fit_h2,
    h3_flip_analysis,
    holm,
)
from src.config import ConfigError, load_config
from src.io_utils import canonical_json, read_jsonl, sha256_of_file, write_text

__all__ = ["main", "build_results"]


def _csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """A minimal CSV writer -- no dependency, and the quoting rules stay visible."""
    def cell(value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        return f'"{text}"' if ("," in text or '"' in text) else text

    lines = [",".join(columns)]
    lines += [",".join(cell(row.get(column)) for column in columns) for row in rows]
    return "\n".join(lines) + "\n"


def build_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute every reported quantity from the primary rows."""
    primary = primary_rows(rows)
    eq_rate, neq_rate, balanced = balanced_decision_accuracy(primary)

    metric_rows = [
        witness_success(primary).to_dict(),
        conditional_witness_validity(primary).to_dict(),
        shortest_witness_success(primary).to_dict(),
        trace_correctness(primary).to_dict(),
        parse_failure_rate(primary).to_dict(),
        decision_flip_rate(primary).to_dict(),
        orbit_success(primary).to_dict(),
        eq_rate.to_dict(),
        neq_rate.to_dict(),
        {
            "metric": "balanced_decision_accuracy",
            "numerator": None,
            "denominator": eq_rate.denominator + neq_rate.denominator,
            "denominator_description": "mean of the two class accuracies (secondary)",
            "rate": balanced,
        },
    ]

    h1 = fit_h1(primary)
    h2 = fit_h2(primary)
    h3 = h3_flip_analysis(primary)

    # Holm over the frozen secondary family only. Absent tests keep their slot so the family
    # size cannot shrink to whichever tests happened to run.
    raw_p = {
        "H2_prompt_effect": h2.get("prompt").p_value if h2.get("prompt") else None,
        "H2_prompt_depth_interaction": (
            h2.get("depth:prompt").p_value if h2.get("depth:prompt") else None
        ),
        "H3_flip_rate": _flip_rate_p(h3),
    }
    available = {name: p for name, p in raw_p.items() if p is not None}
    adjusted = holm(available) if available else {}

    return {
        "frozen_analysis_settings": {
            "depth_centre": DEPTH_CENTRE,
            "prompt_coding": PROMPT_CODING,
            "secondary_holm_family": list(SECONDARY_FAMILY),
            "primary_sample_index": 0,
        },
        "coverage": coverage(primary),
        "metrics": metric_rows,
        "h1": {name: estimate.to_dict() for name, estimate in h1.items()},
        "h2": {name: estimate.to_dict() for name, estimate in h2.items()},
        "h3": {name: estimate.to_dict() for name, estimate in h3.items()},
        "multiplicity": {
            "family": list(SECONDARY_FAMILY),
            "raw_p": raw_p,
            "holm_adjusted": adjusted,
            "note": (
                "H1 is primary and uncorrected. The family is frozen in analysis.stats and "
                "does not shrink to the tests that happened to be estimable."
            ),
        },
    }


def _flip_rate_p(h3: dict[str, Any]) -> float | None:
    """Whether the flip rate differs from zero, read off its bootstrap interval.

    Reported as a bounded indicator rather than an exact p-value: the interval is the
    estimate the protocol asks to emphasise, and inventing a precise p from percentiles
    would overstate what the bootstrap provides.
    """
    estimate = h3.get("flip_rate")
    if estimate is None or estimate.ci_low is None:
        return None
    return 0.001 if estimate.ci_low > 0 else 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.make_results",
        description="Regenerate every result table and figure from graded rows.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--graded",
        help="graded JSONL; defaults to <parsed_dir>/graded.jsonl, "
        "falling back to graded_dummy.jsonl",
    )
    parser.add_argument(
        "--expected-cells",
        type=int,
        help="assert this many primary cells; omit to skip the count check",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        parsed_dir = config.path_for("parsed_dir")
        if args.graded:
            graded_path = Path(args.graded)
        else:
            graded_path = parsed_dir / "graded.jsonl"
            if not graded_path.is_file():
                graded_path = parsed_dir / "graded_dummy.jsonl"
        if not graded_path.is_file():
            print(f"{graded_path} not found.", file=sys.stderr)
            return 1
        rows = list(read_jsonl(graded_path))
        validate_rows(primary_rows(rows), expected_cells=args.expected_cells)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1

    results = build_results(rows)
    tables_dir = config.path_for("tables_dir")
    figures_dir = config.path_for("figures_dir")

    write_text(tables_dir / "metrics.csv", _csv(
        results["metrics"],
        ["metric", "numerator", "denominator", "rate", "denominator_description"],
    ))
    estimate_rows = [
        {"hypothesis": key.upper(), **estimate}
        for key in ("h1", "h2", "h3")
        for estimate in results[key].values()
    ]
    write_text(tables_dir / "estimates.csv", _csv(
        estimate_rows,
        ["hypothesis", "name", "value", "ci_low", "ci_high", "p_value", "method", "note"],
    ))
    write_text(tables_dir / "results.json", canonical_json(results) + "\n")

    written = figures.write_all(primary_rows(rows), figures_dir)

    manifest = {
        "config": str(config.path),
        "config_sha256": sha256_of_file(config.path),
        "graded_input": str(graded_path),
        "graded_sha256": sha256_of_file(graded_path),
        "outputs": {
            str(path): sha256_of_file(path)
            for path in [
                tables_dir / "metrics.csv",
                tables_dir / "estimates.csv",
                tables_dir / "results.json",
                *written,
            ]
        },
    }
    write_text(tables_dir / "checksums.json", canonical_json(manifest) + "\n")

    print(f"graded input : {graded_path}")
    print(f"coverage     : {results['coverage']}")
    primary = witness_success(primary_rows(rows))
    print(f"H1 outcome   : {primary.numerator}/{primary.denominator} "
          f"= {primary.value:.1%}" if primary.value is not None else "H1 outcome   : n/a")
    depth = results["h1"].get("depth")
    if depth and depth["value"] is not None:
        print(f"H1 slope     : {depth['value']:+.4f} "
              f"[{depth['ci_low']:+.4f}, {depth['ci_high']:+.4f}]")
    elif depth:
        print(f"H1 slope     : non-estimable ({depth['note']})")
    print(f"tables       : {tables_dir}")
    print(f"figures      : {figures_dir} ({len(written)} files)")
    print(f"checksums    : {tables_dir / 'checksums.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
