"""Operational repeatability audit over the frozen repeat samples (§10.5, D14).

**What this measures, and what it does not.** The frozen config sends a stratified 20% of
pairs three times under identical requests. `temperature=0` is *accepted but measurably not
effective* on all three deployments, so the models sample. This module quantifies how much
the graded answer moves under that resampling.

It is an **operational repeatability audit, never a determinism check**:

- Agreement here is **not** evidence of determinism. Three identical answers are consistent
  with sampling that happens to concentrate, and prove nothing about the decoder.
- Disagreement is **not** attributable to any single cause. Sampling, server-side batching,
  quantization and routing are all live and none is separately observable from here, so no
  cause is named (§10.5).
- These rows **never enter a primary estimate**. `analysis.metrics.primary_rows` drops every
  `sample_index > 0` before any confirmatory rate is computed; pooling them would weight the
  repeated pairs more heavily and understate every interval.

**Denominators are reported everywhere**, because two honest definitions of "agreement"
differ and the difference matters:

- *unconditional* treats an unparseable answer as its own outcome, so a cell whose three
  samples all failed to parse counts as agreeing. That is a true statement about the
  deployment and it is what rule 5 implies -- a parse failure is an observed outcome, not a
  missing one.
- *given all parsed* restricts to cells where every sample produced a strict-valid answer.
  It is the more intuitive number and the smaller, selected one.

Both are emitted with their coverage. Quoting either alone would be misleading.

Intervals cluster on `pair_id` (rule 9): the repeats of one pair are not independent
observations of anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analysis.metrics import _decision
from analysis.stats import cluster_bootstrap_rate
from src.backends.adapters import extract
from src.config import load_config
from src.grading import strict_parse
from src.io_utils import read_jsonl, write_text
from src.parse_and_grade import _terminal_attempts

__all__ = ["build_cells", "repeatability_report", "main"]

#: Cell key: everything that must be identical for two calls to be a repeat of each other.
_CELL_FIELDS = ("pair_id", "rendering", "prompt_condition", "model_id")


def _cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in _CELL_FIELDS)


def _witness_strings(
    raw_log: Path, alphabet: tuple[str, ...]
) -> dict[str, str | None]:
    """Recover each cell's proposed witness by re-running the **frozen** parser.

    The graded record keeps `witness_length`, not the witness itself, and two different
    witnesses of equal length would be indistinguishable by length alone. Rather than
    approximate, this re-derives the string through the same adapter and the same
    `strict_parse` the grader used -- reuse, not a second implementation, so it cannot drift
    from what was actually scored.
    """
    witnesses: dict[str, str | None] = {}
    for run_id, row in _terminal_attempts(list(read_jsonl(raw_log))).items():
        if not row.get("ok"):
            continue
        answer = extract(row.get("raw_body"), row.get("answer_adapter") or "openai_chat_content_v1")
        condition = str(row.get("prompt_condition") or "")
        if condition not in ("P0", "P1"):
            continue
        parsed = strict_parse(answer.final_text, condition, alphabet)
        witnesses[run_id] = parsed.witness if parsed.ok else None
    return witnesses


def build_cells(
    graded: Sequence[dict[str, Any]],
    expected_samples: int,
    witnesses: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """One row per repeated cell, carrying the per-sample outcomes to be compared.

    Only cells with **every** expected sample are returned. A partly-answered cell cannot
    support an agreement statement, and silently comparing two samples where three were
    scheduled would overstate agreement.
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in graded:
        grouped.setdefault(_cell_key(row), []).append(row)

    cells: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        indices = {int(r["sample_index"]) for r in rows}
        if indices != set(range(expected_samples)):
            continue
        rows = sorted(rows, key=lambda r: int(r["sample_index"]))
        tokens = [
            (r.get("usage") or {}).get("completion_tokens") for r in rows
        ]
        cell = {
            "pair_id": key[0],
            "rendering": key[1],
            "prompt_condition": key[2],
            "model_id": key[3],
            "family": rows[0]["family"],
            "oracle_depth": rows[0]["oracle_depth"],
            "decisions": [_decision(r) for r in rows],
            "parsed": [bool(r["parse_valid"]) for r in rows],
            "witness_success": [r["witness_success"] for r in rows],
            "witness_valid": [r["witness_valid"] for r in rows],
            "completion_tokens": [t for t in tokens if isinstance(t, int)],
        }
        if witnesses is not None:
            cell["witnesses"] = [witnesses.get(str(r["run_id"])) for r in rows]
        cells.append(cell)
    return cells


def _rate(cells: Sequence[dict[str, Any]], predicate, eligible) -> dict[str, Any] | None:
    """A rate over the eligible cells, with a pair-clustered interval and its denominator."""
    subset = [cell for cell in cells if eligible(cell)]
    if not subset:
        return None

    def statistic(rows: Sequence[dict[str, Any]]) -> float | None:
        chosen = [row for row in rows if eligible(row)]
        if not chosen:
            return None
        return sum(1 for row in chosen if predicate(row)) / len(chosen)

    value = statistic(subset)
    low, high = cluster_bootstrap_rate(subset, statistic)
    return {
        "numerator": sum(1 for cell in subset if predicate(cell)),
        "denominator": len(subset),
        "rate": value,
        "ci_low": low,
        "ci_high": high,
    }


def _all_same(values: Sequence[Any]) -> bool:
    return len(set(values)) == 1


def repeatability_report(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every agreement measure, overall and per model, with denominators."""

    def measures(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        all_parsed = lambda c: all(c["parsed"])  # noqa: E731
        neq = lambda c: c["family"] == "NEQ_DEPTH"  # noqa: E731
        all_witnessed = lambda c: all(w is True for w in c["witness_valid"])  # noqa: E731

        token_sets = [c["completion_tokens"] for c in subset if len(c["completion_tokens"]) > 1]
        spreads = [
            (max(t) - min(t)) / statistics.median(t)
            for t in token_sets
            if statistics.median(t) > 0
        ]
        return {
            "cells": len(subset),
            "decision_agreement_unconditional": _rate(
                subset, lambda c: _all_same(c["decisions"]), lambda c: True
            ),
            "decision_agreement_given_all_parsed": _rate(
                subset, lambda c: _all_same(c["decisions"]), all_parsed
            ),
            "all_samples_parsed": _rate(subset, all_parsed, lambda c: True),
            "witness_success_agreement": _rate(
                subset, lambda c: _all_same(c["witness_success"]), neq
            ),
            "identical_witness_string": (
                _rate(
                    subset,
                    lambda c: _all_same(c.get("witnesses") or [None]),
                    lambda c: neq(c) and all_witnessed(c) and "witnesses" in c,
                )
                if any("witnesses" in c for c in subset)
                else None
            ),
            "identical_completion_token_counts": _rate(
                subset,
                lambda c: len(c["completion_tokens"]) > 1
                and _all_same(c["completion_tokens"]),
                lambda c: len(c["completion_tokens"]) > 1,
            ),
            "completion_token_spread": {
                "median_relative_range": statistics.median(spreads) if spreads else None,
                "max_relative_range": max(spreads) if spreads else None,
                "cells": len(spreads),
                "note": "(max - min) / median completion tokens, per cell",
            },
        }

    models = sorted({cell["model_id"] for cell in cells})
    return {
        "note": (
            "Operational repeatability audit, NEVER a determinism check. Agreement is not "
            "evidence of determinism, and disagreement is not attributed to any single "
            "cause: sampling, batching, quantization and routing are all live and none is "
            "separately observable here. These rows never enter a primary estimate."
        ),
        "unit": "one cell = (pair_id, rendering, prompt_condition, model_id), compared across "
        "its repeat samples; only cells with every scheduled sample are included",
        "intervals": "95% percentile bootstrap, resampling pair_id (rule 9)",
        "overall": measures(cells),
        "by_model": {
            model: measures([c for c in cells if c["model_id"] == model]) for model in models
        },
    }


def _fmt(entry: dict[str, Any] | None) -> str:
    if not entry or entry.get("rate") is None:
        return "     n/a"
    ci = ""
    if entry.get("ci_low") is not None:
        ci = f" [{100 * entry['ci_low']:.1f}, {100 * entry['ci_high']:.1f}]"
    return f"{100 * entry['rate']:6.1f}%{ci}  ({entry['numerator']}/{entry['denominator']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operational repeatability audit (§10.5).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--graded", default=None)
    parser.add_argument("--raw", default=None, help="raw log, for witness strings")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    spec = config.get("samples.repeatability") or {}
    expected = int(config.get("samples.main", 1)) + int(spec.get("extra_samples") or 0)
    if expected < 2:
        print("no repeat samples are declared; nothing to audit", file=sys.stderr)
        return 1

    graded_path = Path(args.graded or (Path("data/parsed") / config.config_id / "graded.jsonl"))
    graded = list(read_jsonl(graded_path))

    witnesses = None
    raw_path = Path(args.raw or str(config.require("execution.raw_log")))
    if raw_path.is_file():
        witnesses = _witness_strings(raw_path, config.alphabet)

    cells = build_cells(graded, expected, witnesses)
    if not cells:
        print(
            f"no cell has all {expected} samples in {graded_path}; "
            f"run the audit calls and re-grade first",
            file=sys.stderr,
        )
        return 1

    report = repeatability_report(cells)
    out = Path(args.out or (Path("analysis/tables") / config.config_id / "repeatability.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text(out, json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"graded input : {graded_path}")
    print(f"repeated cells : {len(cells)} with all {expected} samples")
    print()
    labels = (
        ("decision agreement, unconditional", "decision_agreement_unconditional"),
        ("decision agreement, all parsed", "decision_agreement_given_all_parsed"),
        (f"all {expected} samples strict-parsed", "all_samples_parsed"),
        ("witness success agreement (NEQ)", "witness_success_agreement"),
        ("identical witness string", "identical_witness_string"),
        ("identical completion-token counts", "identical_completion_token_counts"),
    )
    rows = [("ALL", report["overall"])] + sorted(report["by_model"].items())
    for name, block in rows:
        print(f"{name}  (n={block['cells']} cells)")
        for label, key in labels:
            print(f"   {label:<36}: {_fmt(block[key])}")
        spread = block["completion_token_spread"]
        if spread["median_relative_range"] is not None:
            print(
                f"   completion-token spread             : "
                f"median {spread['median_relative_range']:.2f}, "
                f"max {spread['max_relative_range']:.2f} (relative range)"
            )
        print()
    print(f"report       : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
