"""Post-review output-cap diagnostics and selected-sample H1/H2 sensitivity.

Reads existing primary grades and the same terminal raw attempts used by the frozen
grader. Never changes a grade, retries a call, or replaces the original analysis.
Cap exclusion is outcome-dependent selection, not a simulated larger-budget experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from analysis.preregistered_secondaries import _ame_from_fit
from analysis.stats import fit_h1
from src.backends.adapters import extract
from src.io_utils import read_jsonl
from src.parse_and_grade import _terminal_attempts

RESAMPLES = 2000  # Exploratory analysis; not the original confirmatory bootstrap.
SEED = 20261028
DEPTH_BINS = ((1, 2), (3, 4), (5, 8), (9, 12))


def depth_bin(depth: int | None) -> str:
    if depth is None:
        return "equivalent"
    for low, high in DEPTH_BINS:
        if low <= depth <= high:
            return f"{low}-{high}"
    raise ValueError(f"Depth outside the declared bins: {depth}")


def join_primary(graded: list[dict], raw: list[dict]) -> list[dict]:
    """Validate the raw/grade join and attach observable cap/answer diagnostics."""
    terminal = _terminal_attempts(raw)
    enriched = []
    seen = set()
    for grade in graded:
        if int(grade["sample_index"]) != 0:
            continue
        run_id = grade["run_id"]
        if run_id in seen:
            raise ValueError(f"Duplicate primary grade: {run_id}")
        seen.add(run_id)
        attempt = terminal.get(run_id)
        if attempt is None:
            raise ValueError(f"Missing terminal raw attempt: {run_id}")
        for field in (
            "pair_id",
            "rendering",
            "prompt_condition",
            "sample_index",
            "response_id",
            "finish_reason",
            "usage",
        ):
            if grade.get(field) != attempt.get(field):
                raise ValueError(f"Raw/grade mismatch in {field}: {run_id}")
        if grade["model_id"] != attempt.get("requested_model"):
            raise ValueError(f"Raw/grade model mismatch: {run_id}")
        cap = (attempt.get("decoding") or {}).get("max_tokens")
        if not isinstance(cap, int) or cap <= 0:
            raise ValueError(f"Missing positive output cap: {run_id}")
        answer = extract(attempt.get("raw_body"), grade["answer_adapter"])
        if answer.finish_reason != grade["finish_reason"]:
            raise ValueError(f"Body/metadata finish reason mismatch: {run_id}")
        tokens = (grade.get("usage") or {}).get("completion_tokens")
        length_stop = answer.finish_reason == "length"
        tokens_at_cap = isinstance(tokens, (int, float)) and tokens >= cap
        empty = answer.final_text is None or not answer.final_text.strip()
        row = {
            **grade,
            "output_cap": cap,
            "cap_hit": length_stop or tokens_at_cap,
            "length_stop": length_stop,
            "tokens_at_cap": tokens_at_cap,
            "cap_signals_disagree": length_stop != tokens_at_cap,
            "empty_final": empty,
            "completion_tokens": tokens,
            "final_characters": len(answer.final_text) if answer.final_text is not None else 0,
            "depth_bin": depth_bin(grade["oracle_depth"]),
        }
        row["failure_category"] = outcome_category(row)
        enriched.append(row)
    return sorted(enriched, key=lambda r: r["run_id"])


def outcome_category(row: dict) -> str:
    """Mutually exclusive outcome categories; trace errors are a separate dimension."""
    if row["status"] == "transport_failed":
        return "transport_failure"
    if not row["parse_valid"]:
        suffix = "capped" if row["cap_hit"] else "uncapped"
        if row["empty_final"]:
            return "empty_" + suffix
        if row["parse_outcome"] == "not_json":
            return "non_json_" + suffix
        return "other_parse_failure_" + suffix
    if row["oracle_equivalent"]:
        return "correct_equivalence" if row["decision_correct"] else "false_nonequivalence"
    if row["witness_success"]:
        return "shortest_valid_witness" if row["witness_shortest"] else "longer_valid_witness"
    return "invalid_witness" if row["decision_correct"] else "false_equivalence"


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    neq = [r for r in rows if not r["oracle_equivalent"]]
    eq = [r for r in rows if r["oracle_equivalent"]]
    traces = [r for r in neq if r["trace_valid"] is not None]
    tokens = [
        r["completion_tokens"] for r in rows if isinstance(r["completion_tokens"], (float, int))
    ]
    finals = [r["final_characters"] for r in rows if not r["empty_final"]]
    caps = sum(r["cap_hit"] for r in rows)
    return {
        "calls": n,
        "pairs": len({r["pair_id"] for r in rows}),
        "cap_hits": caps,
        "cap_rate": caps / n if n else None,
        "cap_signals_disagree": sum(r["cap_signals_disagree"] for r in rows),
        "parse_failures": sum(not r["parse_valid"] for r in rows),
        "empty_capped": sum(r["cap_hit"] and r["empty_final"] for r in rows),
        "non_json_capped": sum(r["cap_hit"] and r["parse_outcome"] == "not_json" for r in rows),
        "non_json_uncapped": sum(
            not r["cap_hit"] and r["parse_outcome"] == "not_json" for r in rows
        ),
        "neq_calls": len(neq),
        "witness_successes": sum(r["witness_success"] for r in neq),
        "witness_rate": sum(r["witness_success"] for r in neq) / len(neq) if neq else None,
        "eq_calls": len(eq),
        "eq_correct": sum(r["decision_correct"] for r in eq),
        "eq_accuracy": sum(r["decision_correct"] for r in eq) / len(eq) if eq else None,
        "neq_correct": sum(r["decision_correct"] for r in neq),
        "neq_accuracy": sum(r["decision_correct"] for r in neq) / len(neq) if neq else None,
        "neq_traces_checked": len(traces),
        "neq_trace_errors": sum(r["trace_valid"] is False for r in traces),
        "invalid_witness_with_correct_traces": sum(
            r["failure_category"] == "invalid_witness" and r["trace_valid"] is True for r in neq
        ),
        "completion_tokens_median": statistics.median(tokens) if tokens else None,
        "nonempty_final_characters_median": statistics.median(finals) if finals else None,
        "nonempty_final_count": len(finals),
        "outcomes": dict(sorted(Counter(r["failure_category"] for r in rows).items())),
    }


def breakdown(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return [
        {**dict(zip(fields, key, strict=True)), **summarize(group)}
        for key, group in sorted(groups.items())
    ]


def resample_pairs(blocks: list[list[dict]], picked: np.ndarray) -> list[dict]:
    """Give each sampled copy its own cluster id, even when a pair is drawn twice."""
    return [
        {**row, "pair_id": f"bootstrap_{copy}"}
        for copy, index in enumerate(picked)
        for row in blocks[int(index)]
    ]


def h2_sensitivity(
    rows: list[dict],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    estimator: Callable = _ame_from_fit,
    progress: Callable | None = None,
) -> dict:
    """Refit the frozen mean model; average P1-P0 predictions over retained NEQ calls."""
    if resamples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    eligible = [r for r in rows if not r["oracle_equivalent"] and r["sample_index"] == 0]
    point = estimator(eligible)
    if point is None or not math.isfinite(point):
        return {"estimable": False, "reason": "Non-estimable point fit"}
    groups = defaultdict(list)
    for row in eligible:
        groups[row["pair_id"]].append(row)
    blocks = [groups[key] for key in sorted(groups)]
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(resamples):
        subset = resample_pairs(blocks, rng.integers(0, len(blocks), size=len(blocks)))
        estimate = estimator(subset)
        if estimate is not None and math.isfinite(estimate):
            draws.append(float(estimate))
        if progress and (i + 1) % 200 == 0:
            progress(f"H2 bootstrap {i + 1}/{resamples}; successful fits {len(draws)}")
    if len(draws) < resamples / 2:
        return {
            "estimable": False,
            "reason": "Fewer than half of bootstrap fits estimable",
            "successful_fits": len(draws),
            "attempted_fits": resamples,
        }
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "estimable": True,
        "percentage_points": 100 * point,
        "ci_low_pp": 100 * float(low),
        "ci_high_pp": 100 * float(high),
        "n_cells": len(eligible),
        "n_pairs": len(blocks),
        "resamples": resamples,
        "seed": seed,
        "successful_fits": len(draws),
        "dropped_fits": resamples - len(draws),
        "method": "Exploratory AME; frozen H1 mean model; retained-cell standardization; "
        "pair-cluster percentile bootstrap with distinct cluster IDs per sampled copy",
    }


def build_report(rows: list[dict], resamples: int = RESAMPLES, progress=None) -> dict:
    retained = [r for r in rows if not r["cap_hit"]]
    report = {
        "status": "Post-review exploratory sensitivity; frozen primary estimates unchanged",
        "cap_definition": "finish_reason=length OR reported completion_tokens >= requested "
        "decoding.max_tokens; report agreement between the two signals",
        "selection_warning": "Cap exclusion selects completed calls using an observed outcome. "
        "It does not estimate performance under a larger output budget or isolate a reasoning "
        "mechanism. Non-cap parse failures still count as errors.",
        "overall": summarize(rows),
        "without_caps": summarize(retained),
        "output_caps": sorted({r["output_cap"] for r in rows}),
        "breakdowns": {},
        "without_caps_breakdowns": {},
    }
    groups = [
        ("family",),
        ("oracle_depth",),
        ("depth_bin",),
        ("prompt_condition",),
        ("model_id",),
        ("rendering",),
        ("family", "prompt_condition"),
        ("family", "model_id"),
        ("model_id", "prompt_condition"),
        ("depth_bin", "prompt_condition", "model_id"),
        ("family", "depth_bin", "prompt_condition", "model_id"),
    ]
    for fields in groups:
        key = "__".join(fields)
        report["breakdowns"][key] = breakdown(rows, fields)
        report["without_caps_breakdowns"][key] = breakdown(retained, fields)
    report["h1_primary_refit"] = {k: v.to_dict() for k, v in fit_h1(rows).items()}
    report["h1_without_caps"] = {k: v.to_dict() for k, v in fit_h1(retained).items()}
    report["h2_primary_point_refit_pp"] = 100 * _ame_from_fit(rows)
    report["h2_without_caps"] = h2_sensitivity(retained, resamples, progress=progress)
    report["neq_prompt_outcomes"] = breakdown(
        [r for r in rows if not r["oracle_equivalent"]], ("prompt_condition",)
    )
    return report


def markdown_report(report: dict) -> str:
    total, kept = report["overall"], report["without_caps"]
    h1, primary_h1 = report["h1_without_caps"]["depth"], report["h1_primary_refit"]["depth"]
    h2 = report["h2_without_caps"]
    if not h2["estimable"]:
        raise ValueError("Do not publish a summary of a non-estimable H2 fit")
    lines = [
        "# Output-cap diagnostics and trace-prompt sensitivity",
        "",
        report["status"],
        "",
        "## Definitions and coverage",
        "",
        report["cap_definition"],
        "",
        f"Primary sample 0 only: {total['calls']:,} calls from {total['pairs']} DFA pairs. "
        f"Requested cap: {report['output_caps']}. There are {total['cap_hits']} cap hits; "
        f"the two cap signals disagree on {total['cap_signals_disagree']} calls. "
        f"Of {total['parse_failures']} parse failures, {total['empty_capped']} are empty capped "
        f"answers, {total['non_json_capped']} are capped non-JSON answers, and "
        f"{total['non_json_uncapped']} are uncapped non-JSON answers.",
        "",
        "Grades are joined by run_id to the terminal raw attempt using the frozen grader's "
        "chronological selection. Identity, response, finish reason, and usage fields must "
        "agree. Final answers are extracted with the frozen adapter; no reasoning text "
        "is promoted into an answer and no response is reparsed or repaired.",
        "",
        "## Cap hits and failure types",
        "",
        "| Group | Calls | Capped | Cap rate | Empty capped | Non-JSON capped | "
        "Non-JSON uncapped | All parse failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for field in ("family", "depth_bin", "prompt_condition", "model_id"):
        for row in report["breakdowns"][field]:
            lines.append(
                f"| {row[field]} | {row['calls']} | {row['cap_hits']} | "
                f"{100 * row['cap_rate']:.2f}% | {row['empty_capped']} | "
                f"{row['non_json_capped']} | {row['non_json_uncapped']} | "
                f"{row['parse_failures']} |"
            )
    lines += [
        "",
        "Exact-depth counts and model/prompt/family intersections are in `caps.json` "
        "and `cap_breakdowns.csv`, with both original and cap-excluded denominators.",
        "",
        "## H1/H2 sensitivity",
        "",
        "The same frozen GEE mean model is fitted after excluding capped calls. "
        "H1 uses exact oracle depth, the frozen prompt interaction, rendering and model "
        "indicators, and pair-cluster robust uncertainty. H2 averages P1-P0 predicted "
        "probability differences over the retained non-equivalent calls, with a "
        "pair-clustered percentile bootstrap. Each repeated draw of a pair is assigned "
        "a distinct bootstrap cluster ID. No new multiplicity-adjusted test is claimed.",
        "",
        f"H1 primary refit: {primary_h1['value']:.4f} log-odds/depth "
        f"[{primary_h1['ci_low']:.4f}, {primary_h1['ci_high']:.4f}]. "
        f"Without caps: {h1['value']:.4f} "
        f"[{h1['ci_low']:.4f}, {h1['ci_high']:.4f}].",
        "",
        f"H2 original point refit: {report['h2_primary_point_refit_pp']:.2f} pp. "
        f"Without caps: {h2['percentage_points']:.2f} pp "
        f"[{h2['ci_low_pp']:.2f}, {h2['ci_high_pp']:.2f}]; "
        f"{h2['n_cells']} retained non-equivalent calls from {h2['n_pairs']} pairs. "
        f"Bootstrap: {h2['successful_fits']}/{h2['resamples']} fits used; "
        f"seed {h2['seed']}. These intervals are exploratory; the original H2 interval "
        "and Holm correction remain as reported in the frozen-results artifacts.",
        "",
        "| Population | Non-equivalent decision accuracy | Equivalent decision accuracy | "
        "Valid-witness success |",
        "|---|---:|---:|---:|",
    ]
    for name, row in (("Primary", total), ("Cap-excluded", kept)):
        lines.append(
            f"| {name} | {row['neq_correct']}/{row['neq_calls']} "
            f"({100 * row['neq_accuracy']:.2f}%) | {row['eq_correct']}/{row['eq_calls']} "
            f"({100 * row['eq_accuracy']:.2f}%) | "
            f"{row['witness_successes']}/{row['neq_calls']} "
            f"({100 * row['witness_rate']:.2f}%) |"
        )
    lines += [
        "",
        "## Trace-prompt decomposition on non-equivalent pairs",
        "",
        "| Outcome | P0 | P1 | P1-P0 count |",
        "|---|---:|---:|---:|",
    ]
    prompts = {r["prompt_condition"]: r for r in report["neq_prompt_outcomes"]}
    for category in sorted({c for r in prompts.values() for c in r["outcomes"]}):
        a, b = (prompts[p]["outcomes"].get(category, 0) for p in ("P0", "P1"))
        lines.append(f"| {category} | {a} | {b} | {b - a:+d} |")
    p1 = prompts["P1"]
    lines += [
        "",
        f"P1 traces were checked on {p1['neq_traces_checked']} non-equivalent "
        f"responses: {p1['neq_trace_errors']} were incorrect; "
        f"{p1['invalid_witness_with_correct_traces']} invalid witnesses nevertheless "
        "had correct traces. Trace correctness is not required for the primary "
        "valid-witness metric. Trace errors overlap witness errors and are not another "
        "disjoint term in the failure table.",
        "",
        "| Prompt | Non-equivalent calls | Median completion tokens | "
        "Median nonempty final-answer characters | Nonempty answers |",
        "|---|---:|---:|---:|---:|",
    ]
    for prompt, row in prompts.items():
        lines.append(
            f"| {prompt} | {row['calls']} | {row['completion_tokens_median']} | "
            f"{row['nonempty_final_characters_median']} | {row['nonempty_final_count']} |"
        )
    lines += [
        "",
        "Character length is a directly measured description of visible answers, "
        "not a token count or a measure of reasoning. Completion usage includes provider "
        "accounting; its allocation to hidden reasoning versus final text is not "
        "identified here. Nonempty-answer summaries are conditional on an answer existing.",
        "",
        "## Interpretation and limits",
        "",
        report["selection_warning"],
        "",
        "The cap-excluded depth coefficient remains negative, but is smaller in magnitude. "
        "The trace-prompt penalty remains negative. These observations show that the "
        "original patterns are not solely due to the observed capped calls; they do not "
        "identify how uncapped reasoning, formatting demands, or a larger budget would "
        "change outcomes. Correct traces of an invalid word do not certify a distinction.",
        "",
        "## Reproduction",
        "",
        "Run `python -m analysis.review_caps`. "
        "The original raw log, grades, configuration, and confirmatory modules are unchanged. "
        "`caps.json` contains all estimates, subgroup counts, input digests, and runtime "
        "versions; `cap_calls.csv` contains per-call diagnostics without response text.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded", type=Path, default=Path("data/parsed/main_v2/graded.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/main_v2_responses.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/tables/review_20261028"))
    args = parser.parse_args()
    rows = join_primary(list(read_jsonl(args.graded)), list(read_jsonl(args.raw)))
    print("Joined primary calls:", json.dumps(summarize(rows)), flush=True)
    report = build_report(rows, progress=lambda message: print(message, flush=True))
    report["inputs"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.graded, args.raw)
    }
    report["runtime"] = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "statsmodels")
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "caps.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "caps.md").write_text(markdown_report(report), encoding="utf-8")
    fields = [
        "run_id",
        "pair_id",
        "family",
        "oracle_depth",
        "depth_bin",
        "model_id",
        "prompt_condition",
        "rendering",
        "output_cap",
        "completion_tokens",
        "cap_hit",
        "length_stop",
        "tokens_at_cap",
        "cap_signals_disagree",
        "empty_final",
        "parse_valid",
        "parse_outcome",
        "failure_category",
        "witness_success",
        "trace_valid",
        "final_characters",
    ]
    with (args.out_dir / "cap_calls.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    flat = []
    for population, key in (("primary", "breakdowns"), ("cap_excluded", "without_caps_breakdowns")):
        for grouping, entries in report[key].items():
            for entry in entries:
                flat.append(
                    {
                        "population": population,
                        "grouping": grouping,
                        **{k: v for k, v in entry.items() if k != "outcomes"},
                    }
                )
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (args.out_dir / "cap_breakdowns.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print("H1 without caps:", report["h1_without_caps"]["depth"], flush=True)
    print("H2 without caps:", report["h2_without_caps"], flush=True)


if __name__ == "__main__":
    main()
