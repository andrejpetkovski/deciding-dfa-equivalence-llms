"""Post-review exploratory comparison of rendering and repeated-call disagreement.

The original confirmatory analysis is unchanged. Each comparison group is one
(pair_id, model_id, prompt_condition) with both renderings. Restrict both rates to
the SAME groups with every selected call strictly parsed. Average all cross-rendering
pairs and all distinct within-rendering sample pairs; bootstrap whole DFA pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

from analysis.metrics import _decision
from analysis.stats import cluster_bootstrap_rate
from src.io_utils import read_jsonl


def matched_groups(rows: list[dict[str, Any]], samples: tuple[int, ...]) -> dict[str, Any]:
    """Require complete matched calls, rejecting duplicates rather than choosing one."""
    if len(samples) < 2 or len(set(samples)) != len(samples):
        raise ValueError("Need at least two distinct sample indices")
    grouped: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for row in rows:
        index = int(row["sample_index"])
        if index not in samples or row["rendering"] not in ("R0", "R1"):
            continue
        key = (row["pair_id"], row["model_id"], row["prompt_condition"])
        call = (row["rendering"], index)
        if call in grouped[key]:
            raise ValueError(f"Duplicate graded call: {key}, {call}")
        grouped[key][call] = row
    required = set(product(("R0", "R1"), samples))
    # Audit groups are identified by an observed repeat, not a primary-only observation.
    audit = {key: calls for key, calls in grouped.items() if any(i > 0 for _, i in calls)}
    complete = {key: calls for key, calls in audit.items() if set(calls) == required}
    eligible = []
    for key, calls in sorted(complete.items()):
        if not all(row["parse_valid"] for row in calls.values()):
            continue
        decisions = {call: _decision(row) for call, row in calls.items()}
        cross = [
            decisions[("R0", i)] != decisions[("R1", j)] for i, j in product(samples, repeat=2)
        ]
        within = [
            decisions[(rendering, i)] != decisions[(rendering, j)]
            for rendering in ("R0", "R1")
            for i, j in combinations(samples, 2)
        ]
        eligible.append(
            {
                "pair_id": key[0],
                "model_id": key[1],
                "prompt_condition": key[2],
                "family": calls[("R0", samples[0])]["family"],
                "cross_disagreements": sum(cross),
                "cross_comparisons": len(cross),
                "within_disagreements": sum(within),
                "within_comparisons": len(within),
                "cross_rate": sum(cross) / len(cross),
                "within_rate": sum(within) / len(within),
                "difference": sum(cross) / len(cross) - sum(within) / len(within),
            }
        )
    return {
        "sample_indices": list(samples),
        "observed_audit_groups": len(audit),
        "complete_groups": len(complete),
        "incomplete_groups": len(audit) - len(complete),
        "excluded_for_parse_failure": len(complete) - len(eligible),
        "eligible_groups": len(eligible),
        "groups": eligible,
    }


def summarize(groups: list[dict], resamples: int = 10000) -> dict[str, Any]:
    """Paired difference and both rates, with pair-clustered percentile intervals."""
    result: dict[str, Any] = {
        "groups": len(groups),
        "dfa_pairs": len({row["pair_id"] for row in groups}),
    }
    for metric in ("cross_rate", "within_rate", "difference"):

        def statistic(rows, key=metric):
            return sum(row[key] for row in rows) / len(rows) if rows else None

        low, high = cluster_bootstrap_rate(
            groups,
            statistic,
            resamples=resamples,
            seed=20261028,
        )
        result[metric] = {"estimate": statistic(groups), "ci_low": low, "ci_high": high}
    for kind in ("cross", "within"):
        result[kind + "_rate"].update(
            {
                "numerator": sum(row[kind + "_disagreements"] for row in groups),
                "denominator": sum(row[kind + "_comparisons"] for row in groups),
            }
        )
    return result


def build_report(rows: list[dict], resamples: int = 10000) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "Exploratory post-review analysis; frozen primary results unchanged",
        "method": "Matched complete-case groups; all cross pairs versus distinct within pairs; "
        "95% percentile bootstrap resampling pair_id; difference = cross minus within",
        "limitations": "Conditional on all selected calls parsing; audit subset only. "
        "Calls were not a newly randomized rendering-versus-repeat experiment; "
        "timing and deployment drift remain possible. A CI including zero does not prove equality.",
        "bootstrap_seed": 20261028,
        "bootstrap_resamples": resamples,
    }
    for name, samples in (("first_two_calls", (0, 1)), ("all_three_calls", (0, 1, 2))):
        block = matched_groups(rows, samples)
        groups = block["groups"]
        block["overall"] = summarize(groups, resamples)
        for field in ("model_id", "prompt_condition", "family"):
            block["by_" + field] = {
                value: summarize([row for row in groups if row[field] == value], resamples)
                for value in sorted({row[field] for row in groups})
            }
        report[name] = block
    return report


def markdown_report(report: dict[str, Any]) -> str:
    """Human-readable evidence generated from the same numeric report."""
    lines = [
        "# H3: rendering changes versus repeated identical requests",
        "",
        report["status"],
        "",
        "## Method",
        "",
        "A group is one DFA pair, model, and prompt, observed under both renderings. "
        "The first comparison uses samples 0 and 1 from each rendering. The sensitivity "
        "uses samples 0, 1, and 2. Each analysis retains only groups where every selected "
        "call parses, so its cross-rendering and within-rendering rates use the same groups. "
        "No parse failure is reclassified in the primary results.",
        "",
        "For two calls per rendering, each group contributes four cross-rendering and two "
        "within-rendering comparisons; with three calls, these counts are nine and six. "
        "These comparisons are correlated, not independent observations. "
        "The paired difference is cross-rendering minus within-rendering disagreement. "
        "Intervals resample whole DFA pairs, retaining all their models and prompts.",
        "",
        f"Bootstrap: {report['bootstrap_resamples']:,} resamples; seed "
        f"{report['bootstrap_seed']}; 95% percentile intervals.",
        "",
        "## Results",
        "",
        "| Selected calls | Eligible groups / complete groups | DFA pairs | "
        "Cross-rendering disagreement | Within-rendering disagreement | Difference (pp), 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def rate(entry):
        return f"{100 * entry['estimate']:.2f}% ({entry['numerator']}/{entry['denominator']})"

    def difference(entry):
        return (
            f"{100 * entry['estimate']:+.2f} "
            f"[{100 * entry['ci_low']:+.2f}, {100 * entry['ci_high']:+.2f}]"
        )

    for name in ("first_two_calls", "all_three_calls"):
        block = report[name]
        overall = block["overall"]
        lines.append(
            f"| {block['sample_indices']} | {block['eligible_groups']}/{block['complete_groups']} "
            f"| {overall['dfa_pairs']} | {rate(overall['cross_rate'])} "
            f"| {rate(overall['within_rate'])} | {difference(overall['difference'])} |"
        )
    lines += [
        "",
        "The matched estimates do not establish additional disagreement attributable "
        "to rendering beyond repeated-call variability. This does not prove equivalence "
        "of the rates. The original 58/1,047 primary cross-rendering flips remain a valid "
        "description, but a nonzero flip rate alone does not isolate a rendering effect.",
        "",
        "The review's conversion of 8.5% three-call disagreement to about 5.7% pairwise "
        "disagreement is a useful diagnostic: for three Boolean answers that are not all "
        "equal, exactly two of the three unordered pairs disagree. However, that overall "
        "audit denominator differs from the primary rendering denominator. The matched "
        "analysis above avoids comparing those different subsets.",
        "",
        "## Descriptive breakdowns",
        "",
    ]
    for name in ("first_two_calls", "all_three_calls"):
        block = report[name]
        lines += [
            f"### Samples {block['sample_indices']}",
            "",
            f"Observed audit groups: {block['observed_audit_groups']}; incomplete: "
            f"{block['incomplete_groups']}; excluded for a parse failure: "
            f"{block['excluded_for_parse_failure']}.",
            "",
            "| Subgroup | Groups | Cross-rendering | Within-rendering | Difference (pp), 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
        for field in ("model_id", "prompt_condition", "family"):
            for value, entry in block["by_" + field].items():
                lines.append(
                    f"| {value} | {entry['groups']} | {rate(entry['cross_rate'])} "
                    f"| {rate(entry['within_rate'])} | {difference(entry['difference'])} |"
                )
        lines.append("")
    lines += [
        "## Limits and provenance",
        "",
        report["limitations"],
        "",
        "The choice of this comparison was made after review, not before inference. "
        "Model/prompt/family breakdowns are descriptive, without multiplicity-adjusted tests. "
        "The complete-case filter may select easier calls and is not evidence about "
        "unparseable decisions. No new model calls were made.",
        "",
        f"Input: `{report['input']}`",
        "",
        f"Input SHA-256: `{report['input_sha256']}`",
        "",
        "Reproduce from the repository root: `python -m analysis.review_h3`. "
        "Machine-readable results and group-level counts: `h3.json` alongside this file.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded", type=Path, default=Path("data/parsed/main_v2/graded.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("analysis/tables/review_20261028/h3.json"))
    args = parser.parse_args()
    report = build_report(list(read_jsonl(args.graded)))
    report["input"] = str(args.graded)
    report["input_sha256"] = hashlib.sha256(args.graded.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    for name in ("first_two_calls", "all_three_calls"):
        print(
            name,
            json.dumps(
                {
                    k: v
                    for k, v in report[name].items()
                    if k not in ("groups", "by_model_id", "by_prompt_condition", "by_family")
                }
            ),
        )


if __name__ == "__main__":
    main()
