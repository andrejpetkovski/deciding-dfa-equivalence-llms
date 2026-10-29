"""Exploratory configuration-specific depth slopes and categorical-depth sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.review_caps import depth_bin
from analysis.stats import _design, _fit_gee, fit_h1
from src.io_utils import read_jsonl


def categorical_design(rows):
    """Replace exact linear depth and its prompt interaction by bin indicators."""
    matrix, outcome, clusters, names = _design(rows, with_interaction=True)
    keep = [i for i, name in enumerate(names) if name not in ("depth", "depth:prompt")]
    columns = [matrix[:, i] for i in keep]
    labels = [names[i] for i in keep]
    prompt = matrix[:, names.index("prompt")]
    for band in ("3-4", "5-8", "9-12"):
        indicator = np.array([depth_bin(r["oracle_depth"]) == band for r in rows], dtype=float)
        columns.extend((indicator, indicator * prompt))
        labels.extend((f"bin[{band}]", f"bin[{band}]:prompt"))
    return np.column_stack(columns), outcome, clusters, labels


def main():
    path = Path("data/parsed/main_v2/graded.jsonl")
    rows = [r for r in read_jsonl(path) if r["sample_index"] == 0 and not r["oracle_equivalent"]]
    models = sorted({r["model_id"] for r in rows})
    report = {
        "status": "Exploratory post-review analysis; frozen primary model unchanged",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "per_model": {},
        "bin_rates": [],
        "categorical_pooled": {
            k: v.to_dict() for k, v in _fit_gee(*categorical_design(rows)).items()
        },
    }
    for model in models:
        subset = [r for r in rows if r["model_id"] == model]
        report["per_model"][model] = {k: v.to_dict() for k, v in fit_h1(subset).items()}
        for prompt in ("P0", "P1"):
            for band in ("1-2", "3-4", "5-8", "9-12"):
                group = [
                    r
                    for r in subset
                    if r["prompt_condition"] == prompt and depth_bin(r["oracle_depth"]) == band
                ]
                report["bin_rates"].append(
                    {
                        "model": model,
                        "prompt": prompt,
                        "bin": band,
                        "calls": len(group),
                        "successes": sum(r["witness_success"] for r in group),
                    }
                )
    lines = [
        "# Configuration-specific depth analysis",
        "",
        "Exploratory primary-sample analysis: 100 NEQ pairs, 400 calls per model. "
        "All parse failures remain errors. Each model uses exact centered depth, "
        "prompt, rendering, and depth-by-prompt interaction, with exchangeable GEE "
        "and pair-cluster robust 95% intervals. Slopes average across the centered "
        "prompt coding. No multiplicity-adjusted heterogeneity test is claimed.",
        "",
        "| Configuration | Depth slope | 95% CI |",
        "|---|---:|---:|",
    ]
    for model, estimates in report["per_model"].items():
        e = estimates["depth"]
        assert all(e[k] is not None and np.isfinite(e[k]) for k in ("value", "ci_low", "ci_high"))
        lines.append(f"| {model} | {e['value']:.4f} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
    lines += [
        "",
        "| Configuration | Prompt | Depth bin | Successes / calls |",
        "|---|---|---|---:|",
    ]
    for r in report["bin_rates"]:
        lines.append(
            f"| {r['model']} | {r['prompt']} | {r['bin']} | {r['successes']}/{r['calls']} |"
        )
    lines += [
        "",
        "## Categorical-depth sensitivity",
        "",
        "The pooled alternative replaces the linear depth term and its prompt interaction "
        "with three depth-bin indicators and their prompt interactions, retaining "
        "model/rendering controls and pair clustering. Reference bin: 1-2. "
        "This relaxes linearity across bins but does not identify a within-bin slope.",
        "",
        "| Bin contrast versus 1-2 | Log-odds contrast | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, e in report["categorical_pooled"].items():
        if name.startswith("bin[") and ":" not in name:
            assert all(
                e[k] is not None and np.isfinite(e[k]) for k in ("value", "ci_low", "ci_high")
            )
            lines.append(f"| {name} | {e['value']:.4f} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
    lines += [
        "",
        "Unequal bin widths do not imply midpoint coding: the original regression "
        "uses exact oracle depth minus 5.5. Its linear log-odds assumption is a summary "
        "constraint, not a proven law. Configuration-specific intervals are descriptive; "
        "differences in their significance are not a test of differences between models. "
        "Model and provider remain confounded. Full coefficients are in depth.json.",
        "",
        "Reproduce with `python -m analysis.review_depth`.",
        "",
    ]
    out = Path("analysis/tables/review_20261028")
    (out / "depth.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "depth.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
