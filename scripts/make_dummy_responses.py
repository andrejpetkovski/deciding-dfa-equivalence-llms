"""Synthesize graded rows so the analysis runs end to end before any model call.

Usage::

    python -m scripts.make_dummy_responses --config configs/pilot_revised.yaml

``docs/experiment_revision.md`` requires the runner, grader and analysis to be complete on
dummy responses *before* the experimental pilot. This produces the graded-row shape that
``src/parse_and_grade.py`` will emit once M7 fixes the wire format, so `analysis/` can be
exercised, and its failure paths checked, against data that costs nothing.

The synthetic model is deliberately crude but not degenerate: witness success falls with
depth, P1 helps a little, R1 hurts a little, and the three models differ. **Nothing here is a
prediction.** Its only job is to exercise every branch, so it also emits the cases that break
naive analysis code — transport failures, parse failures, valid-but-not-shortest witnesses,
and wrong traces on otherwise successful cells.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from src.config import ConfigError, load_config
from src.io_utils import append_jsonl, read_jsonl

__all__ = ["main", "synthesize"]

#: Stand-in model identities. Real ones are unverified until M7 (D4).
DUMMY_MODELS = ("dummy/model-a", "dummy/model-b", "dummy/model-c")

#: Per-model skill on the log-odds scale, so the three differ without any of them being
#: perfect or hopeless.
_MODEL_SKILL = {"dummy/model-a": 1.2, "dummy/model-b": 0.4, "dummy/model-c": -0.4}

_TRANSPORT_FAILURE_RATE = 0.01
_PARSE_FAILURE_RATE = 0.06


def synthesize(
    pair_records: list[dict[str, Any]],
    models: tuple[str, ...] = DUMMY_MODELS,
    renderings: tuple[str, ...] = ("R0", "R1"),
    prompts: tuple[str, ...] = ("P0", "P1"),
    seed: int = 20260908,
    dataset_id: str = "dummy",
) -> list[dict[str, Any]]:
    """Produce one graded row per scheduled cell, including failure cases."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for record in pair_records:
        equivalent = record["oracle_equivalent"]
        depth = record["oracle_depth"]
        for model in models:
            for rendering in renderings:
                for prompt in prompts:
                    row = _one_cell(
                        rng, record, model, rendering, prompt, equivalent, depth, dataset_id
                    )
                    rows.append(row)
    return rows


def _one_cell(
    rng: np.random.Generator,
    record: dict[str, Any],
    model: str,
    rendering: str,
    prompt: str,
    equivalent: bool,
    depth: int | None,
    dataset_id: str,
) -> dict[str, Any]:
    run_id = (
        f"{dataset_id}|{record['pair_id']}|{rendering}|{prompt}|{model}|s0"
    )
    base = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "pair_id": record["pair_id"],
        "family": record["family"],
        "oracle_equivalent": equivalent,
        "oracle_depth": depth,
        "rendering": rendering,
        "prompt_condition": prompt,
        "model_id": model,
        "sample_index": 0,
    }

    # A cell that never returned. It must survive the whole analysis as a failure rather than
    # an absent row, which is the shape most likely to break a naive denominator.
    if rng.random() < _TRANSPORT_FAILURE_RATE:
        return {
            **base,
            "status": "transport_failed",
            "parse_valid": False,
            "parse_outcome": "transport_failed",
            "decision_correct": False,
            "witness_success": None if equivalent else False,
            "witness_valid": None,
            "witness_shortest": None,
            "trace_valid": None,
            "first_wrong_step": None,
        }

    if rng.random() < _PARSE_FAILURE_RATE:
        return {
            **base,
            "status": "received",
            "parse_valid": False,
            "parse_outcome": "not_json",
            "decision_correct": False,
            "witness_success": None if equivalent else False,
            "witness_valid": None,
            "witness_shortest": None,
            "trace_valid": None,
            "first_wrong_step": None,
        }

    skill = _MODEL_SKILL[model]
    prompt_bonus = 0.35 if prompt == "P1" else 0.0
    rendering_penalty = -0.30 if rendering == "R1" else 0.0

    if equivalent:
        # No witness outcome exists on an equivalent cell.
        logit = skill + prompt_bonus + rendering_penalty + 0.2
        correct = bool(rng.random() < 1 / (1 + np.exp(-logit)))
        return {
            **base,
            "status": "received",
            "parse_valid": True,
            "parse_outcome": "ok",
            "decision_correct": correct,
            "witness_success": None,
            "witness_valid": None,
            "witness_shortest": None,
            "trace_valid": None,
            "first_wrong_step": None,
        }

    # Witness success decays with depth: the effect H1 is designed to detect.
    logit = skill + prompt_bonus + rendering_penalty + 1.4 - 0.22 * float(depth)
    success = bool(rng.random() < 1 / (1 + np.exp(-logit)))
    # A model can call the pair non-equivalent and still fail to certify it.
    decided_neq = success or rng.random() < 0.6
    witness_valid = success if decided_neq else None
    # Among successes, some witnesses are valid but longer than the oracle's.
    shortest = bool(rng.random() < 0.7) if success else (False if decided_neq else None)

    trace_valid = None
    first_wrong = None
    if prompt == "P1" and decided_neq:
        # Traces fail more often than witnesses: a wrong trace must not disturb success (D3).
        trace_valid = bool(rng.random() < (0.6 if success else 0.2))
        if not trace_valid:
            first_wrong = int(rng.integers(0, max(1, int(depth)) + 1))

    return {
        **base,
        "status": "received",
        "parse_valid": True,
        "parse_outcome": "ok",
        "decision_correct": decided_neq,
        "witness_success": success,
        "witness_valid": witness_valid,
        "witness_shortest": shortest,
        "trace_valid": trace_valid,
        "first_wrong_step": first_wrong,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.make_dummy_responses",
        description="Synthesize graded rows for analysis development (no model calls).",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
        if not pairs_path.is_file():
            print(f"{pairs_path} not found. Run python -m src.generate first.",
                  file=sys.stderr)
            return 1
        records = list(read_jsonl(pairs_path))
        destination = config.path_for("parsed_dir") / "graded_dummy.jsonl"
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    rows = synthesize(
        records, seed=args.seed, dataset_id=f"{config.config_id}_dummy"
    )
    if destination.exists():
        destination.unlink()
    for row in rows:
        append_jsonl(destination, row)

    print(f"pairs        : {len(records)}")
    print(f"graded rows  : {len(rows)} ({destination})")
    print(
        "NOTE: synthetic rows for analysis development. Not model output, not evidence "
        "about any model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
