"""Turn the append-only raw log into one graded row per scheduled cell (Step 10).

Usage::

    python -m src.parse_and_grade --config configs/pilot_revised.yaml

The join is the point of this module. Four artifacts have to meet without any of them being
adjusted to fit: the schedule (which cells were supposed to exist), the raw log (what came
back, including what did not), ``pairs.jsonl`` (the oracle's verdict and the machines), and
``manifest.jsonl`` (the label bijection the model was actually shown).

That last one is the subtle one. A P1 trace arrives as surface labels -- ``zer, gur, vum``
under R1 -- and means nothing until it is mapped back through *that rendering's* bijection.
Grading an R1 answer through R0's labels produces a confident, wrong score, which is what
QA gate 6 exists to prevent and what this module must not reintroduce.

Nothing here repairs an answer. The adapter picks the answer field, ``strict_parse`` accepts
or rejects it, and a rejection is recorded as a failure (rule 5). ``forgiving_parse`` runs
alongside and its verdict is written into clearly separate ``sensitivity_*`` columns that no
primary metric reads.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.backends.adapters import DEFAULT_ADAPTER, extract
from src.config import PLACEHOLDERS, Config, ConfigError, load_config
from src.dfa import from_record
from src.grading import forgiving_parse, grade, strict_parse
from src.io_utils import append_jsonl, read_jsonl

__all__ = ["build_rows", "main"]

#: Statuses a graded row can carry. ``received`` means a final answer channel existed to
#: grade, whatever it contained.
RECEIVED = "received"
TRANSPORT_FAILED = "transport_failed"


def _terminal_attempts(raw_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The last attempt for each run id, plus how many attempts it took.

    Earlier attempts are never discarded from the log; they are simply not the outcome of
    the cell. Their count is carried onto the row so a cell that needed four tries is
    visible in the record rather than looking like a clean single call.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        run_id = row.get("run_id")
        if run_id:
            grouped[str(run_id)].append(row)

    terminal: dict[str, dict[str, Any]] = {}
    for run_id, attempts in grouped.items():
        # Chronological, not by attempt number. A resumed run restarts its numbering at 1,
        # so a cell retried after an outage would otherwise have two "attempt 2" rows and the
        # winner would depend on file order. The clock is unambiguous across invocations.
        attempts.sort(
            key=lambda row: (str(row.get("started_at_utc", "")), int(row.get("attempt", 0)))
        )
        last = dict(attempts[-1])
        last["_attempt_count"] = len(attempts)
        last["_transport_failures"] = sum(1 for row in attempts if not row.get("ok"))
        terminal[run_id] = last
    return terminal


def _labels_by_pair_rendering(manifest_rows: list[dict[str, Any]]) -> dict[tuple, dict]:
    return {
        (row["pair_id"], row["rendering"]): row
        for row in manifest_rows
    }


def _declared_models(config: Config) -> set[str]:
    """The exact request ids the config's design contains."""
    models: set[str] = set()
    for entry in config.require("models"):
        for key in ("id", "candidate_alias"):
            value = entry.get(key)
            if isinstance(value, str) and value not in PLACEHOLDERS:
                models.add(value)
                break
    return models


def build_rows(
    config: Config,
    raw_rows: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    *,
    adapter: str = DEFAULT_ADAPTER,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Grade every cell the config's design contains. Returns ``(rows, problems)``.

    Rows for models the config no longer declares are **skipped, loudly**. The raw log is
    append-only and keeps every attempt ever made (rule 3), but a model withdrawn from the
    design is not part of the dataset: grading its cells would put a model into the
    denominators that no schedule contains. That is not hypothetical -- when
    ``nemotron-3-ultra-free`` was withdrawn on 2026-09-08 the log already held 13 of its
    cells, every one a transport failure, which would have entered the analysis as 13
    failures for a model under no test.
    """
    alphabet = config.alphabet
    pairs = {record["pair_id"]: record for record in pair_records}
    labels = _labels_by_pair_rendering(manifest_rows)
    terminal = _terminal_attempts(raw_rows)
    declared = _declared_models(config)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    undeclared: dict[str, int] = {}
    for run_id, attempt in list(terminal.items()):
        model = str(attempt.get("requested_model"))
        if model not in declared:
            undeclared[model] = undeclared.get(model, 0) + 1
            del terminal[run_id]
    for model, count in sorted(undeclared.items()):
        problems.append(
            f"skipped {count} cell(s) for {model!r}: the config no longer declares that "
            f"model, so its cells are not part of this dataset. The raw attempts remain in "
            f"the log."
        )

    for run_id in sorted(terminal):
        attempt = terminal[run_id]
        pair_id = attempt.get("pair_id")
        record = pairs.get(str(pair_id))
        if record is None:
            problems.append(f"{run_id}: pair {pair_id!r} is not in pairs.jsonl")
            continue
        manifest = labels.get((pair_id, attempt.get("rendering")))
        if manifest is None:
            problems.append(
                f"{run_id}: no manifest row for {pair_id} / {attempt.get('rendering')}; "
                f"the label bijection is unknown, so the answer cannot be graded"
            )
            continue

        rows.append(
            _grade_one(
                attempt,
                record,
                manifest,
                alphabet=alphabet,
                adapter=adapter,
            )
        )
    return rows, problems


def _grade_one(
    attempt: dict[str, Any],
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    alphabet: tuple[str, ...],
    adapter: str,
) -> dict[str, Any]:
    equivalent = bool(record["oracle_equivalent"])
    depth = record.get("oracle_depth")
    condition = str(attempt["prompt_condition"])

    base: dict[str, Any] = {
        "run_id": attempt["run_id"],
        "dataset_id": attempt.get("dataset_id"),
        "pair_id": record["pair_id"],
        "family": record["family"],
        "oracle_equivalent": equivalent,
        "oracle_depth": depth,
        "rendering": attempt.get("rendering"),
        "prompt_condition": condition,
        "model_id": attempt.get("requested_model"),
        "returned_model": attempt.get("returned_model"),
        "sample_index": int(attempt.get("sample_index", 0)),
        "attempts": attempt.get("_attempt_count", 1),
        "transport_failures": attempt.get("_transport_failures", 0),
        "response_id": attempt.get("response_id"),
        "finish_reason": attempt.get("finish_reason"),
        "usage": attempt.get("usage"),
        "latency_ms": attempt.get("latency_ms"),
        "answer_adapter": adapter,
    }

    # A cell that never returned a response. It stays in the schedule as a row so it cannot
    # vanish from a denominator, and its witness outcome is False rather than None on a
    # non-equivalent pair (D3).
    if not attempt.get("ok"):
        return {
            **base,
            "status": TRANSPORT_FAILED,
            "parse_valid": False,
            "parse_outcome": TRANSPORT_FAILED,
            "decision_correct": False,
            "witness_success": None if equivalent else False,
            "witness_valid": None,
            "witness_shortest": None,
            "trace_valid": None,
            "first_wrong_step": None,
            "truncated": False,
            "reasoning_present": False,
            "failure_kind": attempt.get("failure_kind"),
            "failure_detail": attempt.get("failure_detail"),
            "sensitivity_parse_valid": False,
            "sensitivity_decision_correct": False,
            "sensitivity_witness_success": None if equivalent else False,
        }

    answer = extract(attempt.get("raw_body"), adapter)
    parsed = strict_parse(answer.final_text, condition, alphabet)
    dfa_a = from_record(record["dfa_A"])
    dfa_b = from_record(record["dfa_B"])
    labels_a = tuple(manifest["labels_A"])
    labels_b = tuple(manifest["labels_B"])
    scored = grade(parsed, dfa_a, dfa_b, equivalent, depth, labels_a, labels_b)

    # Sensitivity only. Written to columns no primary metric reads, so a more permissive
    # parser can never silently become the reported result (rule 5).
    lenient = forgiving_parse(answer.final_text, condition, alphabet)
    lenient_scored = grade(lenient, dfa_a, dfa_b, equivalent, depth, labels_a, labels_b)

    return {
        **base,
        "status": RECEIVED,
        "parse_valid": scored.parse_valid,
        "parse_outcome": scored.parse_outcome,
        "decision_correct": scored.decision_correct,
        "witness_success": scored.witness_success,
        "witness_valid": scored.witness_valid,
        "witness_shortest": scored.witness_shortest,
        "trace_valid": scored.trace_valid,
        "first_wrong_step": scored.first_wrong_step,
        "truncated": answer.truncated,
        "reasoning_present": answer.reasoning_text is not None,
        "answer_problem": answer.problem,
        "grade_detail": scored.detail,
        "sensitivity_parse_valid": lenient_scored.parse_valid,
        "sensitivity_decision_correct": lenient_scored.decision_correct,
        "sensitivity_witness_success": lenient_scored.witness_success,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.parse_and_grade",
        description="Strict-parse and grade the raw responses into graded.jsonl.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw", help="raw JSONL; defaults to execution.raw_log")
    parser.add_argument("--out", help="output path; defaults to <parsed_dir>/graded.jsonl")
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="frozen final-answer adapter name (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        raw_path = Path(args.raw) if args.raw else Path(str(config.require("execution.raw_log")))
        pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
        manifest_path = config.path_for("renderings_dir") / "manifest.jsonl"
        out_path = Path(args.out) if args.out else config.path_for("parsed_dir") / "graded.jsonl"
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    for path in (raw_path, pairs_path, manifest_path):
        if not path.is_file():
            print(f"{path} not found.", file=sys.stderr)
            return 1

    raw_rows = list(read_jsonl(raw_path))
    pair_records = list(read_jsonl(pairs_path))
    manifest_rows = list(read_jsonl(manifest_path))

    try:
        rows, problems = build_rows(
            config, raw_rows, pair_records, manifest_rows, adapter=args.adapter
        )
    except KeyError as error:
        print(f"adapter error: {error}", file=sys.stderr)
        return 2

    if problems:
        print("Join problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)

    # Graded output is derived, not raw: regenerating it from the append-only log must be
    # reproducible, so a stale file is replaced rather than appended to.
    if out_path.exists():
        out_path.unlink()
    for row in rows:
        append_jsonl(out_path, row)

    received = sum(1 for row in rows if row["status"] == RECEIVED)
    strict_ok = sum(1 for row in rows if row["parse_valid"])
    lenient_ok = sum(1 for row in rows if row["sensitivity_parse_valid"])
    neq = [row for row in rows if not row["oracle_equivalent"]]
    successes = sum(1 for row in neq if row["witness_success"])

    print(f"raw attempts    : {len(raw_rows)}")
    print(f"graded cells    : {len(rows)} ({received} received, "
          f"{len(rows) - received} transport-failed)")
    print(f"strict parse ok : {strict_ok}/{len(rows)}")
    print(f"forgiving parse : {lenient_ok}/{len(rows)}  (sensitivity only, never primary)")
    if neq:
        print(f"witness success : {successes}/{len(neq)} = {successes / len(neq):.1%} "
              f"(end-to-end, over all scheduled non-equivalent cells)")
    print(f"written         : {out_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
