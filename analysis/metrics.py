"""Outcome metrics and dataset validation (§9, D3).

Every metric here is defined by its **denominator** as much as by its numerator, and the
denominators differ deliberately. Two of them are easy to confuse and mean different things:

- *End-to-end witness success* is over **all scheduled** ground-truth non-equivalent cells, so
  a parse failure or a transport failure counts against it. This is H1/H2's primary outcome.
- *Conditional witness validity* is over **parsed outputs that predicted non-equivalent**, so
  it says nothing about how often the model got that far.

Reporting the second where the first belongs would inflate the headline rate by exactly the
failures the study is trying to measure. :func:`coverage` exists so the gap between scheduled,
attempted, received and strict-valid is always visible next to any conditional rate.

Primary rates use ``sample_index == 0``; repeats are an operational audit analysed separately
(D4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "REQUIRED_COLUMNS",
    "Rate",
    "ValidationError",
    "balanced_decision_accuracy",
    "conditional_witness_validity",
    "coverage",
    "decision_flip_rate",
    "orbit_success",
    "parse_failure_rate",
    "primary_rows",
    "shortest_witness_success",
    "trace_correctness",
    "validate_rows",
    "witness_success",
]

#: Every graded row must carry these. ``parse_and_grade`` (M7/M8) emits them; the dummy
#: generator produces the same shape so the analysis can be exercised before any model call.
REQUIRED_COLUMNS = (
    "run_id",
    "dataset_id",
    "pair_id",
    "family",
    "oracle_equivalent",
    "oracle_depth",
    "rendering",
    "prompt_condition",
    "model_id",
    "sample_index",
    "status",
    "parse_valid",
    "decision_correct",
    "witness_success",
    "witness_valid",
    "witness_shortest",
    "trace_valid",
)

#: A cell that never produced a final answer. Kept distinct from a parse failure so §9's
#: "report transport failures separately" is possible, while both still count against the
#: conservative end-to-end rates.
TRANSPORT_FAILED = "transport_failed"


class ValidationError(ValueError):
    """A graded dataset violates a structural assumption the analysis depends on."""


@dataclass(frozen=True, slots=True)
class Rate:
    """One metric: numerator, denominator, and the rate, with the denominator named."""

    name: str
    numerator: int
    denominator: int
    denominator_description: str

    @property
    def value(self) -> float | None:
        """``None`` when the denominator is empty -- never silently 0."""
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "denominator_description": self.denominator_description,
            "rate": self.value,
        }


# --------------------------------------------------------------------------- selection


def primary_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The primary analysis set: sample 0 only (§10.5).

    Repeats exist to audit operational repeatability, not to add observations. Pooling them
    would weight repeated pairs more heavily and understate every interval.
    """
    return [row for row in rows if int(row["sample_index"]) == 0]


def _neq(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row["oracle_equivalent"]]


# --------------------------------------------------------------------------- validation


def validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_cells: int | None = None,
    declared_samples: Sequence[int] = (0,),
) -> None:
    """Check the structural assumptions every metric below relies on (§Step 11).

    Raises:
        ValidationError: naming the violation. These are assertions rather than warnings
            because each one silently corrupts a rate: a duplicated ``run_id`` double-counts
            a cell, a second ``dataset_id`` pools splits the protocol keeps separate, and a
            missing oracle label makes a row ungradeable while still occupying a denominator.
    """
    if not rows:
        raise ValidationError("no graded rows")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(rows[0]))
    if missing_columns:
        raise ValidationError(f"graded rows are missing columns: {missing_columns}")

    run_ids = Counter(row["run_id"] for row in rows)
    duplicated = [run_id for run_id, count in run_ids.items() if count > 1]
    if duplicated:
        raise ValidationError(f"duplicate run_id values: {duplicated[:5]}")

    datasets = {row["dataset_id"] for row in rows}
    if len(datasets) > 1:
        raise ValidationError(
            f"rows span more than one dataset: {sorted(datasets)}. Pilot, main and audit "
            f"data are not pooled in confirmatory results."
        )

    undeclared = sorted(
        {int(row["sample_index"]) for row in rows} - set(int(s) for s in declared_samples)
    )
    if undeclared:
        raise ValidationError(f"undeclared sample_index values present: {undeclared}")

    for row in rows:
        if row["oracle_equivalent"] is None:
            raise ValidationError(f"{row['run_id']}: missing oracle_equivalent")
        if not row["oracle_equivalent"] and row["oracle_depth"] is None:
            raise ValidationError(f"{row['run_id']}: non-equivalent pair with no depth")

    primary = primary_rows(rows)
    cells = Counter(
        (row["pair_id"], row["rendering"], row["prompt_condition"], row["model_id"])
        for row in primary
    )
    repeated = [cell for cell, count in cells.items() if count > 1]
    if repeated:
        raise ValidationError(f"duplicate primary cells: {repeated[:5]}")

    if expected_cells is not None and len(primary) != expected_cells:
        raise ValidationError(
            f"expected {expected_cells} primary cells, found {len(primary)}"
        )


def coverage(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Scheduled, received, strict-valid and transport-failed counts (§9).

    Printed beside every conditional rate. Without it a high conditional-validity number is
    unreadable: it cannot be told apart from a model that answered twice and got both right.
    """
    received = [row for row in rows if row["status"] != TRANSPORT_FAILED]
    return {
        "scheduled": len(rows),
        "received": len(received),
        "transport_failed": len(rows) - len(received),
        "strict_valid": sum(1 for row in received if row["parse_valid"]),
        "non_equivalent_scheduled": len(_neq(rows)),
    }


# --------------------------------------------------------------------------- metrics


def witness_success(rows: Sequence[dict[str, Any]]) -> Rate:
    """**H1/H2 primary outcome.** Over all scheduled ground-truth non-equivalent cells.

    Parse and transport failures count as failures, not as absent rows -- otherwise the rate
    rises precisely when the model fails more often.
    """
    scheduled = _neq(rows)
    return Rate(
        "end_to_end_witness_success",
        sum(1 for row in scheduled if row["witness_success"] is True),
        len(scheduled),
        "all scheduled ground-truth non-equivalent primary cells",
    )


def conditional_witness_validity(rows: Sequence[dict[str, Any]]) -> Rate:
    """Over parsed outputs that predicted non-equivalent. **Not** an end-to-end rate."""
    eligible = [
        row for row in rows if row["parse_valid"] and row["witness_valid"] is not None
    ]
    return Rate(
        "conditional_witness_validity",
        sum(1 for row in eligible if row["witness_valid"] is True),
        len(eligible),
        "parsed outputs predicting non-equivalent",
    )


def shortest_witness_success(rows: Sequence[dict[str, Any]]) -> Rate:
    """Descriptive only (D3). A valid longer witness still succeeds on H1 and H2."""
    scheduled = _neq(rows)
    return Rate(
        "shortest_witness_success",
        sum(
            1
            for row in scheduled
            if row["witness_success"] is True and row["witness_shortest"] is True
        ),
        len(scheduled),
        "all scheduled ground-truth non-equivalent primary cells (descriptive)",
    )


def balanced_decision_accuracy(rows: Sequence[dict[str, Any]]) -> tuple[Rate, Rate, float | None]:
    """Secondary metric: the mean of the two class accuracies.

    Returned as both class rates plus their mean, because the mean alone hides the case the
    baselines exploit -- a constant answer scores 100% on one class and 0% on the other.
    """
    equivalent = [row for row in rows if row["oracle_equivalent"]]
    non_equivalent = _neq(rows)
    eq_rate = Rate(
        "decision_accuracy_equivalent",
        sum(1 for row in equivalent if row["decision_correct"]),
        len(equivalent),
        "all scheduled ground-truth equivalent primary cells",
    )
    neq_rate = Rate(
        "decision_accuracy_non_equivalent",
        sum(1 for row in non_equivalent if row["decision_correct"]),
        len(non_equivalent),
        "all scheduled ground-truth non-equivalent primary cells",
    )
    if eq_rate.value is None or neq_rate.value is None:
        return eq_rate, neq_rate, None
    return eq_rate, neq_rate, (eq_rate.value + neq_rate.value) / 2


def trace_correctness(rows: Sequence[dict[str, Any]]) -> Rate:
    """P1 non-equivalent cells that supplied a witness. Separate from witness success."""
    eligible = [
        row
        for row in rows
        if row["prompt_condition"] == "P1"
        and not row["oracle_equivalent"]
        and row["trace_valid"] is not None
    ]
    return Rate(
        "trace_correctness",
        sum(1 for row in eligible if row["trace_valid"] is True),
        len(eligible),
        "P1 non-equivalent cells with a witness",
    )


def parse_failure_rate(rows: Sequence[dict[str, Any]]) -> Rate:
    """Over *received* outputs. Transport failures are reported separately (§9)."""
    received = [row for row in rows if row["status"] != TRANSPORT_FAILED]
    return Rate(
        "parse_failure_rate",
        sum(1 for row in received if not row["parse_valid"]),
        len(received),
        "received final-answer outputs, including empty answers",
    )


# --------------------------------------------------------------------------- paired R0/R1


def _paired_cells(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Group rows into R0/R1 pairs keyed by (pair, prompt, model)."""
    pairs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["pair_id"], row["prompt_condition"], row["model_id"])
        pairs.setdefault(key, {})[row["rendering"]] = row
    return {key: cell for key, cell in pairs.items() if {"R0", "R1"} <= set(cell)}


def decision_flip_rate(rows: Sequence[dict[str, Any]]) -> Rate:
    """H3: over R0/R1 cells where **both** sides produced a strict-valid Boolean.

    A flip is only meaningful between two answers that exist, so the denominator is
    conditional -- which is exactly why §9 requires its coverage to be reported alongside.
    """
    eligible = [
        cell
        for cell in _paired_cells(rows).values()
        if cell["R0"]["parse_valid"] and cell["R1"]["parse_valid"]
    ]
    flips = sum(
        1
        for cell in eligible
        if cell["R0"]["decision_correct"] != cell["R1"]["decision_correct"]
        or _decision(cell["R0"]) != _decision(cell["R1"])
    )
    return Rate(
        "decision_flip_rate",
        flips,
        len(eligible),
        "R0/R1 cell pairs with two strict-valid Boolean decisions",
    )


def _decision(row: dict[str, Any]) -> bool | None:
    """The model's Boolean, reconstructed from correctness and ground truth."""
    if not row["parse_valid"]:
        return None
    return row["oracle_equivalent"] if row["decision_correct"] else (
        not row["oracle_equivalent"]
    )


def orbit_success(rows: Sequence[dict[str, Any]]) -> Rate:
    """H3: both representations correct, over **all** scheduled R0/R1 cell pairs.

    Unlike the flip rate this denominator is unconditional, so a parse failure cannot help.
    """
    pairs = _paired_cells(rows)
    return Rate(
        "orbit_success",
        sum(
            1
            for cell in pairs.values()
            if cell["R0"]["decision_correct"] and cell["R1"]["decision_correct"]
        ),
        len(pairs),
        "all scheduled R0/R1 cell pairs",
    )
