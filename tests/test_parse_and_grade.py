"""The raw-log-to-graded-row join (M8 CLI).

Every response body here is wrapped in the **real** gateway envelope taken from a committed
probe response, with only ``message.content`` replaced. A test that fed bare answer strings to
the grader would skip the adapter entirely, which is the layer most likely to be wrong.

The envelope fixture is still nemotron's, because the wire format is what it exercises and
that has not changed. The default *model id* is `mimo-v2.5-free`, one of the two the design
retains after D5.5 -- cells for an undeclared model are deliberately skipped, so a helper
defaulting to a withdrawn model would silently produce empty results.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.dfa import from_record
from src.io_utils import read_jsonl
from src.parse_and_grade import build_rows
from src.simulate import accepts

CONFIG_PATH = "configs/pilot_revised.yaml"
ENVELOPE = json.loads(
    (Path(__file__).parent / "fixtures" / "gateway" / "nemotron_ok.json").read_text(
        encoding="utf-8"
    )
)["body"]


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def pairs() -> list[dict[str, Any]]:
    return list(read_jsonl(Path("data/pairs/pilot_v2/pairs.jsonl")))


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, Any]]:
    return list(read_jsonl(Path("data/renderings/pilot_v2/manifest.jsonl")))


def neq_pair(pairs: list[dict[str, Any]], min_depth: int = 1) -> dict[str, Any]:
    for record in pairs:
        if not record["oracle_equivalent"] and (record["oracle_depth"] or 0) >= min_depth:
            return record
    raise AssertionError("no non-equivalent pair in the pilot set")


def eq_pair(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    for record in pairs:
        if record["oracle_equivalent"]:
            return record
    raise AssertionError("no equivalent pair in the pilot set")


def attempt(
    record: dict[str, Any],
    content: str | None,
    *,
    rendering: str = "R0",
    condition: str = "P0",
    ok: bool = True,
    model: str = "mimo-v2.5-free",
    attempt_no: int = 1,
    terminal: bool = True,
) -> dict[str, Any]:
    """One raw-log line carrying a real gateway envelope."""
    body = copy.deepcopy(ENVELOPE)
    body["choices"][0]["message"]["content"] = content
    return {
        "run_id": f"pilot_v2|{record['pair_id']}|{rendering}|{condition}|{model}|s0",
        "dataset_id": "pilot_v2",
        "pair_id": record["pair_id"],
        "rendering": rendering,
        "prompt_condition": condition,
        "sample_index": 0,
        "attempt": attempt_no,
        "terminal": terminal,
        "requested_model": model,
        "returned_model": model,
        "ok": ok,
        "status": 200 if ok else 503,
        "finish_reason": "stop",
        "usage": body.get("usage"),
        "latency_ms": 42.0,
        "response_id": body.get("id"),
        "raw_body": body if ok else {"error": {"message": "Endpoint is unavailable."}},
        "failure_kind": None if ok else "server_unavailable:503",
    }


def grade_one(config, pairs, manifest, raw: list[dict[str, Any]]):
    rows, problems = build_rows(config, raw, pairs, manifest)
    return rows, problems


# --------------------------------------------------------------------------- happy path


def test_a_correct_shortest_witness_succeeds(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    answer = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    rows, problems = grade_one(config, pairs, manifest, [attempt(record, answer)])

    assert problems == []
    row = rows[0]
    assert row["status"] == "received"
    assert row["parse_valid"]
    assert row["decision_correct"]
    assert row["witness_success"]
    assert row["witness_valid"]
    assert row["witness_shortest"]


def test_a_valid_but_longer_witness_still_succeeds(config, pairs, manifest) -> None:
    """D3: the prompt asks for *a* witness. Shortestness is descriptive, not required."""
    record = neq_pair(pairs, min_depth=1)
    dfa_a, dfa_b = from_record(record["dfa_A"]), from_record(record["dfa_B"])
    depth = record["oracle_depth"]

    longer = None
    for length in range(depth + 1, depth + 6):
        for letters in itertools.product("01", repeat=length):
            word = "".join(letters)
            if accepts(dfa_a, word) != accepts(dfa_b, word):
                longer = word
                break
        if longer:
            break
    assert longer is not None and len(longer) > depth

    answer = json.dumps({"equivalent": False, "witness": longer})
    rows, _ = grade_one(config, pairs, manifest, [attempt(record, answer)])

    assert rows[0]["witness_success"] is True
    assert rows[0]["witness_shortest"] is False


def test_an_equivalent_pair_has_no_witness_outcome(config, pairs, manifest) -> None:
    record = eq_pair(pairs)
    answer = json.dumps({"equivalent": True, "witness": None})
    rows, _ = grade_one(config, pairs, manifest, [attempt(record, answer)])

    assert rows[0]["decision_correct"]
    # The outcome is undefined, not False: there is no witness to produce.
    assert rows[0]["witness_success"] is None


# --------------------------------------------------------------------------- failures


def test_a_fenced_answer_fails_strictly_and_passes_only_in_sensitivity(
    config, pairs, manifest
) -> None:
    record = neq_pair(pairs)
    inner = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    rows, _ = grade_one(
        config, pairs, manifest, [attempt(record, f"```json\n{inner}\n```")]
    )
    row = rows[0]

    assert not row["parse_valid"]
    assert row["witness_success"] is False  # counts as wrong (rule 5)
    # The forgiving verdict exists but lives in columns no primary metric reads.
    assert row["sensitivity_parse_valid"]
    assert row["sensitivity_witness_success"] is True


def test_an_invalid_witness_is_a_failed_certificate_not_a_parse_error(
    config, pairs, manifest
) -> None:
    record = neq_pair(pairs)
    # A legal-alphabet word that does not distinguish the machines.
    dfa_a, dfa_b = from_record(record["dfa_A"]), from_record(record["dfa_B"])
    bad = next(
        "".join(letters)
        for length in range(1, 6)
        for letters in itertools.product("01", repeat=length)
        if accepts(dfa_a, "".join(letters)) == accepts(dfa_b, "".join(letters))
    )
    answer = json.dumps({"equivalent": False, "witness": bad})
    rows, _ = grade_one(config, pairs, manifest, [attempt(record, answer)])
    row = rows[0]

    assert row["parse_valid"]
    assert row["decision_correct"]  # it did say non-equivalent
    assert row["witness_valid"] is False
    assert row["witness_success"] is False


def test_a_transport_failure_stays_in_the_denominator(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    rows, _ = grade_one(config, pairs, manifest, [attempt(record, None, ok=False)])
    row = rows[0]

    assert row["status"] == "transport_failed"
    # False, never None: a missing row would silently shrink the denominator.
    assert row["witness_success"] is False
    assert row["parse_outcome"] == "transport_failed"
    assert row["failure_kind"] == "server_unavailable:503"


def test_a_reasoning_only_response_is_a_parse_failure(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    raw = attempt(record, None)
    raw["raw_body"]["choices"][0]["finish_reason"] = "length"
    raw["raw_body"]["choices"][0]["message"]["reasoning_content"] = "thinking..."
    rows, _ = grade_one(config, pairs, manifest, [raw])
    row = rows[0]

    assert row["status"] == "received"
    assert not row["parse_valid"]
    assert row["witness_success"] is False
    assert row["truncated"]
    assert row["reasoning_present"]


# --------------------------------------------------------------------------- bijection


def test_a_p1_trace_is_graded_through_the_rendering_the_model_saw(
    config, pairs, manifest
) -> None:
    """The same trace under the wrong bijection must not score as correct (gate 6)."""
    from src.simulate import trace as simulate_trace

    record = neq_pair(pairs)
    witness = record["oracle_witness"]
    dfa_a, dfa_b = from_record(record["dfa_A"]), from_record(record["dfa_B"])
    labels = {row["rendering"]: row for row in manifest if row["pair_id"] == record["pair_id"]}

    r1 = labels["R1"]
    answer = json.dumps(
        {
            "equivalent": False,
            "witness": witness,
            "trace_A": [r1["labels_A"][state] for state in simulate_trace(dfa_a, witness)],
            "trace_B": [r1["labels_B"][state] for state in simulate_trace(dfa_b, witness)],
        }
    )

    correct, _ = grade_one(
        config, pairs, manifest, [attempt(record, answer, rendering="R1", condition="P1")]
    )
    assert correct[0]["trace_valid"] is True

    # The identical answer, claimed to be an R0 cell, is graded through R0's labels and fails.
    wrong, _ = grade_one(
        config, pairs, manifest, [attempt(record, answer, rendering="R0", condition="P1")]
    )
    assert wrong[0]["trace_valid"] is False


# --------------------------------------------------------------------------- joining


def test_retries_are_summarised_onto_the_row(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    answer = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    raw = [
        attempt(record, None, ok=False, attempt_no=1, terminal=False),
        attempt(record, None, ok=False, attempt_no=2, terminal=False),
        attempt(record, answer, attempt_no=3, terminal=True),
    ]
    rows, _ = grade_one(config, pairs, manifest, raw)

    assert len(rows) == 1  # three attempts, one cell
    assert rows[0]["attempts"] == 3
    assert rows[0]["transport_failures"] == 2
    assert rows[0]["witness_success"]


def test_a_missing_manifest_row_is_a_reported_problem_not_a_guess(
    config, pairs, manifest
) -> None:
    record = neq_pair(pairs)
    answer = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    without_r1 = [row for row in manifest if row["rendering"] != "R1"]

    rows, problems = build_rows(
        config, [attempt(record, answer, rendering="R1")], pairs, without_r1
    )
    assert rows == []
    assert any("label bijection is unknown" in problem for problem in problems)


def test_an_unknown_pair_is_reported(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    raw = attempt(record, "{}")
    raw["pair_id"] = "p_999999"
    rows, problems = build_rows(config, [raw], pairs, manifest)

    assert rows == []
    assert any("not in pairs.jsonl" in problem for problem in problems)


# ------------------------------------------------- a model withdrawn from the design


def test_cells_for_an_undeclared_model_are_skipped_loudly(config, pairs, manifest) -> None:
    """A model withdrawn from the design must not enter the denominators.

    When `nemotron-3-ultra-free` was withdrawn on 2026-09-08 the raw log already held 13 of
    its cells, every one a transport failure. Grading them would have contributed 13 failures
    for a model under no test -- and silently, since they look like ordinary rows.
    """
    record = neq_pair(pairs)
    answer = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    kept = attempt(record, answer, model="mimo-v2.5-free")
    dropped = attempt(record, answer, model="nemotron-3-ultra-free")

    rows, problems = build_rows(config, [kept, dropped], pairs, manifest)

    assert [row["model_id"] for row in rows] == ["mimo-v2.5-free"]
    assert any("nemotron-3-ultra-free" in problem for problem in problems)
    assert any("no longer declares" in problem for problem in problems)


def test_the_skip_counts_the_cells_it_dropped(config, pairs, manifest) -> None:
    record = neq_pair(pairs)
    answer = json.dumps({"equivalent": False, "witness": record["oracle_witness"]})
    dropped = [
        attempt(record, answer, model="nemotron-3-ultra-free", rendering=r, condition=c)
        for r, c in (("R0", "P0"), ("R1", "P0"), ("R0", "P1"))
    ]

    rows, problems = build_rows(config, dropped, pairs, manifest)
    assert rows == []
    assert any("skipped 3 cell(s)" in problem for problem in problems)


def test_the_two_retained_models_are_the_declared_design(config) -> None:
    """D5.5: two official releases from two distinct developers."""
    from src.parse_and_grade import _declared_models

    assert _declared_models(config) == {"mimo-v2.5-free", "qwen3.8-27b"}
