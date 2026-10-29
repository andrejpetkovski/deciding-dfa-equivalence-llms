"""Tests for strict parsing and deterministic grading (M8, Step 10, D3)."""

from __future__ import annotations

import json

import pytest

from src.dfa import DFA
from src.grading import (
    ParseOutcome,
    forgiving_parse,
    grade,
    strict_parse,
)
from src.rendering import render_R0
from tests.helpers import BINARY

# A minimal non-equivalent pair: A accepts words ending in 1, B accepts nothing.
# The shortest distinguishing word is "1", at depth 1.
DFA_A = DFA.build(n_states=2, alphabet=BINARY, start=0, accepting=[1], delta=[[0, 1], [0, 1]])
DFA_B = DFA.build(n_states=2, alphabet=BINARY, start=0, accepting=[], delta=[[0, 1], [0, 1]])
LABELS_A = render_R0(DFA_A, "A").labels
LABELS_B = render_R0(DFA_B, "B").labels


def answer(**fields) -> str:
    return json.dumps(fields)


def parse_p0(text):
    return strict_parse(text, "P0", BINARY)


def parse_p1(text):
    return strict_parse(text, "P1", BINARY)


# --------------------------------------------------------------------------- strict parser


def test_a_clean_p0_answer_parses():
    parsed = parse_p0(answer(equivalent=False, witness="1"))
    assert parsed.ok and parsed.outcome == ParseOutcome.OK
    assert parsed.equivalent is False and parsed.witness == "1"


def test_a_clean_equivalent_answer_parses():
    parsed = parse_p0(answer(equivalent=True, witness=None))
    assert parsed.ok and parsed.equivalent is True and parsed.witness is None


def test_surrounding_whitespace_is_tolerated():
    """Whitespace carries no content; a fence or a sentence does."""
    assert parse_p0("\n  " + answer(equivalent=True, witness=None) + "  \n").ok


def test_a_code_fence_is_a_parse_failure():
    """§8.5: fences are a strict-parse failure, however common they are in practice."""
    fenced = "```json\n" + answer(equivalent=True, witness=None) + "\n```"
    assert parse_p0(fenced).outcome == ParseOutcome.NOT_JSON


def test_trailing_prose_is_a_parse_failure():
    text = answer(equivalent=True, witness=None) + "\n\nHope this helps!"
    assert parse_p0(text).outcome == ParseOutcome.NOT_JSON


def test_an_empty_answer_is_a_parse_failure():
    assert parse_p0("").outcome == ParseOutcome.EMPTY
    assert parse_p0(None).outcome == ParseOutcome.EMPTY
    assert parse_p0("   ").outcome == ParseOutcome.EMPTY


def test_a_json_array_is_not_an_object():
    assert parse_p0("[1, 2]").outcome == ParseOutcome.NOT_AN_OBJECT


def test_a_missing_key_fails():
    assert parse_p0(json.dumps({"equivalent": True})).outcome == ParseOutcome.WRONG_KEYS


def test_an_extra_key_fails():
    """The schema sets additionalProperties false; an added field is not the contract."""
    text = answer(equivalent=True, witness=None, confidence=0.9)
    assert parse_p0(text).outcome == ParseOutcome.WRONG_KEYS


def test_p1_keys_are_required_under_p1():
    assert parse_p1(answer(equivalent=True, witness=None)).outcome == ParseOutcome.WRONG_KEYS


def test_p0_rejects_p1_keys():
    text = answer(equivalent=True, witness=None, trace_A=None, trace_B=None)
    assert parse_p0(text).outcome == ParseOutcome.WRONG_KEYS


@pytest.mark.parametrize("value", ["true", 1, None, "yes"])
def test_a_non_boolean_decision_fails(value):
    assert parse_p0(answer(equivalent=value, witness=None)).outcome in (
        ParseOutcome.WRONG_TYPE,
        ParseOutcome.INCONSISTENT_FIELDS,
    )


def test_a_non_string_witness_fails():
    assert parse_p0(answer(equivalent=False, witness=101)).outcome == ParseOutcome.WRONG_TYPE


def test_a_witness_outside_the_alphabet_fails():
    parsed = parse_p0(answer(equivalent=False, witness="012"))
    assert parsed.outcome == ParseOutcome.ILLEGAL_SYMBOL
    assert "2" in parsed.detail


def test_the_empty_word_is_a_legal_witness():
    """`""` is the empty word, not a missing one."""
    parsed = parse_p0(answer(equivalent=False, witness=""))
    assert parsed.ok and parsed.witness == ""


# --------------------------------------------------------------------------- consistency


def test_equivalent_with_a_witness_is_inconsistent():
    parsed = parse_p0(answer(equivalent=True, witness="1"))
    assert parsed.outcome == ParseOutcome.INCONSISTENT_FIELDS


def test_non_equivalent_without_a_witness_is_inconsistent():
    parsed = parse_p0(answer(equivalent=False, witness=None))
    assert parsed.outcome == ParseOutcome.INCONSISTENT_FIELDS


def test_p1_non_equivalent_needs_both_traces():
    text = answer(equivalent=False, witness="1", trace_A=["q00", "q01"], trace_B=None)
    assert parse_p1(text).outcome == ParseOutcome.INCONSISTENT_FIELDS


def test_p1_equivalent_must_not_carry_traces():
    text = answer(equivalent=True, witness=None, trace_A=["q00"], trace_B=None)
    assert parse_p1(text).outcome == ParseOutcome.INCONSISTENT_FIELDS


def test_a_non_string_trace_entry_fails():
    text = answer(equivalent=False, witness="1", trace_A=[0, 1], trace_B=["q00", "q01"])
    assert parse_p1(text).outcome == ParseOutcome.WRONG_TYPE


def test_an_unknown_condition_raises():
    with pytest.raises(ValueError, match=r"unknown prompt condition"):
        strict_parse("{}", "P9", BINARY)


# --------------------------------------------------------------------------- forgiving


def test_the_forgiving_parser_strips_one_fence():
    fenced = "```json\n" + answer(equivalent=True, witness=None) + "\n```"
    assert not parse_p0(fenced).ok
    assert forgiving_parse(fenced, "P0", BINARY).ok


def test_the_forgiving_parser_does_not_repair_anything_else():
    """One fence is the only concession; it is not a general rescue path."""
    assert not forgiving_parse(
        answer(equivalent=True, witness=None) + " thanks!", "P0", BINARY
    ).ok
    assert not forgiving_parse(json.dumps({"equivalent": True}), "P0", BINARY).ok
    assert not forgiving_parse("Sure! " + answer(equivalent=True, witness=None), "P0", BINARY).ok


# --------------------------------------------------------------------------- grading


def _grade(text, condition="P0", oracle_equivalent=False, oracle_depth=1):
    parsed = strict_parse(text, condition, BINARY)
    return grade(parsed, DFA_A, DFA_B, oracle_equivalent, oracle_depth, LABELS_A, LABELS_B)


def test_a_correct_shortest_witness_succeeds():
    result = _grade(answer(equivalent=False, witness="1"))
    assert result.decision_correct
    assert result.witness_success is True
    assert result.witness_valid is True
    assert result.witness_shortest is True


def test_a_valid_longer_witness_still_succeeds():
    """D3: the prompt asks for *a* witness. Scoring shortestness as failure would grade
    against an instruction that was never given."""
    result = _grade(answer(equivalent=False, witness="001"))
    assert result.witness_success is True
    assert result.witness_valid is True
    assert result.witness_shortest is False  # descriptive only


def test_an_invalid_witness_fails_even_with_the_right_decision():
    """Both machines reject "0", so it distinguishes nothing."""
    result = _grade(answer(equivalent=False, witness="0"))
    assert result.decision_correct is True
    assert result.witness_valid is False
    assert result.witness_success is False


def test_claiming_equivalence_on_a_non_equivalent_pair_fails():
    result = _grade(answer(equivalent=True, witness=None))
    assert result.decision_correct is False
    assert result.witness_success is False
    assert result.witness_valid is None


def test_a_parse_failure_is_a_failed_scheduled_cell_not_a_missing_row():
    """Rule 5 and D3: it must stay in the denominator."""
    result = _grade("```\n{}\n```")
    assert result.parse_valid is False
    assert result.decision_correct is False
    assert result.witness_success is False  # counted, not None


def test_equivalent_cells_have_no_witness_outcome():
    """H1's outcome is defined only on ground-truth non-equivalent pairs."""
    result = _grade(
        answer(equivalent=True, witness=None), oracle_equivalent=True, oracle_depth=None
    )
    assert result.decision_correct is True
    assert result.witness_success is None


def test_a_parse_failure_on_an_equivalent_cell_has_no_witness_outcome():
    result = _grade("garbage", oracle_equivalent=True, oracle_depth=None)
    assert result.parse_valid is False
    assert result.witness_success is None
    assert result.decision_correct is False


# --------------------------------------------------------------------------- traces


def test_correct_traces_validate():
    text = answer(
        equivalent=False,
        witness="1",
        trace_A=[LABELS_A[0], LABELS_A[1]],
        trace_B=[LABELS_B[0], LABELS_B[1]],
    )
    result = _grade(text, condition="P1")
    assert result.trace_valid is True
    assert result.first_wrong_step is None
    assert result.witness_success is True


def test_a_wrong_trace_does_not_change_witness_success():
    """D3: requiring a valid trace would make the outcome mean something different under P1
    than under P0 -- exactly the comparison H2 is trying to make."""
    text = answer(
        equivalent=False,
        witness="1",
        trace_A=[LABELS_A[0], LABELS_A[0]],  # wrong second state
        trace_B=[LABELS_B[0], LABELS_B[1]],
    )
    result = _grade(text, condition="P1")
    assert result.trace_valid is False
    assert result.first_wrong_step == 1
    assert result.witness_success is True  # unchanged


def test_an_invented_state_label_is_a_wrong_trace_not_a_crash():
    text = answer(
        equivalent=False,
        witness="1",
        trace_A=["qZZ", "qYY"],
        trace_B=[LABELS_B[0], LABELS_B[1]],
    )
    result = _grade(text, condition="P1")
    assert result.trace_valid is False
    assert result.detail.get("trace_A_unmapped") is True


def test_a_short_trace_is_wrong_at_the_right_index():
    """A trace of a word w has length |w|+1; a missing final state diverges at that index."""
    text = answer(
        equivalent=False,
        witness="1",
        trace_A=[LABELS_A[0]],
        trace_B=[LABELS_B[0], LABELS_B[1]],
    )
    result = _grade(text, condition="P1")
    assert result.trace_valid is False
    assert result.first_wrong_step == 1


def test_traces_are_mapped_through_the_rendering_bijection():
    """The model answers in the labels it was shown. Under R1 those are nonces, and grading
    against R0 labels would mark every correct trace wrong."""
    from src.rendering import derive_rng, render_R1

    r1_a = render_R1(DFA_A, "A", derive_rng(1, "R1", "A"))
    r1_b = render_R1(DFA_B, "B", derive_rng(1, "R1", "B"))
    text = answer(
        equivalent=False,
        witness="1",
        trace_A=[r1_a.labels[0], r1_a.labels[1]],
        trace_B=[r1_b.labels[0], r1_b.labels[1]],
    )
    parsed = strict_parse(text, "P1", BINARY)
    correct = grade(parsed, DFA_A, DFA_B, False, 1, r1_a.labels, r1_b.labels)
    assert correct.trace_valid is True

    # The same answer graded against the R0 bijection is wrong, which is why gate 6 verifies
    # the manifest's mapping rather than mere isomorphism.
    mismatched = grade(parsed, DFA_A, DFA_B, False, 1, LABELS_A, LABELS_B)
    assert mismatched.trace_valid is False


def test_grade_serializes_flatly():
    result = _grade(answer(equivalent=False, witness="1"))
    row = result.to_dict()
    assert row["witness_success"] is True
    assert row["parse_outcome"] == ParseOutcome.OK
    assert json.loads(json.dumps(row))["witness_length"] == 1
