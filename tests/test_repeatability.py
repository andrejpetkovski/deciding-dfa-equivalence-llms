"""Repeatability audit measures (D14).

Every case here is hand-built with a known answer, because the whole value of this module is
that its numbers are trustworthy. The measures it reports are easy to get subtly wrong in
ways that would inflate agreement -- comparing an incomplete cell, silently dropping
unparseable samples, or treating equal witness *lengths* as equal witnesses -- so each of
those is pinned by a test rather than assumed.
"""

from __future__ import annotations

from analysis.repeatability import build_cells, repeatability_report


def _row(sample: int, **over):
    """One graded row with the fields the audit reads."""
    row = {
        "pair_id": "p_000001",
        "rendering": "R0",
        "prompt_condition": "P0",
        "model_id": "m/one",
        "family": "NEQ_DEPTH",
        "oracle_depth": 4,
        "oracle_equivalent": False,
        "sample_index": sample,
        "parse_valid": True,
        "decision_correct": True,
        "witness_success": True,
        "witness_valid": True,
        "usage": {"completion_tokens": 100},
        "run_id": f"main|p_000001|R0|P0|m/one|s{sample}",
    }
    row.update(over)
    return row


def _cells(rows, witnesses=None):
    return build_cells(rows, expected_samples=3, witnesses=witnesses)


# --------------------------------------------------------------------------- completeness


def test_a_cell_missing_a_sample_is_excluded_entirely():
    """Comparing 2 of 3 scheduled samples would overstate agreement."""
    assert _cells([_row(0), _row(1)]) == []


def test_a_complete_cell_is_included():
    cells = _cells([_row(0), _row(1), _row(2)])
    assert len(cells) == 1
    assert cells[0]["pair_id"] == "p_000001"


def test_cells_are_keyed_on_all_four_fields():
    """Two different renderings of one pair are different cells, not repeats."""
    rows = [_row(i) for i in range(3)] + [_row(i, rendering="R1") for i in range(3)]
    assert len({(c["rendering"]) for c in _cells(rows)}) == 2


# --------------------------------------------------------------------------- agreement


def test_three_identical_decisions_agree():
    report = repeatability_report(_cells([_row(0), _row(1), _row(2)]))
    entry = report["overall"]["decision_agreement_unconditional"]
    assert entry["numerator"] == 1 and entry["denominator"] == 1


def test_one_differing_decision_breaks_agreement():
    rows = [_row(0), _row(1), _row(2, decision_correct=False)]
    report = repeatability_report(_cells(rows))
    assert report["overall"]["decision_agreement_unconditional"]["numerator"] == 0


def test_all_three_unparsed_agree_unconditionally_but_are_excluded_when_conditioning():
    """Rule 5: a parse failure is an observed outcome, not a missing one.

    Consistently failing to parse IS a true statement about the deployment, so the
    unconditional measure counts it -- and the conditional measure must not, or the two
    numbers would be the same and the distinction pointless.
    """
    rows = [_row(i, parse_valid=False, decision_correct=False) for i in range(3)]
    overall = repeatability_report(_cells(rows))["overall"]

    assert overall["decision_agreement_unconditional"]["numerator"] == 1
    assert overall["decision_agreement_given_all_parsed"] is None  # no eligible cell
    assert overall["all_samples_parsed"]["numerator"] == 0


def test_a_parse_failure_among_parsed_samples_counts_as_disagreement():
    """An unparseable sample is a different outcome from a Boolean, not a tie-breaker."""
    rows = [_row(0), _row(1), _row(2, parse_valid=False, decision_correct=False)]
    overall = repeatability_report(_cells(rows))["overall"]
    assert overall["decision_agreement_unconditional"]["numerator"] == 0


def test_witness_success_agreement_is_restricted_to_non_equivalent_cells():
    """witness_success is None on equivalent pairs; including them would fake agreement."""
    eq = [
        _row(i, family="EQ_ISO", oracle_equivalent=True, witness_success=None,
             witness_valid=None, pair_id="p_eq", run_id=f"r{i}")
        for i in range(3)
    ]
    overall = repeatability_report(_cells(eq))["overall"]
    assert overall["witness_success_agreement"] is None
    assert overall["cells"] == 1


# --------------------------------------------------------------------------- witnesses


def test_same_verdict_with_different_witnesses_is_not_an_identical_witness():
    """The distinction this module exists to make: stable answer, unstable generation."""
    rows = [_row(0), _row(1), _row(2)]
    witnesses = {
        "main|p_000001|R0|P0|m/one|s0": "01",
        "main|p_000001|R0|P0|m/one|s1": "10",  # same length, different string
        "main|p_000001|R0|P0|m/one|s2": "01",
    }
    overall = repeatability_report(_cells(rows, witnesses))["overall"]

    assert overall["witness_success_agreement"]["numerator"] == 1  # all succeeded
    assert overall["identical_witness_string"]["numerator"] == 0  # but not the same one


def test_identical_witnesses_are_reported_as_identical():
    rows = [_row(0), _row(1), _row(2)]
    witnesses = {f"main|p_000001|R0|P0|m/one|s{i}": "0110" for i in range(3)}
    overall = repeatability_report(_cells(rows, witnesses))["overall"]
    assert overall["identical_witness_string"]["numerator"] == 1


def test_witness_comparison_needs_every_sample_to_have_a_valid_witness():
    rows = [_row(0), _row(1), _row(2, witness_valid=False, witness_success=False)]
    witnesses = {f"main|p_000001|R0|P0|m/one|s{i}": "01" for i in range(3)}
    overall = repeatability_report(_cells(rows, witnesses))["overall"]
    assert overall["identical_witness_string"] is None


# --------------------------------------------------------------------------- generation


def test_identical_token_counts_are_detected():
    overall = repeatability_report(_cells([_row(0), _row(1), _row(2)]))["overall"]
    assert overall["identical_completion_token_counts"]["numerator"] == 1
    assert overall["completion_token_spread"]["median_relative_range"] == 0.0


def test_token_spread_is_the_relative_range():
    """452 / 1222 / 2414 were the real calibration counts for one identical request."""
    rows = [
        _row(0, usage={"completion_tokens": 452}),
        _row(1, usage={"completion_tokens": 1222}),
        _row(2, usage={"completion_tokens": 2414}),
    ]
    overall = repeatability_report(_cells(rows))["overall"]

    assert overall["identical_completion_token_counts"]["numerator"] == 0
    # (2414 - 452) / 1222
    assert abs(overall["completion_token_spread"]["median_relative_range"] - 1.6056) < 1e-3


# --------------------------------------------------------------------------- reporting


def test_per_model_blocks_partition_the_cells():
    rows = [_row(i) for i in range(3)] + [
        _row(i, model_id="m/two", run_id=f"t{i}") for i in range(3)
    ]
    report = repeatability_report(_cells(rows))

    assert set(report["by_model"]) == {"m/one", "m/two"}
    assert sum(b["cells"] for b in report["by_model"].values()) == report["overall"]["cells"]


def test_the_report_states_that_it_is_not_a_determinism_check():
    """The framing is load-bearing and must travel with the numbers."""
    note = repeatability_report(_cells([_row(0), _row(1), _row(2)]))["note"]
    assert "never a determinism check" in note.lower()
