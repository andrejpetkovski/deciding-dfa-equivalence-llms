"""The pre-registered H2/H3 secondaries (D13).

The property under test is the primary-sample rule. Every confirmatory estimate is computed on
``sample_index = 0``; the repeatability audit's calls measure operational variation and are not
extra observations. The H3 McNemar test once took its R0/R1 pairs from *every* graded row, and
because pairs are keyed by (pair, prompt, model) without the sample index, a repeat silently
replaced the primary answer in each audited cell. The committed artifact reported 1,057 pairs
where the declared population has 1,047, and a p-value of 0.6989 where the declared test gives
0.8957. Nothing covered this module, so nothing noticed. These tests make that failure loud.
"""

from __future__ import annotations

from analysis.metrics import decision_flip_rate, primary_rows
from analysis.preregistered_secondaries import _neq_primary, h3_pooled_mcnemar


def _row(rendering: str, sample: int, correct: bool, **over):
    """One graded cell carrying the fields the McNemar test and the flip rate read."""
    row = {
        "run_id": f"main|p_000150|{rendering}|P1|m/one|s{sample}",
        "pair_id": "p_000150",
        "family": "EQ_EXPAND",
        "rendering": rendering,
        "prompt_condition": "P1",
        "model_id": "m/one",
        "sample_index": sample,
        "status": "received",
        "parse_valid": True,
        "oracle_equivalent": True,
        "oracle_depth": None,
        "decision_correct": correct,
        "witness_success": None,
    }
    row.update(over)
    return row


def _tally(result):
    return (
        result["both_correct"],
        result["only_r0_correct"],
        result["only_r1_correct"],
        result["neither_correct"],
    )


def _audited_cell():
    """Sample 0 has R0 right and R1 wrong; both repeats have it the other way round."""
    primary = [_row("R0", 0, True), _row("R1", 0, False)]
    repeats = [_row("R0", s, False) for s in (1, 2)] + [_row("R1", s, True) for s in (1, 2)]
    return primary, repeats


# --------------------------------------------------------------------------- H3 McNemar


def test_repeat_samples_do_not_replace_the_primary_answer():
    """The regression itself: an audited cell is scored on its primary answer."""
    primary, repeats = _audited_cell()
    assert _tally(h3_pooled_mcnemar(primary + repeats)) == (0, 1, 0, 0)


def test_the_result_does_not_depend_on_where_repeats_sit_in_the_input():
    """The graded file lists repeats after sample 0; the estimate must not care."""
    primary, repeats = _audited_cell()
    assert h3_pooled_mcnemar(repeats + primary) == h3_pooled_mcnemar(primary + repeats)


def test_adding_repeats_leaves_the_test_identical_to_the_primary_only_test():
    cells, repeats = [], []
    for index, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
        pair = f"p_{index:06d}"
        cells += [_row("R0", 0, a, pair_id=pair), _row("R1", 0, b, pair_id=pair)]
        # Every repeat is right on both renderings. Flipped copies of these four cells would
        # be a useless fixture: they permute the tally onto itself, so a leak would pass
        # unnoticed (it did, on the first draft). All-correct repeats turn a leak into (4,0,0,0).
        repeats += [
            _row(rendering, s, True, pair_id=pair) for rendering in ("R0", "R1") for s in (1, 2)
        ]
    assert _tally(h3_pooled_mcnemar(cells)) == (1, 1, 1, 1)
    assert h3_pooled_mcnemar(cells + repeats) == h3_pooled_mcnemar(cells)


def test_repeats_alone_are_not_a_population():
    result = h3_pooled_mcnemar([_row("R0", 1, True), _row("R1", 1, False)])
    assert _tally(result) == (0, 0, 0, 0)
    assert result["estimable"] is False


def test_a_pair_enters_only_when_both_renderings_parsed():
    rows = [_row("R0", 0, True), _row("R1", 0, False, parse_valid=False)]
    assert _tally(h3_pooled_mcnemar(rows)) == (0, 0, 0, 0)


def test_the_mcnemar_population_is_the_flip_rate_population():
    """The invariant whose violation exposed the bug: 1,057 McNemar pairs against 1,047 flips.

    Here sample 0's R1 failed to parse, so the pair is outside both populations -- but its
    repeats parsed, so a test that let repeats through would count one pair the flip rate
    does not.
    """
    rows = [
        _row("R0", 0, True, pair_id="p_000001"),
        _row("R1", 0, True, pair_id="p_000001"),
        _row("R0", 0, True, pair_id="p_000002"),
        _row("R1", 0, False, pair_id="p_000002", parse_valid=False),
        _row("R0", 1, True, pair_id="p_000002"),
        _row("R1", 1, False, pair_id="p_000002"),
    ]
    mcnemar_pairs = sum(_tally(h3_pooled_mcnemar(rows)))
    assert mcnemar_pairs == decision_flip_rate(primary_rows(rows)).denominator == 1


# --------------------------------------------------------------------------- H2 eligibility


def test_h2_eligibility_is_non_equivalent_cells_at_sample_zero():
    """H2 was always guarded; pinned here so the two estimators cannot drift apart again."""
    rows = [
        {"id": "keep", "oracle_equivalent": False, "sample_index": 0},
        {"id": "repeat", "oracle_equivalent": False, "sample_index": 1},
        {"id": "equivalent", "oracle_equivalent": True, "sample_index": 0},
    ]
    assert [row["id"] for row in _neq_primary(rows)] == ["keep"]
