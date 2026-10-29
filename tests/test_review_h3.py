"""Known-answer checks for the post-review matched H3 comparison."""

import pytest

from analysis.review_h3 import matched_groups, summarize


def rows(a=(True, False, True), b=(True, True, False), pair="p1"):
    return [
        {
            "pair_id": pair,
            "model_id": "m",
            "prompt_condition": "P0",
            "rendering": rendering,
            "sample_index": i,
            "parse_valid": True,
            "oracle_equivalent": True,
            "decision_correct": answer,
            "family": "EQ_ISO",
        }
        for rendering, answers in (("R0", a), ("R1", b))
        for i, answer in enumerate(answers)
    ]


def test_all_pairs_counts_are_exact_and_not_three_call_disagreement():
    group = matched_groups(rows(), (0, 1, 2))["groups"][0]
    assert group["cross_disagreements"] == 4
    assert group["cross_comparisons"] == 9
    assert group["within_disagreements"] == 4
    assert group["within_comparisons"] == 6
    assert group["difference"] == pytest.approx(-2 / 9)


def test_first_two_excludes_third_call_and_uses_same_groups_for_both_rates():
    data = rows()
    data[-1]["parse_valid"] = False
    block = matched_groups(data, (0, 1))
    assert block["eligible_groups"] == 1
    assert block["groups"][0]["cross_rate"] == 0.5
    assert block["groups"][0]["within_rate"] == 0.5
    assert matched_groups(data, (0, 1, 2))["excluded_for_parse_failure"] == 1
    data[0]["parse_valid"] = False
    assert matched_groups(data, (0, 1))["eligible_groups"] == 0


def test_missing_repeat_is_reported_and_primary_only_groups_are_not_audit():
    block = matched_groups(rows()[:-1] + rows(a=(True,), b=(True,), pair="p2"), (0, 1, 2))
    assert block["observed_audit_groups"] == 1
    assert block["incomplete_groups"] == 1
    assert block["eligible_groups"] == 0


def test_duplicate_calls_raise_instead_of_overwriting():
    data = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        matched_groups(data + [data[0]], (0, 1))


def test_persistent_rendering_effect_is_distinct_from_repeat_variability():
    groups = matched_groups(rows(a=(True, True), b=(False, False)), (0, 1))["groups"]
    report = summarize(groups, resamples=20)
    assert report["difference"] == {"estimate": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert report["within_rate"]["numerator"] == 0


def test_bootstrap_resamples_whole_dfa_pairs_and_retains_repeated_draws():
    # Opposite effects in two prompts of the same pair must cancel in every draw.
    groups = [
        {
            "pair_id": "p1",
            "cross_rate": 1.0,
            "within_rate": 0.0,
            "difference": 1.0,
            "cross_disagreements": 4,
            "cross_comparisons": 4,
            "within_disagreements": 0,
            "within_comparisons": 2,
        },
        {
            "pair_id": "p1",
            "cross_rate": 0.0,
            "within_rate": 1.0,
            "difference": -1.0,
            "cross_disagreements": 0,
            "cross_comparisons": 4,
            "within_disagreements": 2,
            "within_comparisons": 2,
        },
    ]
    report = summarize(groups, resamples=30)
    assert report["dfa_pairs"] == 1
    assert report["difference"] == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    assert summarize([], resamples=30)["difference"]["estimate"] is None
