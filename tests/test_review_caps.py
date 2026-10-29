"""Guard cap classification, raw joins, selection, and pair-cluster bootstrapping."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from analysis.review_caps import (
    breakdown,
    h2_sensitivity,
    join_primary,
    resample_pairs,
    summarize,
)


def example(finish="stop", tokens=20, content='{"equivalent": false}', sample=0):
    grade = {
        "run_id": f"run|s{sample}",
        "pair_id": "p1",
        "rendering": "R0",
        "prompt_condition": "P0",
        "sample_index": sample,
        "model_id": "model",
        "response_id": "response",
        "finish_reason": finish,
        "usage": {"completion_tokens": tokens},
        "answer_adapter": "openai_chat_content_v1",
        "family": "NEQ_DEPTH",
        "oracle_equivalent": False,
        "oracle_depth": 3,
        "status": "received",
        "parse_valid": True,
        "parse_outcome": "ok",
        "decision_correct": True,
        "witness_success": True,
        "witness_shortest": True,
        "trace_valid": None,
    }
    raw = {
        key: grade[key]
        for key in (
            "run_id",
            "pair_id",
            "rendering",
            "prompt_condition",
            "sample_index",
            "response_id",
            "finish_reason",
            "usage",
        )
    }
    raw.update(
        {
            "requested_model": "model",
            "ok": True,
            "attempt": 1,
            "started_at_utc": "2026-09-09T12:00:00Z",
            "decoding": {"max_tokens": 100},
            "raw_body": {"choices": [{"message": {"content": content}, "finish_reason": finish}]},
        }
    )
    return grade, raw


@pytest.mark.parametrize(
    "finish,tokens,expected",
    [
        ("stop", 99, False),
        ("length", 99, True),
        ("stop", 100, True),
        ("length", 100, True),
    ],
)
def test_cap_signals_are_observable_and_not_inferred_from_parsing(finish, tokens, expected):
    grade, raw = example(finish, tokens)
    row = join_primary([grade], [raw])[0]
    assert row["cap_hit"] is expected
    assert row["cap_signals_disagree"] == ((finish == "length") != (tokens >= 100))
    # A valid answer at the cap retains its grade; exclusion is a separate sensitivity.
    assert row["witness_success"] is True
    assert row["failure_category"] == "shortest_valid_witness"


def test_empty_and_non_json_cap_hits_are_distinct_and_repeats_do_not_enter():
    grade, raw = example("length", 100, "")
    grade.update(parse_valid=False, parse_outcome="empty", witness_success=False)
    repeat, repeat_raw = example(sample=1)
    rows = join_primary([grade, repeat], [raw, repeat_raw])
    assert len(rows) == 1
    assert rows[0]["failure_category"] == "empty_capped"
    assert summarize(rows)["empty_capped"] == 1
    raw["raw_body"]["choices"][0]["message"]["content"] = "partial prose"
    grade["parse_outcome"] = "not_json"
    rows = join_primary([grade], [raw])
    assert rows[0]["failure_category"] == "non_json_capped"
    assert summarize(rows)["empty_capped"] == 0
    assert summarize(rows)["non_json_capped"] == 1


def test_terminal_attempt_matches_frozen_chronology_not_attempt_number():
    grade, last = example()
    old = deepcopy(last)
    old.update(started_at_utc="2026-09-08T12:00:00Z", attempt=9, response_id="old")
    assert len(join_primary([grade], [last, old])) == 1


def test_duplicate_missing_and_stale_raw_joins_fail_loudly():
    grade, raw = example()
    with pytest.raises(ValueError, match="Duplicate"):
        join_primary([grade, grade], [raw])
    with pytest.raises(ValueError, match="Missing terminal"):
        join_primary([grade], [])
    raw["response_id"] = "different"
    with pytest.raises(ValueError, match="mismatch in response_id"):
        join_primary([grade], [raw])


def test_trace_errors_are_not_counted_twice_as_primary_failures():
    grade, raw = example()
    grade.update(prompt_condition="P1", trace_valid=False)
    raw["prompt_condition"] = "P1"
    rows = join_primary([grade], [raw])
    summary = summarize(rows)
    assert summary["witness_successes"] == 1
    assert summary["neq_trace_errors"] == 1
    assert summary["outcomes"] == {"shortest_valid_witness": 1}
    grade.update(trace_valid=True, witness_success=False, witness_shortest=False)
    summary = summarize(join_primary([grade], [raw]))
    assert summary["invalid_witness_with_correct_traces"] == 1


def test_breakdown_denominators_account_for_every_call():
    grade, raw = example()
    row = join_primary([grade], [raw])[0]
    rows = [row, {**row, "pair_id": "p2", "prompt_condition": "P1"}]
    by_prompt = breakdown(rows, ("prompt_condition",))
    assert sum(r["calls"] for r in by_prompt) == 2
    assert sum(sum(r["outcomes"].values()) for r in by_prompt) == 2
    assert summarize([])["cap_rate"] is None


def test_resampled_copies_have_distinct_cluster_ids_and_preserve_all_conditions():
    original = [[{"pair_id": "p1", "prompt_condition": p} for p in ("P0", "P1")]]
    sampled = resample_pairs(original, np.array([0, 0]))
    assert [r["pair_id"] for r in sampled] == ["bootstrap_0"] * 2 + ["bootstrap_1"] * 2
    assert [r["prompt_condition"] for r in sampled] == ["P0", "P1"] * 2
    assert original[0][0]["pair_id"] == "p1"


def test_h2_bootstrap_ignores_repeats_and_is_seeded():
    grade, raw = example()
    row = join_primary([grade], [raw])[0]
    rows = [row, {**row, "pair_id": "p2"}, {**row, "sample_index": 1}]

    def constant_estimate(sample):
        assert len(sample) == 2
        assert len({r["pair_id"] for r in sample}) == 2
        return -0.05

    report = h2_sensitivity(rows, resamples=20, estimator=constant_estimate)
    assert report["n_cells"] == 2
    assert report["percentage_points"] == report["ci_low_pp"] == report["ci_high_pp"] == -5
    assert report["successful_fits"] == 20
    assert report == h2_sensitivity(rows, resamples=20, estimator=constant_estimate)


def test_nonfinite_fit_cannot_be_published_as_an_estimate():
    grade, raw = example()
    rows = join_primary([grade], [raw])
    assert not h2_sensitivity(rows, resamples=20, estimator=lambda _: float("nan"))["estimable"]
