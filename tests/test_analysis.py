"""Tests for metrics, statistical models and the results pipeline (M9, Step 11)."""

from __future__ import annotations

import json

import pytest

from analysis.make_results import build_results
from analysis.make_results import main as results_main
from analysis.metrics import (
    ValidationError,
    balanced_decision_accuracy,
    conditional_witness_validity,
    coverage,
    decision_flip_rate,
    orbit_success,
    parse_failure_rate,
    primary_rows,
    trace_correctness,
    validate_rows,
    witness_success,
)
from analysis.stats import (
    DEPTH_CENTRE,
    PROMPT_CODING,
    SECONDARY_FAMILY,
    cluster_bootstrap_rate,
    fit_h1,
    holm,
    mcnemar_exact,
)
from scripts.make_dummy_responses import synthesize
from src.io_utils import read_jsonl, read_text, write_text


def row(**overrides):
    """A graded row with sensible defaults; override only what a test is about."""
    base = {
        "run_id": overrides.pop("run_id", "r0"),
        "dataset_id": "d",
        "pair_id": "p0",
        "family": "NEQ_DEPTH",
        "oracle_equivalent": False,
        "oracle_depth": 4,
        "rendering": "R0",
        "prompt_condition": "P0",
        "model_id": "m",
        "sample_index": 0,
        "status": "received",
        "parse_valid": True,
        "decision_correct": True,
        "witness_success": True,
        "witness_valid": True,
        "witness_shortest": True,
        "trace_valid": None,
        "first_wrong_step": None,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- denominators


def test_witness_success_counts_failures_against_the_scheduled_denominator():
    """The distinction the whole metric module exists for: a parse failure must lower the
    end-to-end rate, not vanish from it."""
    rows = [
        row(run_id="a"),
        row(run_id="b", parse_valid=False, witness_success=False, witness_valid=None),
        row(run_id="c", status="transport_failed", parse_valid=False,
            witness_success=False, witness_valid=None),
    ]
    end_to_end = witness_success(rows)
    assert (end_to_end.numerator, end_to_end.denominator) == (1, 3)

    # Conditional validity sees only the one parsed row that predicted non-equivalent.
    conditional = conditional_witness_validity(rows)
    assert (conditional.numerator, conditional.denominator) == (1, 1)
    assert conditional.value > end_to_end.value  # the gap coverage() exists to expose


def test_equivalent_cells_are_excluded_from_the_witness_denominator():
    rows = [row(run_id="a"), row(run_id="b", oracle_equivalent=True, witness_success=None)]
    assert witness_success(rows).denominator == 1


def test_parse_failure_rate_excludes_transport_failures():
    rows = [
        row(run_id="a"),
        row(run_id="b", parse_valid=False),
        row(run_id="c", status="transport_failed", parse_valid=False),
    ]
    rate = parse_failure_rate(rows)
    assert (rate.numerator, rate.denominator) == (1, 2)
    assert coverage(rows)["transport_failed"] == 1


def test_balanced_accuracy_exposes_a_constant_answer():
    """A constant-non-equivalent rule: perfect on one class, hopeless on the other."""
    rows = [row(run_id=f"n{i}", decision_correct=True) for i in range(5)]
    rows += [
        row(run_id=f"e{i}", oracle_equivalent=True, witness_success=None,
            decision_correct=False)
        for i in range(5)
    ]
    eq, neq, balanced = balanced_decision_accuracy(rows)
    assert neq.value == 1.0 and eq.value == 0.0
    assert balanced == 0.5


def test_trace_correctness_only_covers_p1_cells_with_a_witness():
    rows = [
        row(run_id="a", prompt_condition="P1", trace_valid=True),
        row(run_id="b", prompt_condition="P1", trace_valid=False),
        row(run_id="c", prompt_condition="P1", trace_valid=None),  # no witness
        row(run_id="d", prompt_condition="P0", trace_valid=True),  # not P1
    ]
    rate = trace_correctness(rows)
    assert (rate.numerator, rate.denominator) == (1, 2)


def test_a_rate_with_an_empty_denominator_is_none_not_zero():
    assert witness_success([row(oracle_equivalent=True, witness_success=None)]).value is None


# --------------------------------------------------------------------------- paired R0/R1


def _paired(**overrides):
    return [
        row(run_id="r0", rendering="R0", **overrides.get("R0", {})),
        row(run_id="r1", rendering="R1", **overrides.get("R1", {})),
    ]


def test_a_flip_is_counted_when_the_two_decisions_differ():
    rows = _paired(R0={"decision_correct": True}, R1={"decision_correct": False})
    assert decision_flip_rate(rows).numerator == 1
    assert orbit_success(rows).numerator == 0


def test_no_flip_when_both_sides_agree():
    rows = _paired(R0={"decision_correct": True}, R1={"decision_correct": True})
    assert decision_flip_rate(rows).numerator == 0
    assert orbit_success(rows).numerator == 1


def test_the_flip_denominator_needs_two_strict_valid_decisions():
    """A flip between an answer and a non-answer is not a flip."""
    rows = _paired(R1={"parse_valid": False, "decision_correct": False})
    assert decision_flip_rate(rows).denominator == 0
    # Orbit success is unconditional, so the failed side still costs it.
    assert orbit_success(rows).denominator == 1
    assert orbit_success(rows).numerator == 0


def test_an_unpaired_cell_enters_neither_paired_metric():
    rows = [row(run_id="only", rendering="R0")]
    assert decision_flip_rate(rows).denominator == 0
    assert orbit_success(rows).denominator == 0


# --------------------------------------------------------------------------- validation


def test_primary_rows_excludes_audit_repeats():
    rows = [row(run_id="a"), row(run_id="b", sample_index=1)]
    assert len(primary_rows(rows)) == 1


def test_duplicate_run_ids_are_rejected():
    with pytest.raises(ValidationError, match=r"duplicate run_id"):
        validate_rows([row(run_id="same"), row(run_id="same", pair_id="p1")])


def test_pooling_two_datasets_is_rejected():
    """Pilot, main and audit data are never pooled in confirmatory results."""
    with pytest.raises(ValidationError, match=r"more than one dataset"):
        validate_rows([row(run_id="a"), row(run_id="b", dataset_id="other")])


def test_undeclared_repeats_are_rejected():
    with pytest.raises(ValidationError, match=r"undeclared sample_index"):
        validate_rows([row(run_id="a"), row(run_id="b", sample_index=2)])


def test_duplicate_primary_cells_are_rejected():
    """Two rows for one scheduled cell would double-count it."""
    with pytest.raises(ValidationError, match=r"duplicate primary cells"):
        validate_rows([row(run_id="a"), row(run_id="b")])


def test_a_missing_cell_count_is_rejected():
    with pytest.raises(ValidationError, match=r"expected 5 primary cells"):
        validate_rows([row(run_id="a")], expected_cells=5)


def test_a_non_equivalent_row_without_a_depth_is_rejected():
    with pytest.raises(ValidationError, match=r"no depth"):
        validate_rows([row(run_id="a", oracle_depth=None)])


def test_missing_columns_are_rejected():
    incomplete = row()
    del incomplete["witness_success"]
    with pytest.raises(ValidationError, match=r"missing columns"):
        validate_rows([incomplete])


# --------------------------------------------------------------------------- statistics


def test_frozen_analysis_settings_are_constants():
    """M9 freezes these; they are not arguments that could be chosen after seeing results."""
    assert DEPTH_CENTRE == 5.5
    assert PROMPT_CODING == {"P0": -0.5, "P1": 0.5}
    assert SECONDARY_FAMILY == (
        "H2_prompt_effect",
        "H2_prompt_depth_interaction",
        "H3_flip_rate",
    )


def test_h1_recovers_an_injected_depth_effect():
    """The dummy generator injects a -0.22 log-odds slope; the model must find it."""
    pairs = [
        {
            "pair_id": f"p{i:03d}",
            "family": "NEQ_DEPTH",
            "oracle_equivalent": False,
            "oracle_depth": [1, 2, 3, 4, 6, 8, 9, 11][i % 8],
        }
        for i in range(160)
    ]
    rows = synthesize(pairs, seed=99)
    fitted = fit_h1(primary_rows(rows))
    depth = fitted["depth"]
    assert depth.estimable
    assert depth.value < 0
    assert depth.ci_high < 0, "a real negative slope should exclude zero"


def test_a_constant_outcome_is_reported_as_non_estimable():
    """Perfect separation has no finite maximum; a fitter that 'succeeds' reports an
    artefact of its stopping rule."""
    rows = [
        row(run_id=f"r{i}", pair_id=f"p{i}", oracle_depth=1 + i % 4, witness_success=False)
        for i in range(40)
    ]
    fitted = fit_h1(rows)
    assert not fitted["depth"].estimable
    assert "constant" in fitted["depth"].note


def test_the_bootstrap_resamples_pairs_not_calls():
    """All 12 cells of a pair move together, so an interval reflects pair-level uncertainty."""
    rows = [
        row(run_id=f"{p}_{c}", pair_id=p, rendering="R0", prompt_condition="P0",
            witness_success=(p == "p1"))
        for p in ("p0", "p1")
        for c in range(6)
    ]
    low, high = cluster_bootstrap_rate(
        rows, lambda subset: witness_success(subset).value, resamples=500
    )
    # With two pairs, resampling pairs can only give 0, 0.5 or 1 -- never anything between.
    assert low == 0.0 and high == 1.0


def test_holm_adjusts_and_stays_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["c"] == pytest.approx(0.20)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_mcnemar_uses_discordant_pairs_only():
    assert mcnemar_exact(both_correct=50, only_r0=0, only_r1=0, neither=50) == 1.0
    assert mcnemar_exact(both_correct=0, only_r0=10, only_r1=0, neither=0) < 0.01


# --------------------------------------------------------------------------- pipeline


def test_results_regenerate_from_dummy_responses(tmp_path):
    """The contract requires the whole analysis to run on dummy data before any model call."""
    import yaml

    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["paths"]["parsed_dir"] = str(tmp_path / "parsed")
    data["paths"]["tables_dir"] = str(tmp_path / "tables")
    data["paths"]["figures_dir"] = str(tmp_path / "figures")
    config_path = tmp_path / "config.yaml"
    write_text(config_path, yaml.safe_dump(data, sort_keys=False))

    records = list(read_jsonl("data/pairs/pilot_v2/pairs.jsonl"))
    rows = synthesize(records, dataset_id="test")
    graded = tmp_path / "graded.jsonl"
    write_text(graded, "".join(json.dumps(r) + "\n" for r in rows))

    assert results_main([
        "--config", str(config_path), "--graded", str(graded), "--expected-cells", "288"
    ]) == 0

    for name in ("metrics.csv", "estimates.csv", "results.json", "checksums.json"):
        assert (tmp_path / "tables" / name).is_file()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4

    manifest = json.loads(read_text(tmp_path / "tables" / "checksums.json"))
    assert manifest["graded_sha256"].startswith("sha256:")
    assert len(manifest["outputs"]) == 7


def test_the_pipeline_refuses_a_wrong_cell_count(tmp_path, capsys):
    import yaml

    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["paths"]["parsed_dir"] = str(tmp_path / "parsed")
    data["paths"]["tables_dir"] = str(tmp_path / "tables")
    data["paths"]["figures_dir"] = str(tmp_path / "figures")
    config_path = tmp_path / "config.yaml"
    write_text(config_path, yaml.safe_dump(data, sort_keys=False))

    records = list(read_jsonl("data/pairs/pilot_v2/pairs.jsonl"))
    rows = synthesize(records[:4], dataset_id="test")
    graded = tmp_path / "graded.jsonl"
    write_text(graded, "".join(json.dumps(r) + "\n" for r in rows))

    assert results_main([
        "--config", str(config_path), "--graded", str(graded), "--expected-cells", "288"
    ]) == 1
    assert "validation failed" in capsys.readouterr().err


def test_build_results_records_the_frozen_settings():
    records = list(read_jsonl("data/pairs/pilot_v2/pairs.jsonl"))
    results = build_results(synthesize(records, dataset_id="test"))
    frozen = results["frozen_analysis_settings"]
    assert frozen["depth_centre"] == 5.5
    assert frozen["primary_sample_index"] == 0
    assert results["multiplicity"]["family"] == list(SECONDARY_FAMILY)
    # The Holm family never shrinks to whichever tests happened to be estimable.
    assert set(results["multiplicity"]["raw_p"]) == set(SECONDARY_FAMILY)
