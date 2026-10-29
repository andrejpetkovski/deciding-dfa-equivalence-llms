"""Tests for the descriptive breakdowns (D15).

The risk this module carries is not that a number is slightly off -- it is that a
descriptive table quietly *disagrees* with the frozen analysis it sits next to in the
paper. Most of what follows therefore checks agreement with `analysis.metrics` rather than
recomputing rates independently: a breakdown that splits rows differently from the frozen
metric is the failure mode worth catching.
"""

from __future__ import annotations

import json

import pytest

from analysis.descriptives import (
    ERROR_CATEGORIES,
    build_descriptives,
    cell_table,
    depth_bin_table,
    error_taxonomy,
    family_table,
    forgiving_parser_sensitivity,
    operational,
    qualitative_examples,
    representation_table,
    trace_errors,
    witness_lengths,
)
from analysis.descriptives import (
    main as descriptives_main,
)
from analysis.figures import DEPTH_BINS
from analysis.metrics import (
    balanced_decision_accuracy,
    decision_flip_rate,
    orbit_success,
    witness_success,
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


# --------------------------------------------------------------------------- scope


def test_repeat_samples_never_enter_a_descriptive_table():
    """Rule 9 and D14: repeats are an audit, not extra observations -- here too.

    A descriptive table is no more entitled to pool them than a confirmatory one; pooling
    would weight the audited pairs three times in every breakdown.
    """
    rows = [row(run_id="a"), row(run_id="b", sample_index=1, witness_success=False)]
    report = build_descriptives(rows)
    assert report["primary_cells"] == 1
    assert report["error_taxonomy"]["overall"]["cells"] == 1


# --------------------------------------------------------------------------- agreement


def test_depth_bins_agree_with_the_frozen_metric_on_the_same_subset():
    rows = [
        row(run_id=f"r{i}", pair_id=f"p{i}", oracle_depth=depth, witness_success=(i % 3 != 0))
        for i, depth in enumerate([1, 2, 3, 4, 5, 8, 9, 12] * 3)
    ]
    table = depth_bin_table(rows)
    pooled = {entry["depth_bin"]: entry for entry in table if entry["model_id"] == "ALL"
              and entry["prompt_condition"] == "ALL"}
    assert set(pooled) == {label for label, _, _ in DEPTH_BINS}
    for label, low, high in DEPTH_BINS:
        subset = [r for r in rows if low <= r["oracle_depth"] <= high]
        frozen = witness_success(subset)
        assert pooled[label]["numerator"] == frozen.numerator
        assert pooled[label]["denominator"] == frozen.denominator
        assert pooled[label]["rate"] == frozen.value


def test_representation_table_agrees_with_the_frozen_flip_and_orbit_metrics():
    rows = []
    for index in range(6):
        for rendering in ("R0", "R1"):
            rows.append(
                row(
                    run_id=f"{index}{rendering}",
                    pair_id=f"p{index}",
                    rendering=rendering,
                    decision_correct=not (rendering == "R1" and index < 2),
                )
            )
    table = {entry["model_id"]: entry for entry in representation_table(rows)}
    assert table["ALL"]["flips"] == decision_flip_rate(rows).numerator
    assert table["ALL"]["flip_denominator"] == decision_flip_rate(rows).denominator
    assert table["ALL"]["orbit_numerator"] == orbit_success(rows).numerator
    assert table["m"]["flip_rate"] == table["ALL"]["flip_rate"]


def test_family_table_reports_each_family_pooled_and_per_model():
    """Families are reported separately because EQ_ISO and EQ_EXPAND are not equally easy."""
    rows = [
        row(run_id=f"{fam}{i}", pair_id=f"p{fam}{i}", family=fam, model_id=f"m{i}",
            oracle_equivalent=fam != "NEQ_DEPTH",
            witness_success=None if fam != "NEQ_DEPTH" else True,
            decision_correct=(i == 0))
        for fam in ("EQ_ISO", "EQ_EXPAND", "NEQ_DEPTH")
        for i in (0, 1)
    ]
    table = family_table(rows)
    pooled = {e["family"]: e for e in table if e["model_id"] == "ALL"}
    assert set(pooled) == {"EQ_ISO", "EQ_EXPAND", "NEQ_DEPTH"}
    for entry in pooled.values():
        assert entry["numerator"] == 1
        assert entry["denominator"] == 2
        assert entry["rate"] == 0.5
    # A model absent from a family contributes no row rather than a zero-denominator one.
    assert all(e["denominator"] > 0 for e in table)


def test_family_accuracies_reconcile_with_the_frozen_class_accuracy():
    """The two equivalent families must sum to the frozen equivalent-class numerator."""
    rows = [
        row(run_id=f"e{i}", pair_id=f"pe{i}", family=fam, oracle_equivalent=True,
            witness_success=None, decision_correct=(i % 2 == 0))
        for fam in ("EQ_ISO", "EQ_EXPAND")
        for i in range(4)
    ]
    pooled = {e["family"]: e for e in family_table(rows) if e["model_id"] == "ALL"}
    eq_rate, _, _ = balanced_decision_accuracy(rows)
    assert pooled["EQ_ISO"]["numerator"] + pooled["EQ_EXPAND"]["numerator"] == eq_rate.numerator


def test_cell_table_splits_by_model_prompt_and_rendering():
    rows = [
        row(run_id=f"{m}{p}{r}", pair_id=f"p{i}", model_id=m, prompt_condition=p, rendering=r)
        for i, (m, p, r) in enumerate(
            (m, p, r) for m in ("m1", "m2") for p in ("P0", "P1") for r in ("R0", "R1")
        )
    ]
    table = cell_table(rows)
    assert len(table) == 8
    assert {(e["model_id"], e["prompt_condition"], e["rendering"]) for e in table} == {
        (m, p, r) for m in ("m1", "m2") for p in ("P0", "P1") for r in ("R0", "R1")
    }
    assert all(entry["cells"] == 1 for entry in table)


# --------------------------------------------------------------------------- taxonomy


def test_error_categories_partition_the_non_equivalent_cells():
    """Every scheduled cell lands in exactly one bucket, so the shares sum to one.

    A taxonomy that double-counts or drops a cell is worse than no taxonomy: it reads as a
    complete account of the failures while hiding some of them.
    """
    rows = [
        row(run_id="success"),
        row(run_id="longer", witness_shortest=False),
        row(run_id="invalid", witness_success=False, witness_valid=False),
        row(run_id="said_eq", witness_success=False, decision_correct=False),
        row(run_id="unparsed", parse_valid=False, witness_success=False),
        row(run_id="transport", status="transport_failed", parse_valid=False,
            witness_success=False),
        row(run_id="equivalent_family", oracle_equivalent=True, witness_success=None),
    ]
    taxonomy = error_taxonomy(rows)
    counts = taxonomy["overall"]["categories"]
    assert set(counts) == set(ERROR_CATEGORIES)
    assert sum(counts.values()) == taxonomy["overall"]["cells"] == 6
    assert counts == {
        "success": 1,
        "valid but not shortest": 1,
        "invalid witness": 1,
        "said equivalent": 1,
        "parse failure": 1,
        "transport failure": 1,
    }


def test_a_transport_failure_is_not_counted_as_a_parse_failure():
    """The two are different observations even though rule 5 fails both."""
    rows = [
        row(run_id="t", status="transport_failed", parse_valid=False, witness_success=False)
    ]
    counts = error_taxonomy(rows)["overall"]["categories"]
    assert counts["transport failure"] == 1
    assert counts["parse failure"] == 0


def test_parse_failures_are_split_by_recorded_reason():
    rows = [
        row(run_id="a", parse_valid=False, witness_success=False, parse_outcome="empty"),
        row(run_id="b", parse_valid=False, witness_success=False, parse_outcome="not_json"),
        row(run_id="c", parse_valid=False, witness_success=False, parse_outcome="empty"),
    ]
    assert error_taxonomy(rows)["overall"]["parse_failure_reason"] == {
        "empty": 2,
        "not_json": 1,
    }


# --------------------------------------------------------------------------- witnesses


def test_excess_length_is_measured_against_the_oracle_depth():
    rows = [
        row(run_id="exact", pair_id="p0", oracle_depth=4, grade_detail={"witness_length": 4}),
        row(run_id="over1", pair_id="p1", oracle_depth=4, witness_shortest=False,
            grade_detail={"witness_length": 5}),
        row(run_id="over3", pair_id="p2", oracle_depth=2, witness_shortest=False,
            grade_detail={"witness_length": 5}),
    ]
    valid = witness_lengths(rows)["valid"]
    assert valid["n"] == 3
    assert valid["excess_over_shortest_histogram"] == {"0": 1, "1": 1, "3": 1}


def test_an_empty_witness_is_counted_separately_among_invalid_ones():
    """Gate 3 makes the empty word provably invalid here, so it is its own failure mode."""
    rows = [
        row(run_id="empty", pair_id="p0", witness_success=False, witness_valid=False,
            witness_shortest=False, grade_detail={"witness_length": 0}),
        row(run_id="wrong", pair_id="p1", witness_success=False, witness_valid=False,
            witness_shortest=False, grade_detail={"witness_length": 3}),
    ]
    invalid = witness_lengths(rows)["invalid"]
    assert invalid["n"] == 2
    assert invalid["empty_witness"] == 1
    assert invalid["length_histogram"] == {"0": 1, "3": 1}


# --------------------------------------------------------------------------- examples


def _flip_rows():
    """Two pairs: p0 flips in one cell, p1 flips in two."""
    rows = []
    for pair, flipping in (("p0", {("P0", "m")}), ("p1", {("P0", "m"), ("P1", "m")})):
        for prompt in ("P0", "P1"):
            for rendering in ("R0", "R1"):
                flipped = (prompt, "m") in flipping and rendering == "R1"
                rows.append(
                    row(run_id=f"{pair}{prompt}{rendering}", pair_id=pair, family="EQ_ISO",
                        oracle_equivalent=True, witness_success=None,
                        prompt_condition=prompt, rendering=rendering,
                        decision_correct=not flipped)
                )
    return rows


def test_examples_are_selected_by_a_stated_rule_not_by_interest():
    """§11.4: the rule must be mechanical, so the selection cannot be steered."""
    rows = [
        row(run_id="zz_first_success", pair_id="p0", oracle_depth=1),
        row(run_id="aa_first_success", pair_id="p1", oracle_depth=1),
        row(run_id="mm_failure", pair_id="p2", oracle_depth=1, witness_success=False),
    ]
    examples = qualitative_examples(rows)
    assert "lexicographically first" in examples["selection_rule"]
    bin12 = next(b for b in examples["by_depth_bin"] if b["depth_bin"] == "1-2")
    # Lexicographically first success, not the first encountered in input order.
    assert bin12["success"]["run_id"] == "aa_first_success"
    assert bin12["failure"]["run_id"] == "mm_failure"


def test_flips_are_ranked_by_how_many_cells_flipped():
    ranked = qualitative_examples(_flip_rows())["largest_representation_flips"]
    assert [entry["pair_id"] for entry in ranked] == ["p1", "p0"]
    assert ranked[0]["flipped_cells"] == 2
    assert ranked[1]["flipped_cells"] == 1


def test_flip_denominators_match_the_frozen_flip_rate():
    """The per-family split must reconcile with the frozen metric, or it is a second number."""
    rows = _flip_rows()
    dist = qualitative_examples(rows)["flip_distribution"]
    frozen = decision_flip_rate(rows)
    assert sum(dist["flipped_cells_by_family"].values()) == frozen.numerator
    assert sum(dist["eligible_cells_by_family"].values()) == frozen.denominator
    assert dist["pairs_with_at_least_one_flip"] == 2
    assert dist["max_flipped_cells_for_one_pair"] == 2


def test_examples_carry_raw_text_when_supplied_and_degrade_without_it():
    rows = [row(run_id="r", pair_id="p0", oracle_depth=1)]
    without = qualitative_examples(rows)
    bin12 = next(b for b in without["by_depth_bin"] if b["depth_bin"] == "1-2")
    assert bin12["success"]["raw"] is None

    withtext = qualitative_examples(rows, {"r": "x" * 50}, max_chars=10)
    raw = next(
        b for b in withtext["by_depth_bin"] if b["depth_bin"] == "1-2"
    )["success"]["raw"]
    assert raw["final_answer"] == "x" * 10
    assert raw["chars"] == 50
    assert raw["truncated"] is True


# --------------------------------------------------------------------------- traces


def test_first_wrong_step_histogram_covers_every_invalid_trace():
    rows = [
        row(run_id="ok", prompt_condition="P1", trace_valid=True),
        row(run_id="bad1", prompt_condition="P1", trace_valid=False, first_wrong_step=2),
        row(run_id="bad2", prompt_condition="P1", trace_valid=False, first_wrong_step=2),
        row(run_id="bad3", prompt_condition="P1", trace_valid=False, first_wrong_step=None),
    ]
    errors = trace_errors(rows)
    assert errors["invalid_traces"] == 3
    assert errors["first_wrong_step_histogram"] == {"2": 2}
    # A missing index is reported rather than silently dropped from the denominator.
    assert errors["first_wrong_step_unrecorded"] == 1


# --------------------------------------------------------------------------- sensitivity


def test_the_forgiving_parser_result_is_labelled_and_never_replaces_the_strict_one():
    """Rule 5: the strict parser is primary and the artifact has to say so itself."""
    rows = [
        row(run_id="a", parse_valid=False, witness_success=False,
            sensitivity_parse_valid=True, sensitivity_witness_success=True),
        row(run_id="b"),
    ]
    block = forgiving_parser_sensitivity(rows)
    assert "SENSITIVITY" in block["label"]
    assert block["strict_is_primary"] is True
    assert block["parse_valid"]["strict"] == 1
    assert block["parse_valid"]["forgiving"] == 2
    assert block["parse_valid"]["additional_cells_recovered"] == 1
    # The strict rate is reported alongside, so the two can never be confused.
    assert block["end_to_end_witness_success"]["strict"] == 1
    assert block["end_to_end_witness_success"]["forgiving"] == 2


def test_a_missing_sensitivity_field_falls_back_to_the_strict_outcome():
    """The forgiving parser accepts everything the strict one does, so it cannot lose cells.

    Without the fallback an ungraded field would read as a forgiving *failure*, inventing
    a recovery deficit and flattering the strict parser with data that does not exist.
    """
    rows = [row(run_id="a"), row(run_id="b")]
    block = forgiving_parser_sensitivity(rows)
    assert block["parse_valid"]["strict"] == block["parse_valid"]["forgiving"] == 2
    assert block["parse_valid"]["additional_cells_recovered"] == 0
    assert block["end_to_end_witness_success"]["forgiving"] == 2


# --------------------------------------------------------------------------- operational


def test_unknown_usage_is_omitted_not_recorded_as_zero():
    """D4: an unavailable price is unknown, never free."""
    rows = [
        row(run_id="a", usage={"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.5},
            latency_ms=1000.0),
        row(run_id="b", usage={"prompt_tokens": 200, "completion_tokens": 20},
            latency_ms=None),
    ]
    block = operational(rows)["overall"]
    assert block["cost_usd"]["n"] == 1
    assert block["cost_usd"]["total"] == 0.5
    assert block["latency_ms"]["n"] == 1
    assert block["prompt_tokens"]["n"] == 2
    assert block["prompt_tokens"]["median"] == 150.0


def test_an_empty_distribution_reports_none_rather_than_zero():
    block = operational([row(usage={}, latency_ms=None)])["overall"]
    assert block["cost_usd"] == {
        "n": 0, "median": None, "mean": None, "min": None, "max": None, "total": None
    }


# --------------------------------------------------------------------------- provenance


def test_the_report_declares_itself_descriptive():
    """The label travels with the artifact, not only with the module docstring."""
    report = build_descriptives([row()])
    assert "DESCRIPTIVE AND EXPLORATORY ONLY" in report["note"]
    assert "D15" in report["note"]
    assert "confirmatory" in report["note"]


def test_intervals_state_the_resample_count_they_used():
    report = build_descriptives([row()])
    assert "2000 resamples" in report["interval_method"]
    assert "10,000" in report["interval_method"]


# --------------------------------------------------------------------------- CLI


def test_cli_writes_every_table_from_dummy_responses(tmp_path):
    import yaml

    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["paths"]["parsed_dir"] = str(tmp_path / "parsed")
    data["paths"]["tables_dir"] = str(tmp_path / "tables")
    config_path = tmp_path / "config.yaml"
    write_text(config_path, yaml.safe_dump(data, sort_keys=False))

    records = list(read_jsonl("data/pairs/pilot_v2/pairs.jsonl"))
    graded = tmp_path / "graded.jsonl"
    write_text(
        graded,
        "".join(json.dumps(r) + "\n" for r in synthesize(records, dataset_id="test")),
    )

    assert descriptives_main(["--config", str(config_path), "--graded", str(graded)]) == 0
    for name in (
        "descriptives.json",
        "table_cells.csv",
        "table_depth_bins.csv",
        "table_families.csv",
        "table_representation.csv",
        "table_error_taxonomy.csv",
        "descriptives_checksums.json",
    ):
        assert (tmp_path / "tables" / name).is_file()

    manifest = json.loads(read_text(tmp_path / "tables" / "descriptives_checksums.json"))
    assert manifest["graded_sha256"].startswith("sha256:")
    assert len(manifest["outputs"]) == 6


def test_cli_refuses_a_structurally_invalid_graded_file(tmp_path, capsys):
    """The same validation make_results runs: a duplicated run_id double-counts a cell."""
    import yaml

    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["paths"]["parsed_dir"] = str(tmp_path / "parsed")
    data["paths"]["tables_dir"] = str(tmp_path / "tables")
    config_path = tmp_path / "config.yaml"
    write_text(config_path, yaml.safe_dump(data, sort_keys=False))

    graded = tmp_path / "graded.jsonl"
    duplicated = [row(run_id="same"), row(run_id="same", pair_id="p1")]
    write_text(graded, "".join(json.dumps(r) + "\n" for r in duplicated))

    assert descriptives_main(["--config", str(config_path), "--graded", str(graded)]) == 1
    assert "validation failed" in capsys.readouterr().err


def test_cli_reports_a_missing_graded_file(tmp_path, capsys):
    import yaml

    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["paths"]["parsed_dir"] = str(tmp_path / "parsed")
    data["paths"]["tables_dir"] = str(tmp_path / "tables")
    config_path = tmp_path / "config.yaml"
    write_text(config_path, yaml.safe_dump(data, sort_keys=False))

    assert descriptives_main(
        ["--config", str(config_path), "--graded", str(tmp_path / "nope.jsonl")]
    ) == 1
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- frozen files


@pytest.mark.parametrize("module", ["analysis/make_results.py", "analysis/metrics.py",
                                    "analysis/stats.py"])
def test_d15_did_not_touch_the_frozen_analysis(module):
    """D15's whole claim is that it is *additional*. This is that claim, as a test.

    The frozen modules must not import or call into the descriptive layer: if they did,
    a change here could move a confirmatory number, which is exactly what rule 2 forbids.
    """
    assert "descriptives" not in read_text(module)


def test_cap_exhaustion_is_cross_tabulated_against_the_parse_outcome():
    """Truncated and empty usually coincide; the table must say exactly where they do not."""
    failed = {"parse_valid": False, "witness_success": False, "decision_correct": False}
    rows = [
        row(run_id="a", pair_id="p0", truncated=True, parse_outcome="empty", **failed),
        row(run_id="b", pair_id="p1", truncated=True, parse_outcome="not_json", **failed),
        row(run_id="c", pair_id="p2", truncated=False, parse_outcome="empty", **failed),
        row(run_id="d", pair_id="p3", truncated=False, parse_outcome="ok"),
    ]
    overall = operational(rows)["overall"]
    assert overall["truncated"] == 2
    assert overall["truncated_by_parse_outcome"] == {"empty": 1, "not_json": 1}
    assert overall["empty_without_truncation"] == 1
