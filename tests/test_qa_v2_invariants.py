"""M5R: QA must enforce the v2 construction invariants and a verified exclusion union.

Before M5R the gates validated v1 properties only. A v1 pair could sit in a v2 dataset, a
family could carry the wrong accepting count, pair ids could repeat, and an exclusion list
could name a file that covered nothing — and every gate still reported PASS. Each test below
corrupts exactly one of those and asserts the gate now fails.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from scripts.make_exclusions import build_union
from scripts.make_exclusions import main as exclusions_main
from src.config import ConfigError, load_config
from src.dataset_qa import (
    exclusion_union_failures,
    gate_2_dataset_balance,
    gate_7_deduplication,
    run_all_gates,
)
from src.generate import main as generate_main
from src.generators_v2 import CONSTRUCTION_VERSION
from src.io_utils import read_jsonl, read_text, write_text
from src.render_prompts import main as render_main

ACCEPTING = 7


def _v2_config(tmp_path, mutate=None):
    data = {
        "config_id": "v2_qa_test",
        "seed": 24680,
        "alphabet": ["0", "1"],
        "pairs": {
            "total": 8,
            "construction_version": CONSTRUCTION_VERSION,
            "accepting_count_per_dfa": ACCEPTING,
            "neq_depth": {"state_count": 14, "bins": {"1-2": 2, "9-12": 2}},
            "eq_iso": {"count": 2, "state_count": 14},
            "eq_expand": {
                "count": 2,
                "state_count": 14,
                "core_states": [10, 12],
                "require_non_isomorphic": True,
            },
            "max_generation_attempts_per_pair": 4000,
        },
        "representations": [{"id": "R0"}, {"id": "R1"}],
        "prompts": {
            "system": "prompts/system_v1.txt",
            "conditions": [
                {
                    "id": "P0",
                    "template": "prompts/p0_direct_v1.txt",
                    "schema": "prompts/schema_p0.json",
                },
                {
                    "id": "P1",
                    "template": "prompts/p1_trace_v1.txt",
                    "schema": "prompts/schema_p1.json",
                },
            ],
        },
        "qa": {
            "cross_check_libraries": ["pyformlang", "automata-lib"],
            "token_balance_tolerance": 0.10,
        },
        "paths": {
            "pairs_dir": str(tmp_path / "pairs"),
            "renderings_dir": str(tmp_path / "renderings"),
        },
    }
    if mutate:
        mutate(data)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def v2_dataset(tmp_path):
    """A generated and rendered v2 dataset, plus its config and records."""
    path = _v2_config(tmp_path)
    assert generate_main(["--config", str(path)]) == 0
    assert render_main(["--config", str(path)]) == 0
    config = load_config(path)
    records = list(read_jsonl(tmp_path / "pairs" / "pairs.jsonl"))
    return config, records, tmp_path


def test_a_clean_v2_dataset_passes(v2_dataset):
    config, records, _ = v2_dataset
    assert gate_2_dataset_balance(records, config).status == "PASS"
    statuses = [result.status for result in run_all_gates(config)]
    assert statuses == ["PASS"] * 7 + ["SKIP"]  # gate 8 needs a tokenizer


# --------------------------------------------------------------------------- gate 2, v2


def test_a_v1_pair_cannot_hide_in_a_v2_dataset(v2_dataset):
    """The failure the construction-version guard cannot catch on its own: the generator
    ran correctly, but a record claims a different construction than the config."""
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[0]["construction_version"] = "single_toggle_v1"
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("construction_version" in example for example in result.examples)


def test_a_broken_accepting_count_fails(v2_dataset):
    """If the count control lapses, the accepting-count shortcut comes straight back."""
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[1]["dfa_A"]["accepting"] = corrupted[1]["dfa_A"]["accepting"][:-1]
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("v2 count control" in example for example in result.examples)


def test_duplicate_pair_ids_fail(v2_dataset):
    """Repeated ids silently collapse rows when responses are joined back to pairs."""
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[2]["pair_id"] = corrupted[3]["pair_id"]
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("appears 2 times" in example for example in result.examples)


def test_a_wrong_state_count_fails(v2_dataset):
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[4]["dfa_B"]["n_states"] = 13
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("config declares 14" in example for example in result.examples)


def test_a_wrong_alphabet_fails(v2_dataset):
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[0]["dfa_A"]["alphabet"] = ["a", "b"]
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("alphabet" in example for example in result.examples)


def test_a_short_total_fails(v2_dataset):
    config, records, _ = v2_dataset
    result = gate_2_dataset_balance(records[:-1], config)
    assert result.status == "FAIL"
    assert any("total: expected 8" in example for example in result.examples)


def test_a_foreign_dataset_id_fails(v2_dataset):
    """``p_000000`` is not unique across splits, so the dataset must be named."""
    config, records, _ = v2_dataset
    corrupted = copy.deepcopy(records)
    corrupted[5]["dataset_id"] = "some_other_split"
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("dataset_id" in example for example in result.examples)


def test_v1_datasets_are_not_held_to_v2_invariants():
    """The legacy pilot declares no version or count, and must still pass."""
    config = load_config("configs/pilot.yaml")
    records = list(read_jsonl("data/pairs/pilot/pairs.jsonl"))
    assert gate_2_dataset_balance(records, config).status == "PASS"


# --------------------------------------------------------------------------- gate 7 union


def test_an_exclusion_list_that_covers_nothing_fails():
    """File existence proves nothing about contents: a stale list excludes nothing."""
    failures = exclusion_union_failures(
        set(), "stale.txt", ["data/pairs/pilot/pairs.jsonl"]
    )
    assert failures
    assert "missing 24 of 24" in failures[0]


def test_a_partial_exclusion_union_fails():
    records = list(read_jsonl("data/pairs/pilot/pairs.jsonl"))
    from src.dfa import from_record, pair_canonical_hash

    partial = {
        pair_canonical_hash(from_record(r["dfa_A"]), from_record(r["dfa_B"]))
        for r in records[:-3]
    }
    failures = exclusion_union_failures(
        partial, "partial.txt", ["data/pairs/pilot/pairs.jsonl"]
    )
    assert failures
    assert "missing 3 of 24" in failures[0]


def test_a_complete_union_passes():
    union, per_source = build_union(["data/pairs/pilot/pairs.jsonl"])
    assert per_source["data/pairs/pilot/pairs.jsonl"] == 24
    assert not exclusion_union_failures(set(union), "union.txt", ["data/pairs/pilot/pairs.jsonl"])


def test_an_undeclared_source_list_fails():
    """A configured exclusion file whose coverage nobody declared cannot be verified."""
    failures = exclusion_union_failures({"sha256:abc"}, "list.txt", [])
    assert failures
    assert "exclusion_sources" in failures[0]


def test_a_missing_source_file_fails():
    failures = exclusion_union_failures({"sha256:abc"}, "list.txt", ["data/pairs/absent.jsonl"])
    assert failures
    assert "does not exist" in failures[0]


def test_gate_7_reports_union_failures(v2_dataset):
    _, records, _ = v2_dataset
    result = gate_7_deduplication(
        records, set(), "stale.txt", ["data/pairs/pilot/pairs.jsonl"]
    )
    assert result.status == "FAIL"
    assert result.detail["exclusion_sources"] == ["data/pairs/pilot/pairs.jsonl"]


def test_build_union_refuses_a_missing_source():
    with pytest.raises(FileNotFoundError, match=r"does not exist"):
        build_union(["data/pairs/nope.jsonl"])


# --------------------------------------------------------------------------- union builder


def test_the_union_builder_recomputes_rather_than_copying(tmp_path):
    """A stored hash is an assertion; copying it would produce a list that fails to exclude
    the very pair it names."""
    source = tmp_path / "pairs.jsonl"
    records = list(read_jsonl("data/pairs/pilot/pairs.jsonl"))
    import json

    for record in records:
        record["canonical_hash"] = "sha256:" + "0" * 64  # a lie
    write_text(source, "".join(json.dumps(r) + "\n" for r in records))

    union, _ = build_union([str(source)])
    assert "sha256:" + "0" * 64 not in union
    assert len(union) == 24


def test_the_builder_refuses_to_overwrite_a_generated_artifact(tmp_path, capsys):
    """``src.generate`` owns ``canonical_hashes.txt``; a union must not land on one.

    This is a regression: the first run of this script rewrote the committed legacy pilot's
    hash file, which D2 requires be preserved.
    """
    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["pairs"]["exclude_hashes_from"] = "data/pairs/pilot/canonical_hashes.txt"
    path = tmp_path / "bad.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert exclusions_main(["--config", str(path)]) == 2
    assert "refusing to write" in capsys.readouterr().err


def test_the_builder_refuses_a_destination_that_is_its_own_source(tmp_path, capsys):
    data = yaml.safe_load(read_text("configs/pilot_revised.yaml"))
    data["pairs"]["exclusion_sources"] = ["data/pairs/pilot_v2/exclusion_self.txt"]
    data["pairs"]["exclude_hashes_from"] = "data/pairs/pilot_v2/exclusion_self.txt"
    path = tmp_path / "self.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert exclusions_main(["--config", str(path)]) == 2
    assert "its own sources" in capsys.readouterr().err


def test_generation_refuses_a_missing_exclusion_file(tmp_path):
    """Already covered for QA; asserted here for the v2 config path too."""
    path = _v2_config(
        tmp_path,
        lambda d: d["pairs"].update(exclude_hashes_from=str(tmp_path / "absent.txt")),
    )
    from src.generate import generate_pairs

    with pytest.raises(ConfigError, match=r"does not exist"):
        generate_pairs(load_config(path))
