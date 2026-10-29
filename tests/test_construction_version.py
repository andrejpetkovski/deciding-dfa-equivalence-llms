"""Tests for the construction-version guard and v2 end-to-end generation (M3R).

``docs/experiment_revision.md``: "A v2 config must never silently invoke the v1 single-toggle
generator." Nothing downstream could detect that -- the pairs would be valid, the oracle would
confirm them, every QA gate would pass, and the records would carry a construction label they
did not match. The guard is the only thing standing between that and the paper.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from src.config import ConfigError, load_config
from src.dfa import from_record
from src.generate import (
    CONSTRUCTION_VERSION_V1,
    generate_pairs,
    resolve_construction_version,
)
from src.generators_v2 import CONSTRUCTION_VERSION
from src.io_utils import read_text, write_text
from src.oracle import equivalence_and_witness

BASE = {
    "config_id": "version_test",
    "seed": 4321,
    "alphabet": ["0", "1"],
    "pairs": {
        "construction_version": CONSTRUCTION_VERSION,
        "accepting_count_per_dfa": 7,
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
    "paths": {"pairs_dir": "PLACEHOLDER"},
}


def _config(tmp_path, mutate=None):
    data = copy.deepcopy(BASE)
    data["paths"]["pairs_dir"] = str(tmp_path / "pairs")
    if mutate:
        mutate(data)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    return load_config(path)


# --------------------------------------------------------------------------- the guard


def test_an_absent_version_means_v1():
    """The legacy pilot config predates the field and must keep reproducing byte-identically."""
    assert resolve_construction_version(load_config("configs/pilot.yaml")) == (
        CONSTRUCTION_VERSION_V1
    )


def test_the_revised_configs_declare_v2():
    for path in ("configs/pilot_revised.yaml", "configs/main_frozen.yaml"):
        assert resolve_construction_version(load_config(path)) == CONSTRUCTION_VERSION


def test_an_unknown_version_is_refused(tmp_path):
    """The core guard: refuse, never fall back."""
    config = _config(
        tmp_path, lambda d: d["pairs"].update(construction_version="acceptance_swap_v3")
    )
    with pytest.raises(ConfigError, match=r"is not implemented"):
        resolve_construction_version(config)
    with pytest.raises(ConfigError, match=r"Refusing to fall back"):
        generate_pairs(config)


def test_a_placeholder_version_is_refused(tmp_path):
    config = _config(
        tmp_path, lambda d: d["pairs"].update(construction_version="FILL_AT_FREEZE")
    )
    with pytest.raises(ConfigError, match=r"still 'FILL_AT_FREEZE'"):
        resolve_construction_version(config)


def test_v2_without_an_accepting_count_is_refused(tmp_path):
    """v2 is defined by the count control; running it unset would silently drop the fix."""
    config = _config(tmp_path, lambda d: d["pairs"].pop("accepting_count_per_dfa"))
    with pytest.raises(ConfigError, match=r"requires pairs.accepting_count_per_dfa"):
        generate_pairs(config)


def test_v2_needs_two_more_states_than_v1_for_the_deepest_bin(tmp_path):
    """The branch costs one extra state, so a 13-state config cannot hold bin 9-12."""
    config = _config(tmp_path, lambda d: d["pairs"]["neq_depth"].update(state_count=13))
    with pytest.raises(ConfigError, match=r"needs at least 14 states"):
        generate_pairs(config)


# --------------------------------------------------------------------------- v2 end to end


def test_v2_generation_produces_count_matched_pairs(tmp_path):
    outcome = generate_pairs(_config(tmp_path))
    assert outcome.complete
    assert len(outcome.records) == 8
    for record in outcome.records:
        dfa_a = from_record(record["dfa_A"])
        dfa_b = from_record(record["dfa_B"])
        assert len(dfa_a.accepting) == len(dfa_b.accepting) == 7


def test_the_count_only_shortcut_is_dead_on_v2_data(tmp_path):
    """The finding that motivated D2: on v1 data this rule scored 87.5% balanced."""
    outcome = generate_pairs(_config(tmp_path))
    for record in outcome.records:
        dfa_a = from_record(record["dfa_A"])
        dfa_b = from_record(record["dfa_B"])
        # The rule "counts differ by one" now fires on nothing at all.
        assert abs(len(dfa_a.accepting) - len(dfa_b.accepting)) == 0


def test_v2_records_carry_their_construction_version_and_dataset(tmp_path):
    """A pair with no construction label cannot be checked against the config that claims
    to have produced it, and ``p_000000`` is not unique across splits."""
    outcome = generate_pairs(_config(tmp_path))
    for record in outcome.records:
        assert record["construction_version"] == CONSTRUCTION_VERSION
        assert record["dataset_id"] == "version_test"


def test_v1_records_are_labelled_v1(tmp_path):
    config = _config(
        tmp_path,
        lambda d: (
            d["pairs"].pop("construction_version"),
            d["pairs"].pop("accepting_count_per_dfa"),
        ),
    )
    outcome = generate_pairs(config)
    assert all(r["construction_version"] == CONSTRUCTION_VERSION_V1 for r in outcome.records)


def test_v2_depths_are_exact_and_oracle_confirmed(tmp_path):
    outcome = generate_pairs(_config(tmp_path))
    for record in outcome.records:
        if record["family"] != "NEQ_DEPTH":
            continue
        result = equivalence_and_witness(
            from_record(record["dfa_A"]), from_record(record["dfa_B"])
        )
        assert result.shortest_depth == record["requested_depth"] == record["oracle_depth"]


def test_v2_generation_is_reproducible(tmp_path):
    first = generate_pairs(_config(tmp_path))
    second = generate_pairs(_config(tmp_path / "second"))
    assert [r["canonical_hash"] for r in first.records] == [
        r["canonical_hash"] for r in second.records
    ]


def test_the_legacy_pilot_still_regenerates_identically(tmp_path):
    """v2 must not disturb the committed v1 artifacts (D2, legacy isolation)."""
    config = load_config("configs/pilot.yaml")
    committed = [
        line for line in read_text("data/pairs/pilot/pairs.jsonl").splitlines() if line
    ]
    outcome = generate_pairs(config)
    assert len(outcome.records) == len(committed)
    import json

    for fresh, stored in zip(outcome.records, committed, strict=True):
        old = json.loads(stored)
        assert fresh["canonical_hash"] == old["canonical_hash"]
        assert fresh["oracle_depth"] == old["oracle_depth"]
