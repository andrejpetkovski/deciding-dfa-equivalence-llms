"""Tests for config loading and the generation CLI (M3, Step 4)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import yaml

from src.config import PLACEHOLDERS, Config, ConfigError, load_config
from src.dfa import from_record, pair_canonical_hash
from src.generate import generate_pairs, main
from src.io_utils import read_text, sha256_of_file, write_text
from src.oracle import equivalence_and_witness

PILOT = "configs/pilot.yaml"


def _small_config(tmp_path, **overrides):
    """A miniature config: two pairs per family, so a full run is fast."""
    data = {
        "config_id": "test",
        "seed": 4242,
        "alphabet": ["0", "1"],
        "pairs": {
            "neq_depth": {
                "state_count": 14,
                "bins": {"1-2": 2, "9-12": 2},
                "exclude_depth_zero": True,
                "forbid_epsilon_cue": True,
            },
            "eq_iso": {"count": 2, "state_count": 14},
            "eq_expand": {
                "count": 2,
                "state_count": 14,
                "core_states": [10, 12],
                "require_non_isomorphic": True,
            },
            "max_generation_attempts_per_pair": 2000,
        },
        "paths": {"pairs_dir": str(tmp_path / "pairs")},
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    return path


# --------------------------------------------------------------------------- config


def test_load_config_reads_the_pilot():
    config = load_config(PILOT)
    assert config.config_id == "pilot_v1"
    assert config.alphabet == ("0", "1")


def test_missing_config_file_is_an_error():
    with pytest.raises(ConfigError, match=r"no such config file"):
        load_config("configs/does_not_exist.yaml")


def test_dotted_lookup():
    config = load_config(PILOT)
    assert config.get("pairs.neq_depth.state_count") == 14
    assert config.get("pairs.nope.deeper", "fallback") == "fallback"


def test_require_names_the_missing_key():
    config = load_config(PILOT)
    with pytest.raises(ConfigError, match=r"missing required key 'pairs.nope'"):
        config.require("pairs.nope")


def test_require_rejects_an_unfilled_placeholder():
    """A config must not be runnable while a required value is still a placeholder (§14.2).

    Asserted against a synthetic config rather than against `main_frozen.yaml`, which was
    frozen on 2026-09-09 (D11) and now carries a real seed. Pinning this property to a
    particular file made the test lapse the moment that file was legitimately filled in.
    """
    config = Config(path=load_config(PILOT).path, data={"seed": "FILL_AT_FREEZE"})
    with pytest.raises(ConfigError, match=r"still 'FILL_AT_FREEZE'"):
        config.require("seed")


def test_the_frozen_main_config_has_no_placeholders_left():
    """The other half: after the freeze, the main config must be complete.

    `frozen_commit` and `frozen_at_utc` are recorded in docs/freeze_log.md at tag time and
    are the only two allowed to remain, since a commit cannot contain its own SHA.
    """
    config = load_config("configs/main_frozen.yaml")
    for key in ("seed", "execution.order_seed", "samples.repeatability.selection_seed"):
        assert isinstance(config.require(key), int)
    for key in ("analysis.h2_contrast_specification", "analysis.secondary_test_family"):
        assert str(config.require(key)) not in PLACEHOLDERS


def test_generation_refuses_an_unfrozen_config():
    """The strongest form of the guard: generation cannot start on a template.

    Uses the shipped template markers directly, so the guard keeps biting after
    `main_frozen.yaml` itself was frozen (D11).
    """
    config = load_config("configs/main_frozen.yaml")
    unfrozen = Config(path=config.path, data={**config.data, "seed": "FILL_AT_FREEZE"})
    with pytest.raises(ConfigError):
        generate_pairs(unfrozen)


def test_nested_placeholders_are_found():
    config = Config(
        path=load_config(PILOT).path,
        data={"models": [{"id": "FILL_AT_FREEZE", "role": "ok"}]},
    )
    with pytest.raises(ConfigError, match=r"models\[0\].id"):
        config.require_no_placeholders("models")


def test_alphabet_validation_rejects_multi_character_symbols(tmp_path):
    path = tmp_path / "bad.yaml"
    write_text(path, yaml.safe_dump({"alphabet": ["0", "10"]}))
    with pytest.raises(ConfigError, match=r"single character"):
        _ = load_config(path).alphabet


def test_alphabet_validation_rejects_duplicates(tmp_path):
    path = tmp_path / "dup.yaml"
    write_text(path, yaml.safe_dump({"alphabet": ["0", "0"]}))
    with pytest.raises(ConfigError, match=r"duplicate symbols"):
        _ = load_config(path).alphabet


# --------------------------------------------------------------------------- generation


def test_generation_fills_every_quota(tmp_path):
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    assert outcome.complete
    assert len(outcome.records) == 8
    assert Counter(r["family"] for r in outcome.records) == {
        "NEQ_DEPTH": 4,
        "EQ_ISO": 2,
        "EQ_EXPAND": 2,
    }


def test_every_record_matches_a_fresh_oracle_run(tmp_path):
    """The recorded label must be reproducible from the recorded automata alone."""
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    for record in outcome.records:
        dfa_a = from_record(record["dfa_A"])
        dfa_b = from_record(record["dfa_B"])
        result = equivalence_and_witness(dfa_a, dfa_b)
        assert result.equivalent == record["oracle_equivalent"]
        assert result.shortest_witness == record["oracle_witness"]
        assert result.shortest_depth == record["oracle_depth"]
        assert pair_canonical_hash(dfa_a, dfa_b) == record["canonical_hash"]


def test_neq_depth_records_land_in_their_requested_bin(tmp_path):
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    for record in outcome.records:
        if record["family"] != "NEQ_DEPTH":
            continue
        assert record["oracle_depth"] == record["requested_depth"]
        assert record["oracle_depth"] >= 1  # d = 0 is excluded (§Step 3)


def test_equivalent_families_carry_no_witness(tmp_path):
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    for record in outcome.records:
        if record["family"] == "NEQ_DEPTH":
            continue
        assert record["oracle_equivalent"] is True
        assert record["oracle_witness"] is None
        assert record["oracle_depth"] is None


def test_pair_ids_and_hashes_are_unique(tmp_path):
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    ids = [r["pair_id"] for r in outcome.records]
    hashes = [r["canonical_hash"] for r in outcome.records]
    assert len(set(ids)) == len(ids)
    assert len(set(hashes)) == len(hashes)


def test_records_carry_graph_statistics(tmp_path):
    """§13 needs these to rule out depth confounding."""
    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    for record in outcome.records:
        stats = record["graph_statistics"]
        assert stats["n_states_a"] == 14
        assert stats["n_reachable_a"] == 14  # QA gate 5
        assert 1 <= stats["minimal_states_a"] <= 14


def test_a_recorded_seed_regenerates_its_own_pair(tmp_path):
    """§12.5: every pair must be reproducible from its seed alone."""
    import random

    from src.generators import neq_depth_pair, permute_independently

    outcome = generate_pairs(load_config(_small_config(tmp_path)))
    record = next(r for r in outcome.records if r["family"] == "NEQ_DEPTH")

    rng = random.Random(record["seed"])
    depth = rng.randint(*[int(p) for p in _bin_for(record).split("-")])
    assert depth == record["requested_depth"]
    draw = neq_depth_pair(rng, 14, depth, ("0", "1"))
    assert draw.pair is not None
    dfa_a, dfa_b = permute_independently(rng, *draw.pair)
    assert pair_canonical_hash(dfa_a, dfa_b) == record["canonical_hash"]


def _bin_for(record) -> str:
    """The stratum a record's requested depth came from, per the test config."""
    return "1-2" if record["requested_depth"] <= 2 else "9-12"


def test_generation_is_reproducible(tmp_path):
    first = generate_pairs(load_config(_small_config(tmp_path)))
    second = generate_pairs(load_config(_small_config(tmp_path)))
    assert first.records == second.records
    assert first.attempts == second.attempts


def test_a_different_seed_gives_different_pairs(tmp_path):
    base = generate_pairs(load_config(_small_config(tmp_path)))
    other = generate_pairs(load_config(_small_config(tmp_path / "b", seed=999)))
    assert {r["canonical_hash"] for r in base.records} != {
        r["canonical_hash"] for r in other.records
    }


def test_excluded_hashes_are_never_reused(tmp_path):
    """§Step 7: pilot pairs must not reappear in the main set."""
    first = generate_pairs(load_config(_small_config(tmp_path)))
    exclusions = tmp_path / "exclude.txt"
    write_text(
        exclusions, "".join(f"{r['canonical_hash']}\n" for r in first.records)
    )

    path = _small_config(tmp_path / "second")
    data = yaml.safe_load(read_text(path))
    data["pairs"]["exclude_hashes_from"] = str(exclusions)
    write_text(path, yaml.safe_dump(data, sort_keys=False))

    second = generate_pairs(load_config(path))
    assert second.complete
    excluded = {r["canonical_hash"] for r in first.records}
    assert not excluded & {r["canonical_hash"] for r in second.records}


def test_unfillable_quota_reports_incomplete(tmp_path):
    """A quota that cannot be met must fail loudly rather than return a short set."""
    path = _small_config(tmp_path)
    data = yaml.safe_load(read_text(path))
    # A core already at the target size leaves nothing to expand, so both sides come out
    # identical and the non-isomorphism requirement rejects every draw.
    data["pairs"]["neq_depth"]["bins"] = {}
    data["pairs"]["eq_iso"]["count"] = 0
    data["pairs"]["eq_expand"]["count"] = 1
    data["pairs"]["eq_expand"]["core_states"] = [14, 14]
    data["pairs"]["max_generation_attempts_per_pair"] = 30
    write_text(path, yaml.safe_dump(data, sort_keys=False))

    outcome = generate_pairs(load_config(path))
    assert not outcome.complete
    assert any(reason.endswith("quota_unfilled") for reason in outcome.rejections)
    assert outcome.rejections["EQ_EXPAND:isomorphic_expansions"] > 0


def test_a_depth_bin_too_deep_for_the_state_count_is_a_config_error(tmp_path):
    """Caught before any work, with a message that says which knob to change."""
    path = _small_config(tmp_path)
    data = yaml.safe_load(read_text(path))
    data["pairs"]["neq_depth"]["state_count"] = 10
    data["pairs"]["neq_depth"]["bins"] = {"9-12": 1}
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match=r"needs at least 13 states"):
        generate_pairs(load_config(path))


def test_malformed_depth_bin_is_rejected(tmp_path):
    path = _small_config(tmp_path)
    data = yaml.safe_load(read_text(path))
    data["pairs"]["neq_depth"]["bins"] = {"one to two": 1}
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match=r"must look like"):
        generate_pairs(load_config(path))


# --------------------------------------------------------------------------- CLI


def test_cli_writes_every_artifact(tmp_path):
    path = _small_config(tmp_path)
    assert main(["--config", str(path)]) == 0
    pairs_dir = tmp_path / "pairs"
    assert (pairs_dir / "pairs.jsonl").is_file()
    assert (pairs_dir / "canonical_hashes.txt").is_file()
    assert (pairs_dir / "summary.json").is_file()
    assert (pairs_dir / "rejections.json").is_file()

    summary = json.loads(read_text(pairs_dir / "summary.json"))
    assert summary["pairs_generated"] == 8
    assert summary["complete"] is True


def test_cli_hash_list_matches_the_records(tmp_path):
    path = _small_config(tmp_path)
    main(["--config", str(path)])
    pairs_dir = tmp_path / "pairs"
    listed = read_text(pairs_dir / "canonical_hashes.txt").split()
    recorded = [
        json.loads(line)["canonical_hash"]
        for line in read_text(pairs_dir / "pairs.jsonl").splitlines()
        if line.strip()
    ]
    assert listed == recorded


def test_cli_refuses_to_overwrite_by_default(tmp_path, capsys):
    """Generated pairs are immutable: silently regenerating would orphan collected
    responses (research rule 3)."""
    path = _small_config(tmp_path)
    assert main(["--config", str(path)]) == 0
    before = sha256_of_file(tmp_path / "pairs" / "pairs.jsonl")

    assert main(["--config", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert sha256_of_file(tmp_path / "pairs" / "pairs.jsonl") == before


def test_cli_overwrite_reproduces_identical_bytes(tmp_path):
    path = _small_config(tmp_path)
    main(["--config", str(path)])
    before = sha256_of_file(tmp_path / "pairs" / "pairs.jsonl")
    assert main(["--config", str(path), "--overwrite"]) == 0
    assert sha256_of_file(tmp_path / "pairs" / "pairs.jsonl") == before


def test_cli_reports_a_config_error(tmp_path, capsys):
    """A config error exits 2 with a message, rather than a traceback or a silent pass."""
    path = tmp_path / "unfrozen.yaml"
    write_text(path, yaml.safe_dump({"config_id": "x", "seed": "FILL_AT_FREEZE"}))
    assert main(["--config", str(path)]) == 2
    assert "config error" in capsys.readouterr().err
