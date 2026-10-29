"""Tests for the eight dataset QA gates (M5, Step 6).

A gate that cannot fail is worthless, so every gate here is checked twice: once against a
clean dataset, and once against a dataset corrupted in precisely the way that gate exists to
catch.
"""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from src.config import PLACEHOLDERS, ConfigError, load_config
from src.dataset_qa import (
    _load_excluded,
    gate_1_oracle_agreement,
    gate_2_dataset_balance,
    gate_3_no_epsilon_leak,
    gate_4_completeness,
    gate_5_reachability,
    gate_6_representation_fidelity,
    gate_7_deduplication,
    gate_8_token_balance,
    main,
    run_all_gates,
    token_counter_from_config,
)
from src.generate import main as generate_main
from src.io_utils import read_jsonl, read_text, write_text
from src.render_prompts import main as render_main

LIBRARIES = ["pyformlang", "automata-lib"]


def _config(tmp_path, **pairs_overrides):
    data = {
        "config_id": "qa_test",
        "seed": 8080,
        "alphabet": ["0", "1"],
        "pairs": {
            "neq_depth": {"state_count": 14, "bins": {"1-2": 2, "9-12": 2}},
            "eq_iso": {"count": 1, "state_count": 14},
            "eq_expand": {
                "count": 1,
                "state_count": 14,
                "core_states": [10, 12],
                "require_non_isomorphic": True,
            },
            "max_generation_attempts_per_pair": 2000,
        },
        "representations": [{"id": "R0"}, {"id": "R1"}],
        "prompts": {
            "system": "prompts/system_v1.txt",
            "conditions": [
                {
                    "id": "P0",
                    "name": "P0_direct",
                    "template": "prompts/p0_direct_v1.txt",
                    "schema": "prompts/schema_p0.json",
                },
                {
                    "id": "P1",
                    "name": "P1_trace",
                    "template": "prompts/p1_trace_v1.txt",
                    "schema": "prompts/schema_p1.json",
                },
            ],
        },
        "qa": {
            "cross_check_libraries": LIBRARIES,
            "token_balance_tolerance": 0.10,
        },
        "paths": {
            "pairs_dir": str(tmp_path / "pairs"),
            "renderings_dir": str(tmp_path / "renderings"),
        },
    }
    data["pairs"].update(pairs_overrides)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def dataset(tmp_path):
    """A small generated and rendered dataset, plus its config path."""
    path = _config(tmp_path)
    assert generate_main(["--config", str(path)]) == 0
    assert render_main(["--config", str(path)]) == 0
    config = load_config(path)
    records = [
        json.loads(line)
        for line in read_text(tmp_path / "pairs" / "pairs.jsonl").splitlines()
        if line.strip()
    ]
    return config, records, tmp_path


# --------------------------------------------------------------------------- clean run


def test_a_clean_dataset_passes_every_runnable_gate(dataset):
    config, _, _ = dataset
    results = run_all_gates(config)
    assert [r.status for r in results] == ["PASS"] * 7 + ["SKIP"]
    assert results[7].number == 8  # only token balance skips, for want of a tokenizer


# --------------------------------------------------------------------------- gate 1


def test_gate_1_catches_a_flipped_stored_label(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    corrupted[0]["oracle_equivalent"] = not corrupted[0]["oracle_equivalent"]
    result = gate_1_oracle_agreement(corrupted, LIBRARIES)
    assert result.status == "FAIL"
    assert "fresh oracle run" in result.examples[0]


def test_gate_1_catches_a_tampered_transition(dataset):
    """Editing the automaton without editing its label must not slip through."""
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    target = next(r for r in corrupted if r["family"] == "EQ_ISO")
    target["dfa_A"]["accepting"] = []  # now almost certainly a different language
    result = gate_1_oracle_agreement(corrupted, LIBRARIES)
    assert result.status == "FAIL"


def test_gate_1_skips_when_no_library_is_configured(dataset):
    _, records, _ = dataset
    assert gate_1_oracle_agreement(records, []).status == "SKIP"


# --------------------------------------------------------------------------- gate 2


def test_gate_2_catches_a_short_stratum(dataset):
    config, records, _ = dataset
    result = gate_2_dataset_balance([r for r in records if r["family"] != "NEQ_DEPTH"], config)
    assert result.status == "FAIL"
    assert any("expected 2 pairs, found 0" in example for example in result.examples)


def test_gate_2_catches_a_depth_outside_every_bin(dataset):
    config, records, _ = dataset
    corrupted = copy.deepcopy(records)
    target = next(r for r in corrupted if r["family"] == "NEQ_DEPTH")
    target["oracle_depth"] = 7  # the test config declares only 1-2 and 9-12
    result = gate_2_dataset_balance(corrupted, config)
    assert result.status == "FAIL"
    assert any("in no declared bin" in example for example in result.examples)


# --------------------------------------------------------------------------- gate 3


def test_gate_3_catches_an_epsilon_cue(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    target = corrupted[0]
    start = target["dfa_A"]["start"]
    accepting = set(target["dfa_A"]["accepting"]) ^ {start}
    target["dfa_A"]["accepting"] = sorted(accepting)
    result = gate_3_no_epsilon_leak(corrupted, )
    assert result.status == "FAIL"
    assert "start states disagree" in result.examples[0]


def test_gate_3_catches_a_zero_depth_pair(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    next(r for r in corrupted if r["family"] == "NEQ_DEPTH")["oracle_depth"] = 0
    result = gate_3_no_epsilon_leak(corrupted)
    assert result.status == "FAIL"
    assert any("depth is 0" in example for example in result.examples)


# --------------------------------------------------------------------------- gate 4


def test_gate_4_catches_a_missing_transition(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    corrupted[0]["dfa_A"]["delta"][3] = [0]  # one column instead of two
    result = gate_4_completeness(corrupted)
    assert result.status == "FAIL"
    assert "has 1 transitions" in result.examples[0]


def test_gate_4_catches_a_missing_row(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    corrupted[0]["dfa_A"]["delta"].pop()
    result = gate_4_completeness(corrupted)
    assert result.status == "FAIL"
    assert "rows for 14 states" in result.examples[0]


# --------------------------------------------------------------------------- gate 5


def test_gate_5_catches_an_unreachable_state(dataset):
    """Point every edge away from state 13 so nothing can reach it."""
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    delta = corrupted[0]["dfa_A"]["delta"]
    corrupted[0]["dfa_A"]["start"] = 0
    for row in delta:
        for column, destination in enumerate(row):
            if destination == 13:
                row[column] = 0
    result = gate_5_reachability(corrupted)
    assert result.status == "FAIL"
    assert "unreachable states [13]" in result.examples[0]


# --------------------------------------------------------------------------- gate 6


def test_gate_6_catches_an_edited_rendering(dataset):
    """A rendering that no longer denotes its stored automaton invalidates H3."""
    config, records, tmp_path = dataset
    renderings = tmp_path / "renderings"
    target = records[0]["pair_id"]
    path = renderings / f"{target}_A_R0.txt"
    text = read_text(path)
    # Redirect one transition, changing the language the rendering describes.
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if " -> " in line)
    source, _, _ = lines[index].partition(" -> ")
    lines[index] = f"{source} -> q00"
    write_text(path, "\n".join(lines) + "\n")

    result = gate_6_representation_fidelity(records, renderings, ["R0", "R1"])
    assert result.status == "FAIL"
    assert target in result.examples[0]


def test_gate_6_catches_a_corrupt_rendering(dataset):
    config, records, tmp_path = dataset
    renderings = tmp_path / "renderings"
    path = renderings / f"{records[0]['pair_id']}_B_R1.txt"
    write_text(path, "DFA B\nalphabet: 0, 1\ninitial: aaa\ntransitions:\n")
    result = gate_6_representation_fidelity(records, renderings, ["R0", "R1"])
    assert result.status == "FAIL"
    assert "DFAValidationError" in result.examples[0]


def test_gate_6_skips_without_renderings(dataset, tmp_path):
    _, records, _ = dataset
    result = gate_6_representation_fidelity(records, tmp_path / "absent", ["R0", "R1"])
    assert result.status == "SKIP"
    assert "render_prompts" in result.detail["reason"]


# --------------------------------------------------------------------------- gate 7


def test_gate_7_catches_a_duplicate_pair(dataset):
    _, records, _ = dataset
    corrupted = [*copy.deepcopy(records), copy.deepcopy(records[0])]
    result = gate_7_deduplication(corrupted, set(), None)
    assert result.status == "FAIL"
    assert "duplicate canonical hash" in result.examples[0]


def test_gate_7_catches_a_pair_reused_from_the_pilot(dataset):
    """§Step 7: a pilot pair reappearing in the main set is contamination."""
    _, records, _ = dataset
    excluded = {records[2]["canonical_hash"]}
    result = gate_7_deduplication(records, excluded, "pilot_hashes.txt")
    assert result.status == "FAIL"
    assert "pilot_hashes.txt" in result.examples[0]


def test_gate_7_passes_on_a_clean_set(dataset):
    _, records, _ = dataset
    result = gate_7_deduplication(records, set(), None)
    assert result.status == "PASS"
    assert result.detail["distinct_hashes"] == len(records)


# --------------------------------------------------------------------------- gate 8


def test_gate_8_passes_with_a_balanced_tokenizer(dataset):
    """A whitespace tokenizer stands in for a model's: R0 and R1 have identical shape."""
    _, _, tmp_path = dataset
    result = gate_8_token_balance(
        tmp_path / "renderings", lambda text: len(text.split()), 0.10
    )
    assert result.status == "PASS"
    assert result.detail["mean_ratio"] == pytest.approx(1.0)
    assert result.checked > 0


def test_gate_8_catches_a_systematic_length_difference(dataset):
    """A tokenizer whose per-label cost varies must still trip the tolerance.

    Under the shared-vocabulary scheme (D5.4) R0 and R1 use the same label strings, so a
    cost difference can only come from the derangement *reweighting* how often each label
    appears. This charges heavily for one label to make that reweighting visible -- the
    residual the structural check bounds but cannot eliminate.
    """
    _, _, tmp_path = dataset

    def biased(text: str) -> int:
        return sum(50 if word.startswith("q00") else 1 for word in text.split())

    result = gate_8_token_balance(tmp_path / "renderings", biased, 0.10)
    assert result.status == "FAIL"
    assert "outside" in result.examples[0]


def test_gate_8_structural_check_is_not_inert(dataset):
    """It must actually find labels, or it passes by inspecting nothing.

    An earlier revision of this check compiled a broken regex, matched zero labels, and
    reported a clean structural pass on every dataset -- the same failure mode gate 4 had.
    Asserting a *non-zero* observation is what distinguishes checking from pretending.
    """
    from src.dataset_qa import structural_label_balance

    _, _, tmp_path = dataset
    pair_ids = [r["pair_id"] for r in read_jsonl(tmp_path / "pairs" / "pairs.jsonl")]
    failures, detail = structural_label_balance(tmp_path / "renderings", pair_ids)

    assert failures == []
    assert detail["structural_machines_checked"] > 0
    # The derangement necessarily reweights label frequencies, so a zero here would mean the
    # counter never saw a label.
    assert detail["worst_label_frequency_l1"] > 0


def test_gate_8_fails_when_r1_reverts_to_a_different_vocabulary(dataset):
    """The regression this guards: someone reintroduces nonce labels under a v2 config."""
    from src.dataset_qa import structural_label_balance

    _, _, tmp_path = dataset
    renderings = tmp_path / "renderings"
    pair_ids = [r["pair_id"] for r in read_jsonl(tmp_path / "pairs" / "pairs.jsonl")]
    target = renderings / f"{pair_ids[0]}_A_R1.txt"
    target.write_text(
        target.read_text(encoding="utf-8").replace("q0", "zu"), encoding="utf-8"
    )

    failures, _ = structural_label_balance(renderings, pair_ids)
    assert any("different label vocabularies" in f for f in failures)

    # And the gate itself fails on it, with no tokenizer available.
    result = gate_8_token_balance(renderings, None, 0.10, pair_ids)
    assert result.status == "FAIL"


def test_gate_8_does_not_apply_the_structural_check_to_legacy_nonce_data(dataset):
    """The v1 pilot was rendered before D5.4 and must not be condemned by it (D2)."""
    _, _, tmp_path = dataset
    renderings = tmp_path / "renderings"
    pair_ids = [r["pair_id"] for r in read_jsonl(tmp_path / "pairs" / "pairs.jsonl")]
    target = renderings / f"{pair_ids[0]}_A_R1.txt"
    target.write_text(
        target.read_text(encoding="utf-8").replace("q0", "zu"), encoding="utf-8"
    )

    result = gate_8_token_balance(renderings, None, 0.10, pair_ids, "nonce_cvc")
    assert result.status == "SKIP"
    assert "not applicable" in str(result.detail.get("structural_check"))


def test_gate_8_skips_without_a_tokenizer(dataset):
    _, _, tmp_path = dataset
    result = gate_8_token_balance(tmp_path / "renderings", None, 0.10)
    assert result.status == "SKIP"
    assert "tokenizer_for_token_balance" in result.detail["reason"]


def test_gate_8_skips_without_renderings(tmp_path):
    result = gate_8_token_balance(tmp_path / "absent", lambda t: len(t), 0.10)
    assert result.status == "SKIP"


def test_token_counter_is_none_for_a_placeholder():
    """A config that cannot name a tokenizer yet gets a skip, not a crash.

    Asserted against configs that still hold a placeholder rather than against whichever
    config happens to be unfilled today: the property under test belongs to
    ``token_counter_from_config``, not to the contents of one file.
    """
    assert token_counter_from_config(load_config("configs/pilot.yaml")) is None
    assert token_counter_from_config(load_config("configs/pilot_revised.yaml")) is None


def test_main_config_names_a_real_tokenizer():
    """Gate 8 must have a real instrument before the freeze.

    Checked at the config level so the test stays offline; building the counter would
    download from the HF hub. A placeholder here would make gate 8 report SKIP, and
    ``--strict`` exists precisely so a skipped gate 8 cannot be mistaken for a pass.
    """
    name = load_config("configs/main_frozen.yaml").get("qa.tokenizer_for_token_balance")
    assert name and str(name) not in PLACEHOLDERS


# --------------------------------------------------------------------------- CLI


def test_cli_passes_on_a_clean_dataset(tmp_path, capsys):
    path = _config(tmp_path)
    generate_main(["--config", str(path)])
    render_main(["--config", str(path)])
    assert main(["--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("[PASS]") == 7
    assert "[SKIP]" in captured.out


def test_cli_strict_mode_fails_on_a_skip(tmp_path, capsys):
    """Required before the freeze: every gate must actually have run."""
    path = _config(tmp_path)
    generate_main(["--config", str(path)])
    render_main(["--config", str(path)])
    assert main(["--config", str(path), "--strict"]) == 1
    assert "FAILED (--strict)" in capsys.readouterr().err


def test_cli_fails_when_a_gate_fails(tmp_path, capsys):
    path = _config(tmp_path)
    generate_main(["--config", str(path)])
    render_main(["--config", str(path)])
    # Break a rendering so gate 6 fires.
    renderings = tmp_path / "renderings"
    target = next(renderings.glob("*_A_R0.txt"))
    write_text(target, "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\ntransitions:\n")
    assert main(["--config", str(path)]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_cli_writes_a_report(tmp_path):
    path = _config(tmp_path)
    generate_main(["--config", str(path)])
    render_main(["--config", str(path)])
    main(["--config", str(path)])
    report = json.loads(read_text(tmp_path / "pairs" / "qa_report.json"))
    assert len(report["gates"]) == 8
    assert report["n_failed"] == 0
    assert report["n_skipped"] == 1
    assert report["all_gates_ran"] is False
    assert [g["gate"] for g in report["gates"]] == list(range(1, 9))


def test_cli_reports_a_missing_pair_set(tmp_path, capsys):
    path = _config(tmp_path)
    assert main(["--config", str(path)]) == 2
    assert "src.generate" in capsys.readouterr().err


# --------------------------------------------------------------------------- review regressions
#
# Six ways QA could approve a broken dataset, each found by an external review of M1-M6 and
# each reproduced before being fixed. They are grouped here so the class of failure stays
# visible: in every case the gate returned PASS while the thing it exists to protect was
# absent or wrong.


def test_a_configured_but_missing_exclusion_file_is_fatal(tmp_path):
    """Review finding 1. An empty exclusion set turns gate 7's contamination check into a
    no-op precisely when it matters: the main run would regenerate the pilot pairs and
    nothing downstream would notice."""
    path = _config(tmp_path)
    data = yaml.safe_load(read_text(path))
    data["pairs"]["exclude_hashes_from"] = str(tmp_path / "absent.txt")
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    generate_main(["--config", str(path)])
    with pytest.raises(ConfigError, match=r"does not exist"):
        _load_excluded(load_config(path))


def test_generation_also_refuses_a_missing_exclusion_file(tmp_path, capsys):
    path = _config(tmp_path)
    data = yaml.safe_load(read_text(path))
    data["pairs"]["exclude_hashes_from"] = str(tmp_path / "absent.txt")
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert generate_main(["--config", str(path)]) == 2
    assert "contamination guard" in capsys.readouterr().err


def test_an_empty_manifest_fails_gates_6_and_8(dataset, tmp_path):
    """Review finding 2a. Passing after checking zero renderings is the most dangerous way
    for QA to succeed."""
    _, records, _ = dataset
    empty = tmp_path / "empty_renderings"
    empty.mkdir()
    write_text(empty / "manifest.jsonl", "")
    ids = [r["pair_id"] for r in records]
    assert gate_6_representation_fidelity(records, empty, ["R0", "R1"]).status == "FAIL"
    assert gate_8_token_balance(empty, lambda t: len(t.split()), 0.10, ids).status == "FAIL"


def test_a_missing_family_fails_gate_2(dataset):
    """Review finding 2b. Balanced accuracy over one class is not balanced accuracy."""
    config, records, _ = dataset
    result = gate_2_dataset_balance(
        [r for r in records if r["family"] == "NEQ_DEPTH"], config
    )
    assert result.status == "FAIL"
    assert any("family EQ_ISO" in example for example in result.examples)


def test_a_partial_rendering_set_fails_gate_6(dataset, tmp_path):
    """Coverage, not just fidelity: a pair rendered under R0 but not R1 has no H3 cell."""
    _, records, source = dataset
    partial = tmp_path / "partial"
    partial.mkdir()
    rows = [
        row
        for row in read_jsonl(source / "renderings" / "manifest.jsonl")
        if row["rendering"] == "R0"
    ]
    for row in rows:
        for name in ("A", "B"):
            filename = f"{row['pair_id']}_{name}_R0.txt"
            write_text(partial / filename, read_text(source / "renderings" / filename))
    write_text(
        partial / "manifest.jsonl",
        "".join(json.dumps(row) + "\n" for row in rows),
    )
    result = gate_6_representation_fidelity(records, partial, ["R0", "R1"])
    assert result.status == "FAIL"
    assert any("never rendered" in example for example in result.examples)


def test_a_reversed_label_bijection_fails_gate_6(dataset, tmp_path):
    """Review finding 3. A reversed mapping still describes an isomorphic automaton, so
    comparing canonical forms cannot see it -- but M8 maps model traces through this
    mapping, and a wrong one would mark correct traces wrong."""
    _, records, source = dataset
    mangled = tmp_path / "mangled"
    mangled.mkdir()
    rows = []
    for row in read_jsonl(source / "renderings" / "manifest.jsonl"):
        for name in ("A", "B"):
            filename = f"{row['pair_id']}_{name}_{row['rendering']}.txt"
            write_text(mangled / filename, read_text(source / "renderings" / filename))
        bad = dict(row)
        bad["labels_A"] = list(reversed(row["labels_A"]))
        rows.append(bad)
    write_text(
        mangled / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in rows)
    )
    result = gate_6_representation_fidelity(records, mangled, ["R0", "R1"])
    assert result.status == "FAIL"
    assert any("label mapping" in example for example in result.examples)


def test_gate_6_rejects_a_manifest_with_no_labels(dataset, tmp_path):
    _, records, source = dataset
    stripped = tmp_path / "stripped"
    stripped.mkdir()
    rows = []
    for row in read_jsonl(source / "renderings" / "manifest.jsonl"):
        for name in ("A", "B"):
            filename = f"{row['pair_id']}_{name}_{row['rendering']}.txt"
            write_text(stripped / filename, read_text(source / "renderings" / filename))
        rows.append({k: v for k, v in row.items() if k != "labels_A"})
    write_text(
        stripped / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in rows)
    )
    result = gate_6_representation_fidelity(records, stripped, ["R0", "R1"])
    assert result.status == "FAIL"
    assert any("no label list" in example for example in result.examples)


def test_a_forged_hash_cannot_hide_a_duplicate(dataset):
    """Review finding 4a. A stored hash is an assertion about a pair, not evidence."""
    _, records, _ = dataset
    duplicated = [*copy.deepcopy(records), copy.deepcopy(records[0])]
    duplicated[-1]["pair_id"] = "p_999999"
    duplicated[-1]["canonical_hash"] = "sha256:" + "f" * 64
    result = gate_7_deduplication(duplicated, set(), None)
    assert result.status == "FAIL"
    assert any("hashes to" in example for example in result.examples)


def test_a_forged_hash_cannot_evade_the_exclusion_list(dataset):
    _, records, _ = dataset
    excluded = {records[0]["canonical_hash"]}
    tampered = copy.deepcopy(records)
    tampered[0]["canonical_hash"] = "sha256:" + "a" * 64  # pretend to be a different pair
    result = gate_7_deduplication(tampered, excluded, "pilot.txt")
    assert result.status == "FAIL"


def test_a_corrupted_witness_fails_gate_1(dataset):
    """Review finding 4b. The witness grades §10.1's shortest-witness metric and the depth
    *is* H1's explanatory variable; checking only the boolean protects neither."""
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    for record in corrupted:
        if not record["oracle_equivalent"]:
            record["oracle_witness"] = ""
    result = gate_1_oracle_agreement(corrupted, LIBRARIES)
    assert result.status == "FAIL"
    assert any("fresh oracle run" in example for example in result.examples)


def test_a_corrupted_depth_fails_gate_1(dataset):
    _, records, _ = dataset
    corrupted = copy.deepcopy(records)
    target = next(r for r in corrupted if not r["oracle_equivalent"])
    target["oracle_depth"] = target["oracle_depth"] + 3
    result = gate_1_oracle_agreement(corrupted, LIBRARIES)
    assert result.status == "FAIL"


def test_pilot_dataset_passes_qa():
    """The committed pilot set must stay green as the code evolves."""
    config = load_config("configs/pilot.yaml")
    if not (config.path_for("pairs_dir") / "pairs.jsonl").is_file():
        pytest.skip("pilot set not generated in this checkout")
    results = run_all_gates(config)
    failures = [r for r in results if r.status == "FAIL"]
    assert not failures, [r.examples for r in failures]


def test_a_dataset_may_be_reused_under_a_differently_named_config() -> None:
    """One dataset, several runs. A provider change is a new run over the same pairs.

    Gate 2 checks dataset identity because `p_000000` is not unique across splits. Forcing
    the config_id to equal the dataset_id would have meant either renaming the dataset for
    every execution environment, or letting two runs share run_ids.
    """
    from src.config import load_config
    from src.dataset_qa import gate_2_dataset_balance

    config = load_config("configs/pilot_openrouter.yaml")
    assert config.config_id == "pilot_v2_openrouter"
    assert config.get("dataset_id") == "pilot_v2"

    records = list(read_jsonl("data/pairs/pilot_v2/pairs.jsonl"))
    result = gate_2_dataset_balance(records, config)
    assert result.status == "PASS", result.examples
