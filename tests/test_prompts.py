"""Tests for prompt assembly and the leak guard (M6, §8)."""

from __future__ import annotations

import json

import pytest
import yaml

from src.config import ConfigError, load_config
from src.io_utils import read_text, sha256_hex, write_text
from src.prompts import (
    PLACEHOLDER_A,
    PLACEHOLDER_B,
    Prompt,
    assert_no_leak,
    build_messages,
    build_user_message,
    load_prompt_assets,
)

PILOT = "configs/pilot.yaml"
RENDER_A = "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: q01\ntransitions:\nq00, 0 -> q01"
RENDER_B = "DFA B\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\ntransitions:\nq00, 0 -> q00"

CLEAN_RECORD = {
    "pair_id": "p_000042",
    "seed": 918273645,
    "family": "NEQ_DEPTH",
    "oracle_witness": "0110",
    "oracle_depth": 4,
}


# --------------------------------------------------------------------------- substitution


def test_substitution_places_both_renderings():
    template = f"before\n{PLACEHOLDER_A}\nmiddle\n{PLACEHOLDER_B}\nafter"
    user = build_user_message(template, RENDER_A, RENDER_B)
    assert RENDER_A in user
    assert RENDER_B in user
    assert PLACEHOLDER_A not in user and PLACEHOLDER_B not in user


def test_literal_json_braces_survive_substitution():
    """The reason substitution uses str.replace and never str.format.

    Both frozen templates end with a JSON skeleton. ``format`` would read those braces as
    replacement fields and either raise or silently mangle the frozen prompt text.
    """
    assets = load_prompt_assets(load_config(PILOT))
    for condition in assets.conditions:
        user = build_user_message(condition.template, RENDER_A, RENDER_B)
        assert '{"equivalent": true|false' in user

    template = f'{PLACEHOLDER_A} {PLACEHOLDER_B} {{"equivalent": true|false}}'
    with pytest.raises((KeyError, IndexError, ValueError)):
        template.format(x=1)  # the failure mode being avoided
    assert '{"equivalent": true|false}' in build_user_message(
        template, RENDER_A, RENDER_B
    )


def test_substitution_rejects_a_template_missing_a_placeholder():
    with pytest.raises(ValueError, match=r"missing the \{DFA_B_RENDERING\} placeholder"):
        build_user_message(f"only {PLACEHOLDER_A}", RENDER_A, RENDER_B)


def test_substitution_rejects_a_rendering_that_reintroduces_a_placeholder():
    """A surviving placeholder would mean showing a model the template, not the problem.

    Substitution replaces A and then B, so a *B* rendering carrying the A placeholder is the
    case that survives; the reverse is consumed by the second pass.
    """
    template = f"{PLACEHOLDER_A}\n{PLACEHOLDER_B}"
    with pytest.raises(ValueError, match=r"survived substitution"):
        build_user_message(template, RENDER_A, PLACEHOLDER_A)


# --------------------------------------------------------------------------- messages


def test_system_role_placement_is_the_default():
    prompt = build_messages("SYS", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B)
    assert [m["role"] for m in prompt.messages] == ["system", "user"]
    assert prompt.messages[0]["content"] == "SYS"


def test_prepend_placement_folds_the_system_message_into_the_user_turn():
    """For chat templates that reject a system role -- a frozen, study-wide setting."""
    prompt = build_messages(
        "SYS", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B, "prepend_to_user"
    )
    assert [m["role"] for m in prompt.messages] == ["user"]
    assert prompt.user.startswith("SYS\n\n")
    assert RENDER_A in prompt.user


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError, match=r"unknown system_placement"):
        build_messages("SYS", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B, "nope")


def test_prompt_hash_is_stable_and_content_sensitive():
    first = build_messages("SYS", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B)
    same = build_messages("SYS", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B)
    other = build_messages("SYS!", f"{PLACEHOLDER_A}{PLACEHOLDER_B}", RENDER_A, RENDER_B)
    assert first.sha256 == same.sha256
    assert first.sha256 != other.sha256
    assert first.sha256.startswith("sha256:")


def test_placement_changes_the_hash():
    """The two placements are different interfaces and must not be confusable."""
    template = f"{PLACEHOLDER_A}{PLACEHOLDER_B}"
    a = build_messages("SYS", template, RENDER_A, RENDER_B, "system_role")
    b = build_messages("SYS", template, RENDER_A, RENDER_B, "prepend_to_user")
    assert a.sha256 != b.sha256


# --------------------------------------------------------------------------- leak guard


def test_a_clean_prompt_passes():
    assets = load_prompt_assets(load_config(PILOT))
    for condition in assets.conditions:
        prompt = build_messages(assets.system, condition.template, RENDER_A, RENDER_B)
        assert_no_leak(prompt, CLEAN_RECORD)


def test_the_word_witness_is_allowed():
    """Both frozen templates name the output field 'witness'; flagging it would fire on
    every legitimate prompt. The witness *value* is what must not appear."""
    assets = load_prompt_assets(load_config(PILOT))
    p1 = assets.condition("P1")
    assert "witness" in p1.template
    assert_no_leak(
        build_messages(assets.system, p1.template, RENDER_A, RENDER_B), CLEAN_RECORD
    )


@pytest.mark.parametrize(
    ("injected", "expected"),
    [
        ("this pair_id is 7", "forbidden word 'pair_id'"),
        ("oracle says no", "forbidden word 'oracle'"),
        ("shortest depth 4", "forbidden word 'depth'"),
        ("generation seed", "forbidden word 'seed'"),
        ("canonical_hash abc", "forbidden word 'canonical_hash'"),
    ],
)
def test_forbidden_words_are_caught(injected, expected):
    prompt = Prompt(system="SYS", user=f"{RENDER_A}\n{injected}\n{RENDER_B}")
    with pytest.raises(ValueError, match=expected):
        assert_no_leak(prompt, CLEAN_RECORD)


def test_a_leaked_pair_id_is_caught():
    prompt = Prompt(system=None, user=f"{RENDER_A}\nfrom p_000042\n{RENDER_B}")
    with pytest.raises(ValueError, match=r"contains its pair_id"):
        assert_no_leak(prompt, {**CLEAN_RECORD, "pair_id": "P_000042"})


def test_a_leaked_family_is_caught():
    """EQ_ISO rather than NEQ_DEPTH: the latter contains "depth", so the forbidden-word
    check fires first and this branch would never be exercised."""
    prompt = Prompt(system=None, user=f"{RENDER_A}\neq_iso stratum\n{RENDER_B}")
    with pytest.raises(ValueError, match=r"contains its family"):
        assert_no_leak(prompt, {**CLEAN_RECORD, "family": "EQ_ISO"})


def test_a_leaked_seed_is_caught():
    prompt = Prompt(system=None, user=f"{RENDER_A}\nbuilt from 918273645\n{RENDER_B}")
    with pytest.raises(ValueError, match=r"contains its generation seed"):
        assert_no_leak(prompt, CLEAN_RECORD)


def test_a_leaked_witness_is_caught():
    """The worst possible leak: the answer itself."""
    prompt = Prompt(system=None, user=f"{RENDER_A}\ntry 0110\n{RENDER_B}")
    with pytest.raises(ValueError, match=r"contains the oracle witness"):
        assert_no_leak(prompt, CLEAN_RECORD)


def test_a_short_witness_is_not_checked_by_value():
    """A one- or two-symbol witness would collide with ordinary transition rows, so
    checking it by value would fire on every clean prompt."""
    prompt = Prompt(system=None, user=f"{RENDER_A}\n{RENDER_B}")
    assert_no_leak(prompt, {**CLEAN_RECORD, "oracle_witness": "0"})
    assert_no_leak(prompt, {**CLEAN_RECORD, "oracle_witness": "01"})


def test_the_leak_guard_inspects_the_system_message_too():
    prompt = Prompt(system="the oracle knows", user=RENDER_A)
    with pytest.raises(ValueError, match=r"forbidden word 'oracle'"):
        assert_no_leak(prompt, CLEAN_RECORD)


# --------------------------------------------------------------------------- asset loading


def test_pilot_assets_load():
    assets = load_prompt_assets(load_config(PILOT))
    assert [c.id for c in assets.conditions] == ["P0", "P1"]
    assert assets.system_placement == "system_role"
    assert assets.system_sha256 == sha256_hex(read_text("prompts/system_v1.txt"))
    assert assets.condition("P1").schema["required"] == [
        "equivalent",
        "witness",
        "trace_A",
        "trace_B",
    ]


def test_unknown_condition_lookup_raises():
    assets = load_prompt_assets(load_config(PILOT))
    with pytest.raises(KeyError, match=r"no prompt condition 'P9'"):
        assets.condition("P9")


def _config_with(tmp_path, **prompt_overrides):
    data = yaml.safe_load(read_text(PILOT))
    data["prompts"].update(prompt_overrides)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    return load_config(path)


def test_a_frozen_system_hash_is_verified(tmp_path):
    """§14.2 requires the prompt text to be frozen *and hashed*. Recording a hash and never
    checking it would make the freeze decorative."""
    config = _config_with(tmp_path, system_sha256="sha256:" + "0" * 64)
    with pytest.raises(ConfigError, match=r"does not match the hash frozen"):
        load_prompt_assets(config)


def test_a_correct_frozen_system_hash_passes(tmp_path):
    actual = sha256_hex(read_text("prompts/system_v1.txt"))
    config = _config_with(tmp_path, system_sha256=actual)
    assert load_prompt_assets(config).system_sha256 == actual


def test_a_frozen_template_hash_is_verified(tmp_path):
    data = yaml.safe_load(read_text(PILOT))
    data["prompts"]["conditions"][0]["template_sha256"] = "sha256:" + "1" * 64
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match=r"does not match its frozen template_sha256"):
        load_prompt_assets(load_config(path))


def test_placeholder_hashes_are_not_enforced():
    """The main config carries FILL_AT_FREEZE until the freeze; that must still load."""
    assets = load_prompt_assets(load_config("configs/main_frozen.yaml"))
    assert len(assets.conditions) == 2


def test_a_missing_template_file_is_reported(tmp_path):
    data = yaml.safe_load(read_text(PILOT))
    data["prompts"]["conditions"][0]["template"] = "prompts/absent.txt"
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match=r"prompt asset not found"):
        load_prompt_assets(load_config(path))


def test_a_template_without_placeholders_is_rejected(tmp_path):
    bad = tmp_path / "bad_template.txt"
    write_text(bad, "no placeholders here")
    data = yaml.safe_load(read_text(PILOT))
    data["prompts"]["conditions"][0]["template"] = str(bad)
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ConfigError, match=r"is missing the \{DFA_A_RENDERING\} placeholder"):
        load_prompt_assets(load_config(path))


def test_an_unknown_system_placement_is_rejected(tmp_path):
    config = _config_with(tmp_path, system_placement="inline")
    with pytest.raises(ConfigError, match=r"system_placement must be"):
        load_prompt_assets(config)


def test_schemas_are_strict_contracts():
    """§8.5: extra keys are a parse failure, so the schema must forbid them."""
    for name in ("p0", "p1"):
        schema = json.loads(read_text(f"prompts/schema_{name}.json"))
        assert schema["additionalProperties"] is False
        assert schema["properties"]["witness"]["pattern"] == "^[01]*$"
