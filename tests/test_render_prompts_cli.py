"""Tests for the rendering CLI (M4, Step 5)."""

from __future__ import annotations

import json

import yaml

from src.config import load_config
from src.generate import main as generate_main
from src.io_utils import read_jsonl, read_text, sha256_of_file, write_text
from src.oracle import equivalence_and_witness
from src.render_prompts import main as render_main
from src.rendering import parse_rendering


def _prepared(tmp_path):
    """A tiny generated pair set plus the config that produced it."""
    data = {
        "config_id": "test",
        "seed": 606,
        "alphabet": ["0", "1"],
        "pairs": {
            "neq_depth": {"state_count": 14, "bins": {"1-2": 1, "9-12": 1}},
            "eq_iso": {"count": 1, "state_count": 14},
            "eq_expand": {
                "count": 1,
                "state_count": 14,
                "core_states": [10, 12],
                "require_non_isomorphic": True,
            },
            "max_generation_attempts_per_pair": 2000,
        },
        "representations": [
            {"id": "R0", "name": "R0_canonical"},
            {"id": "R1", "name": "R1_nonce_shuffled"},
        ],
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
        "paths": {
            "pairs_dir": str(tmp_path / "pairs"),
            "renderings_dir": str(tmp_path / "renderings"),
        },
    }
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert generate_main(["--config", str(path)]) == 0
    return path


def test_cli_writes_a_file_per_dfa_and_condition(tmp_path):
    path = _prepared(tmp_path)
    assert render_main(["--config", str(path)]) == 0
    renderings = tmp_path / "renderings"
    # 4 pairs x 2 machines x 2 conditions
    assert len(list(renderings.glob("p_*.txt"))) == 16
    assert (renderings / "manifest.jsonl").is_file()
    assert (renderings / "summary.json").is_file()


def test_manifest_covers_every_pair_and_condition(tmp_path):
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    rows = list(read_jsonl(tmp_path / "renderings" / "manifest.jsonl"))
    assert len(rows) == 8
    assert {row["rendering"] for row in rows} == {"R0", "R1"}
    for row in rows:
        assert len(row["labels_A"]) == 14
        assert len(set(row["labels_A"])) == 14
        assert row["text_sha256_A"].startswith("sha256:")


def test_manifest_hashes_match_the_written_files(tmp_path):
    """The manifest is what M8 grades against, so it must describe the actual bytes."""
    import hashlib

    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    renderings = tmp_path / "renderings"
    for row in read_jsonl(renderings / "manifest.jsonl"):
        for name in ("A", "B"):
            text = read_text(renderings / f"{row['pair_id']}_{name}_{row['rendering']}.txt")
            # The file carries a trailing newline the hashed rendering does not.
            digest = hashlib.sha256(text.rstrip("\n").encode("utf-8")).hexdigest()
            assert row[f"text_sha256_{name}"] == f"sha256:{digest}"


def test_manifest_never_contains_the_answer(tmp_path):
    """The bijection belongs in the manifest; the oracle verdict does not belong anywhere
    near the rendering pipeline's output (research rule 8)."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    for row in read_jsonl(tmp_path / "renderings" / "manifest.jsonl"):
        assert set(row) == {
            "pair_id",
            "rendering",
            "labels_A",
            "labels_B",
            "text_sha256_A",
            "text_sha256_B",
        }


def test_rendered_files_round_trip_to_the_recorded_oracle_verdict(tmp_path):
    """QA gate 6, end to end through the CLI."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    pairs = {
        record["pair_id"]: record
        for record in read_jsonl(tmp_path / "pairs" / "pairs.jsonl")
    }
    renderings = tmp_path / "renderings"
    for row in read_jsonl(renderings / "manifest.jsonl"):
        parsed = [
            parse_rendering(
                read_text(renderings / f"{row['pair_id']}_{name}_{row['rendering']}.txt")
            ).dfa
            for name in ("A", "B")
        ]
        result = equivalence_and_witness(*parsed)
        record = pairs[row["pair_id"]]
        assert result.equivalent == record["oracle_equivalent"]
        assert result.shortest_depth == record["oracle_depth"]


def test_rendering_is_reproducible(tmp_path):
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    manifest = tmp_path / "renderings" / "manifest.jsonl"
    before = sha256_of_file(manifest)
    assert render_main(["--config", str(path), "--overwrite"]) == 0
    assert sha256_of_file(manifest) == before


def test_cli_refuses_to_overwrite_by_default(tmp_path, capsys):
    path = _prepared(tmp_path)
    assert render_main(["--config", str(path)]) == 0
    assert render_main(["--config", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_reports_missing_pairs(tmp_path, capsys):
    data = {
        "config_id": "test",
        "seed": 1,
        "alphabet": ["0", "1"],
        "representations": [{"id": "R0"}],
        "paths": {
            "pairs_dir": str(tmp_path / "nothing"),
            "renderings_dir": str(tmp_path / "renderings"),
        },
    }
    path = tmp_path / "config.yaml"
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert render_main(["--config", str(path)]) == 1
    assert "src.generate" in capsys.readouterr().err


def test_summary_counts_match(tmp_path):
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    summary = json.loads(read_text(tmp_path / "renderings" / "summary.json"))
    assert summary["by_condition"] == {"R0": 4, "R1": 4}
    assert summary["files_written"] == 16


def test_representation_conditions_come_from_the_config(tmp_path):
    """Rendering only what the config declares, so a config edit cannot be bypassed."""
    path = _prepared(tmp_path)
    data = yaml.safe_load(read_text(path))
    data["representations"] = [{"id": "R0"}]
    write_text(path, yaml.safe_dump(data, sort_keys=False))
    assert render_main(["--config", str(path)]) == 0
    rows = list(read_jsonl(tmp_path / "renderings" / "manifest.jsonl"))
    assert {row["rendering"] for row in rows} == {"R0"}


def test_cli_writes_one_prompt_per_cell(tmp_path):
    """4 pairs x 2 representations x 2 prompt conditions."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    prompts = list(read_jsonl(tmp_path / "renderings" / "prompts.jsonl"))
    assert len(prompts) == 16
    assert len({p["prompt_id"] for p in prompts}) == 16
    assert {p["prompt_condition"] for p in prompts} == {"P0", "P1"}
    assert {p["rendering"] for p in prompts} == {"R0", "R1"}


def test_each_prompt_embeds_its_two_renderings_verbatim(tmp_path):
    """The prompt must contain exactly the bytes that were written as renderings."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    renderings = tmp_path / "renderings"
    for prompt in read_jsonl(renderings / "prompts.jsonl"):
        for name in ("A", "B"):
            text = read_text(
                renderings / f"{prompt['pair_id']}_{name}_{prompt['rendering']}.txt"
            ).rstrip("\n")
            assert text in prompt["user"]


def test_prompt_hashes_are_distinct_across_cells(tmp_path):
    """Every cell is a different prompt; a collision would mean a cell was duplicated."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    prompts = list(read_jsonl(tmp_path / "renderings" / "prompts.jsonl"))
    assert len({p["prompt_sha256"] for p in prompts}) == len(prompts)


def test_p0_and_p1_share_a_rendering_but_not_a_template(tmp_path):
    """§8.4: the same serialization in both prompt conditions, differing only in the ask."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    prompts = {
        p["prompt_id"]: p
        for p in read_jsonl(tmp_path / "renderings" / "prompts.jsonl")
    }
    pair = next(iter({p["pair_id"] for p in prompts.values()}))
    p0 = prompts[f"{pair}|R0|P0"]
    p1 = prompts[f"{pair}|R0|P1"]
    assert p0["system"] == p1["system"]
    assert p0["template_sha256"] != p1["template_sha256"]
    assert "trace_A" in p1["user"] and "trace_A" not in p0["user"]


def test_no_prompt_leaks_experiment_metadata(tmp_path):
    """Research rule 8, checked over the full assembled text of every cell."""
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    records = {
        r["pair_id"]: r for r in read_jsonl(tmp_path / "pairs" / "pairs.jsonl")
    }
    for prompt in read_jsonl(tmp_path / "renderings" / "prompts.jsonl"):
        record = records[prompt["pair_id"]]
        blob = f"{prompt['system']}\n{prompt['user']}"
        assert record["pair_id"] not in blob
        assert record["family"] not in blob
        assert str(record["seed"]) not in blob
        witness = record["oracle_witness"]
        if witness and len(witness) >= 3:
            assert witness not in blob


def test_prompts_are_reproducible(tmp_path):
    path = _prepared(tmp_path)
    render_main(["--config", str(path)])
    prompts_path = tmp_path / "renderings" / "prompts.jsonl"
    before = sha256_of_file(prompts_path)
    render_main(["--config", str(path), "--overwrite"])
    assert sha256_of_file(prompts_path) == before


def test_pilot_renderings_are_current():
    """The committed pilot renderings must match what the current code produces."""
    config = load_config("configs/pilot.yaml")
    pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
    renderings_dir = config.path_for("renderings_dir")
    if not (renderings_dir / "manifest.jsonl").is_file():
        return  # not generated in this checkout

    from src.render_prompts import render_all

    rows, _ = render_all(config, pairs_path)
    committed = list(read_jsonl(renderings_dir / "manifest.jsonl"))
    assert len(rows) == len(committed)
    for fresh, stored in zip(rows, committed, strict=True):
        assert fresh["text_sha256_A"] == stored["text_sha256_A"]
        assert fresh["text_sha256_B"] == stored["text_sha256_B"]
