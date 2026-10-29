"""Tests for UTF-8 I/O, canonical JSON, and hashing (M1)."""

from __future__ import annotations

import json

import pytest

from src.io_utils import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    read_text,
    sha256_hex,
    sha256_of_file,
    sha256_of_json,
    write_text,
)

# --------------------------------------------------------------------------- canonical JSON


def test_canonical_json_is_key_order_independent():
    """The property the whole hashing scheme rests on."""
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_has_no_incidental_whitespace():
    assert canonical_json({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_canonical_json_keeps_non_ascii_literal():
    """Protocol text carries Sigma and delta; escaping them would change the digest."""
    assert canonical_json({"sym": "Σδ"}) == '{"sym":"Σδ"}'


def test_canonical_json_round_trips():
    obj = {"z": [1, {"y": None}], "a": True, "n": 1.5}
    assert json.loads(canonical_json(obj)) == obj


# --------------------------------------------------------------------------- hashing


def test_hashes_carry_the_recorded_prefix():
    digest = sha256_hex("abc")
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_known_sha256_vector():
    """Guards against an accidental change of digest or encoding."""
    assert sha256_hex("abc") == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_str_and_bytes_hash_identically():
    assert sha256_hex("Σ") == sha256_hex("Σ".encode())


def test_sha256_of_json_ignores_key_order():
    assert sha256_of_json({"b": 1, "a": 2}) == sha256_of_json({"a": 2, "b": 1})


def test_sha256_of_file_matches_the_bytes(tmp_path):
    path = tmp_path / "sample.txt"
    write_text(path, "Σδ line\n")
    assert sha256_of_file(path) == sha256_hex("Σδ line\n".encode())


def test_file_hash_is_line_ending_stable(tmp_path):
    """write_text must emit LF so freeze hashes match between Windows and the cluster."""
    path = tmp_path / "prompt.txt"
    write_text(path, "line one\nline two\n")
    assert path.read_bytes() == b"line one\nline two\n"


# --------------------------------------------------------------------------- text I/O


def test_write_then_read_round_trips_non_ascii(tmp_path):
    path = tmp_path / "nested" / "dir" / "protocol.txt"
    text = "A ≡ B iff L(A) △ L(B) = ∅\n"
    write_text(path, text)
    assert read_text(path) == text


def test_write_text_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c.txt"
    write_text(path, "x")
    assert path.exists()


# --------------------------------------------------------------------------- JSONL


def test_append_jsonl_accumulates_records(tmp_path):
    """Raw responses are append-only (research rule 3): a second write must not truncate."""
    path = tmp_path / "raw" / "responses.jsonl"
    append_jsonl(path, {"run_id": "a", "raw": "first"})
    append_jsonl(path, {"run_id": "b", "raw": "second"})
    records = list(read_jsonl(path))
    assert [r["run_id"] for r in records] == ["a", "b"]


def test_append_jsonl_writes_one_line_per_record(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})
    assert path.read_bytes() == b'{"a":1}\n{"b":2}\n'


def test_append_jsonl_survives_embedded_newlines(tmp_path):
    """A raw model response containing newlines must stay one JSONL record."""
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"raw": "line one\nline two"})
    records = list(read_jsonl(path))
    assert len(records) == 1
    assert records[0]["raw"] == "line one\nline two"


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    write_text(path, '{"a":1}\n\n{"b":2}\n')
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_reports_the_offending_line(tmp_path):
    """A corrupted raw log must fail loudly, never be silently skipped."""
    path = tmp_path / "log.jsonl"
    write_text(path, '{"a":1}\nnot json\n')
    with pytest.raises(json.JSONDecodeError, match=r"line 2"):
        list(read_jsonl(path))
