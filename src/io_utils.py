"""UTF-8 file helpers, canonical JSON serialization, and hashing.

Every byte this project hashes goes through :func:`canonical_json`, so that a record
serialized on the Windows dev machine and on the Linux cluster produces the same digest
(project code conventions).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "append_jsonl",
    "canonical_json",
    "read_jsonl",
    "read_text",
    "sha256_hex",
    "sha256_of_file",
    "sha256_of_json",
    "write_text",
]


# --------------------------------------------------------------------------- text I/O


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a text file as UTF-8, with newlines normalized to ``\\n``.

    The explicit encoding matters: this project is developed on Windows, where the
    default is cp1252 and any Sigma or delta in a protocol file would raise.
    """
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write UTF-8 text with ``\\n`` line endings, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- JSONL


def append_jsonl(path: str | os.PathLike[str], record: Any) -> None:
    """Append one record to a JSONL file.

    Append-only by construction: this is the sole supported way to write raw model
    responses, which must never be overwritten or deleted (research rule 3).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_json(record))
        fh.write("\n")


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank lines.

    Raises :class:`json.JSONDecodeError` with the line number attached if a record is
    malformed -- a corrupted raw log must fail loudly rather than be silently skipped.
    """
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{exc.msg} (in {path!s} line {lineno})", exc.doc, exc.pos
                ) from None


# --------------------------------------------------------------------------- hashing


def canonical_json(obj: Any) -> str:
    """Serialize to the one JSON form this project hashes.

    Sorted keys and no incidental whitespace, so logically equal objects always produce
    equal bytes. ``ensure_ascii=False`` keeps non-ASCII characters literal rather than
    escaped, which is stable as long as the output is written as UTF-8 -- which
    :func:`write_text` and :func:`append_jsonl` guarantee.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    """Return the ``sha256:<hex>`` digest of a string (UTF-8 encoded) or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_of_json(obj: Any) -> str:
    """Return the ``sha256:<hex>`` digest of an object's canonical JSON form."""
    return sha256_hex(canonical_json(obj))


def sha256_of_file(path: str | os.PathLike[str]) -> str:
    """Return the ``sha256:<hex>`` digest of a file's raw bytes.

    Reads bytes, not text, so the digest covers exactly what is on disk. Prompt template
    hashes recorded at the protocol freeze use this.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_of_files(paths: Iterable[str | os.PathLike[str]]) -> dict[str, str]:
    """Return a ``{path: digest}`` manifest, with POSIX-style keys for cross-platform use."""
    return {Path(p).as_posix(): sha256_of_file(p) for p in paths}
