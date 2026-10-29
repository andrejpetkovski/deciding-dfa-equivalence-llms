"""Emit the renderings and the assembled prompts for every generated pair.

Protocol Step 5 and §8. Usage::

    python -m src.render_prompts --config configs/pilot.yaml

Produces three things:

- one text file per (pair, DFA, representation), the exact bytes describing one machine;
- ``manifest.jsonl``, carrying the label bijections and text hashes. The bijection is what
  the grader needs to map a model's trace back to internal state ids, and it must never
  appear in a prompt (research rule 8), so it lives here and nowhere else;
- ``prompts.jsonl``, one row per (pair, representation, prompt condition) holding the exact
  assembled messages. The runner sends these verbatim, which is what guarantees the prompt
  that was graded is the prompt that was sent; §9.3 requires the exact prompt text to be
  retained in any case.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import Config, ConfigError, load_config
from src.dfa import from_record
from src.io_utils import append_jsonl, canonical_json, read_jsonl, write_text
from src.prompts import PromptAssets, assert_no_leak, build_messages, load_prompt_assets
from src.rendering import derive_rng, render

__all__ = ["build_prompts", "main", "render_all"]

#: Fields a *rendering* must never contain. The prompt-level check in :mod:`src.prompts` is
#: deliberately different: the frozen templates legitimately use the word "witness" to name an
#: output field, so there it is checked by value rather than by name.
_FORBIDDEN_KEYS = ("pair_id", "oracle", "family", "seed", "depth", "witness")


def _assert_rendering_has_no_leak(text: str, record: dict[str, Any]) -> None:
    """Fail loudly if experiment metadata reached the text a model will read.

    Cheap insurance against the failure mode that would silently invalidate every result:
    a prompt that discloses the answer or the stratum (§5.6, §8.4).
    """
    lowered = text.lower()
    for key in _FORBIDDEN_KEYS:
        if key in lowered:
            raise ValueError(f"rendering leaks {key!r}: {text[:200]!r}")
    if record["pair_id"].lower() in lowered:
        raise ValueError(f"rendering leaks its pair_id {record['pair_id']!r}")


def render_all(
    config: Config, pairs_path: Path
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Render every pair under every configured representation condition.

    Returns the manifest rows and a per-condition count.
    """
    representations = config.require("representations")
    conditions = [str(c["id"]) for c in representations]
    # The R1 label scheme is a config decision, not a code default: the legacy v1 pilot was
    # rendered with nonce labels and must stay byte-reproducible (D2), while v2 datasets use
    # the shared-vocabulary derangement that keeps gate 8 satisfiable (D5.4).
    r1_scheme = next(
        (str(c.get("labels")) for c in representations if str(c["id"]) == "R1" and c.get("labels")),
        "shared_vocabulary_deranged",
    )
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for record in read_jsonl(pairs_path):
        dfas = {"A": from_record(record["dfa_A"]), "B": from_record(record["dfa_B"])}
        for condition in conditions:
            entry: dict[str, Any] = {
                "pair_id": record["pair_id"],
                "rendering": condition,
            }
            for name, dfa in dfas.items():
                # Seeded from the pair's own seed, so a rendering is reproducible from the
                # record alone and is independent of the order pairs are processed in.
                rng = derive_rng(record["seed"], condition, name)
                rendered = render(dfa, name, condition, rng, r1_scheme)
                _assert_rendering_has_no_leak(rendered.text, record)
                entry[f"labels_{name}"] = list(rendered.labels)
                entry[f"text_sha256_{name}"] = rendered.text_sha256
                entry[f"text_{name}"] = rendered.text
            rows.append(entry)
            counts[condition] += 1
    return rows, counts


def build_prompts(
    assets: PromptAssets,
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble one prompt per (pair, representation, prompt condition).

    Every assembled prompt is leak-checked before it is written. The check runs here, not
    only in a test, because a prompt that names its stratum or contains its answer still
    parses and still grades: the failure would be silent, and it would invalidate every
    number in the paper.
    """
    prompts: list[dict[str, Any]] = []
    for row in rows:
        record = records[row["pair_id"]]
        for condition in assets.conditions:
            prompt = build_messages(
                assets.system,
                condition.template,
                row["text_A"],
                row["text_B"],
                assets.system_placement,
            )
            assert_no_leak(prompt, record)
            prompts.append(
                {
                    "prompt_id": f"{row['pair_id']}|{row['rendering']}|{condition.id}",
                    "pair_id": row["pair_id"],
                    "rendering": row["rendering"],
                    "prompt_condition": condition.id,
                    "system": prompt.system,
                    "user": prompt.user,
                    "prompt_sha256": prompt.sha256,
                    "system_sha256": assets.system_sha256,
                    "template_sha256": condition.template_sha256,
                }
            )
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.render_prompts",
        description="Serialize every pair and assemble the prompts that embed them.",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing rendering set"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
        renderings_dir = config.path_for("renderings_dir")
        manifest_path = renderings_dir / "manifest.jsonl"
        prompts_path = renderings_dir / "prompts.jsonl"

        if not pairs_path.is_file():
            print(
                f"{pairs_path} not found. Run python -m src.generate --config "
                f"{args.config} first.",
                file=sys.stderr,
            )
            return 1
        if manifest_path.exists() and not args.overwrite:
            print(
                f"{manifest_path} already exists; pass --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1

        assets = load_prompt_assets(config)
        rows, counts = render_all(config, pairs_path)
        records = {r["pair_id"]: r for r in read_jsonl(pairs_path)}
        prompts = build_prompts(assets, rows, records)
    except (ConfigError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for path in (manifest_path, prompts_path):
        if path.exists():
            path.unlink()

    for row in rows:
        # The manifest keeps the bijection and the hashes; the text lives in its own file so
        # the exact bytes a model saw can be inspected directly.
        for name in ("A", "B"):
            write_text(
                renderings_dir / f"{row['pair_id']}_{name}_{row['rendering']}.txt",
                row[f"text_{name}"] + "\n",
            )
        # Drop only the full text, which lives in its own file; the hashes stay.
        append_jsonl(
            manifest_path,
            {k: v for k, v in row.items() if k not in ("text_A", "text_B")},
        )

    for prompt in prompts:
        append_jsonl(prompts_path, prompt)

    by_prompt_condition = Counter(p["prompt_condition"] for p in prompts)
    write_text(
        renderings_dir / "summary.json",
        canonical_json(
            {
                "config_id": config.config_id,
                "pairs_rendered": len(rows) // max(len(counts), 1),
                "by_condition": dict(sorted(counts.items())),
                "files_written": len(rows) * 2,
                "prompts_built": len(prompts),
                "by_prompt_condition": dict(sorted(by_prompt_condition.items())),
                "system_placement": assets.system_placement,
                "system_sha256": assets.system_sha256,
            }
        )
        + "\n",
    )

    print(f"renderings   : {renderings_dir} ({len(rows) * 2} files)")
    print(f"manifest     : {manifest_path} ({len(rows)} rows)")
    print(f"prompts      : {prompts_path} ({len(prompts)} rows)")
    print(
        f"by condition : {dict(sorted(counts.items()))} x "
        f"{dict(sorted(by_prompt_condition.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
