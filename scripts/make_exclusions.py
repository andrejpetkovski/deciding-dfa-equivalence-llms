"""Build the verified exclusion-hash union that keeps earlier pairs out of a later set.

Usage::

    python -m scripts.make_exclusions --config configs/main_frozen.yaml

Reads every dataset named in ``pairs.exclusion_sources``, **recomputes** each pair's
canonical hash from its automata, and writes the union to ``pairs.exclude_hashes_from``.

Recomputing rather than copying the stored ``canonical_hash`` field is the point of the
script. A stored hash is an assertion a record makes about itself; if a record were ever
edited without its hash being updated, copying that hash would produce an exclusion list
that silently fails to exclude the pair it names. Gate 7 performs the same recomputation
independently, so the list and the check cannot drift together.

§Step 7 and D2: pilot and calibration pairs must never reappear in the main set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ConfigError, load_config
from src.dataset_qa import exclusion_union_failures
from src.dfa import from_record, pair_canonical_hash
from src.io_utils import read_jsonl, write_text

__all__ = ["EXCLUSION_PREFIX", "build_union", "main"]

#: Exclusion lists are *derived* artifacts and live in their own filename namespace.
#: ``src.generate`` owns ``canonical_hashes.txt`` for each dataset it produces, and writing a
#: union over one of those would silently rewrite another dataset's generated output -- which
#: is exactly what happened the first time this script ran against the pilot config.
EXCLUSION_PREFIX = "exclusion_"


def build_union(sources: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return the sorted hash union and a per-source count.

    Raises:
        FileNotFoundError: if a declared source is absent. A missing source cannot be
            silently skipped: the resulting list would look complete while excluding
            nothing from that dataset.
    """
    union: set[str] = set()
    per_source: dict[str, int] = {}
    for source in sources:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"exclusion source {source} does not exist; generate it first or remove it "
                f"from pairs.exclusion_sources"
            )
        count = 0
        for record in read_jsonl(path):
            digest = pair_canonical_hash(
                from_record(record["dfa_A"]), from_record(record["dfa_B"])
            )
            union.add(digest)
            count += 1
        per_source[source] = count
    return sorted(union), per_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.make_exclusions",
        description="Write the verified exclusion-hash union for a config.",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        sources = [str(s) for s in (config.get("pairs.exclusion_sources") or [])]
        destination = config.get("pairs.exclude_hashes_from")
        if not sources:
            print(
                f"{args.config}: pairs.exclusion_sources is not declared, so there is no "
                f"union to build.",
                file=sys.stderr,
            )
            return 2
        if not destination:
            print(
                f"{args.config}: pairs.exclude_hashes_from is not declared, so the union "
                f"has nowhere to go.",
                file=sys.stderr,
            )
            return 2
        destination_path = Path(str(destination))
        if not destination_path.name.startswith(EXCLUSION_PREFIX):
            print(
                f"refusing to write the union to {destination}: an exclusion list must be "
                f"named {EXCLUSION_PREFIX}*, so it cannot overwrite a dataset artifact that "
                f"src.generate owns (such as canonical_hashes.txt).",
                file=sys.stderr,
            )
            return 2
        if str(destination_path) in {str(Path(s)) for s in sources}:
            print(
                f"refusing to write the union to {destination}: it is also one of its own "
                f"sources.",
                file=sys.stderr,
            )
            return 2
        union, per_source = build_union(sources)
    except (ConfigError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    write_text(Path(str(destination)), "".join(f"{digest}\n" for digest in union))

    for source, count in per_source.items():
        print(f"  {count:>5} pairs from {source}")
    print(f"union written : {destination} ({len(union)} distinct hashes)")
    if sum(per_source.values()) != len(union):
        print(
            f"note          : {sum(per_source.values()) - len(union)} hash(es) appear in "
            f"more than one source"
        )

    # Verify with the same check gate 7 will run, so a bad union fails here rather than
    # during QA on a dataset that has already been generated against it.
    remaining = exclusion_union_failures(set(union), str(destination), sources)
    if remaining:
        print("\nverification FAILED:", file=sys.stderr)
        for failure in remaining:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("verification  : OK (gate 7's own check passes against this union)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
