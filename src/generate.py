"""Build the base-pair set: quota-driven generation under oracle-gated acceptance.

Protocol Step 4. Usage::

    python -m src.generate --config configs/pilot.yaml

The loop is deliberately simple and deliberately strict. A candidate is *requested* with a
family and, for NEQ_DEPTH, a target depth. The oracle then decides what it actually is, and
the candidate is accepted only if every acceptance rule in §7.3 holds. Generator intent never
becomes a label (research rule 1).

Every rejection is counted by reason and written to ``rejections.json``: §Step 4 asks for
these so that a stratum which is unusually hard to sample shows up as evidence rather than as
a silent bias.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PLACEHOLDERS, Config, ConfigError, load_config
from src.dfa import DFA, pair_canonical_hash, to_record
from src.generators import (
    eq_expand_pair,
    eq_iso_pair,
    neq_depth_pair,
    permute_independently,
)
from src.generators_v2 import (
    CONSTRUCTION_VERSION as CONSTRUCTION_VERSION_V2,
)
from src.generators_v2 import (
    eq_expand_pair_v2,
    eq_iso_pair_v2,
    neq_depth_pair_v2,
)
from src.io_utils import append_jsonl, canonical_json, read_text, write_text
from src.minimize import minimal_state_count
from src.oracle import OracleResult, equivalence_and_witness
from src.simulate import access_distances

__all__ = [
    "GenerationOutcome",
    "generate_pairs",
    "main",
    "resolve_construction_version",
]


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """What one generation run produced."""

    records: list[dict[str, Any]]
    rejections: Counter[str]
    attempts: int
    complete: bool


#: The v1 construction: one accepting state toggled between A and B.
CONSTRUCTION_VERSION_V1 = "single_toggle_v1"

#: Every construction this module can actually build. A config naming anything else is
#: refused rather than defaulted, so a v2 config can never silently run the v1 generator --
#: the two produce datasets that differ in exactly the property v2 exists to fix.
IMPLEMENTED_CONSTRUCTIONS = (CONSTRUCTION_VERSION_V1, CONSTRUCTION_VERSION_V2)


def resolve_construction_version(config: Config) -> str:
    """Return the construction the config asks for, refusing anything unimplemented.

    Absent means v1: the legacy pilot config predates the field and must keep reproducing
    byte-identically. Anything present must be a version this module implements.

    Raises:
        ConfigError: naming the unknown version and what is available. This is the guard
            ``docs/experiment_revision.md`` requires -- without it a v2 config would quietly
            produce v1 pairs carrying a v2 label, which no downstream check would catch.
    """
    declared = config.get("pairs.construction_version") or config.get(
        "construction_version"
    )
    if declared is None:
        return CONSTRUCTION_VERSION_V1
    declared = str(declared)
    if declared in PLACEHOLDERS:
        raise ConfigError(
            f"{config.path}: pairs.construction_version is still {declared!r}"
        )
    if declared not in IMPLEMENTED_CONSTRUCTIONS:
        raise ConfigError(
            f"{config.path}: construction_version {declared!r} is not implemented.\n"
            f"Available: {', '.join(IMPLEMENTED_CONSTRUCTIONS)}.\n"
            f"Refusing to fall back to a different construction: the generated pairs would "
            f"carry a version label they do not match."
        )
    return declared


def _parse_bin(label: str) -> tuple[int, int]:
    """Parse a depth-stratum label such as ``"5-8"`` into inclusive bounds."""
    try:
        low, high = (int(part) for part in str(label).split("-"))
    except ValueError:
        raise ConfigError(
            f"depth bin {label!r} must look like '5-8'"
        ) from None
    if low < 1 or high < low:
        raise ConfigError(f"depth bin {label!r} must satisfy 1 <= low <= high")
    return low, high


def _graph_statistics(dfa_a: DFA, dfa_b: DFA) -> dict[str, Any]:
    """Structural descriptors recorded per pair (§9.1, §13 depth-confounding safeguard).

    ``d`` is meant to be the difficulty variable, so anything else that plausibly drives
    difficulty has to be recorded alongside it or the H1 model cannot be adjusted for it.
    The minimal state count is the most informative of these: it separates a machine that
    genuinely needs its states from one padded with redundant clones.
    """
    stats: dict[str, Any] = {}
    for name, dfa in (("a", dfa_a), ("b", dfa_b)):
        distances = [d for d in access_distances(dfa) if d is not None]
        stats[f"n_states_{name}"] = dfa.n_states
        stats[f"n_accepting_{name}"] = len(dfa.accepting)
        stats[f"accepting_fraction_{name}"] = round(
            len(dfa.accepting) / dfa.n_states, 6
        )
        stats[f"n_reachable_{name}"] = len(distances)
        stats[f"max_access_distance_{name}"] = max(distances) if distances else None
        stats[f"mean_access_distance_{name}"] = (
            round(sum(distances) / len(distances), 6) if distances else None
        )
        stats[f"n_self_loops_{name}"] = sum(
            1
            for state, row in enumerate(dfa.delta)
            for destination in row
            if destination == state
        )
        stats[f"minimal_states_{name}"] = minimal_state_count(dfa)
    return stats


def _build_record(
    pair_id: str,
    seed: int,
    family: str,
    requested_depth: int | None,
    dfa_a: DFA,
    dfa_b: DFA,
    oracle: OracleResult,
    construction_version: str,
    dataset_id: str,
) -> dict[str, Any]:
    """Assemble one base-pair record (§9.1).

    The DFAs are stored *as generated*, not canonicalized: the record is meant to reproduce
    exactly what the generator built from its seed (§12.5). ``canonical_hash`` is separately
    computed over the canonical form, which is what makes it a renaming-proof dedup key.

    ``construction_version`` and ``dataset_id`` are recorded per pair, not just per run:
    ``p_000000`` is not unique across splits, and a pair carrying no construction label
    cannot be checked against the config that claims to have produced it.
    """
    return {
        "pair_id": pair_id,
        "dataset_id": dataset_id,
        "construction_version": construction_version,
        "seed": seed,
        "family": family,
        "requested_depth": requested_depth,
        "dfa_A": to_record(dfa_a, canonical=False),
        "dfa_B": to_record(dfa_b, canonical=False),
        "oracle_equivalent": oracle.equivalent,
        "oracle_witness": oracle.shortest_witness,
        "oracle_depth": oracle.shortest_depth,
        "graph_statistics": _graph_statistics(dfa_a, dfa_b),
        "canonical_hash": pair_canonical_hash(dfa_a, dfa_b),
    }


def _load_excluded_hashes(config: Config) -> set[str]:
    """Read the hash list that keeps pilot pairs out of the main set (§Step 7).

    A configured-but-missing file is a hard error. Treating it as an empty exclusion set
    would silently disable the contamination guard at exactly the moment it matters: the
    main run would regenerate the pilot pairs and every downstream check would still pass,
    because there is nothing left to compare against.

    Raises:
        ConfigError: if ``pairs.exclude_hashes_from`` names a file that does not exist.
    """
    location = config.get("pairs.exclude_hashes_from")
    if not location:
        return set()
    path = Path(str(location))
    if not path.is_file():
        raise ConfigError(
            f"pairs.exclude_hashes_from points at {path}, which does not exist.\n"
            f"Generate that set first, or remove the key if no exclusion is intended. "
            f"Continuing would silently disable the contamination guard (§Step 7)."
        )
    return {line.strip() for line in read_text(path).splitlines() if line.strip()}


def generate_pairs(config: Config) -> GenerationOutcome:
    """Fill every family quota, accepting only candidates the oracle confirms.

    Each attempt draws a fresh seed from the master RNG and builds its candidate from a
    ``random.Random`` seeded with it, so the recorded ``seed`` regenerates that exact pair
    on its own (§12.5). Rejections are counted by reason, never silently retried away.
    """
    alphabet = config.alphabet
    master_seed = int(config.require("seed"))
    master = random.Random(master_seed)
    max_attempts_per_pair = int(config.get("pairs.max_generation_attempts_per_pair", 5000))

    neq = config.get("pairs.neq_depth") or {}
    eq_iso = config.get("pairs.eq_iso") or {}
    eq_expand = config.get("pairs.eq_expand") or {}

    version = resolve_construction_version(config)
    accepting_count = config.get("pairs.accepting_count_per_dfa")
    if version == CONSTRUCTION_VERSION_V2:
        if accepting_count is None:
            raise ConfigError(
                f"{config.path}: construction_version {version!r} requires "
                f"pairs.accepting_count_per_dfa (the count held equal across families)"
            )
        accepting_count = int(accepting_count)

    # One work item per pair to produce: (family, depth-bin label or None).
    # Validate every bin up front: a stratum the state count cannot express is a config
    # error, not a rejection, and it should say so before any work is done.
    neq_states = int(neq.get("state_count", 14))
    # v2 branches into two designated states, so the deepest bin costs one more state than
    # v1's single chain: depth 12 needs 14, which is exactly the configured count.
    backbone_overhead = 2 if version == CONSTRUCTION_VERSION_V2 else 1
    todo: list[tuple[str, str | None]] = []
    for label, count in (neq.get("bins") or {}).items():
        _low, high = _parse_bin(label)
        if neq_states < high + backbone_overhead:
            raise ConfigError(
                f"depth bin {label!r} needs at least {high + backbone_overhead} states to "
                f"hold its {version} backbone, but pairs.neq_depth.state_count is "
                f"{neq_states}"
            )
        todo += [("NEQ_DEPTH", str(label))] * int(count)
    todo += [("EQ_ISO", None)] * int(eq_iso.get("count", 0))
    todo += [("EQ_EXPAND", None)] * int(eq_expand.get("count", 0))

    seen: set[str] = _load_excluded_hashes(config)
    records: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    attempts = 0

    for index, (family, bin_label) in enumerate(todo):
        accepted = False
        for _ in range(max_attempts_per_pair):
            attempts += 1
            seed = master.getrandbits(63)
            rng = random.Random(seed)

            requested_depth: int | None = None
            core_range = eq_expand.get("core_states") or [10, 12]
            if family == "NEQ_DEPTH":
                assert bin_label is not None
                low, high = _parse_bin(bin_label)
                requested_depth = rng.randint(low, high)
                if version == CONSTRUCTION_VERSION_V2:
                    draw = neq_depth_pair_v2(
                        rng,
                        n_states=neq_states,
                        depth=requested_depth,
                        alphabet=alphabet,
                        accepting_count=accepting_count,
                    )
                else:
                    draw = neq_depth_pair(
                        rng,
                        n_states=neq_states,
                        depth=requested_depth,
                        alphabet=alphabet,
                    )
            elif family == "EQ_ISO":
                if version == CONSTRUCTION_VERSION_V2:
                    draw = eq_iso_pair_v2(
                        rng,
                        n_states=int(eq_iso.get("state_count", 14)),
                        alphabet=alphabet,
                        accepting_count=accepting_count,
                    )
                else:
                    draw = eq_iso_pair(
                        rng, n_states=int(eq_iso.get("state_count", 14)), alphabet=alphabet
                    )
            elif version == CONSTRUCTION_VERSION_V2:
                draw = eq_expand_pair_v2(
                    rng,
                    n_states=int(eq_expand.get("state_count", 14)),
                    alphabet=alphabet,
                    core_states=(int(core_range[0]), int(core_range[1])),
                    accepting_count=accepting_count,
                    require_non_isomorphic=bool(
                        eq_expand.get("require_non_isomorphic", True)
                    ),
                )
            else:
                draw = eq_expand_pair(
                    rng,
                    n_states=int(eq_expand.get("state_count", 14)),
                    alphabet=alphabet,
                    core_states=(int(core_range[0]), int(core_range[1])),
                    require_non_isomorphic=bool(
                        eq_expand.get("require_non_isomorphic", True)
                    ),
                )

            if draw.pair is None:
                rejections[f"{family}:{draw.reason}"] += 1
                continue

            dfa_a, dfa_b = permute_independently(rng, *draw.pair)

            # ---- the oracle decides, and only then ----
            oracle = equivalence_and_witness(dfa_a, dfa_b)

            expected_equivalent = family != "NEQ_DEPTH"
            if oracle.equivalent != expected_equivalent:
                rejections[f"{family}:oracle_label_mismatch"] += 1
                continue

            if version == CONSTRUCTION_VERSION_V2:
                # The whole point of v2: no family may be identifiable by its accepting
                # count. Enforced at acceptance rather than trusted from the builder, for
                # the same reason the oracle re-decides every label.
                counts = (len(dfa_a.accepting), len(dfa_b.accepting))
                if counts != (accepting_count, accepting_count):
                    rejections[f"{family}:accepting_count_violation"] += 1
                    continue

            if family == "NEQ_DEPTH":
                if oracle.shortest_depth != requested_depth:
                    rejections[f"{family}:depth_mismatch"] += 1
                    continue
                if bool(neq.get("exclude_depth_zero", True)) and oracle.shortest_depth == 0:
                    rejections[f"{family}:depth_zero"] += 1
                    continue
                if bool(neq.get("forbid_epsilon_cue", True)) and (
                    (dfa_a.start in dfa_a.accepting) != (dfa_b.start in dfa_b.accepting)
                ):
                    rejections[f"{family}:epsilon_cue"] += 1
                    continue

            digest = pair_canonical_hash(dfa_a, dfa_b)
            if digest in seen:
                rejections[f"{family}:duplicate_hash"] += 1
                continue

            seen.add(digest)
            records.append(
                _build_record(
                    pair_id=f"p_{index:06d}",
                    seed=seed,
                    family=family,
                    requested_depth=requested_depth,
                    dfa_a=dfa_a,
                    dfa_b=dfa_b,
                    oracle=oracle,
                    construction_version=version,
                    dataset_id=config.config_id,
                )
            )
            accepted = True
            break

        if not accepted:
            rejections[f"{family}:quota_unfilled"] += 1
            return GenerationOutcome(records, rejections, attempts, complete=False)

    return GenerationOutcome(records, rejections, attempts, complete=True)


def _depth_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    """Accepted NEQ_DEPTH pairs per oracle depth, for the run summary."""
    histogram: Counter[str] = Counter()
    for record in records:
        if record["family"] == "NEQ_DEPTH":
            histogram[str(record["oracle_depth"])] += 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generate",
        description="Generate the base DFA pair set under oracle-gated acceptance.",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace an existing pair set. Generated pairs are immutable by default: "
            "regenerating silently would invalidate any raw responses already collected "
            "against the old set."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        pairs_dir = config.path_for("pairs_dir")
        pairs_path = pairs_dir / "pairs.jsonl"

        if pairs_path.exists() and not args.overwrite:
            print(
                f"{pairs_path} already exists. Generated pairs are immutable; "
                f"pass --overwrite only if no responses have been collected against them.",
                file=sys.stderr,
            )
            return 1

        outcome = generate_pairs(config)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    if pairs_path.exists():
        pairs_path.unlink()
    for record in outcome.records:
        append_jsonl(pairs_path, record)

    write_text(
        pairs_dir / "canonical_hashes.txt",
        "".join(f"{record['canonical_hash']}\n" for record in outcome.records),
    )

    by_family = Counter(record["family"] for record in outcome.records)
    summary = {
        "config_id": config.config_id,
        "seed": config.require("seed"),
        "pairs_generated": len(outcome.records),
        "by_family": dict(sorted(by_family.items())),
        "oracle_depth_histogram": _depth_histogram(outcome.records),
        "attempts": outcome.attempts,
        "acceptance_rate": (
            round(len(outcome.records) / outcome.attempts, 6) if outcome.attempts else None
        ),
        "complete": outcome.complete,
    }
    write_text(pairs_dir / "summary.json", canonical_json(summary) + "\n")
    write_text(
        pairs_dir / "rejections.json",
        canonical_json(dict(sorted(outcome.rejections.items()))) + "\n",
    )

    print(f"pairs written : {pairs_path} ({len(outcome.records)} records)")
    print(f"by family     : {dict(sorted(by_family.items()))}")
    print(f"depths        : {summary['oracle_depth_histogram']}")
    print(f"attempts      : {outcome.attempts} (acceptance {summary['acceptance_rate']})")
    if outcome.rejections:
        print(f"rejections    : {dict(sorted(outcome.rejections.items()))}")

    if not outcome.complete:
        print("QUOTA UNFILLED: see rejections.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
