"""Automated dataset quality assurance: the eight gates of Protocol Step 6.

Usage::

    python -m src.dataset_qa --config configs/pilot.yaml

Exits non-zero if any gate fails. A gate whose inputs are unavailable reports ``SKIP`` rather
than quietly passing -- a silent pass would let an unchecked dataset reach the main run, which
is exactly what these gates exist to prevent. Use ``--strict`` (required before the freeze) to
make a skip fatal too.

**Ordering note.** §Step 5 (renderings) precedes §Step 6 (QA), but the command sequence in
§9.4 lists ``dataset_qa`` *before* ``render_prompts``. Gates 6 and 8 read renderings, so they
can only run in the Step 5 -> Step 6 order. This module works either way: run it after
``render_prompts`` for all eight gates, or earlier to get gates 1-5 and 7 with 6 and 8 marked
``SKIP``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PLACEHOLDERS, Config, ConfigError, load_config
from src.crosscheck import library_equivalent
from src.dfa import DFA, canonical_form, from_record, pair_canonical_hash
from src.io_utils import canonical_json, read_jsonl, read_text, write_text
from src.oracle import equivalence_and_witness
from src.rendering import parse_rendering
from src.simulate import reachable

__all__ = [
    "GateResult",
    "exclusion_union_failures",
    "main",
    "run_all_gates",
    "token_counter_from_config",
]

#: Failure examples kept per gate. The full count is always reported; the list is capped so a
#: systematically broken dataset produces a readable report rather than a megabyte of noise.
_MAX_EXAMPLES = 20


@dataclass(frozen=True, slots=True)
class GateResult:
    """The outcome of one QA gate."""

    number: int
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    checked: int = 0
    n_failures: int = 0
    examples: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        number: int,
        name: str,
        failures: list[str],
        checked: int,
        **detail: Any,
    ) -> GateResult:
        return cls(
            number=number,
            name=name,
            status="PASS" if not failures else "FAIL",
            checked=checked,
            n_failures=len(failures),
            examples=tuple(failures[:_MAX_EXAMPLES]),
            detail=detail,
        )

    @classmethod
    def skipped(cls, number: int, name: str, reason: str, **detail: Any) -> GateResult:
        """A gate that could not run. ``detail`` carries whatever *was* established.

        A skip is not always a blank: gate 8 verifies its structural preconditions even when
        no tokenizer is available, and losing that evidence would make the report less
        informative than the run.
        """
        return cls(
            number=number, name=name, status="SKIP", detail={"reason": reason, **detail}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.number,
            "name": self.name,
            "status": self.status,
            "checked": self.checked,
            "n_failures": self.n_failures,
            "examples": list(self.examples),
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- gates


def gate_1_oracle_agreement(
    records: list[dict[str, Any]], libraries: list[str]
) -> GateResult:
    """The hand-written product BFS must agree with every configured library.

    The oracle assigns every label in the study, so an independent confirmation of each
    verdict is the single most valuable check here.

    The **whole** stored oracle result is recomputed and compared, not just the boolean.
    Checking only ``oracle_equivalent`` would let a corrupted witness or depth through, and
    both are load-bearing: the witness is what §10.1's shortest-witness metric grades
    against, and the depth *is* H1's explanatory variable.
    """
    if not libraries:
        return GateResult.skipped(1, "oracle_agreement", "no cross-check libraries configured")
    failures: list[str] = []
    for record in records:
        dfa_a, error_a = _load_dfa(record, "A")
        dfa_b, error_b = _load_dfa(record, "B")
        if dfa_a is None or dfa_b is None:
            failures.append(error_a or error_b or "unreadable record")
            continue

        fresh = equivalence_and_witness(dfa_a, dfa_b)
        stored = (
            record["oracle_equivalent"],
            record["oracle_witness"],
            record["oracle_depth"],
        )
        recomputed = (fresh.equivalent, fresh.shortest_witness, fresh.shortest_depth)
        if stored != recomputed:
            failures.append(
                f"{record['pair_id']}: stored (equivalent, witness, depth) = {stored!r} "
                f"but a fresh oracle run gives {recomputed!r}"
            )
            continue

        for library in libraries:
            theirs = library_equivalent(dfa_a, dfa_b, library)
            if theirs != fresh.equivalent:
                failures.append(
                    f"{record['pair_id']}: oracle={fresh.equivalent} {library}={theirs}"
                )
    return GateResult.of(
        1, "oracle_agreement", failures, len(records), libraries=libraries
    )


def gate_2_dataset_balance(records: list[dict[str, Any]], config: Config) -> GateResult:
    """Dataset balance and identity: quotas, unique IDs, and the v2 construction invariants.

    The original gate 2 covered only depth balance. The amended spec (protocol §8)
    widens it to everything that makes a dataset *the one the config describes*, because no
    other gate covered any of it:

    - **Total and family quotas.** A set missing every equivalent pair passed QA intact, and
      balanced accuracy over one class is not balanced accuracy.
    - **Unique pair IDs.** Duplicated ids silently collapse rows when responses are joined
      back to pairs.
    - **Alphabet and state counts** match what the config declares, so a 12-state pair cannot
      sit unnoticed in a 14-state study.
    - **v2 invariants** (D2): both machines accept exactly ``accepting_count_per_dfa``, and
      every record's ``construction_version`` matches the config's. Without the second check
      a v1 pair could sit in a v2 dataset and nothing else would notice -- the pairs are
      valid, the oracle confirms them, and only the label is wrong.
    - **Dataset id** matches the config, since ``p_000000`` is not unique across splits.
    """
    bins = config.get("pairs.neq_depth.bins") or {}
    family_quotas = {
        "NEQ_DEPTH": sum(int(v) for v in bins.values()) if bins else None,
        "EQ_ISO": config.get("pairs.eq_iso.count"),
        "EQ_EXPAND": config.get("pairs.eq_expand.count"),
    }
    if not bins:
        return GateResult.skipped(2, "dataset_balance", "no depth bins configured")

    observed: Counter[str] = Counter()
    unbinned: list[str] = []
    for record in records:
        if record["family"] != "NEQ_DEPTH":
            continue
        depth = record["oracle_depth"]
        for label in bins:
            low, high = (int(part) for part in str(label).split("-"))
            if low <= depth <= high:
                observed[label] += 1
                break
        else:
            unbinned.append(f"{record['pair_id']}: depth {depth} is in no declared bin")

    failures = list(unbinned)
    for label, expected in bins.items():
        if observed[label] != int(expected):
            failures.append(
                f"bin {label!r}: expected {expected} pairs, found {observed[label]}"
            )

    by_family = Counter(record["family"] for record in records)
    for family, quota in family_quotas.items():
        if quota is None:
            continue
        if by_family[family] != int(quota):
            failures.append(
                f"family {family}: expected {quota} pairs, found {by_family[family]}"
            )
    unexpected = set(by_family) - set(family_quotas)
    if unexpected:
        failures.append(f"undeclared families present: {sorted(unexpected)}")

    declared_total = config.get("pairs.total")
    if declared_total is not None and len(records) != int(declared_total):
        failures.append(
            f"total: expected {declared_total} pairs, found {len(records)}"
        )

    seen_ids: Counter[str] = Counter(record["pair_id"] for record in records)
    for pair_id, count in seen_ids.items():
        if count > 1:
            failures.append(f"pair_id {pair_id!r} appears {count} times")

    expected_alphabet = list(config.alphabet)
    state_counts = {
        "NEQ_DEPTH": config.get("pairs.neq_depth.state_count"),
        "EQ_ISO": config.get("pairs.eq_iso.state_count"),
        "EQ_EXPAND": config.get("pairs.eq_expand.state_count"),
    }
    version = config.get("pairs.construction_version")
    accepting_count = config.get("pairs.accepting_count_per_dfa")
    # The DATASET's identity, which is not the RUN's. One dataset can be executed under
    # several configs -- a different provider is a different run over the same pairs -- so an
    # execution config names the dataset it reuses rather than being forced to rename it.
    dataset_id = str(config.get("dataset_id") or config.config_id)

    for record in records:
        label = record["pair_id"]
        if version is not None and record.get("construction_version") != str(version):
            failures.append(
                f"{label}: construction_version {record.get('construction_version')!r} "
                f"but the config declares {str(version)!r}"
            )
        if record.get("dataset_id") not in (None, dataset_id):
            failures.append(
                f"{label}: dataset_id {record.get('dataset_id')!r} != config {dataset_id!r}"
            )
        expected_states = state_counts.get(record["family"])
        for name in ("A", "B"):
            spec = record[f"dfa_{name}"]
            if spec.get("alphabet") != expected_alphabet:
                failures.append(
                    f"{label}/{name}: alphabet {spec.get('alphabet')!r} != "
                    f"configured {expected_alphabet!r}"
                )
            if expected_states is not None and spec.get("n_states") != int(expected_states):
                failures.append(
                    f"{label}/{name}: {spec.get('n_states')} states, config declares "
                    f"{expected_states}"
                )
            if accepting_count is not None:
                observed_accepting = len(spec.get("accepting") or [])
                if observed_accepting != int(accepting_count):
                    failures.append(
                        f"{label}/{name}: {observed_accepting} accepting states, config "
                        f"declares {accepting_count} (v2 count control)"
                    )

    return GateResult.of(
        2,
        "dataset_balance",
        failures,
        len(records),
        expected=dict(bins),
        observed=dict(sorted(observed.items())),
        expected_families={k: v for k, v in family_quotas.items() if v is not None},
        observed_families=dict(sorted(by_family.items())),
        expected_total=declared_total,
        construction_version=version,
        accepting_count_per_dfa=accepting_count,
    )


def gate_3_no_epsilon_leak(records: list[dict[str, Any]]) -> GateResult:
    """Both machines of a pair must agree on the empty word.

    Otherwise the answer is visible from the accepting-state lines alone, with no traversal
    at all, and the pair's depth would be 0 rather than its stratum's.
    """
    failures = []
    for record in records:
        dfa_a, error_a = _load_dfa(record, "A")
        dfa_b, error_b = _load_dfa(record, "B")
        if dfa_a is None or dfa_b is None:
            failures.append(error_a or error_b or "unreadable record")
            continue
        if (dfa_a.start in dfa_a.accepting) != (dfa_b.start in dfa_b.accepting):
            failures.append(f"{record['pair_id']}: start states disagree on acceptance")
        if record["oracle_depth"] == 0:
            failures.append(f"{record['pair_id']}: oracle depth is 0")
    return GateResult.of(3, "no_epsilon_leak", failures, len(records))


def _load_dfa(record: dict[str, Any], name: str) -> tuple[DFA | None, str | None]:
    """Build one DFA from a record, returning an error message instead of raising.

    A structurally broken record is a *finding* for gate 4, not a crash: every other gate
    has to survive it and report, or one bad row would abort the whole QA run and hide the
    state of the remaining pairs.
    """
    try:
        return from_record(record[f"dfa_{name}"]), None
    except Exception as error:  # noqa: BLE001 - the message is the finding
        return None, f"{record['pair_id']}/{name}: {type(error).__name__}: {error}"


def _each_dfa(records: list[dict[str, Any]]):
    """Yield ``(pair_id, name, dfa_or_None, error_or_None)`` for every machine."""
    for record in records:
        for name in ("A", "B"):
            dfa, error = _load_dfa(record, name)
            yield record["pair_id"], name, dfa, error


def gate_4_completeness(records: list[dict[str, Any]]) -> GateResult:
    """Every state must have exactly one in-range transition per alphabet symbol.

    Deliberately inspects the *raw record*, not a constructed :class:`~src.dfa.DFA`. The DFA
    type already enforces totality, so building one first would raise on exactly the records
    this gate is meant to report -- the gate would be unable to fail. What is being checked
    is that the serialized form on disk is still well-formed: a write or read bug, not a
    builder bug.
    """
    failures = []
    checked = 0
    for record in records:
        for name in ("A", "B"):
            checked += 1
            spec = record[f"dfa_{name}"]
            label = f"{record['pair_id']}/{name}"
            n_states = spec.get("n_states")
            alphabet = spec.get("alphabet") or []
            delta = spec.get("delta") or []

            if not isinstance(n_states, int) or n_states <= 0:
                failures.append(f"{label}: n_states is {n_states!r}")
                continue
            if len(delta) != n_states:
                failures.append(f"{label}: {len(delta)} rows for {n_states} states")
                continue
            for state, row in enumerate(delta):
                if len(row) != len(alphabet):
                    failures.append(
                        f"{label}: state {state} has {len(row)} transitions, "
                        f"expected {len(alphabet)}"
                    )
                    continue
                for column, destination in enumerate(row):
                    if not isinstance(destination, int) or not 0 <= destination < n_states:
                        failures.append(
                            f"{label}: delta[{state}][{column}] = {destination!r} is not a "
                            f"state in 0..{n_states - 1}"
                        )
            start = spec.get("start")
            if not isinstance(start, int) or not 0 <= start < n_states:
                failures.append(f"{label}: start state {start!r} out of range")
            for accepting in spec.get("accepting") or []:
                if not isinstance(accepting, int) or not 0 <= accepting < n_states:
                    failures.append(f"{label}: accepting state {accepting!r} out of range")
    return GateResult.of(4, "completeness", failures, checked)


def gate_5_reachability(records: list[dict[str, Any]]) -> GateResult:
    """Every state of a main-set machine must be reachable from its start."""
    failures = []
    checked = 0
    for pair_id, name, dfa, error in _each_dfa(records):
        checked += 1
        if dfa is None:
            failures.append(error or f"{pair_id}/{name}: unreadable")
            continue
        found = reachable(dfa)
        if len(found) != dfa.n_states:
            missing = sorted(set(range(dfa.n_states)) - found)
            failures.append(f"{pair_id}/{name}: unreachable states {missing}")
    return GateResult.of(5, "reachability", failures, checked)


def gate_6_representation_fidelity(
    records: list[dict[str, Any]], renderings_dir: Path, conditions: list[str]
) -> GateResult:
    """Every rendering must read back to the same pair, with the same oracle verdict, and
    every pair must actually be rendered under every condition.

    Three things are checked, and the second two were added after a review showed the first
    alone was not enough:

    1. **Fidelity.** The rendering denotes the stored automaton. This is what licenses
       treating R0 and R1 as the *same instance* under H3 -- otherwise a decision flip
       between them would be a generation bug, not an invariance violation.
    2. **Coverage.** Every (pair, condition) cell is present exactly once. Without this an
       empty manifest passes the gate after checking nothing, which is the most dangerous
       possible way for QA to succeed.
    3. **The label bijection.** The manifest's ``labels_X`` must map each *internal* state to
       the label the rendering actually used. Comparing canonical forms cannot see this: a
       reversed bijection still describes an isomorphic automaton and passes. M8 maps model
       traces back through this mapping, so a wrong one would silently mark correct traces
       wrong.
    """
    manifest_path = renderings_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return GateResult.skipped(
            6,
            "representation_fidelity",
            f"no renderings at {manifest_path}; run python -m src.render_prompts first",
        )

    by_id = {record["pair_id"]: record for record in records}
    failures: list[str] = []
    checked = 0
    seen_cells: Counter[tuple[str, str]] = Counter()
    for row in read_jsonl(manifest_path):
        pair_id, condition = row["pair_id"], row["rendering"]
        record = by_id.get(pair_id)
        if record is None:
            failures.append(f"{pair_id}: rendered but not in the pair set")
            continue
        seen_cells[(pair_id, condition)] += 1
        checked += 1
        try:
            parsed = [
                parse_rendering(
                    read_text(renderings_dir / f"{pair_id}_{name}_{condition}.txt")
                )
                for name in ("A", "B")
            ]
            stored = [from_record(record[f"dfa_{n}"]) for n in ("A", "B")]
            for name, got, want in zip(("A", "B"), parsed, stored, strict=True):
                if canonical_form(got.dfa) != canonical_form(want):
                    failures.append(
                        f"{pair_id}/{name}/{condition}: rendering does not read back to "
                        f"the stored automaton"
                    )
                    continue
                claimed = row.get(f"labels_{name}")
                problem = _bijection_problem(want, got, claimed)
                if problem:
                    failures.append(f"{pair_id}/{name}/{condition}: {problem}")
            parsed = [p.dfa for p in parsed]  # type: ignore[assignment]
            result = equivalence_and_witness(*parsed)
            if result.equivalent != record["oracle_equivalent"]:
                failures.append(
                    f"{pair_id}/{condition}: oracle verdict changed under rendering"
                )
            elif result.shortest_depth != record["oracle_depth"]:
                failures.append(
                    f"{pair_id}/{condition}: depth {result.shortest_depth} != "
                    f"stored {record['oracle_depth']}"
                )
        except Exception as error:  # noqa: BLE001 - any failure here is a gate failure
            failures.append(f"{pair_id}/{condition}: {type(error).__name__}: {error}")

    failures.extend(_coverage_failures(seen_cells, by_id, conditions))
    return GateResult.of(
        6,
        "representation_fidelity",
        failures,
        checked,
        expected_cells=len(by_id) * len(conditions),
        conditions=conditions,
    )


def _bijection_problem(
    stored: DFA, parsed: Any, claimed: list[str] | None
) -> str | None:
    """Return why ``claimed`` is not the true state-to-label mapping, or ``None`` if it is.

    ``parsed`` numbers states by sorted label, so mapping each stored state through
    ``claimed`` and then through the parsed numbering must reproduce the stored automaton
    *exactly* -- not merely up to isomorphism.
    """
    if not isinstance(claimed, list):
        return "manifest has no label list"
    if len(claimed) != stored.n_states:
        return f"manifest lists {len(claimed)} labels for {stored.n_states} states"
    if len(set(claimed)) != len(claimed):
        return "manifest label list has duplicates"

    position = {label: i for i, label in enumerate(parsed.labels)}
    missing = [label for label in claimed if label not in position]
    if missing:
        return f"labels absent from the rendering: {missing[:3]}"

    permutation = tuple(position[label] for label in claimed)
    if stored.relabel(permutation) != parsed.dfa:
        return "label mapping does not send stored states to the rendered ones"
    return None


def _coverage_failures(
    seen: Counter[tuple[str, str]],
    by_id: dict[str, Any],
    conditions: list[str],
) -> list[str]:
    """Report any (pair, condition) cell that is missing or duplicated."""
    failures: list[str] = []
    missing = [
        f"{pair_id}/{condition}"
        for pair_id in by_id
        for condition in conditions
        if seen[(pair_id, condition)] == 0
    ]
    if missing:
        failures.append(
            f"{len(missing)} of {len(by_id) * len(conditions)} cells were never rendered, "
            f"e.g. {missing[:5]}"
        )
    duplicated = [f"{p}/{c}" for (p, c), n in seen.items() if n > 1]
    if duplicated:
        failures.append(f"cells rendered more than once: {duplicated[:5]}")
    return failures


def exclusion_union_failures(
    excluded: set[str], excluded_from: str | None, sources: list[str]
) -> list[str]:
    """Check that the exclusion list really covers every dataset it claims to.

    ``exclude_hashes_from`` naming an existing file proves nothing about its *contents*: a
    stale list written before the revised pilot existed, or one copied from another split,
    passes a file-existence check and then excludes nothing that matters. So each declared
    source is re-read and its hashes are **recomputed from the automata** -- a source's own
    stored hash is an assertion, not evidence -- and every one must appear in the union.

    Returns a list of failure descriptions, empty when the union is complete.
    """
    if not sources:
        if excluded_from:
            return [
                f"{excluded_from} is configured but pairs.exclusion_sources is not, so its "
                f"coverage cannot be verified. Declare the datasets the union must cover."
            ]
        return []
    if not excluded_from:
        return [
            "pairs.exclusion_sources is declared but pairs.exclude_hashes_from is not, "
            "so nothing is actually excluded"
        ]

    failures: list[str] = []
    for source in sources:
        path = Path(source)
        if not path.is_file():
            failures.append(
                f"exclusion source {source} does not exist; the union cannot be verified"
            )
            continue
        missing: list[str] = []
        total = 0
        for record in read_jsonl(path):
            dfa_a, error_a = _load_dfa(record, "A")
            dfa_b, error_b = _load_dfa(record, "B")
            if dfa_a is None or dfa_b is None:
                failures.append(error_a or error_b or f"{source}: unreadable record")
                continue
            total += 1
            if pair_canonical_hash(dfa_a, dfa_b) not in excluded:
                missing.append(record["pair_id"])
        if missing:
            failures.append(
                f"{excluded_from} is missing {len(missing)} of {total} hashes from "
                f"{source}, e.g. {missing[:5]}"
            )
    return failures


def gate_7_deduplication(
    records: list[dict[str, Any]],
    excluded: set[str],
    excluded_from: str | None,
    exclusion_sources: list[str] | None = None,
) -> GateResult:
    """No pair may repeat inside the set, and none may come from the excluded set.

    Duplicates would silently inflate the sample: §5.1 makes the base pair the independent
    unit, so the same pair twice is one unit counted as two.

    Hashes are **recomputed from the automata**, not read from the record. A stored hash is
    an assertion about a pair; trusting it means a duplicate carrying a different stored
    value passes deduplication, and a tampered record passes the exclusion check. The stored
    value is separately compared against the recomputed one, since a mismatch is itself a
    corruption worth reporting.
    """
    failures = []
    seen: Counter[str] = Counter()
    owners: dict[str, list[str]] = {}
    for record in records:
        dfa_a, error_a = _load_dfa(record, "A")
        dfa_b, error_b = _load_dfa(record, "B")
        if dfa_a is None or dfa_b is None:
            failures.append(error_a or error_b or "unreadable record")
            continue
        digest = pair_canonical_hash(dfa_a, dfa_b)
        if digest != record.get("canonical_hash"):
            failures.append(
                f"{record['pair_id']}: stored canonical_hash "
                f"{record.get('canonical_hash')} but the pair hashes to {digest}"
            )
        seen[digest] += 1
        owners.setdefault(digest, []).append(record["pair_id"])
        if digest in excluded:
            failures.append(f"{record['pair_id']}: hash also present in {excluded_from}")
    for digest, count in seen.items():
        if count > 1:
            failures.append(f"duplicate canonical hash across {owners[digest]}")

    sources = list(exclusion_sources or [])
    failures.extend(exclusion_union_failures(excluded, excluded_from, sources))

    return GateResult.of(
        7,
        "deduplication",
        failures,
        len(records),
        distinct_hashes=len(seen),
        excluded_list=excluded_from,
        excluded_count=len(excluded),
        exclusion_sources=sources,
    )


#: A rendering label: "q" followed by digits. Both conditions share this
#: vocabulary since D5.4, which is what the structural check verifies.
_LABEL_RE = re.compile(r"\bq\d+\b")


def structural_label_balance(
    renderings_dir: Path, pair_ids: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Check the properties that make R0/R1 token balance hold without a tokenizer.

    Three things are true by construction once R1 reuses R0's vocabulary under a derangement
    (D5.4), and each is checked here rather than assumed:

    1. **Identical label vocabulary.** The same multiset of label strings appears in both.
    2. **Identical label-occurrence count.** Each rendering names a state the same number of
       times, so no label is added or dropped.
    3. **Identical character count.** Follows from 1 and 2 with fixed-width labels, and is
       cheap to verify directly.

    What remains is *not* zero: the per-label frequencies are reweighted by the derangement,
    so if two labels tokenize to different lengths the totals can drift slightly. The drift
    is bounded by half the L1 distance between the frequency vectors times the spread in
    per-label token cost -- a few tokens in ~840, far inside the +-10% gate, but a bound
    rather than an identity. This function reports that L1 distance so the bound is visible,
    and it does **not** substitute for the measured ratio: a violation here fails gate 8, but
    passing here does not pass it.
    """
    failures: list[str] = []
    worst_l1 = 0
    checked = 0
    for pair_id in sorted(pair_ids):
        for name in ("A", "B"):
            try:
                text_0 = read_text(renderings_dir / f"{pair_id}_{name}_R0.txt")
                text_1 = read_text(renderings_dir / f"{pair_id}_{name}_R1.txt")
            except FileNotFoundError:
                continue
            checked += 1
            labels_0 = _LABEL_RE.findall(text_0)
            labels_1 = _LABEL_RE.findall(text_1)
            if sorted(set(labels_0)) != sorted(set(labels_1)):
                failures.append(
                    f"{pair_id}/{name}: R0 and R1 use different label vocabularies, so the "
                    f"token balance no longer holds by construction"
                )
                continue
            if len(labels_0) != len(labels_1):
                failures.append(
                    f"{pair_id}/{name}: {len(labels_0)} label occurrences in R0 but "
                    f"{len(labels_1)} in R1"
                )
            if len(text_0) != len(text_1):
                failures.append(
                    f"{pair_id}/{name}: {len(text_0)} characters in R0 but "
                    f"{len(text_1)} in R1"
                )
            counts_0 = Counter(labels_0)
            counts_1 = Counter(labels_1)
            worst_l1 = max(
                worst_l1,
                sum(abs(counts_0[k] - counts_1[k]) for k in set(counts_0) | set(counts_1)),
            )
    return failures, {
        "structural_machines_checked": checked,
        "worst_label_frequency_l1": worst_l1,
    }


def gate_8_token_balance(
    renderings_dir: Path,
    token_counter: Callable[[str], int] | None,
    tolerance: float,
    expected_pair_ids: list[str] | None = None,
    r1_scheme: str = "shared_vocabulary_deranged",
) -> GateResult:
    """R1 and R0 prompts must be within ``tolerance`` of each other in input tokens.

    R1 changes labels and row order and nothing else, so it must not also change prompt
    size: a systematic length difference would confound the H3 comparison, since the two
    conditions would differ in cost and context pressure as well as in presentation.

    Every expected pair must be measured. Without that a missing rendering is skipped
    silently and the gate passes on whatever remains -- an empty manifest passed on zero
    measurements before this check existed.
    """
    manifest_path = renderings_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return GateResult.skipped(
            8, "token_balance", f"no renderings at {manifest_path}"
        )
    pairs_seen: set[str] = {row["pair_id"] for row in read_jsonl(manifest_path)}
    # The structural guarantee only exists under the shared-vocabulary scheme. Applying it to
    # a legacy nonce-labelled dataset would condemn data that was rendered correctly under the
    # scheme in force at the time (D2 keeps the v1 pilot reproducible).
    if r1_scheme == "shared_vocabulary_deranged":
        structural_failures, structural_detail = structural_label_balance(
            renderings_dir, sorted(pairs_seen)
        )
    else:
        structural_failures, structural_detail = [], {
            "structural_check": f"not applicable to the {r1_scheme!r} label scheme"
        }
    if structural_failures:
        # A vocabulary or length mismatch means the by-construction guarantee is gone, and
        # that is a failure whether or not a tokenizer is available to measure the damage.
        return GateResult.of(
            8, "token_balance", structural_failures, 0, **structural_detail
        )
    if token_counter is None:
        if "worst_label_frequency_l1" in structural_detail:
            reason = (
                "structural balance holds (identical label vocabulary, occurrence count and "
                "character count; worst label-frequency L1 "
                f"{structural_detail['worst_label_frequency_l1']}), but no tokenizer is "
                "configured (qa.tokenizer_for_token_balance) and no measured per-model "
                "ratios are recorded. Structure bounds the residual; it does not measure it."
            )
        else:
            reason = (
                "no tokenizer configured (qa.tokenizer_for_token_balance). Token counts are "
                "model-specific, so this gate cannot run until a model is chosen."
            )
        return GateResult.skipped(8, "token_balance", reason, **structural_detail)

    failures: list[str] = []
    if expected_pair_ids is not None:
        absent = sorted(set(expected_pair_ids) - pairs_seen)
        if absent:
            failures.append(
                f"{len(absent)} of {len(expected_pair_ids)} pairs are absent from the "
                f"manifest, e.g. {absent[:5]}"
            )

    ratios: list[float] = []
    for pair_id in sorted(pairs_seen):
        for name in ("A", "B"):
            try:
                r0 = token_counter(
                    read_text(renderings_dir / f"{pair_id}_{name}_R0.txt")
                )
                r1 = token_counter(
                    read_text(renderings_dir / f"{pair_id}_{name}_R1.txt")
                )
            except FileNotFoundError as missing:
                failures.append(f"{pair_id}/{name}: rendering missing ({missing.filename})")
                continue
            if r0 == 0:
                failures.append(f"{pair_id}/{name}: R0 tokenized to zero tokens")
                continue
            ratio = r1 / r0
            ratios.append(ratio)
            if abs(ratio - 1.0) > tolerance:
                failures.append(
                    f"{pair_id}/{name}: R1/R0 token ratio {ratio:.4f} outside "
                    f"±{tolerance:.0%} (R0={r0}, R1={r1})"
                )

    detail: dict[str, Any] = {"tolerance": tolerance, "n_measured": len(ratios)}
    if ratios:
        detail |= {
            "mean_ratio": round(sum(ratios) / len(ratios), 6),
            "min_ratio": round(min(ratios), 6),
            "max_ratio": round(max(ratios), 6),
        }
    return GateResult.of(
        8, "token_balance", failures, len(ratios), **detail, **structural_detail
    )


# --------------------------------------------------------------------------- driver


def token_counter_from_config(config: Config) -> Callable[[str], int] | None:
    """Build a token counter from ``qa.tokenizer_for_token_balance``, if one is configured.

    Returns ``None`` when the setting is absent or still a placeholder, so gate 8 reports a
    skip with a reason instead of failing on a config that legitimately cannot fill it yet.
    """
    name = config.get("qa.tokenizer_for_token_balance")
    if not name or str(name) in PLACEHOLDERS:
        return None
    from transformers import AutoTokenizer  # imported lazily: heavy, and often unneeded

    tokenizer = AutoTokenizer.from_pretrained(str(name))
    return lambda text: len(tokenizer.encode(text))


def _load_excluded(config: Config) -> tuple[set[str], str | None]:
    """Read the exclusion hash list, refusing to treat a missing file as "nothing excluded".

    Raises:
        ConfigError: if ``pairs.exclude_hashes_from`` names a file that does not exist.
            Silently substituting an empty set would make gate 7's contamination check pass
            vacuously -- the failure mode this gate exists to catch.
    """
    location = config.get("pairs.exclude_hashes_from")
    if not location:
        return set(), None
    path = Path(str(location))
    if not path.is_file():
        raise ConfigError(
            f"pairs.exclude_hashes_from points at {path}, which does not exist.\n"
            f"Gate 7 cannot verify that the pilot pairs were excluded. Generate that set "
            f"first, or remove the key if no exclusion is intended."
        )
    return {
        line.strip() for line in read_text(path).splitlines() if line.strip()
    }, str(path)


def run_all_gates(
    config: Config, token_counter: Callable[[str], int] | None = None
) -> list[GateResult]:
    """Run every gate against the configured pair set and renderings."""
    pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
    if not pairs_path.is_file():
        raise ConfigError(
            f"{pairs_path} not found. Run python -m src.generate first."
        )
    records = list(read_jsonl(pairs_path))
    renderings_dir = config.path_for("renderings_dir")
    excluded, excluded_from = _load_excluded(config)
    libraries = [str(x) for x in (config.get("qa.cross_check_libraries") or [])]
    tolerance = float(config.get("qa.token_balance_tolerance", 0.10))
    representations = config.get("representations") or []
    conditions = [str(c["id"]) for c in representations]
    r1_scheme = next(
        (
            str(c.get("labels"))
            for c in representations
            if str(c.get("id")) == "R1" and c.get("labels")
        ),
        "shared_vocabulary_deranged",
    )
    exclusion_sources = [str(s) for s in (config.get("pairs.exclusion_sources") or [])]
    pair_ids = [record["pair_id"] for record in records]

    return [
        gate_1_oracle_agreement(records, libraries),
        gate_2_dataset_balance(records, config),
        gate_3_no_epsilon_leak(records),
        gate_4_completeness(records),
        gate_5_reachability(records),
        gate_6_representation_fidelity(records, renderings_dir, conditions),
        gate_7_deduplication(records, excluded, excluded_from, exclusion_sources),
        gate_8_token_balance(
            renderings_dir, token_counter, tolerance, pair_ids, r1_scheme
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset_qa",
        description="Run the eight dataset quality gates (Protocol Step 6).",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "treat a skipped gate as a failure. Required before the protocol freeze: "
            "every gate must have actually run."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        counter = token_counter_from_config(config)
        results = run_all_gates(config, counter)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - report, do not traceback at a CLI boundary
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    width = max(len(r.name) for r in results)
    for result in results:
        line = f"  [{result.status}] gate {result.number} {result.name:<{width}}"
        if result.status == "SKIP":
            print(f"{line}  ({result.detail.get('reason', '')})")
        else:
            print(f"{line}  checked {result.checked}, failures {result.n_failures}")
            for example in result.examples:
                print(f"           - {example}")

    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]
    report_path = config.path_for("pairs_dir") / "qa_report.json"
    write_text(
        report_path,
        canonical_json(
            {
                "config_id": config.config_id,
                "gates": [r.to_dict() for r in results],
                "n_failed": len(failed),
                "n_skipped": len(skipped),
                "all_gates_ran": not skipped,
                "passed": not failed and (not skipped or not args.strict),
            }
        )
        + "\n",
    )
    print(f"\nreport: {report_path}")

    if failed:
        print(f"FAILED: {len(failed)} gate(s) did not pass.", file=sys.stderr)
        return 1
    if skipped:
        message = f"{len(skipped)} gate(s) could not run."
        if args.strict:
            print(f"FAILED (--strict): {message}", file=sys.stderr)
            return 1
        print(f"WARNING: {message} Re-run with --strict before the freeze.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
