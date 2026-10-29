"""The canonical DFA object, structural validation, and canonical hashing.

Protocol Step 1. States are stable internal integer IDs; surface labels such as ``q00`` or
``miv`` belong only to renderings (:mod:`src.rendering`) and must never appear here, in a
hash input, or in an oracle computation (research rule 10).

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from src.io_utils import canonical_json, sha256_of_json

__all__ = [
    "DFA",
    "DFAValidationError",
    "canonical_form",
    "dfa_canonical_hash",
    "pair_canonical_hash",
    "to_record",
    "validate",
]


class DFAValidationError(ValueError):
    """A DFA violates a structural invariant required by the protocol."""


@dataclass(frozen=True, slots=True)
class DFA:
    """A complete deterministic finite automaton over integer states ``0..n_states-1``.

    Attributes:
        n_states: Number of states. ``Q = range(n_states)``.
        alphabet: Input symbols, in the order that fixes ``delta``'s columns. The oracle
            and every generator iterate symbols in this order, so it must stay stable for
            witnesses to be reproducible.
        start: The initial state.
        accepting: The accepting states ``F``.
        delta: ``delta[state][symbol_index] -> state``. Total by construction: a DFA that
            reaches this type has one destination for every (state, symbol) pair.

    Frozen: a DFA is never mutated in place. Derive a new one with
    :meth:`with_accepting` or :meth:`relabel`.
    """

    n_states: int
    alphabet: tuple[str, ...]
    start: int
    accepting: frozenset[int]
    delta: tuple[tuple[int, ...], ...]

    # ----------------------------------------------------------------- construction

    @classmethod
    def build(
        cls,
        *,
        n_states: int,
        alphabet: tuple[str, ...] | list[str],
        start: int,
        accepting: frozenset[int] | set[int] | list[int],
        delta: tuple[tuple[int, ...], ...] | list[list[int]],
        validate_now: bool = True,
    ) -> DFA:
        """Build a DFA from loose inputs, normalizing container types.

        Validates by default: an invalid DFA should never exist as an object, so that
        every downstream module can assume totality and range-correctness.
        """
        dfa = cls(
            n_states=n_states,
            alphabet=tuple(alphabet),
            start=start,
            accepting=frozenset(accepting),
            delta=tuple(tuple(row) for row in delta),
        )
        if validate_now:
            validate(dfa)
        return dfa

    # ----------------------------------------------------------------- accessors

    @property
    def states(self) -> range:
        """The state set ``Q``."""
        return range(self.n_states)

    def symbol_index(self, symbol: str) -> int:
        """Column index of ``symbol`` in :attr:`delta`.

        Raises:
            DFAValidationError: if the symbol is outside the alphabet.
        """
        try:
            return self.alphabet.index(symbol)
        except ValueError:
            raise DFAValidationError(
                f"symbol {symbol!r} is not in alphabet {self.alphabet!r}"
            ) from None

    def step(self, state: int, symbol: str) -> int:
        """Single transition ``delta(state, symbol)``."""
        return self.delta[state][self.symbol_index(symbol)]

    def is_accepting(self, state: int) -> bool:
        """Whether ``state`` is in ``F``."""
        return state in self.accepting

    # ----------------------------------------------------------------- derivation

    def with_accepting(self, accepting: frozenset[int] | set[int] | list[int]) -> DFA:
        """A copy with a different accepting set and identical transitions.

        This is how the NEQ_DEPTH family toggles acceptance at a single state (§5.3).
        """
        return DFA(
            n_states=self.n_states,
            alphabet=self.alphabet,
            start=self.start,
            accepting=frozenset(accepting),
            delta=self.delta,
        )

    def toggled(self, state: int) -> DFA:
        """A copy with ``state``'s acceptance flipped and everything else identical."""
        if not 0 <= state < self.n_states:
            raise DFAValidationError(f"state {state} outside 0..{self.n_states - 1}")
        accepting = set(self.accepting)
        accepting.symmetric_difference_update({state})
        return self.with_accepting(accepting)

    def relabel(self, permutation: tuple[int, ...] | list[int]) -> DFA:
        """Apply a state bijection: old state ``i`` becomes ``permutation[i]``.

        The result recognizes exactly the same language -- renaming states cannot change
        semantics (§3.4). This is the EQ_ISO construction and the basis of the
        renaming property test in §7.4.

        Raises:
            DFAValidationError: if ``permutation`` is not a bijection on ``Q``.
        """
        perm = tuple(permutation)
        if len(perm) != self.n_states or sorted(perm) != list(range(self.n_states)):
            raise DFAValidationError(
                f"permutation is not a bijection on 0..{self.n_states - 1}: {perm!r}"
            )
        new_delta: list[tuple[int, ...]] = [()] * self.n_states
        for old_state, row in enumerate(self.delta):
            new_delta[perm[old_state]] = tuple(perm[dest] for dest in row)
        return DFA(
            n_states=self.n_states,
            alphabet=self.alphabet,
            start=perm[self.start],
            accepting=frozenset(perm[q] for q in self.accepting),
            delta=tuple(new_delta),
        )


# --------------------------------------------------------------------------- validation


def validate(dfa: DFA, *, require_reachable: bool = False) -> None:
    """Check every structural invariant Step 2 requires.

    Args:
        dfa: The automaton to check.
        require_reachable: Also require that every state is reachable from the start.
            Main-set pairs must satisfy this (QA gate 5), but intermediate objects inside
            a generator legitimately may not, so it is off by default.

    Raises:
        DFAValidationError: on the first violation found, naming the offending element.
    """
    if dfa.n_states <= 0:
        raise DFAValidationError(f"n_states must be positive, got {dfa.n_states}")

    if len(dfa.alphabet) == 0:
        raise DFAValidationError("alphabet must be non-empty")
    if len(set(dfa.alphabet)) != len(dfa.alphabet):
        raise DFAValidationError(f"alphabet has duplicate symbols: {dfa.alphabet!r}")

    if not 0 <= dfa.start < dfa.n_states:
        raise DFAValidationError(
            f"start state {dfa.start} outside 0..{dfa.n_states - 1}"
        )

    for q in sorted(dfa.accepting):
        if not 0 <= q < dfa.n_states:
            raise DFAValidationError(
                f"accepting state {q} outside 0..{dfa.n_states - 1}"
            )

    # Totality: exactly one destination per (state, symbol), no missing rows or columns.
    if len(dfa.delta) != dfa.n_states:
        raise DFAValidationError(
            f"delta has {len(dfa.delta)} rows, expected {dfa.n_states}"
        )
    n_symbols = len(dfa.alphabet)
    for state, row in enumerate(dfa.delta):
        if len(row) != n_symbols:
            raise DFAValidationError(
                f"state {state} has {len(row)} transitions, expected {n_symbols} "
                f"(one per symbol in {dfa.alphabet!r})"
            )
        for idx, dest in enumerate(row):
            if not isinstance(dest, int) or isinstance(dest, bool):
                raise DFAValidationError(
                    f"delta[{state}][{idx}] = {dest!r} is not an integer state id"
                )
            if not 0 <= dest < dfa.n_states:
                raise DFAValidationError(
                    f"delta({state}, {dfa.alphabet[idx]!r}) = {dest}, "
                    f"outside 0..{dfa.n_states - 1}"
                )

    if require_reachable:
        seen = _reachable_states(dfa)
        if len(seen) != dfa.n_states:
            missing = sorted(set(range(dfa.n_states)) - seen)
            raise DFAValidationError(
                f"states unreachable from start {dfa.start}: {missing}"
            )


def _reachable_states(dfa: DFA) -> set[int]:
    """States reachable from the start. Kept here so :func:`validate` stays self-contained."""
    seen = {dfa.start}
    queue = deque([dfa.start])
    while queue:
        state = queue.popleft()
        for dest in dfa.delta[state]:
            if dest not in seen:
                seen.add(dest)
                queue.append(dest)
    return seen


# --------------------------------------------------------------------------- canonical form


def canonical_form(dfa: DFA) -> DFA:
    """Renumber states into BFS order from the start, giving a label-independent form.

    For a *complete* automaton whose states are *all reachable*, breadth-first numbering
    with symbols visited in alphabet order is a canonical form: two such automata are
    isomorphic if and only if their canonical forms are identical. That is what makes
    :func:`dfa_canonical_hash` a sound deduplication key (Step 4) -- it cannot be fooled
    by a different state numbering.

    Raises:
        DFAValidationError: if any state is unreachable, since BFS numbering does not
            define a canonical position for such a state and the resulting hash would
            depend on the incoming numbering.
    """
    validate(dfa, require_reachable=True)

    order: list[int] = [dfa.start]
    position: dict[int, int] = {dfa.start: 0}
    queue = deque([dfa.start])
    while queue:
        state = queue.popleft()
        for dest in dfa.delta[state]:  # alphabet order, by construction of delta
            if dest not in position:
                position[dest] = len(order)
                order.append(dest)
                queue.append(dest)

    permutation = tuple(position[q] for q in range(dfa.n_states))
    return dfa.relabel(permutation)


def to_record(dfa: DFA, *, canonical: bool = True) -> dict[str, Any]:
    """Serialize a DFA to the plain dict used in base-pair records (§9.1).

    Args:
        dfa: The automaton.
        canonical: Convert to :func:`canonical_form` first. Records stored on disk use
            the canonical form so a record and its hash always agree.
    """
    d = canonical_form(dfa) if canonical else dfa
    return {
        "n_states": d.n_states,
        "alphabet": list(d.alphabet),
        "start": d.start,
        "accepting": sorted(d.accepting),
        "delta": [list(row) for row in d.delta],
    }


def from_record(record: dict[str, Any]) -> DFA:
    """Rebuild a DFA from a :func:`to_record` dict."""
    return DFA.build(
        n_states=record["n_states"],
        alphabet=record["alphabet"],
        start=record["start"],
        accepting=record["accepting"],
        delta=record["delta"],
    )


# --------------------------------------------------------------------------- hashing


def dfa_canonical_hash(dfa: DFA) -> str:
    """``sha256:<hex>`` over the canonical form -- invariant under state renumbering."""
    return sha256_of_json(to_record(dfa, canonical=True))


def pair_canonical_hash(dfa_a: DFA, dfa_b: DFA) -> str:
    """Deduplication key for a base pair (Step 4).

    Order-insensitive: ``(A, B)`` and ``(B, A)`` hash the same, because they are the same
    mathematical instance and admitting both would silently duplicate a unit of analysis.
    Each side is canonicalized first, so a renumbering of either automaton does not
    produce a "new" pair.
    """
    sides = sorted(
        canonical_json(to_record(d, canonical=True)) for d in (dfa_a, dfa_b)
    )
    return sha256_of_json(sides)
