"""The exact equivalence oracle: product-automaton BFS with a shortest witness.

Protocol Step 3 and §3.2. This module is the single source of truth for every label in the
dataset. A pair is equivalent because :func:`equivalence_and_witness` said so and for no
other reason -- generator intent is a request, never a label (research rule 1).

The construction. For DFAs A and B over the same alphabet, explore product states ``(p, q)``
from ``(start_A, start_B)``. A product state is *distinguishing* when exactly one component
is accepting::

    F_△ = [F_A × (Q_B − F_B)] ∪ [(Q_A − F_A) × F_B]

The languages differ if and only if a distinguishing product state is reachable. Because the
search is breadth-first, the first one encountered is at minimum distance from the start, so
reconstructing its access path yields a *shortest* distinguishing word.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.dfa import DFA, DFAValidationError
from src.simulate import trace

__all__ = [
    "OracleResult",
    "equivalence_and_witness",
    "witness_upper_bound",
]


@dataclass(frozen=True, slots=True)
class OracleResult:
    """The exact verdict for one DFA pair (§Step 3).

    Attributes:
        equivalent: Whether ``L(A) == L(B)``.
        shortest_witness: A word accepted by exactly one of the two DFAs, of minimum
            length; ``None`` when the pair is equivalent. The empty word is ``""``, which
            is distinct from ``None`` -- see :attr:`shortest_depth`.
        shortest_depth: ``len(shortest_witness)``, the difficulty variable ``d`` of §3.3;
            ``None`` when the pair is equivalent. ``d == 0`` means the start states already
            disagree on acceptance, which the main set excludes (§Step 3).
        product_trace_A: States visited in A while reading the witness, including the
            initial state, so its length is ``d + 1``; ``None`` when equivalent.
        product_trace_B: The same for B.
    """

    equivalent: bool
    shortest_witness: str | None
    shortest_depth: int | None
    product_trace_A: tuple[int, ...] | None
    product_trace_B: tuple[int, ...] | None

    def __post_init__(self) -> None:
        # An OracleResult is written into every base-pair record and drives every metric,
        # so an internally inconsistent one must never exist.
        if self.equivalent:
            if not (
                self.shortest_witness is None
                and self.shortest_depth is None
                and self.product_trace_A is None
                and self.product_trace_B is None
            ):
                raise ValueError("an equivalent result carries no witness or traces")
        else:
            if self.shortest_witness is None or self.shortest_depth is None:
                raise ValueError("a non-equivalent result requires a witness and a depth")
            if self.shortest_depth != len(self.shortest_witness):
                raise ValueError(
                    f"depth {self.shortest_depth} does not match witness "
                    f"{self.shortest_witness!r} of length {len(self.shortest_witness)}"
                )
            for name, tr in (
                ("product_trace_A", self.product_trace_A),
                ("product_trace_B", self.product_trace_B),
            ):
                if tr is None or len(tr) != self.shortest_depth + 1:
                    raise ValueError(
                        f"{name} must have length depth+1 = {self.shortest_depth + 1}"
                    )


def witness_upper_bound(dfa_a: DFA, dfa_b: DFA) -> int:
    """Largest possible shortest-witness length: ``|Q_A|·|Q_B| - 1`` (§3.2).

    A shortest path through the product need not revisit a product state, so it can touch
    at most ``|Q_A|·|Q_B|`` states and therefore uses at most that many minus one symbols.
    Useful as an assertion and to bound the depth strata a generator can request.
    """
    return dfa_a.n_states * dfa_b.n_states - 1


def equivalence_and_witness(dfa_a: DFA, dfa_b: DFA) -> OracleResult:
    """Decide ``L(A) == L(B)`` exactly, returning a shortest witness when they differ.

    Symbols are tried in ``sorted(alphabet)`` order, as the protocol pseudocode specifies.
    That matters: among the several shortest witnesses a pair may admit, BFS returns the
    first one reached, so a fixed symbol order is what makes the returned witness -- and
    hence every ``oracle_witness`` field in the dataset -- reproducible. Sorting rather
    than reusing the alphabet tuple's own order keeps the result independent of how the
    :class:`~src.dfa.DFA` happened to be constructed.

    Args:
        dfa_a: The first automaton.
        dfa_b: The second automaton, over the same alphabet.

    Returns:
        An :class:`OracleResult`. When the pair is equivalent the witness, depth, and
        traces are all ``None``; when it is not, the witness is of minimum length and the
        traces are the deterministic runs of that witness in each DFA.

    Raises:
        DFAValidationError: if the two alphabets differ as sets, which always signals a
            construction bug rather than a legitimate comparison.
    """
    if set(dfa_a.alphabet) != set(dfa_b.alphabet):
        raise DFAValidationError(
            f"alphabets differ: {dfa_a.alphabet!r} vs {dfa_b.alphabet!r}"
        )

    # Symbol order is the protocol's sorted order; column indices are resolved per DFA, so
    # the two automata may legitimately store their alphabets in different orders.
    symbols = sorted(dfa_a.alphabet)
    col_a = {s: dfa_a.alphabet.index(s) for s in symbols}
    col_b = {s: dfa_b.alphabet.index(s) for s in symbols}

    start = (dfa_a.start, dfa_b.start)
    # predecessor maps a product state to (previous product state, symbol taken). The
    # start maps to None, which also marks it as visited.
    predecessor: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        state = queue.popleft()
        p, q = state
        if (p in dfa_a.accepting) != (q in dfa_b.accepting):
            witness = _reconstruct(predecessor, state)
            return OracleResult(
                equivalent=False,
                shortest_witness=witness,
                shortest_depth=len(witness),
                product_trace_A=trace(dfa_a, witness),
                product_trace_B=trace(dfa_b, witness),
            )

        for symbol in symbols:
            nxt = (dfa_a.delta[p][col_a[symbol]], dfa_b.delta[q][col_b[symbol]])
            if nxt not in predecessor:
                predecessor[nxt] = (state, symbol)
                queue.append(nxt)

    return OracleResult(
        equivalent=True,
        shortest_witness=None,
        shortest_depth=None,
        product_trace_A=None,
        product_trace_B=None,
    )


def _reconstruct(
    predecessor: dict[tuple[int, int], tuple[tuple[int, int], str] | None],
    target: tuple[int, int],
) -> str:
    """Walk predecessors back from ``target`` to the start and return the access word.

    Returns ``""`` when ``target`` is the start product state, which is the ``d == 0``
    case: the two start states already disagree on acceptance, so the empty word
    distinguishes them (§Step 3).
    """
    symbols: list[str] = []
    state = target
    while (edge := predecessor[state]) is not None:
        previous, symbol = edge
        symbols.append(symbol)
        state = previous
    symbols.reverse()
    return "".join(symbols)
