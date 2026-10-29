"""DFA simulation, reachability, and shortest access distances.

Protocol Step 2. These are the deterministic ground-truth operations the grader compares
model output against, so they must be exact and side-effect free.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

from collections import deque

from src.dfa import DFA, DFAValidationError

__all__ = [
    "accepts",
    "access_distances",
    "final_state",
    "first_divergence",
    "reachable",
    "trace",
    "validate_word",
]


def validate_word(dfa: DFA, word: str) -> None:
    """Check that every symbol of ``word`` is in the alphabet.

    Called before simulation so an illegal symbol fails with a precise message rather
    than an opaque lookup error. The grader relies on this to classify a model witness
    containing a symbol outside the alphabet as a parse failure (§8.5) instead of
    crashing the run.

    Raises:
        DFAValidationError: naming the first offending symbol and its index.
    """
    for i, symbol in enumerate(word):
        if symbol not in dfa.alphabet:
            raise DFAValidationError(
                f"word[{i}] = {symbol!r} is not in alphabet {dfa.alphabet!r}"
            )


def trace(dfa: DFA, word: str) -> tuple[int, ...]:
    """The full state sequence visited while reading ``word``.

    Includes the initial state, so the result has length ``len(word) + 1`` -- the
    invariant the P1 prompt states to the model and the grader enforces (§7.2).

    Raises:
        DFAValidationError: if ``word`` contains a symbol outside the alphabet.
    """
    validate_word(dfa, word)
    # Resolve symbol columns once: the same word is simulated against both DFAs of a
    # pair, and witnesses can be a dozen symbols long across thousands of calls.
    index = {symbol: i for i, symbol in enumerate(dfa.alphabet)}
    states = [dfa.start]
    state = dfa.start
    for symbol in word:
        state = dfa.delta[state][index[symbol]]
        states.append(state)
    return tuple(states)


def final_state(dfa: DFA, word: str) -> int:
    """``delta*(start, word)`` without materializing the intermediate states."""
    validate_word(dfa, word)
    index = {symbol: i for i, symbol in enumerate(dfa.alphabet)}
    state = dfa.start
    for symbol in word:
        state = dfa.delta[state][index[symbol]]
    return state


def accepts(dfa: DFA, word: str) -> bool:
    """Whether ``word`` is in ``L(dfa)``.

    The empty string is the empty word: ``accepts(dfa, "")`` is whether the start state
    is accepting.
    """
    return final_state(dfa, word) in dfa.accepting


def reachable(dfa: DFA) -> frozenset[int]:
    """States reachable from the start."""
    seen = {dfa.start}
    queue = deque([dfa.start])
    while queue:
        state = queue.popleft()
        for dest in dfa.delta[state]:
            if dest not in seen:
                seen.add(dest)
                queue.append(dest)
    return frozenset(seen)


def access_distances(dfa: DFA) -> tuple[int | None, ...]:
    """Shortest access distance from the start to each state, indexed by state id.

    ``None`` marks an unreachable state. Distance 0 is the start state itself.

    This is the quantity the NEQ_DEPTH generator controls: it needs a designated state
    whose shortest access distance is exactly the target depth ``d`` (§5.4). Note that a
    state's access distance is a *necessary* condition for the pair's witness depth, not
    a sufficient one -- only the product BFS in :mod:`src.oracle` establishes the actual
    ``d``, and it is the oracle that assigns the label (research rule 1).
    """
    distance: list[int | None] = [None] * dfa.n_states
    distance[dfa.start] = 0
    queue = deque([dfa.start])
    while queue:
        state = queue.popleft()
        here = distance[state]
        assert here is not None  # dequeued states always have a distance
        for dest in dfa.delta[state]:
            if distance[dest] is None:
                distance[dest] = here + 1
                queue.append(dest)
    return tuple(distance)


def first_divergence(
    expected: tuple[int, ...] | list[int], reported: tuple[int, ...] | list[int]
) -> int | None:
    """Index of the first position where two traces disagree, or ``None`` if identical.

    A length mismatch diverges at the first index past the shorter trace. Used by the
    grader to record the first wrong symbol index of a model trace, which the error
    taxonomy reports as a "trace transition error" (§11.2).
    """
    for i, (exp, rep) in enumerate(zip(expected, reported, strict=False)):
        if exp != rep:
            return i
    if len(expected) != len(reported):
        return min(len(expected), len(reported))
    return None
