"""Shared test fixtures and deterministic random DFA construction.

Every helper takes an explicit seeded RNG. No module-level seeding (research rule 7):
a test that fails must fail identically on the next run.
"""

from __future__ import annotations

import random

from src.dfa import DFA
from src.simulate import reachable

BINARY = ("0", "1")


def random_dfa(
    rng: random.Random,
    n_states: int = 6,
    alphabet: tuple[str, ...] = BINARY,
    *,
    accept_probability: float = 0.4,
) -> DFA:
    """A complete DFA with uniformly random transitions.

    States may be unreachable -- that is realistic for a raw random draw, and the
    protocol's reachability requirement is enforced by a generator's rejection rule
    rather than by construction (§7.3). Use :func:`random_reachable_dfa` where a test
    needs all states reachable.
    """
    delta = tuple(
        tuple(rng.randrange(n_states) for _ in alphabet) for _ in range(n_states)
    )
    accepting = frozenset(q for q in range(n_states) if rng.random() < accept_probability)
    return DFA.build(
        n_states=n_states,
        alphabet=alphabet,
        start=rng.randrange(n_states),
        accepting=accepting,
        delta=delta,
    )


def random_reachable_dfa(
    rng: random.Random,
    n_states: int = 6,
    alphabet: tuple[str, ...] = BINARY,
    *,
    accept_probability: float = 0.4,
    max_attempts: int = 200,
) -> DFA:
    """A random complete DFA in which every state is reachable, by rejection sampling."""
    for _ in range(max_attempts):
        dfa = random_dfa(rng, n_states, alphabet, accept_probability=accept_probability)
        if len(reachable(dfa)) == n_states:
            return dfa
    raise RuntimeError(
        f"no fully reachable DFA with {n_states} states after {max_attempts} attempts"
    )


def random_permutation(rng: random.Random, n: int) -> tuple[int, ...]:
    """A uniformly random bijection on ``0..n-1``, as a tuple ``old -> new``."""
    perm = list(range(n))
    rng.shuffle(perm)
    return tuple(perm)


def all_words(alphabet: tuple[str, ...], max_length: int) -> list[str]:
    """Every word over ``alphabet`` of length ``0..max_length``, shortest first.

    Used to verify by exhaustion that a BFS witness really is shortest (§7.4).
    """
    words = [""]
    frontier = [""]
    for _ in range(max_length):
        frontier = [w + s for w in frontier for s in alphabet]
        words.extend(frontier)
    return words


# --------------------------------------------------------------------------- fixtures


def even_zeros_dfa() -> DFA:
    """Accepts binary words containing an even number of ``0``s. Start is accepting."""
    return DFA.build(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=[0],
        #        on "0", on "1"
        delta=[[1, 0],  # state 0: even count
               [0, 1]],  # state 1: odd count
    )


def ends_with_one_dfa() -> DFA:
    """Accepts binary words ending in ``1``. The empty word is rejected."""
    return DFA.build(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=[1],
        delta=[[0, 1],
               [0, 1]],
    )
