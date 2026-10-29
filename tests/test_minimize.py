"""Tests for DFA minimization and unreachable-state pruning (M3, §5.5)."""

from __future__ import annotations

import random

from src.crosscheck import automata_lib_equivalent, to_automata_lib
from src.dfa import DFA, canonical_form
from src.minimize import minimal_state_count, minimize, prune_unreachable
from src.oracle import equivalence_and_witness
from tests.helpers import BINARY, even_zeros_dfa, random_dfa, random_reachable_dfa

# --------------------------------------------------------------------------- pruning


def test_prune_preserves_the_language():
    rng = random.Random(3001)
    for _ in range(100):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 9))
        assert equivalence_and_witness(dfa, prune_unreachable(dfa)).equivalent


def test_prune_drops_a_disconnected_sink():
    dfa = DFA.build(
        n_states=3, alphabet=BINARY, start=0, accepting=[2], delta=[[0, 1], [1, 0], [2, 2]]
    )
    pruned = prune_unreachable(dfa)
    assert pruned.n_states == 2
    assert pruned.accepting == frozenset()  # the accepting state was the unreachable one


def test_prune_is_a_no_op_on_a_fully_reachable_dfa():
    rng = random.Random(3002)
    dfa = random_reachable_dfa(rng, n_states=8)
    assert canonical_form(prune_unreachable(dfa)) == canonical_form(dfa)


# --------------------------------------------------------------------------- minimization


def test_minimize_preserves_the_language():
    """The property EQ_EXPAND depends on: the core must recognize the sampled language."""
    rng = random.Random(3003)
    for _ in range(200):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 10))
        assert equivalence_and_witness(dfa, minimize(dfa)).equivalent


def test_minimize_never_grows_the_machine():
    rng = random.Random(3004)
    for _ in range(100):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 10))
        assert minimize(dfa).n_states <= dfa.n_states


def test_minimize_is_idempotent():
    rng = random.Random(3005)
    for _ in range(100):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 10))
        once = minimize(dfa)
        assert canonical_form(minimize(once)) == canonical_form(once)


def test_minimize_is_deterministic():
    """It runs on the generation path, so its output must be a pure function of its input."""
    rng = random.Random(3006)
    for _ in range(50):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 10))
        assert minimize(dfa) == minimize(dfa)


def test_minimize_collapses_a_renamed_copy_to_the_same_machine():
    rng = random.Random(3007)
    for _ in range(50):
        n = rng.randrange(2, 9)
        dfa = random_reachable_dfa(rng, n_states=n)
        permutation = list(range(n))
        rng.shuffle(permutation)
        renamed = dfa.relabel(tuple(permutation))
        assert canonical_form(minimize(dfa)) == canonical_form(minimize(renamed))


def test_minimize_matches_automata_lib_state_count():
    """Independent confirmation that the hand-written Moore refinement reaches the minimum.

    The minimal DFA is unique up to isomorphism, so agreeing on the state count with a
    library implementation is a real check, not a coincidence.
    """
    rng = random.Random(3008)
    for _ in range(200):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 9))
        ours = minimize(dfa)
        theirs = to_automata_lib(dfa).minify(retain_names=False)
        assert ours.n_states == len(theirs.states)
        assert automata_lib_equivalent(dfa, ours)


# --------------------------------------------------------------------------- edge cases


def test_empty_language_minimizes_to_one_state():
    dfa = DFA.build(
        n_states=4,
        alphabet=BINARY,
        start=0,
        accepting=[],
        delta=[[1, 2], [3, 0], [0, 1], [2, 3]],
    )
    assert minimize(dfa).n_states == 1
    assert minimize(dfa).accepting == frozenset()


def test_universal_language_minimizes_to_one_accepting_state():
    dfa = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=[0, 1, 2],
        delta=[[1, 2], [2, 0], [0, 1]],
    )
    minimal = minimize(dfa)
    assert minimal.n_states == 1
    assert minimal.accepting == frozenset({0})


def test_unreachable_accepting_state_does_not_survive_minimization():
    """The shape that broke pyformlang's is_equivalent_to; minimization must see through it."""
    dfa = DFA.build(
        n_states=3, alphabet=BINARY, start=0, accepting=[2], delta=[[0, 1], [0, 0], [0, 1]]
    )
    assert minimize(dfa).n_states == 1
    assert minimize(dfa).accepting == frozenset()


def test_an_already_minimal_machine_is_unchanged():
    dfa = even_zeros_dfa()
    assert canonical_form(minimize(dfa)) == canonical_form(dfa)


def test_duplicate_states_are_merged():
    """Two states with identical acceptance and identical successors are one state."""
    dfa = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=[1, 2],
        # States 1 and 2 behave identically.
        delta=[[1, 2], [1, 2], [1, 2]],
    )
    assert minimize(dfa).n_states == 2


def test_minimal_state_count_matches_minimize():
    rng = random.Random(3009)
    for _ in range(50):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 9))
        assert minimal_state_count(dfa) == minimize(dfa).n_states
