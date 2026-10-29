"""Tests for simulation, reachability, and access distances (M1, Step 2)."""

from __future__ import annotations

import random

import pytest

from src.dfa import DFA, DFAValidationError
from src.simulate import (
    accepts,
    access_distances,
    final_state,
    first_divergence,
    reachable,
    trace,
    validate_word,
)
from tests.helpers import (
    BINARY,
    all_words,
    ends_with_one_dfa,
    even_zeros_dfa,
    random_dfa,
    random_permutation,
    random_reachable_dfa,
)

# --------------------------------------------------------------------------- known languages


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("", True),  # zero 0s
        ("0", False),
        ("00", True),
        ("1", True),
        ("010", True),  # two 0s
        ("0110", True),  # two 0s
        ("01100", False),  # three 0s
    ],
)
def test_even_zeros_language(word, expected):
    assert accepts(even_zeros_dfa(), word) is expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [("", False), ("1", True), ("0", False), ("10", False), ("011", True)],
)
def test_ends_with_one_language(word, expected):
    assert accepts(ends_with_one_dfa(), word) is expected


def test_empty_word_is_the_start_state():
    """The empty word is the empty string; accepting it is exactly start-state acceptance."""
    assert accepts(even_zeros_dfa(), "") is True
    assert accepts(ends_with_one_dfa(), "") is False


# --------------------------------------------------------------------------- trace invariants


def test_trace_length_is_word_length_plus_one():
    """The invariant the P1 prompt states to the model and the grader enforces (§7.2)."""
    rng = random.Random(101)
    dfa = random_reachable_dfa(rng, n_states=9)
    for length in range(0, 13):
        word = "".join(rng.choice(BINARY) for _ in range(length))
        assert len(trace(dfa, word)) == length + 1


def test_trace_starts_at_the_initial_state():
    rng = random.Random(102)
    dfa = random_reachable_dfa(rng, n_states=6)
    assert trace(dfa, "0110")[0] == dfa.start
    assert trace(dfa, "") == (dfa.start,)


def test_trace_final_state_agrees_with_accepts():
    rng = random.Random(103)
    for _ in range(200):
        dfa = random_dfa(rng, n_states=rng.randrange(2, 8))
        word = "".join(rng.choice(BINARY) for _ in range(rng.randrange(0, 10)))
        states = trace(dfa, word)
        assert states[-1] == final_state(dfa, word)
        assert (states[-1] in dfa.accepting) == accepts(dfa, word)


def test_trace_steps_follow_delta():
    rng = random.Random(104)
    dfa = random_reachable_dfa(rng, n_states=7)
    word = "0110100"
    states = trace(dfa, word)
    for i, symbol in enumerate(word):
        assert states[i + 1] == dfa.step(states[i], symbol)


# --------------------------------------------------------------------------- word validation


def test_illegal_symbol_is_rejected_with_its_index():
    """The grader depends on this to classify a bad witness rather than crash the run."""
    dfa = even_zeros_dfa()
    with pytest.raises(DFAValidationError, match=r"word\[2\] = '2'"):
        trace(dfa, "012")
    with pytest.raises(DFAValidationError, match=r"word\[0\] = 'a'"):
        accepts(dfa, "abc")


def test_validate_word_accepts_the_empty_word():
    validate_word(even_zeros_dfa(), "")


# --------------------------------------------------------------------------- renaming


def test_renaming_preserves_the_language():
    """§7.4: a DFA is equivalent to an independently renamed copy of itself."""
    rng = random.Random(105)
    for _ in range(30):
        n = rng.randrange(2, 8)
        dfa = random_reachable_dfa(rng, n_states=n)
        renamed = dfa.relabel(random_permutation(rng, n))
        for word in all_words(BINARY, 8):
            assert accepts(dfa, word) == accepts(renamed, word)


# --------------------------------------------------------------------------- reachability


def test_reachable_finds_every_state_of_a_connected_dfa():
    rng = random.Random(106)
    dfa = random_reachable_dfa(rng, n_states=8)
    assert reachable(dfa) == frozenset(range(8))


def test_reachable_excludes_a_disconnected_sink():
    dfa = DFA.build(
        n_states=3, alphabet=BINARY, start=0, accepting=[2], delta=[[0, 1], [1, 0], [2, 2]]
    )
    assert reachable(dfa) == frozenset({0, 1})


def test_reachable_always_contains_the_start():
    rng = random.Random(107)
    dfa = random_dfa(rng, n_states=6)
    assert dfa.start in reachable(dfa)


# --------------------------------------------------------------------------- access distances


def test_access_distance_of_the_start_is_zero():
    rng = random.Random(108)
    dfa = random_dfa(rng, n_states=6)
    assert access_distances(dfa)[dfa.start] == 0


def test_access_distances_on_a_known_chain():
    """A 0-chain 0->1->2->3 with all 1-edges returning to the start."""
    dfa = DFA.build(
        n_states=4,
        alphabet=BINARY,
        start=0,
        accepting=[3],
        delta=[[1, 0], [2, 0], [3, 0], [3, 0]],
    )
    assert access_distances(dfa) == (0, 1, 2, 3)


def test_unreachable_states_have_no_distance():
    dfa = DFA.build(
        n_states=3, alphabet=BINARY, start=0, accepting=[2], delta=[[0, 1], [1, 0], [2, 2]]
    )
    assert access_distances(dfa) == (0, 1, None)


def test_access_distances_agree_with_shortest_words():
    """Cross-check BFS against exhaustive enumeration of short words."""
    rng = random.Random(109)
    for _ in range(30):
        n = rng.randrange(2, 7)
        dfa = random_dfa(rng, n_states=n)
        distances = access_distances(dfa)
        shortest: dict[int, int] = {}
        for word in all_words(BINARY, 8):
            state = final_state(dfa, word)
            shortest.setdefault(state, len(word))
        for state in range(n):
            assert distances[state] == shortest.get(state)


def test_access_distances_are_renaming_invariant():
    rng = random.Random(110)
    n = 7
    dfa = random_reachable_dfa(rng, n_states=n)
    perm = random_permutation(rng, n)
    renamed = dfa.relabel(perm)
    original = access_distances(dfa)
    moved = access_distances(renamed)
    for old_state in range(n):
        assert original[old_state] == moved[perm[old_state]]


# --------------------------------------------------------------------------- trace comparison


def test_first_divergence_detects_a_one_symbol_corruption_at_the_right_index():
    """§7.4: a one-symbol corruption in a model trace is detected at the correct step."""
    rng = random.Random(111)
    dfa = random_reachable_dfa(rng, n_states=9)
    word = "01101"
    expected = trace(dfa, word)
    for i in range(len(expected)):
        corrupted = list(expected)
        corrupted[i] = (corrupted[i] + 1) % dfa.n_states
        assert first_divergence(expected, corrupted) == i


def test_first_divergence_is_none_for_identical_traces():
    assert first_divergence((0, 1, 2), (0, 1, 2)) is None


def test_first_divergence_flags_a_short_trace():
    """A trace of the wrong length is the P1 failure mode the schema cannot catch."""
    assert first_divergence((0, 1, 2, 3), (0, 1, 2)) == 3
    assert first_divergence((0, 1), (0, 1, 2)) == 2
