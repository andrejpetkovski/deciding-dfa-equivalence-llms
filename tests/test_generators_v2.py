"""Tests for the v2 acceptance-swap generators (M3R, decision D2)."""

from __future__ import annotations

import random

import pytest

from src.dfa import DFA, canonical_form
from src.generators import clone_state
from src.generators_v2 import (
    CONSTRUCTION_VERSION,
    eq_expand_pair_v2,
    eq_iso_pair_v2,
    expand_to_with_count,
    layered_two_target_graph,
    neq_depth_pair_v2,
    random_complete_dfa_with_count,
)
from src.minimize import minimize
from src.oracle import equivalence_and_witness
from src.simulate import access_distances, final_state, reachable
from tests.helpers import BINARY, all_words

N_STATES = 14
ACCEPTING = 7
DEPTHS = list(range(1, 13))


def _as_dfa(graph, n_states: int = N_STATES) -> DFA:
    return DFA.build(
        n_states=n_states,
        alphabet=BINARY,
        start=0,
        accepting=[],
        delta=[list(row) for row in graph.delta],
    )


# --------------------------------------------------------------------------- two-target graph


@pytest.mark.parametrize("depth", DEPTHS)
def test_both_designated_states_sit_at_exactly_the_requested_depth(depth):
    """The v2 analogue of §5.4's guarantee, now for two states at once."""
    rng = random.Random(7000 + depth)
    for _ in range(20):
        graph = layered_two_target_graph(rng, N_STATES, depth, BINARY)
        assert graph is not None
        distances = access_distances(_as_dfa(graph))
        assert distances[graph.target_t] == depth
        assert distances[graph.target_u] == depth
        assert all(d is not None for d in distances), "every state must be reachable"


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_two_designated_states_are_distinct(depth):
    rng = random.Random(7100 + depth)
    graph = layered_two_target_graph(rng, N_STATES, depth, BINARY)
    assert graph is not None
    assert graph.target_t != graph.target_u


@pytest.mark.parametrize("depth", DEPTHS)
def test_access_words_reach_their_designated_states(depth):
    rng = random.Random(7200 + depth)
    graph = layered_two_target_graph(rng, N_STATES, depth, BINARY)
    assert graph is not None
    dfa = _as_dfa(graph)
    for target in (graph.target_t, graph.target_u):
        word = graph.access_word(target)
        assert len(word) == depth
        assert final_state(dfa, word) == target


def test_the_branch_uses_two_different_symbols():
    """t and u are reached from the same state, so the branch cannot reuse a symbol."""
    rng = random.Random(7300)
    for depth in DEPTHS:
        graph = layered_two_target_graph(rng, N_STATES, depth, BINARY)
        assert graph is not None
        symbol_t, symbol_u = graph.branch_symbols
        assert symbol_t != symbol_u


def test_depth_twelve_exactly_fits_fourteen_states():
    """The contract's tightest case: a depth-12 branch needs 14 states."""
    rng = random.Random(7400)
    assert layered_two_target_graph(rng, 14, 12, BINARY) is not None
    with pytest.raises(ValueError, match=r"need 15"):
        layered_two_target_graph(rng, 14, 13, BINARY)


def test_a_single_symbol_alphabet_cannot_branch():
    rng = random.Random(7401)
    with pytest.raises(ValueError, match=r"at least two symbols"):
        layered_two_target_graph(rng, 14, 3, ("0",))


# --------------------------------------------------------------------------- NEQ_DEPTH v2


@pytest.mark.parametrize("depth", DEPTHS)
def test_v2_pairs_have_exactly_the_requested_oracle_depth(depth):
    rng = random.Random(7500 + depth)
    for _ in range(15):
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None, draw.reason
        result = equivalence_and_witness(*draw.pair)
        assert not result.equivalent
        assert result.shortest_depth == depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_v2_pairs_have_equal_accepting_counts(depth):
    """The whole point of the revision: the count carries no information (D2)."""
    rng = random.Random(7600 + depth)
    for _ in range(15):
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert len(dfa_a.accepting) == len(dfa_b.accepting) == ACCEPTING


def test_v2_pairs_differ_at_exactly_two_states():
    """F_A △ F_B = {t, u}: one swap, not one toggle."""
    rng = random.Random(7700)
    for depth in DEPTHS:
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert dfa_a.delta == dfa_b.delta
        assert len(dfa_a.accepting ^ dfa_b.accepting) == 2


def test_distinguishing_words_are_exactly_those_reaching_a_designated_state():
    """Why the depth is exact under the swap: A and B agree everywhere else, so a word
    separates them precisely when it ends at t or at u."""
    rng = random.Random(7800)
    for depth in (1, 2, 3, 4):
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        designated = dfa_a.accepting ^ dfa_b.accepting
        for word in all_words(BINARY, 7):
            end = final_state(dfa_a, word)
            distinguishes = (end in dfa_a.accepting) != (
                final_state(dfa_b, word) in dfa_b.accepting
            )
            assert distinguishes == (end in designated)


@pytest.mark.parametrize("depth", DEPTHS)
def test_v2_pairs_agree_on_the_empty_word(depth):
    rng = random.Random(7900 + depth)
    for _ in range(10):
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert (dfa_a.start in dfa_a.accepting) == (dfa_b.start in dfa_b.accepting)


def test_v2_pairs_are_fully_reachable():
    rng = random.Random(8000)
    for depth in DEPTHS:
        draw = neq_depth_pair_v2(rng, N_STATES, depth, BINARY, ACCEPTING)
        assert draw.pair is not None
        for dfa in draw.pair:
            assert len(reachable(dfa)) == N_STATES


def test_an_impossible_accepting_count_is_rejected_not_raised():
    rng = random.Random(8001)
    assert neq_depth_pair_v2(rng, N_STATES, 3, BINARY, 0).reason
    assert neq_depth_pair_v2(rng, N_STATES, 3, BINARY, N_STATES).reason


# --------------------------------------------------------------------------- equivalent v2


def test_random_dfa_hits_the_exact_accepting_count():
    rng = random.Random(8100)
    for count in (0, 1, 7, 13, 14):
        dfa = random_complete_dfa_with_count(rng, N_STATES, BINARY, count)
        assert dfa is not None
        assert len(dfa.accepting) == count


def test_eq_iso_v2_pairs_are_equivalent_with_the_frozen_count():
    rng = random.Random(8200)
    for _ in range(40):
        draw = eq_iso_pair_v2(rng, N_STATES, BINARY, ACCEPTING)
        assert draw.pair is not None, draw.reason
        dfa_a, dfa_b = draw.pair
        assert equivalence_and_witness(dfa_a, dfa_b).equivalent
        assert len(dfa_a.accepting) == len(dfa_b.accepting) == ACCEPTING


def test_expand_to_with_count_lands_on_the_target():
    rng = random.Random(8300)
    for _ in range(40):
        core = random_complete_dfa_with_count(rng, rng.randrange(9, 13), BINARY, 5)
        assert core is not None
        expanded = expand_to_with_count(rng, core, N_STATES, ACCEPTING)
        if expanded is None:
            continue
        assert expanded.n_states == N_STATES
        assert len(expanded.accepting) == ACCEPTING
        assert equivalence_and_witness(core, expanded).equivalent


def test_expand_to_with_count_refuses_an_unreachable_target():
    """The count can only grow by one per clone, so some targets simply cannot be met."""
    rng = random.Random(8301)
    core = random_complete_dfa_with_count(rng, 12, BINARY, 2)
    assert core is not None
    # 2 accepting + at most 2 clones cannot reach 7.
    assert expand_to_with_count(rng, core, N_STATES, ACCEPTING) is None


def test_eq_expand_v2_pairs_are_equivalent_non_isomorphic_and_count_matched():
    rng = random.Random(8400)
    accepted = 0
    for _ in range(120):
        draw = eq_expand_pair_v2(
            rng, N_STATES, BINARY, core_states=(10, 12), accepting_count=ACCEPTING
        )
        if draw.pair is None:
            continue
        dfa_a, dfa_b = draw.pair
        assert equivalence_and_witness(dfa_a, dfa_b).equivalent
        assert len(dfa_a.accepting) == len(dfa_b.accepting) == ACCEPTING
        assert canonical_form(dfa_a) != canonical_form(dfa_b)
        assert 10 <= minimize(dfa_a).n_states <= 12
        accepted += 1
    assert accepted > 60, f"acceptance rate too low to be usable: {accepted}/120"


def test_cloning_still_preserves_the_language_under_the_count_schedule():
    """§7.4's property, exercised through the count-constrained path."""
    rng = random.Random(8500)
    for _ in range(60):
        core = random_complete_dfa_with_count(rng, 8, BINARY, 4)
        assert core is not None
        expanded = clone_state(rng, core, rng.randrange(core.n_states))
        if expanded is None:
            continue
        assert equivalence_and_witness(core, expanded).equivalent


# --------------------------------------------------------------------------- determinism


def test_v2_generation_is_reproducible():
    for depth in (1, 6, 12):
        first = neq_depth_pair_v2(random.Random(555), N_STATES, depth, BINARY, ACCEPTING)
        second = neq_depth_pair_v2(random.Random(555), N_STATES, depth, BINARY, ACCEPTING)
        assert first == second
    assert eq_iso_pair_v2(random.Random(556), N_STATES, BINARY, ACCEPTING) == eq_iso_pair_v2(
        random.Random(556), N_STATES, BINARY, ACCEPTING
    )
    assert eq_expand_pair_v2(
        random.Random(557), N_STATES, BINARY, (10, 12), ACCEPTING
    ) == eq_expand_pair_v2(random.Random(557), N_STATES, BINARY, (10, 12), ACCEPTING)


def test_construction_version_string_is_the_contract_value():
    assert CONSTRUCTION_VERSION == "acceptance_swap_v2"
