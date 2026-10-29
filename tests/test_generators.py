"""Tests for the three pair-family generators (M3, §5.2-5.5)."""

from __future__ import annotations

import random

import pytest

from src.dfa import DFA, canonical_form
from src.generators import (
    Draw,
    clone_state,
    eq_expand_pair,
    eq_iso_pair,
    expand_to,
    layered_target_depth_graph,
    neq_depth_pair,
    permute_independently,
    random_complete_dfa,
)
from src.minimize import minimize
from src.oracle import equivalence_and_witness
from src.simulate import access_distances, final_state, reachable
from tests.helpers import BINARY, all_words

N_STATES = 14
DEPTHS = list(range(1, 13))  # the range the four protocol strata span


def _as_dfa(graph, n_states: int = N_STATES) -> DFA:
    """Wrap a bare transition structure as a DFA so the simulation helpers apply."""
    return DFA.build(
        n_states=n_states,
        alphabet=BINARY,
        start=0,
        accepting=[],
        delta=[list(row) for row in graph.delta],
    )


def _layers_are_consistent(delta, layers) -> bool:
    """Every edge must lead at most one layer deeper -- the construction's invariant."""
    return all(
        layers[destination] <= layers[state] + 1
        for state, row in enumerate(delta)
        for destination in row
    )


# --------------------------------------------------------------------------- layered graph


@pytest.mark.parametrize("depth", DEPTHS)
def test_layered_graph_puts_the_target_at_exactly_the_requested_depth(depth):
    """The core guarantee of §5.4, checked across the whole protocol depth range."""
    rng = random.Random(4000 + depth)
    for _ in range(20):
        graph = layered_target_depth_graph(rng, N_STATES, depth, BINARY)
        assert graph is not None
        distances = access_distances(_as_dfa(graph))
        assert distances[graph.target] == depth
        assert all(d is not None for d in distances), "every state must be reachable"


@pytest.mark.parametrize("depth", DEPTHS)
def test_layered_graph_layers_equal_true_distances(depth):
    rng = random.Random(4100 + depth)
    graph = layered_target_depth_graph(rng, N_STATES, depth, BINARY)
    assert graph is not None
    assert tuple(access_distances(_as_dfa(graph))) == graph.layers
    assert _layers_are_consistent(graph.delta, graph.layers)


@pytest.mark.parametrize("depth", DEPTHS)
def test_target_word_spells_a_shortest_path_to_the_target(depth):
    rng = random.Random(4200 + depth)
    graph = layered_target_depth_graph(rng, N_STATES, depth, BINARY)
    assert graph is not None
    assert len(graph.target_word) == depth
    assert final_state(_as_dfa(graph), graph.target_word) == graph.target


def test_layered_graph_rejects_a_depth_that_does_not_fit():
    rng = random.Random(4300)
    with pytest.raises(ValueError, match=r"cannot hold a depth-9 backbone"):
        layered_target_depth_graph(rng, 5, 9, BINARY)


def test_layered_graph_rejects_depth_zero():
    """d = 0 is the epsilon case and is excluded from the main set (§Step 3)."""
    rng = random.Random(4301)
    with pytest.raises(ValueError, match=r"depth must be at least 1"):
        layered_target_depth_graph(rng, 14, 0, BINARY)


# --------------------------------------------------------------------------- NEQ_DEPTH


@pytest.mark.parametrize("depth", DEPTHS)
def test_neq_depth_pair_has_exactly_the_requested_oracle_depth(depth):
    """The property the whole H1 design rests on: the requested d is the oracle's d."""
    rng = random.Random(4400 + depth)
    for _ in range(15):
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None, draw.reason
        result = equivalence_and_witness(*draw.pair)
        assert not result.equivalent
        assert result.shortest_depth == depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_neq_depth_pair_agrees_on_the_empty_word(depth):
    """QA gate 3, "no epsilon leak": the start states must agree on acceptance."""
    rng = random.Random(4500 + depth)
    for _ in range(15):
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert (dfa_a.start in dfa_a.accepting) == (dfa_b.start in dfa_b.accepting)


def test_neq_depth_pair_differs_at_exactly_one_state():
    rng = random.Random(4600)
    for depth in DEPTHS:
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert dfa_a.delta == dfa_b.delta
        assert len(dfa_a.accepting ^ dfa_b.accepting) == 1


def test_distinguishing_words_are_exactly_the_words_reaching_the_toggled_state():
    """Why the depth is exact: A and B differ only at one state's acceptance, so a word
    separates them precisely when it ends there."""
    rng = random.Random(4700)
    for depth in (1, 2, 3, 4):
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        (toggled,) = dfa_a.accepting ^ dfa_b.accepting
        for word in all_words(BINARY, 7):
            end = final_state(dfa_a, word)
            distinguishes = (end in dfa_a.accepting) != (
                final_state(dfa_b, word) in dfa_b.accepting
            )
            assert distinguishes == (end == toggled)


def test_neq_depth_pairs_are_fully_reachable():
    rng = random.Random(4800)
    for depth in DEPTHS:
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        for dfa in draw.pair:
            assert len(reachable(dfa)) == N_STATES


# --------------------------------------------------------------------------- EQ_ISO


def test_eq_iso_pairs_are_equivalent():
    rng = random.Random(4900)
    for _ in range(60):
        draw = eq_iso_pair(rng, N_STATES, BINARY)
        assert draw.pair is not None, draw.reason
        assert equivalence_and_witness(*draw.pair).equivalent


def test_eq_iso_pairs_are_isomorphic_by_construction():
    """Deliberate: EQ_ISO is the easy equivalent control, EQ_EXPAND the hard one (§5.2)."""
    rng = random.Random(4901)
    for _ in range(30):
        draw = eq_iso_pair(rng, N_STATES, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = draw.pair
        assert canonical_form(dfa_a) == canonical_form(dfa_b)


# --------------------------------------------------------------------------- cloning


def test_clone_preserves_the_language():
    """§7.4: adding a correct behavioral clone preserves language equivalence."""
    rng = random.Random(5000)
    for _ in range(120):
        dfa = random_complete_dfa(rng, rng.randrange(3, 9), BINARY)
        assert dfa is not None
        expanded = clone_state(rng, dfa, rng.randrange(dfa.n_states))
        if expanded is None:
            continue
        assert expanded.n_states == dfa.n_states + 1
        assert equivalence_and_witness(dfa, expanded).equivalent


def test_clone_keeps_every_state_reachable():
    rng = random.Random(5001)
    for _ in range(120):
        dfa = random_complete_dfa(rng, rng.randrange(3, 9), BINARY)
        assert dfa is not None
        expanded = clone_state(rng, dfa, rng.randrange(dfa.n_states))
        if expanded is None:
            continue
        assert len(reachable(expanded)) == expanded.n_states


def test_clone_copies_acceptance_and_transitions():
    rng = random.Random(5002)
    dfa = random_complete_dfa(rng, 6, BINARY)
    assert dfa is not None
    for state in range(dfa.n_states):
        expanded = clone_state(rng, dfa, state)
        if expanded is None:
            continue
        clone = expanded.n_states - 1
        assert expanded.delta[clone] == dfa.delta[state]
        assert (clone in expanded.accepting) == (state in dfa.accepting)


def test_expand_to_reaches_the_target_size_and_keeps_the_language():
    rng = random.Random(5003)
    for _ in range(40):
        core = random_complete_dfa(rng, rng.randrange(4, 9), BINARY)
        assert core is not None
        expanded = expand_to(rng, core, N_STATES)
        assert expanded is not None
        assert expanded.n_states == N_STATES
        assert equivalence_and_witness(core, expanded).equivalent


def test_expansion_does_not_change_the_minimal_machine():
    """Cloning adds redundancy, never expressive power."""
    rng = random.Random(5004)
    for _ in range(30):
        core = random_complete_dfa(rng, rng.randrange(4, 8), BINARY)
        assert core is not None
        expanded = expand_to(rng, core, N_STATES)
        assert expanded is not None
        assert canonical_form(minimize(core)) == canonical_form(minimize(expanded))


# --------------------------------------------------------------------------- EQ_EXPAND


def test_eq_expand_pairs_are_equivalent_and_the_right_size():
    rng = random.Random(5100)
    accepted = 0
    for _ in range(120):
        draw = eq_expand_pair(rng, N_STATES, BINARY, core_states=(10, 12))
        if draw.pair is None:
            continue
        dfa_a, dfa_b = draw.pair
        assert dfa_a.n_states == N_STATES and dfa_b.n_states == N_STATES
        assert equivalence_and_witness(dfa_a, dfa_b).equivalent
        accepted += 1
    assert accepted > 60, f"acceptance rate too low to be usable: {accepted}/120"


def test_eq_expand_pairs_are_not_isomorphic_when_required():
    """The point of the family: equivalent without being a renaming (§5.2)."""
    rng = random.Random(5101)
    checked = 0
    for _ in range(120):
        draw = eq_expand_pair(
            rng, N_STATES, BINARY, core_states=(10, 12), require_non_isomorphic=True
        )
        if draw.pair is None:
            continue
        assert canonical_form(draw.pair[0]) != canonical_form(draw.pair[1])
        checked += 1
    assert checked > 60


def test_eq_expand_cores_are_genuinely_minimal():
    """Both sides must minimize back into the configured core range."""
    rng = random.Random(5102)
    checked = 0
    for _ in range(80):
        draw = eq_expand_pair(rng, N_STATES, BINARY, core_states=(10, 12))
        if draw.pair is None:
            continue
        for dfa in draw.pair:
            assert 10 <= minimize(dfa).n_states <= 12
        checked += 1
    assert checked > 40


def test_eq_expand_rejections_carry_a_specific_reason():
    """§Step 4 wants counts by reason, not one undifferentiated failure tally."""
    rng = random.Random(5103)
    reasons = {
        draw.reason
        for _ in range(200)
        if (draw := eq_expand_pair(rng, N_STATES, BINARY, core_states=(10, 12))).reason
    }
    assert reasons, "expected at least one rejection in 200 draws"
    assert reasons <= {
        "core_sample_failed",
        "core_not_minimal",
        "core_larger_than_target",
        "expansion_stalled",
        "isomorphic_expansions",
    }


# --------------------------------------------------------------------------- presentation


def test_independent_permutation_preserves_the_oracle_verdict():
    """§5.3 step 6 is semantics-preserving: renaming cannot change a language."""
    rng = random.Random(5200)
    for depth in DEPTHS:
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        before = equivalence_and_witness(*draw.pair)
        after = equivalence_and_witness(*permute_independently(rng, *draw.pair))
        assert before.equivalent == after.equivalent
        assert before.shortest_depth == after.shortest_depth


def test_independent_permutation_usually_misaligns_the_toggled_state():
    """The intent of §5.3 step 6: the differing state should not share an index."""
    rng = random.Random(5201)
    misaligned = 0
    trials = 60
    for _ in range(trials):
        draw = neq_depth_pair(rng, N_STATES, 5, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = permute_independently(rng, *draw.pair)
        if dfa_a.delta != dfa_b.delta:
            misaligned += 1
    assert misaligned > trials * 0.8


# --------------------------------------------------------------------------- determinism


def test_same_seed_reproduces_the_same_pair():
    """Research rule 7, and §12.5's "regenerate every pair from its seed"."""
    for depth in (1, 5, 12):
        first = neq_depth_pair(random.Random(999), N_STATES, depth, BINARY)
        second = neq_depth_pair(random.Random(999), N_STATES, depth, BINARY)
        assert first == second

    assert eq_iso_pair(random.Random(31), N_STATES, BINARY) == eq_iso_pair(
        random.Random(31), N_STATES, BINARY
    )
    assert eq_expand_pair(
        random.Random(32), N_STATES, BINARY, core_states=(10, 12)
    ) == eq_expand_pair(random.Random(32), N_STATES, BINARY, core_states=(10, 12))


def test_draw_helpers():
    rng = random.Random(1)
    draw = eq_iso_pair(rng, 6, BINARY)
    assert draw.accepted and draw.reason is None
    rejected = Draw.rejected("nope")
    assert not rejected.accepted and rejected.pair is None
