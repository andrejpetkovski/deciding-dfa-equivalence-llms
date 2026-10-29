"""Tests for the R0 and R1 surface renderings (M4, §5.6, decision D1)."""

from __future__ import annotations

import random

import pytest

from src.dfa import DFA, DFAValidationError, canonical_form
from src.generators import eq_iso_pair, neq_depth_pair, permute_independently
from src.oracle import equivalence_and_witness
from src.rendering import (
    ParsedRendering,
    derive_rng,
    nonce_labels,
    parse_rendering,
    permuted_sequential_labels,
    render,
    render_R0,
    render_R1,
    sequential_labels,
)
from tests.helpers import BINARY, even_zeros_dfa, random_reachable_dfa

N_STATES = 14


def _body(text: str) -> str:
    """Everything after the ``DFA X`` header line."""
    return text.split("\n", 1)[1]


def _table(text: str) -> str:
    return text.split("transitions:", 1)[1]


# --------------------------------------------------------------------------- labels


def test_sequential_labels_are_zero_padded_and_ordered():
    assert sequential_labels(14)[:3] == ("q00", "q01", "q02")
    assert sequential_labels(14)[-1] == "q13"


def test_nonce_labels_are_length_matched_to_sequential_ones():
    """QA gate 8's ±10% token tolerance is only attainable if the labels are the same size."""
    rng = random.Random(1)
    sequential = sequential_labels(N_STATES)
    nonces = nonce_labels(rng, N_STATES)
    assert {len(label) for label in nonces} == {len(sequential[0])}


def test_nonce_labels_are_distinct_and_not_alphabet_symbols():
    rng = random.Random(2)
    for n in (2, 14, 40):
        labels = nonce_labels(rng, n)
        assert len(set(labels)) == n
        assert not set(labels) & set(BINARY)


def test_nonce_labels_avoid_the_blocklist():
    """These end up in a published artifact, so the filter has to actually apply."""
    from src.rendering import _BLOCKLIST

    rng = random.Random(3)
    seen = {label for _ in range(300) for label in nonce_labels(rng, 14)}
    assert not seen & _BLOCKLIST


def test_derive_rng_is_stable_across_processes():
    """Seeds via sha256, not Python's salted hash(), or renderings would not reproduce."""
    assert derive_rng(123, "R1", "A").random() == derive_rng(123, "R1", "A").random()
    assert derive_rng(123, "R1", "A").random() != derive_rng(123, "R1", "B").random()


# --------------------------------------------------------------------------- R0


def test_r0_labels_follow_stored_order_not_bfs_order():
    """Decision D1: the whole point is that R0 does *not* re-canonicalize.

    This DFA's stored order differs from its BFS order, so a BFS-based renderer would emit
    different labels than a stored-order one.
    """
    dfa = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=2,  # BFS order would be 2, then its successors; stored order is 0, 1, 2
        accepting=[0],
        delta=[[1, 2], [2, 0], [0, 1]],
    )
    rendered = render_R0(dfa, "A")
    assert rendered.labels == ("q00", "q01", "q02")
    assert "initial: q02" in rendered.text  # start keeps its stored index, not q00


def test_r0_puts_the_initial_state_first_then_lexical_order():
    """§5.6's row order, kept verbatim under decision D1."""
    dfa = DFA.build(
        n_states=4,
        alphabet=BINARY,
        start=2,
        accepting=[1],
        delta=[[1, 2], [2, 3], [3, 0], [0, 1]],
    )
    rows = [
        line.split(",")[0]
        for line in _table(render_R0(dfa, "A").text).strip().splitlines()
    ]
    assert rows[:2] == ["q02", "q02"]  # initial state's rows first
    assert rows[2:] == ["q00", "q00", "q01", "q01", "q03", "q03"]


def test_r0_emits_one_row_per_state_and_symbol():
    rng = random.Random(10)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    rows = _table(render_R0(dfa, "A").text).strip().splitlines()
    assert len(rows) == N_STATES * len(BINARY)


def test_r0_is_deterministic():
    rng = random.Random(11)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    assert render_R0(dfa, "A").text == render_R0(dfa, "A").text


def test_header_names_the_machine():
    dfa = even_zeros_dfa()
    assert render_R0(dfa, "A").text.startswith("DFA A\n")
    assert render_R0(dfa, "B").text.startswith("DFA B\n")


def test_empty_accepting_set_is_rendered_explicitly():
    """An empty list would produce a dangling 'accepting:' line the parser could misread."""
    dfa = DFA.build(
        n_states=2, alphabet=BINARY, start=0, accepting=[], delta=[[1, 1], [0, 0]]
    )
    text = render_R0(dfa, "A").text
    assert "accepting: (none)" in text
    assert parse_rendering(text).dfa.accepting == frozenset()


# --------------------------------------------------------------------------- R1


def test_r1_reuses_r0s_vocabulary_as_a_derangement_and_shuffles_rows():
    """D5.4: R1 carries the same label strings as R0, re-assigned and reordered.

    The identical multiset is what makes the token balance hold by construction; the
    derangement is what guarantees no state keeps its R0 label, which the old nonce scheme
    gave for free.
    """
    rng = random.Random(20)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    r0 = render_R0(dfa, "A")
    r1 = render_R1(dfa, "A", derive_rng(1, "R1", "A"))

    assert sorted(r1.labels) == sorted(r0.labels)
    assert all(a != b for a, b in zip(r0.labels, r1.labels, strict=True))
    assert len(r1.text) == len(r0.text)
    rows = [line.split(",")[0] for line in _table(r1.text).strip().splitlines()]
    # A state's two rows almost never remain adjacent under a uniform shuffle.
    adjacent = sum(1 for a, b in zip(rows, rows[1:], strict=False) if a == b)
    assert adjacent < N_STATES // 2


def test_r1_has_the_same_shape_as_r0():
    """Only labels and row order change: same line count, same header keys (§5.6)."""
    rng = random.Random(21)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    r0 = render_R0(dfa, "A").text.splitlines()
    r1 = render_R1(dfa, "A", derive_rng(2, "R1", "A")).text.splitlines()
    assert len(r0) == len(r1)
    assert [line.split(":")[0] for line in r0[:4]] == [line.split(":")[0] for line in r1[:4]]


def test_r1_length_matches_r0():
    """Character-for-character equal, which is the strongest form of the token-balance goal."""
    rng = random.Random(22)
    for _ in range(20):
        dfa = random_reachable_dfa(rng, n_states=N_STATES)
        r0 = render_R0(dfa, "A")
        r1 = render_R1(dfa, "A", derive_rng(rng.random(), "R1", "A"))
        assert len(r0.text) == len(r1.text)


def test_r1_is_deterministic_for_a_given_seed():
    rng_seed = 12345
    dfa = random_reachable_dfa(random.Random(23), n_states=N_STATES)
    first = render_R1(dfa, "A", derive_rng(rng_seed, "R1", "A"))
    second = render_R1(dfa, "A", derive_rng(rng_seed, "R1", "A"))
    assert first.text == second.text
    assert first.labels == second.labels


# --------------------------------------------------------------------------- round trip


def test_round_trip_recovers_the_automaton():
    rng = random.Random(30)
    for _ in range(40):
        dfa = random_reachable_dfa(rng, n_states=rng.randrange(3, 12))
        for condition in ("R0", "R1"):
            rendered = render(dfa, "A", condition, derive_rng(rng.random(), condition, "A"))
            recovered = parse_rendering(rendered.text).dfa
            assert canonical_form(recovered) == canonical_form(dfa)


def test_both_conditions_give_the_same_oracle_verdict():
    """QA gate 6: R0 and R1 must map back to the same pair and the same oracle result."""
    rng = random.Random(31)
    for depth in (1, 4, 8, 12):
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = permute_independently(rng, *draw.pair)
        reference = equivalence_and_witness(dfa_a, dfa_b)
        for condition in ("R0", "R1"):
            parsed = [
                parse_rendering(
                    render(dfa, name, condition, derive_rng(depth, condition, name)).text
                ).dfa
                for name, dfa in (("A", dfa_a), ("B", dfa_b))
            ]
            result = equivalence_and_witness(*parsed)
            assert result.equivalent == reference.equivalent
            assert result.shortest_depth == reference.shortest_depth


def test_label_mapping_inverts_cleanly():
    """M8 grades model traces by mapping labels back through this."""
    rng = random.Random(32)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    for condition in ("R0", "R1"):
        rendered = render(dfa, "A", condition, derive_rng(1, condition, "A"))
        mapping = rendered.label_to_state
        assert len(mapping) == dfa.n_states
        assert all(rendered.labels[state] == label for label, state in mapping.items())


def test_text_hash_is_stable_and_prefixed():
    rendered = render_R0(even_zeros_dfa(), "A")
    assert rendered.text_sha256.startswith("sha256:")
    assert rendered.text_sha256 == render_R0(even_zeros_dfa(), "A").text_sha256


# --------------------------------------------------------------------------- decision D1


def test_neq_depth_pairs_no_longer_share_a_transition_table():
    """The failure decision D1 was taken to avoid (protocol §5.0)."""
    rng = random.Random(40)
    for depth in range(1, 13):
        draw = neq_depth_pair(rng, N_STATES, depth, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = permute_independently(rng, *draw.pair)
        text_a = render_R0(dfa_a, "A").text
        text_b = render_R0(dfa_b, "B").text
        assert _table(text_a) != _table(text_b)


def test_eq_iso_pairs_no_longer_render_identically():
    rng = random.Random(41)
    for _ in range(20):
        draw = eq_iso_pair(rng, N_STATES, BINARY)
        assert draw.pair is not None
        dfa_a, dfa_b = permute_independently(rng, *draw.pair)
        assert _body(render_R0(dfa_a, "A").text) != _body(render_R0(dfa_b, "B").text)


def test_renderings_carry_no_experiment_metadata():
    """Research rule 8: the model sees the machine and nothing else."""
    rng = random.Random(42)
    draw = neq_depth_pair(rng, N_STATES, 7, BINARY)
    assert draw.pair is not None
    for condition in ("R0", "R1"):
        for name, dfa in zip(("A", "B"), draw.pair, strict=True):
            text = render(dfa, name, condition, derive_rng(1, condition, name)).text.lower()
            for forbidden in ("pair_id", "oracle", "family", "seed", "depth", "witness"):
                assert forbidden not in text


# --------------------------------------------------------------------------- dispatch


def test_render_rejects_an_unknown_condition():
    with pytest.raises(ValueError, match=r"unknown representation condition"):
        render(even_zeros_dfa(), "A", "R7")


def test_r1_requires_an_rng():
    with pytest.raises(ValueError, match=r"R1 needs an rng"):
        render(even_zeros_dfa(), "A", "R1")


# --------------------------------------------------------------------------- parser errors


def test_parser_rejects_a_missing_header_line():
    text = "DFA A\nalphabet: 0, 1\ninitial: q00\ntransitions:\nq00, 0 -> q00\nq00, 1 -> q00"
    with pytest.raises(DFAValidationError, match=r"missing the 'accepting' line"):
        parse_rendering(text)


def test_parser_rejects_a_malformed_transition():
    text = (
        "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\n"
        "transitions:\nq00 0 q00\n"
    )
    with pytest.raises(DFAValidationError, match=r"malformed transition line"):
        parse_rendering(text)


def test_parser_rejects_a_duplicate_transition():
    text = (
        "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\n"
        "transitions:\nq00, 0 -> q00\nq00, 0 -> q00\nq00, 1 -> q00\n"
    )
    with pytest.raises(DFAValidationError, match=r"duplicate transition"):
        parse_rendering(text)


def test_parser_rejects_an_incomplete_table():
    text = (
        "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\n"
        "transitions:\nq00, 0 -> q00\n"
    )
    with pytest.raises(DFAValidationError, match=r"no transition for"):
        parse_rendering(text)


def test_parser_rejects_an_unknown_symbol():
    text = (
        "DFA A\nalphabet: 0, 1\ninitial: q00\naccepting: (none)\n"
        "transitions:\nq00, 2 -> q00\nq00, 0 -> q00\nq00, 1 -> q00\n"
    )
    with pytest.raises(DFAValidationError, match=r"not in alphabet"):
        parse_rendering(text)


def test_parser_rejects_an_initial_state_that_never_appears():
    text = (
        "DFA A\nalphabet: 0, 1\ninitial: q99\naccepting: (none)\n"
        "transitions:\nq00, 0 -> q00\nq00, 1 -> q00\n"
    )
    with pytest.raises(DFAValidationError, match=r"never appears in a transition"):
        parse_rendering(text)


def test_parser_keeps_the_machine_name():
    parsed = parse_rendering(render_R0(even_zeros_dfa(), "B").text)
    assert isinstance(parsed, ParsedRendering)
    assert parsed.name == "B"


def test_the_legacy_nonce_scheme_is_still_reachable_for_reproduction():
    """D2 keeps the v1 pilot byte-reproducible, so the old scheme must survive in code."""
    rng = random.Random(20)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    r0 = render_R0(dfa, "A")
    legacy = render_R1(dfa, "A", derive_rng(1, "R1", "A"), "nonce_cvc")

    assert set(legacy.labels).isdisjoint(r0.labels)


def test_an_unknown_label_scheme_raises_rather_than_defaulting():
    """A config typo must not silently change the manipulation H3 measures."""
    rng = random.Random(20)
    dfa = random_reachable_dfa(rng, n_states=N_STATES)
    with pytest.raises(ValueError, match="unknown R1 label scheme"):
        render_R1(dfa, "A", derive_rng(1, "R1", "A"), "nonce_cvd")


def test_a_derangement_needs_at_least_two_states():
    with pytest.raises(ValueError, match="no derangement"):
        permuted_sequential_labels(random.Random(0), 1)


def test_the_derangement_is_reproducible_from_its_seed():
    a = permuted_sequential_labels(random.Random(7), 14)
    b = permuted_sequential_labels(random.Random(7), 14)
    assert a == b
