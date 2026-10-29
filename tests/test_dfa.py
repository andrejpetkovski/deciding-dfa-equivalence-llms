"""Tests for the canonical DFA object, validation, and canonical hashing (M1, Step 1)."""

from __future__ import annotations

import dataclasses
import random

import pytest

from src.dfa import (
    DFA,
    DFAValidationError,
    canonical_form,
    dfa_canonical_hash,
    from_record,
    pair_canonical_hash,
    to_record,
    validate,
)
from tests.helpers import (
    BINARY,
    even_zeros_dfa,
    random_permutation,
    random_reachable_dfa,
)

# --------------------------------------------------------------------------- immutability


def test_dfa_is_frozen():
    dfa = even_zeros_dfa()
    with pytest.raises(dataclasses.FrozenInstanceError):
        dfa.start = 1  # type: ignore[misc]


def test_derivations_do_not_mutate_the_original():
    dfa = even_zeros_dfa()
    before = to_record(dfa)
    dfa.toggled(0)
    dfa.with_accepting([1])
    dfa.relabel((1, 0))
    assert to_record(dfa) == before


# --------------------------------------------------------------------------- validation


def test_validate_accepts_a_well_formed_dfa():
    validate(even_zeros_dfa(), require_reachable=True)


def test_validate_rejects_a_missing_transition():
    """A state with fewer transitions than symbols violates totality (Step 2)."""
    dfa = DFA(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=frozenset({0}),
        delta=((1, 0), (0,)),  # state 1 has no transition on "1"
    )
    with pytest.raises(DFAValidationError, match=r"state 1 has 1 transitions"):
        validate(dfa)


def test_validate_rejects_a_missing_row():
    dfa = DFA(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=frozenset(),
        delta=((1, 0), (0, 1)),  # only two rows for three states
    )
    with pytest.raises(DFAValidationError, match=r"delta has 2 rows, expected 3"):
        validate(dfa)


def test_validate_rejects_an_out_of_range_destination():
    dfa = DFA(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=frozenset(),
        delta=((1, 5), (0, 1)),  # 5 is not a state
    )
    with pytest.raises(DFAValidationError, match=r"delta\(0, '1'\) = 5"):
        validate(dfa)


def test_validate_rejects_an_out_of_range_start():
    dfa = DFA(
        n_states=2, alphabet=BINARY, start=7, accepting=frozenset(), delta=((0, 0), (1, 1))
    )
    with pytest.raises(DFAValidationError, match=r"start state 7"):
        validate(dfa)


def test_validate_rejects_an_out_of_range_accepting_state():
    dfa = DFA(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=frozenset({9}),
        delta=((0, 0), (1, 1)),
    )
    with pytest.raises(DFAValidationError, match=r"accepting state 9"):
        validate(dfa)


def test_validate_rejects_a_duplicate_alphabet_symbol():
    dfa = DFA(
        n_states=1,
        alphabet=("0", "0"),
        start=0,
        accepting=frozenset(),
        delta=((0, 0),),
    )
    with pytest.raises(DFAValidationError, match=r"duplicate symbols"):
        validate(dfa)


def test_validate_rejects_a_boolean_destination():
    """``True`` is an int in Python; a transition table must not smuggle one in."""
    dfa = DFA(
        n_states=2,
        alphabet=BINARY,
        start=0,
        accepting=frozenset(),
        delta=((True, 0), (0, 1)),  # type: ignore[arg-type]
    )
    with pytest.raises(DFAValidationError, match=r"not an integer state id"):
        validate(dfa)


def test_validate_rejects_unreachable_states_only_when_required():
    """Unreachable states are legal in general and illegal for the main set (QA gate 5)."""
    dfa = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=[2],
        delta=[[0, 1], [1, 0], [2, 2]],  # state 2 is a disconnected sink
    )
    validate(dfa)  # permissive by default
    with pytest.raises(DFAValidationError, match=r"unreachable from start 0: \[2\]"):
        validate(dfa, require_reachable=True)


def test_build_validates_by_default():
    with pytest.raises(DFAValidationError):
        DFA.build(
            n_states=2, alphabet=BINARY, start=0, accepting=[], delta=[[0, 0], [9, 0]]
        )


# --------------------------------------------------------------------------- relabelling


def test_relabel_rejects_a_non_bijection():
    dfa = even_zeros_dfa()
    with pytest.raises(DFAValidationError, match=r"not a bijection"):
        dfa.relabel((0, 0))


def test_relabel_is_involutive_under_the_inverse_permutation():
    rng = random.Random(11)
    dfa = random_reachable_dfa(rng, n_states=7)
    perm = random_permutation(rng, 7)
    inverse = tuple(sorted(range(7), key=lambda i: perm[i]))
    assert dfa.relabel(perm).relabel(inverse) == dfa


def test_toggled_flips_exactly_one_state():
    dfa = even_zeros_dfa()
    toggled = dfa.toggled(1)
    assert toggled.accepting == frozenset({0, 1})
    assert toggled.toggled(1) == dfa
    assert toggled.delta == dfa.delta  # transitions untouched


# --------------------------------------------------------------------------- canonical form


def test_canonical_form_is_invariant_under_renaming():
    """The property that makes the hash a sound dedup key (Step 4)."""
    rng = random.Random(20260907)
    for _ in range(50):
        dfa = random_reachable_dfa(rng, n_states=rng.randrange(2, 9))
        renamed = dfa.relabel(random_permutation(rng, dfa.n_states))
        assert canonical_form(dfa) == canonical_form(renamed)
        assert dfa_canonical_hash(dfa) == dfa_canonical_hash(renamed)


def test_canonical_form_is_idempotent():
    rng = random.Random(5)
    dfa = random_reachable_dfa(rng, n_states=8)
    once = canonical_form(dfa)
    assert canonical_form(once) == once


def test_canonical_form_starts_at_state_zero():
    rng = random.Random(6)
    dfa = random_reachable_dfa(rng, n_states=8)
    assert canonical_form(dfa).start == 0


def test_canonical_form_requires_reachability():
    """BFS numbering leaves no canonical slot for an unreachable state, so it must refuse."""
    dfa = DFA.build(
        n_states=3, alphabet=BINARY, start=0, accepting=[], delta=[[0, 1], [1, 0], [2, 2]]
    )
    with pytest.raises(DFAValidationError, match=r"unreachable"):
        canonical_form(dfa)


def test_distinct_languages_hash_differently():
    dfa = even_zeros_dfa()
    assert dfa_canonical_hash(dfa) != dfa_canonical_hash(dfa.toggled(0))


# --------------------------------------------------------------------------- pair hashing


def test_pair_hash_ignores_argument_order():
    """(A, B) and (B, A) are one instance; admitting both would duplicate a unit."""
    rng = random.Random(77)
    a = random_reachable_dfa(rng, n_states=5)
    b = random_reachable_dfa(rng, n_states=5)
    assert pair_canonical_hash(a, b) == pair_canonical_hash(b, a)


def test_pair_hash_ignores_state_renaming():
    rng = random.Random(78)
    a = random_reachable_dfa(rng, n_states=6)
    b = random_reachable_dfa(rng, n_states=6)
    a2 = a.relabel(random_permutation(rng, 6))
    b2 = b.relabel(random_permutation(rng, 6))
    assert pair_canonical_hash(a, b) == pair_canonical_hash(a2, b2)


def test_pair_hash_separates_different_pairs():
    rng = random.Random(79)
    a = random_reachable_dfa(rng, n_states=5)
    b = random_reachable_dfa(rng, n_states=5)
    assert pair_canonical_hash(a, b) != pair_canonical_hash(a, b.toggled(0))


def test_hash_has_the_recorded_prefix():
    """Base-pair records store ``sha256:...`` (§9.1)."""
    digest = dfa_canonical_hash(even_zeros_dfa())
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


# --------------------------------------------------------------------------- records


def test_record_round_trip():
    rng = random.Random(31)
    dfa = random_reachable_dfa(rng, n_states=7)
    assert canonical_form(from_record(to_record(dfa))) == canonical_form(dfa)


def test_record_is_json_native():
    """Records must serialize with no custom encoder: sets and tuples do not survive JSON."""
    import json

    record = to_record(even_zeros_dfa())
    assert json.loads(json.dumps(record)) == record
