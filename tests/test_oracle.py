"""Tests for the exact equivalence oracle (M2, Step 3, §3.2).

The oracle assigns every label in the dataset, so it gets the most scrutiny in the project:
the §7.4 property tests, an exhaustive minimality check, and agreement with two independent
third-party libraries.
"""

from __future__ import annotations

import random

import pytest

from src.crosscheck import automata_lib_equivalent, library_equivalent, pyformlang_equivalent
from src.dfa import DFA, DFAValidationError
from src.oracle import OracleResult, equivalence_and_witness, witness_upper_bound
from src.simulate import accepts, trace
from tests.helpers import (
    BINARY,
    all_words,
    ends_with_one_dfa,
    even_zeros_dfa,
    random_dfa,
    random_permutation,
    random_reachable_dfa,
)


def _random_word(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BINARY) for _ in range(length))


# --------------------------------------------------------------------------- basics


def test_a_dfa_is_equivalent_to_itself():
    result = equivalence_and_witness(even_zeros_dfa(), even_zeros_dfa())
    assert result.equivalent
    assert result.shortest_witness is None
    assert result.shortest_depth is None


def test_renamed_copy_is_equivalent():
    """§7.4: an automaton is equivalent to an independently renamed copy of itself."""
    rng = random.Random(2001)
    for _ in range(100):
        n = rng.randrange(2, 9)
        dfa = random_reachable_dfa(rng, n_states=n)
        renamed = dfa.relabel(random_permutation(rng, n))
        assert equivalence_and_witness(dfa, renamed).equivalent


def test_different_languages_are_not_equivalent():
    result = equivalence_and_witness(even_zeros_dfa(), ends_with_one_dfa())
    assert not result.equivalent
    assert result.shortest_witness is not None


def test_alphabet_mismatch_raises():
    """A mismatch always signals a construction bug, never a legitimate comparison."""
    other = DFA.build(
        n_states=1, alphabet=("a", "b"), start=0, accepting=[], delta=[[0, 0]]
    )
    with pytest.raises(DFAValidationError, match=r"alphabets differ"):
        equivalence_and_witness(even_zeros_dfa(), other)


def test_alphabet_tuple_order_does_not_change_the_verdict():
    """Symbols are tried in sorted order, so column ordering is an implementation detail."""
    a = even_zeros_dfa()
    # Same machine, alphabet stored in the opposite order, delta columns swapped to match.
    a_swapped = DFA.build(
        n_states=a.n_states,
        alphabet=("1", "0"),
        start=a.start,
        accepting=sorted(a.accepting),
        delta=[[row[1], row[0]] for row in a.delta],
    )
    assert equivalence_and_witness(a, a_swapped).equivalent
    b = ends_with_one_dfa()
    assert (
        equivalence_and_witness(a_swapped, b).shortest_witness
        == equivalence_and_witness(a, b).shortest_witness
    )


# --------------------------------------------------------------------------- epsilon case


def test_toggling_the_start_state_yields_the_empty_witness():
    """§7.4: toggling the initial state's acceptance makes epsilon the shortest witness."""
    rng = random.Random(2002)
    for _ in range(50):
        dfa = random_reachable_dfa(rng, n_states=rng.randrange(2, 9))
        result = equivalence_and_witness(dfa, dfa.toggled(dfa.start))
        assert not result.equivalent
        assert result.shortest_witness == ""
        assert result.shortest_depth == 0


def test_empty_witness_is_distinct_from_no_witness():
    """``""`` (d=0) and ``None`` (equivalent) must never be conflated."""
    dfa = even_zeros_dfa()
    epsilon = equivalence_and_witness(dfa, dfa.toggled(dfa.start))
    same = equivalence_and_witness(dfa, dfa)
    assert epsilon.shortest_witness == "" and epsilon.shortest_depth == 0
    assert same.shortest_witness is None and same.shortest_depth is None


def test_epsilon_traces_are_just_the_start_states():
    dfa = even_zeros_dfa()
    result = equivalence_and_witness(dfa, dfa.toggled(dfa.start))
    assert result.product_trace_A == (dfa.start,)
    assert result.product_trace_B == (dfa.start,)


# --------------------------------------------------------------------------- witness validity


def test_every_witness_is_accepted_by_exactly_one_dfa():
    """§7.4, and the definition of a distinguishing word (§3.2)."""
    rng = random.Random(2003)
    checked = 0
    for _ in range(400):
        a = random_dfa(rng, n_states=rng.randrange(2, 7))
        b = random_dfa(rng, n_states=rng.randrange(2, 7))
        result = equivalence_and_witness(a, b)
        if result.equivalent:
            continue
        witness = result.shortest_witness
        assert witness is not None
        assert accepts(a, witness) != accepts(b, witness)
        checked += 1
    assert checked > 50, "too few non-equivalent pairs sampled to be meaningful"


def test_equivalent_pairs_agree_on_every_short_word():
    """An equivalent verdict must survive exhaustive testing of short words."""
    rng = random.Random(2004)
    words = all_words(BINARY, 9)
    for _ in range(60):
        n = rng.randrange(2, 7)
        a = random_reachable_dfa(rng, n_states=n)
        b = a.relabel(random_permutation(rng, n))
        assert equivalence_and_witness(a, b).equivalent
        for word in words:
            assert accepts(a, word) == accepts(b, word)


def test_witness_respects_the_product_size_bound():
    """§3.2: a shortest witness is no longer than ``|Q_A|·|Q_B| - 1``."""
    rng = random.Random(2005)
    for _ in range(200):
        a = random_dfa(rng, n_states=rng.randrange(2, 8))
        b = random_dfa(rng, n_states=rng.randrange(2, 8))
        result = equivalence_and_witness(a, b)
        if not result.equivalent:
            assert result.shortest_depth is not None
            assert result.shortest_depth <= witness_upper_bound(a, b)


# --------------------------------------------------------------------------- minimality


def test_no_shorter_word_distinguishes_the_pair():
    """§7.4: verify shortestness by exhaustive enumeration of every strictly shorter word.

    Enumerating lengths ``0..d-1`` is what minimality actually asserts, and it is cheap
    because ``d`` is small for random small machines. Deep pairs are skipped rather than
    enumerated -- the depth-controlled generator will exercise those in M3.
    """
    rng = random.Random(2006)
    checked = 0
    for _ in range(300):
        a = random_dfa(rng, n_states=rng.randrange(2, 6))
        b = random_dfa(rng, n_states=rng.randrange(2, 6))
        result = equivalence_and_witness(a, b)
        if result.equivalent:
            continue
        depth = result.shortest_depth
        assert depth is not None
        if depth == 0 or depth > 12:
            continue
        for word in all_words(BINARY, depth - 1):
            assert accepts(a, word) == accepts(b, word), (
                f"word {word!r} of length {len(word)} distinguishes the pair, "
                f"but the oracle reported depth {depth}"
            )
        checked += 1
    assert checked > 30, "too few non-trivial witnesses sampled to be meaningful"


def test_witness_is_the_bfs_first_among_equally_short_ones():
    """With symbols tried in sorted order, the witness is the length-first, then
    lexicographically-first distinguishing word."""
    rng = random.Random(2007)
    for _ in range(120):
        a = random_dfa(rng, n_states=rng.randrange(2, 6))
        b = random_dfa(rng, n_states=rng.randrange(2, 6))
        result = equivalence_and_witness(a, b)
        if result.equivalent or result.shortest_depth is None:
            continue
        if result.shortest_depth > 10:
            continue
        expected = next(
            w for w in all_words(BINARY, result.shortest_depth)
            if accepts(a, w) != accepts(b, w)
        )
        assert result.shortest_witness == expected


# --------------------------------------------------------------------------- traces


def test_traces_are_the_deterministic_runs_of_the_witness():
    rng = random.Random(2008)
    for _ in range(200):
        a = random_dfa(rng, n_states=rng.randrange(2, 7))
        b = random_dfa(rng, n_states=rng.randrange(2, 7))
        result = equivalence_and_witness(a, b)
        if result.equivalent:
            continue
        witness = result.shortest_witness
        assert witness is not None
        assert result.product_trace_A == trace(a, witness)
        assert result.product_trace_B == trace(b, witness)


def test_trace_length_is_depth_plus_one():
    """The same invariant P1 states to the model (§7.2)."""
    rng = random.Random(2009)
    for _ in range(200):
        a = random_dfa(rng, n_states=rng.randrange(2, 7))
        b = random_dfa(rng, n_states=rng.randrange(2, 7))
        result = equivalence_and_witness(a, b)
        if result.equivalent:
            continue
        assert result.shortest_depth is not None
        assert result.product_trace_A is not None
        assert result.product_trace_B is not None
        assert len(result.product_trace_A) == result.shortest_depth + 1
        assert len(result.product_trace_B) == result.shortest_depth + 1


def test_trace_endpoints_disagree_on_acceptance():
    """The final product state is distinguishing, by construction."""
    rng = random.Random(2010)
    for _ in range(200):
        a = random_dfa(rng, n_states=rng.randrange(2, 7))
        b = random_dfa(rng, n_states=rng.randrange(2, 7))
        result = equivalence_and_witness(a, b)
        if result.equivalent:
            continue
        assert result.product_trace_A is not None
        assert result.product_trace_B is not None
        end_a = result.product_trace_A[-1] in a.accepting
        end_b = result.product_trace_B[-1] in b.accepting
        assert end_a != end_b


# --------------------------------------------------------------------------- determinism


def test_repeated_calls_return_identical_results():
    rng = random.Random(2011)
    for _ in range(50):
        a = random_dfa(rng, n_states=rng.randrange(2, 8))
        b = random_dfa(rng, n_states=rng.randrange(2, 8))
        assert equivalence_and_witness(a, b) == equivalence_and_witness(a, b)


def test_witness_is_stable_under_state_renaming():
    """Renaming states cannot change the language, so it cannot change the witness."""
    rng = random.Random(2012)
    for _ in range(100):
        n = rng.randrange(2, 8)
        a = random_reachable_dfa(rng, n_states=n)
        b = random_reachable_dfa(rng, n_states=n)
        plain = equivalence_and_witness(a, b)
        renamed = equivalence_and_witness(
            a.relabel(random_permutation(rng, n)), b.relabel(random_permutation(rng, n))
        )
        assert plain.equivalent == renamed.equivalent
        assert plain.shortest_witness == renamed.shortest_witness


def test_swapping_arguments_preserves_the_verdict_and_witness():
    """Distinguishing is symmetric: only the traces swap."""
    rng = random.Random(2013)
    for _ in range(100):
        a = random_dfa(rng, n_states=rng.randrange(2, 7))
        b = random_dfa(rng, n_states=rng.randrange(2, 7))
        forward = equivalence_and_witness(a, b)
        backward = equivalence_and_witness(b, a)
        assert forward.equivalent == backward.equivalent
        assert forward.shortest_witness == backward.shortest_witness
        assert forward.product_trace_A == backward.product_trace_B
        assert forward.product_trace_B == backward.product_trace_A


# --------------------------------------------------------------------------- result invariants


def test_result_rejects_an_equivalent_verdict_carrying_a_witness():
    with pytest.raises(ValueError, match=r"carries no witness"):
        OracleResult(
            equivalent=True,
            shortest_witness="01",
            shortest_depth=2,
            product_trace_A=(0, 0, 0),
            product_trace_B=(0, 0, 0),
        )


def test_result_rejects_a_depth_that_contradicts_the_witness():
    with pytest.raises(ValueError, match=r"does not match witness"):
        OracleResult(
            equivalent=False,
            shortest_witness="01",
            shortest_depth=5,
            product_trace_A=(0, 0, 0),
            product_trace_B=(0, 0, 0),
        )


def test_result_rejects_a_trace_of_the_wrong_length():
    with pytest.raises(ValueError, match=r"length depth\+1"):
        OracleResult(
            equivalent=False,
            shortest_witness="01",
            shortest_depth=2,
            product_trace_A=(0, 0),
            product_trace_B=(0, 0, 0),
        )


def test_result_rejects_a_non_equivalent_verdict_without_a_witness():
    with pytest.raises(ValueError, match=r"requires a witness"):
        OracleResult(
            equivalent=False,
            shortest_witness=None,
            shortest_depth=None,
            product_trace_A=None,
            product_trace_B=None,
        )


# --------------------------------------------------------------------------- cross-checks


def test_agrees_with_pyformlang_and_automata_lib_on_1000_pairs():
    """M2 exit criterion, and a dry run of QA gate 1 (§Step 6).

    Both libraries decide language equivalence independently of this implementation, so
    agreement across a thousand pairs is strong evidence the product BFS is right.
    """
    rng = random.Random(20260907)
    disagreements: list[str] = []
    equivalent_seen = 0
    for i in range(1000):
        n_a = rng.randrange(2, 8)
        n_b = rng.randrange(2, 8)
        a = random_dfa(rng, n_states=n_a)
        # Half the pairs are built to be equivalent, so the sample is not dominated by
        # trivially-distinguishable pairs.
        b = (
            a.relabel(random_permutation(rng, n_a))
            if i % 2 == 0
            else random_dfa(rng, n_states=n_b)
        )

        ours = equivalence_and_witness(a, b).equivalent
        theirs_pfl = pyformlang_equivalent(a, b)
        theirs_alib = automata_lib_equivalent(a, b)
        if not (ours == theirs_pfl == theirs_alib):
            disagreements.append(
                f"pair {i}: oracle={ours} pyformlang={theirs_pfl} automata-lib={theirs_alib}"
            )
        equivalent_seen += int(ours)

    assert not disagreements, "\n".join(disagreements[:10])
    assert 400 < equivalent_seen < 900, (
        f"unbalanced sample: {equivalent_seen}/1000 equivalent"
    )


def test_library_dispatch_rejects_an_unknown_name():
    """A typo in ``qa.cross_check_library`` must fail loudly, not skip the gate."""
    dfa = even_zeros_dfa()
    with pytest.raises(ValueError, match=r"unknown cross-check library"):
        library_equivalent(dfa, dfa, library="nope")


def test_library_dispatch_matches_the_direct_callers():
    rng = random.Random(2014)
    a = random_dfa(rng, n_states=5)
    b = random_dfa(rng, n_states=5)
    assert library_equivalent(a, b, "pyformlang") == pyformlang_equivalent(a, b)
    assert library_equivalent(a, b, "automata-lib") == automata_lib_equivalent(a, b)


def test_cross_check_handles_a_dfa_with_no_accepting_states():
    """The empty language is a legitimate edge case both converters must survive."""
    empty = DFA.build(
        n_states=2, alphabet=BINARY, start=0, accepting=[], delta=[[1, 1], [0, 0]]
    )
    other_empty = DFA.build(
        n_states=1, alphabet=BINARY, start=0, accepting=[], delta=[[0, 0]]
    )
    assert equivalence_and_witness(empty, other_empty).equivalent
    assert pyformlang_equivalent(empty, other_empty)
    assert automata_lib_equivalent(empty, other_empty)


def test_cross_check_handles_a_dfa_accepting_everything():
    universal_a = DFA.build(
        n_states=1, alphabet=BINARY, start=0, accepting=[0], delta=[[0, 0]]
    )
    universal_b = DFA.build(
        n_states=2, alphabet=BINARY, start=0, accepting=[0, 1], delta=[[1, 1], [0, 0]]
    )
    assert equivalence_and_witness(universal_a, universal_b).equivalent
    assert pyformlang_equivalent(universal_a, universal_b)
    assert automata_lib_equivalent(universal_a, universal_b)


def test_pyformlang_checker_survives_the_is_equivalent_to_defect():
    """Regression guard: do not "simplify" ``pyformlang_equivalent`` back to
    ``is_equivalent_to``.

    On pyformlang 1.0.11 that method reports these two empty-language automata as
    inequivalent, because one reaches emptiness through an unreachable accepting state.
    The generators sample accepting sets independently of reachability, so this shape is
    common here and would produce false alarms in QA gate 1.
    """
    from src.crosscheck import to_pyformlang

    unreachable_final = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=[2],  # unreachable, so the language is empty
        delta=[[0, 1], [0, 0], [0, 1]],
    )
    no_finals = DFA.build(
        n_states=2, alphabet=BINARY, start=0, accepting=[], delta=[[1, 1], [0, 0]]
    )

    # Ground truth: both languages are empty.
    assert equivalence_and_witness(unreachable_final, no_finals).equivalent
    assert not any(
        accepts(unreachable_final, w) or accepts(no_finals, w)
        for w in all_words(BINARY, 8)
    )

    # Our checker is right; the library method it deliberately avoids is not.
    assert pyformlang_equivalent(unreachable_final, no_finals)
    assert automata_lib_equivalent(unreachable_final, no_finals)
    assert not to_pyformlang(unreachable_final).is_equivalent_to(to_pyformlang(no_finals))


def test_cross_check_ignores_unreachable_states():
    """An unreachable accepting state changes no language; all three checkers must agree."""
    plain = DFA.build(
        n_states=2, alphabet=BINARY, start=0, accepting=[1], delta=[[0, 1], [0, 1]]
    )
    with_dead_state = DFA.build(
        n_states=3,
        alphabet=BINARY,
        start=0,
        accepting=[1, 2],  # state 2 is accepting but unreachable
        delta=[[0, 1], [0, 1], [2, 2]],
    )
    assert equivalence_and_witness(plain, with_dead_state).equivalent
    assert pyformlang_equivalent(plain, with_dead_state)
    assert automata_lib_equivalent(plain, with_dead_state)
