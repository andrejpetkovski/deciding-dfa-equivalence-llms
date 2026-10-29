"""Independent equivalence checks against third-party automata libraries.

QA gate 1 (§Step 6) requires that the hand-written product BFS in :mod:`src.oracle` and an
independent library agree on every generated pair. The hand-written oracle stays primary --
it is what produces the shortest witness and the depth ``d`` that the whole study is indexed
by, and neither library returns a witness. These converters exist to catch a bug in it, not
to replace it.

Both libraries decide language equivalence, not structural identity, which is exactly the
independent check that is wanted.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

from typing import Any

from src.dfa import DFA

__all__ = [
    "automata_lib_equivalent",
    "library_equivalent",
    "pyformlang_equivalent",
    "to_automata_lib",
    "to_pyformlang",
]


def to_pyformlang(dfa: DFA) -> Any:
    """Convert to a ``pyformlang`` ``DeterministicFiniteAutomaton``."""
    from pyformlang.finite_automaton import DeterministicFiniteAutomaton

    converted = DeterministicFiniteAutomaton()
    for state, row in enumerate(dfa.delta):
        for index, destination in enumerate(row):
            converted.add_transition(state, dfa.alphabet[index], destination)
    converted.add_start_state(dfa.start)
    for state in sorted(dfa.accepting):
        converted.add_final_state(state)
    return converted


def to_automata_lib(dfa: DFA) -> Any:
    """Convert to an ``automata-lib`` ``automata.fa.dfa.DFA``."""
    from automata.fa.dfa import DFA as AutomataLibDFA

    return AutomataLibDFA(
        states=set(range(dfa.n_states)),
        input_symbols=set(dfa.alphabet),
        transitions={
            state: {dfa.alphabet[i]: dest for i, dest in enumerate(row)}
            for state, row in enumerate(dfa.delta)
        },
        initial_state=dfa.start,
        final_states=set(dfa.accepting),
    )


def pyformlang_equivalent(dfa_a: DFA, dfa_b: DFA) -> bool:
    """Whether ``pyformlang`` considers the two automata language-equivalent.

    Decided by emptiness of the symmetric difference -- the protocol's own definition,
    ``L(A) == L(B)  iff  L(A) △ L(B) == ∅`` (§3.1) -- rather than by pyformlang's
    ``is_equivalent_to``.

    That method is not reliable here. On pyformlang 1.0.11 it reports two empty-language
    automata as inequivalent when one of them reaches emptiness through an *unreachable*
    accepting state, even though the same library's ``is_empty()`` and ``minimize()``
    both agree the languages are empty::

        A: 3 states, accepting {2}, but state 2 is unreachable  ->  L(A) = empty
        B: 2 states, no accepting states                        ->  L(B) = empty
        A.is_equivalent_to(B) == False          # wrong
        A.get_difference(B).is_empty() == True  # right

    That configuration is common in this project -- the generators produce automata whose
    accepting sets are sampled independently of reachability -- so the difference-based
    formulation is the one QA gate 1 must use.
    """
    converted_a = to_pyformlang(dfa_a)
    converted_b = to_pyformlang(dfa_b)
    return bool(
        converted_a.get_difference(converted_b).is_empty()
        and converted_b.get_difference(converted_a).is_empty()
    )


def automata_lib_equivalent(dfa_a: DFA, dfa_b: DFA) -> bool:
    """Whether ``automata-lib`` considers the two automata language-equivalent.

    ``automata-lib`` overloads ``==`` on DFAs to mean language equality, not structural
    identity.
    """
    return bool(to_automata_lib(dfa_a) == to_automata_lib(dfa_b))


def library_equivalent(dfa_a: DFA, dfa_b: DFA, library: str = "pyformlang") -> bool:
    """Dispatch to one of the library checkers by name.

    Args:
        dfa_a: The first automaton.
        dfa_b: The second automaton.
        library: ``"pyformlang"`` or ``"automata-lib"``. The frozen config names which one
            QA gate 1 uses (``qa.cross_check_library``).

    Raises:
        ValueError: for an unknown library name, so a typo in the frozen config fails
            loudly rather than silently skipping the gate.
    """
    if library == "pyformlang":
        return pyformlang_equivalent(dfa_a, dfa_b)
    if library in ("automata-lib", "automata_lib"):
        return automata_lib_equivalent(dfa_a, dfa_b)
    raise ValueError(
        f"unknown cross-check library {library!r}; expected 'pyformlang' or 'automata-lib'"
    )
