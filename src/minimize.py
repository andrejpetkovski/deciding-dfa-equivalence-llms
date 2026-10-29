"""DFA minimization by Moore partition refinement, and unreachable-state pruning.

Needed by the EQ_EXPAND family, which starts from a *minimized* core DFA of 10-12 states
and expands it twice (§5.5), and by the graph statistics recorded for each pair, where the
minimal state count measures how much redundancy a machine carries.

Written by hand rather than delegated to ``pyformlang`` or ``automata-lib`` on purpose. This
code sits on the *generation* path, so a library upgrade that changed its output would change
the generated dataset -- and the dataset must be reproducible from a seed and a commit alone
(§12.5). :mod:`src.crosscheck` verifies this implementation against both libraries in the
tests, which is the right place for a third-party dependency.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

from collections import deque

from src.dfa import DFA

__all__ = ["minimal_state_count", "minimize", "prune_unreachable"]


def prune_unreachable(dfa: DFA) -> DFA:
    """Drop states unreachable from the start, renumbering survivors in BFS order.

    The language is unchanged: an unreachable state contributes to no accepted word, even
    when it is accepting.
    """
    order: list[int] = [dfa.start]
    position: dict[int, int] = {dfa.start: 0}
    queue = deque([dfa.start])
    while queue:
        state = queue.popleft()
        for destination in dfa.delta[state]:
            if destination not in position:
                position[destination] = len(order)
                order.append(destination)
                queue.append(destination)

    return DFA(
        n_states=len(order),
        alphabet=dfa.alphabet,
        start=0,
        accepting=frozenset(position[q] for q in dfa.accepting if q in position),
        delta=tuple(
            tuple(position[destination] for destination in dfa.delta[state])
            for state in order
        ),
    )


def minimize(dfa: DFA) -> DFA:
    """Return the unique minimal DFA recognizing ``L(dfa)``.

    Moore's algorithm: prune unreachable states, split on acceptance, then repeatedly
    refine blocks by the tuple of blocks their successors fall into, until the partition
    stops changing. Merging two states is sound exactly when no word separates them, which
    is what refinement to a fixed point establishes.

    Block ids are assigned in ascending order of each block's smallest member, so the
    output is a deterministic function of the input -- required, since this runs on the
    generation path (research rule 7).
    """
    pruned = prune_unreachable(dfa)
    n_symbols = len(pruned.alphabet)

    # Initial partition: accepting vs non-accepting. A machine with no accepting states,
    # or with all states accepting, starts as a single block and refines to one state.
    block = tuple(1 if q in pruned.accepting else 0 for q in range(pruned.n_states))

    while True:
        signatures: dict[tuple[int, ...], int] = {}
        new_block = [0] * pruned.n_states
        # Iterate states in ascending order so block ids follow smallest-member order.
        for state in range(pruned.n_states):
            signature = (block[state],) + tuple(
                block[pruned.delta[state][symbol]] for symbol in range(n_symbols)
            )
            if signature not in signatures:
                signatures[signature] = len(signatures)
            new_block[state] = signatures[signature]
        if tuple(new_block) == block:
            break
        block = tuple(new_block)

    n_blocks = len(set(block))
    # One representative per block; the smallest member, matching the id ordering.
    representative: dict[int, int] = {}
    for state in range(pruned.n_states):
        representative.setdefault(block[state], state)

    return DFA(
        n_states=n_blocks,
        alphabet=pruned.alphabet,
        start=block[pruned.start],
        accepting=frozenset(block[q] for q in pruned.accepting),
        delta=tuple(
            tuple(
                block[pruned.delta[representative[b]][symbol]]
                for symbol in range(n_symbols)
            )
            for b in range(n_blocks)
        ),
    )


def minimal_state_count(dfa: DFA) -> int:
    """Number of states in the minimal DFA for ``L(dfa)``.

    Recorded per pair as a graph statistic: it separates a machine that genuinely needs 14
    states from one padded with redundant clones, which is exactly the difference between
    the EQ_ISO and EQ_EXPAND families (§13, generator artifact).
    """
    return minimize(dfa).n_states
