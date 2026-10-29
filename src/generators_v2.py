"""The v2 acceptance-swap generators (decision D2, ``docs/experiment_revision.md``).

Why this exists. The v1 construction toggles one state's acceptance, so every non-equivalent
pair satisfies ``abs(|F_A| - |F_B|) == 1`` while every EQ_ISO pair satisfies ``== 0``. Counting
two lists then decides equivalence without reading a single transition, at every depth. That
cue is fatal to the *secondary* balanced-accuracy metric and it makes the depth curve on
decisions uninformative, so v2 removes it.

Two changes do that:

- **Acceptance swap.** Two designated states ``t`` and ``u`` both sit at shortest access
  distance exactly ``d``. ``A`` accepts ``t`` and rejects ``u``; ``B`` swaps just those two.
  ``F_A △ F_B = {t, u}``, and the counts are equal *by construction*: each machine accepts the
  shared set plus exactly one designated state.
- **A fixed accepting count** (working value 7 of 14) in every family and bin, so EQ_ISO and
  EQ_EXPAND carry no count signature either.

The depth stays exact for the same reason as in v1. ``A`` and ``B`` share a transition table,
so a word distinguishes them exactly when it ends at ``t`` or at ``u``; the shortest such word
has length ``min(dist(t), dist(u)) = d``. The oracle still assigns every label (rule 1).

Equal counts remove *this* shortcut and do not establish the absence of others -- transition
graphs of a NEQ_DEPTH pair remain isomorphic to each other, for instance. ``scripts/baselines.py``
measures what the remaining ones are worth.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from src.dfa import DFA, canonical_form
from src.generators import Draw, clone_state
from src.minimize import minimize
from src.simulate import access_distances, reachable

__all__ = [
    "CONSTRUCTION_VERSION",
    "TwoTargetGraph",
    "eq_expand_pair_v2",
    "eq_iso_pair_v2",
    "expand_to_with_count",
    "layered_two_target_graph",
    "neq_depth_pair_v2",
    "random_complete_dfa_with_count",
]

#: Identifier every v2 config and record must carry. :mod:`src.generate` refuses to run a
#: config naming a version it cannot implement, so a v2 config can never silently fall back
#: to the v1 single-toggle path.
CONSTRUCTION_VERSION = "acceptance_swap_v2"


@dataclass(frozen=True, slots=True)
class TwoTargetGraph:
    """A transition structure with *two* designated states at the same BFS distance.

    Attributes:
        delta: The transition table, ``delta[state][symbol_index]``.
        target_t: First designated state, at shortest access distance ``depth``.
        target_u: Second designated state, also at exactly ``depth``.
        layers: Intended BFS layer per state, which the construction makes equal to the true
            shortest access distance.
        prefix_word: The backbone word of length ``depth - 1`` reaching the branch state.
        branch_symbols: ``(symbol_to_t, symbol_to_u)`` taken from the branch state.
    """

    delta: tuple[tuple[int, ...], ...]
    target_t: int
    target_u: int
    layers: tuple[int, ...]
    prefix_word: str
    branch_symbols: tuple[str, str]

    def access_word(self, target: int) -> str:
        """A shortest access word for one designated state."""
        symbol_t, symbol_u = self.branch_symbols
        return self.prefix_word + (symbol_t if target == self.target_t else symbol_u)


def layered_two_target_graph(
    rng: random.Random,
    n_states: int,
    depth: int,
    alphabet: tuple[str, ...],
) -> TwoTargetGraph | None:
    """Build a complete, fully reachable graph with two states at exactly ``depth``.

    A shared backbone runs from the start through layer ``depth - 1``, then *branches* on the
    two alphabet symbols into ``t`` and ``u``. That costs ``depth + 2`` states, which is why
    the deepest bin still fits: depth 12 needs 14, exactly the configured state count.

    The layer invariant is the same one v1 relies on -- every edge leads to a state at most
    one layer deeper -- and it yields the same two bounds. Nothing reaches a layer-``d`` state
    in fewer than ``d`` steps, and each state is given an incoming edge from the layer below,
    so both designated distances are exactly ``depth``.

    Returns:
        A :class:`TwoTargetGraph`, or ``None`` if the draw ran out of outgoing edges while
        keeping every state reachable -- a rejection the caller counts and retries.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if len(alphabet) < 2:
        raise ValueError("the acceptance-swap branch needs at least two symbols")
    if n_states < depth + 2:
        raise ValueError(
            f"{n_states} states cannot hold a depth-{depth} two-target backbone; "
            f"need {depth + 2}"
        )

    n_symbols = len(alphabet)
    symbol_index = {symbol: i for i, symbol in enumerate(alphabet)}
    target_t, target_u = depth, depth + 1

    # Backbone p_0 .. p_{depth-1}, then a two-way branch into t and u.
    prefix_word = "".join(rng.choice(alphabet) for _ in range(depth - 1))
    branch_pair = list(rng.sample(alphabet, 2))  # which symbol reaches which target
    symbol_to_t, symbol_to_u = branch_pair

    delta: list[list[int | None]] = [[None] * n_symbols for _ in range(n_states)]
    free: list[set[int]] = [set(range(n_symbols)) for _ in range(n_states)]
    for i, symbol in enumerate(prefix_word):
        column = symbol_index[symbol]
        delta[i][column] = i + 1
        free[i].discard(column)
    branch_state = depth - 1
    for symbol, target in ((symbol_to_t, target_t), (symbol_to_u, target_u)):
        column = symbol_index[symbol]
        delta[branch_state][column] = target
        free[branch_state].discard(column)

    layers: list[int] = [-1] * n_states
    states_by_layer: dict[int, list[int]] = defaultdict(list)
    for state in range(depth):
        layers[state] = state
        states_by_layer[state].append(state)
    for target in (target_t, target_u):
        layers[target] = depth
        states_by_layer[depth].append(target)
    max_layer = depth

    # Hang the remaining states off a state one layer up, choosing only from layers that
    # still have a spare outgoing edge so the draw cannot paint itself into a corner.
    for state in range(depth + 2, n_states):
        eligible = [
            layer
            for layer in range(1, max_layer + 2)
            if any(free[source] for source in states_by_layer[layer - 1])
        ]
        if not eligible:
            return None
        layer = rng.choice(eligible)
        sources = [source for source in states_by_layer[layer - 1] if free[source]]
        source = rng.choice(sources)
        column = rng.choice(sorted(free[source]))
        delta[source][column] = state
        free[source].discard(column)
        layers[state] = layer
        states_by_layer[layer].append(state)
        max_layer = max(max_layer, layer)

    # Fill the rest, respecting the layer invariant. State 0 is always a legal destination,
    # so no slot is ever unfillable.
    designated = {target_t, target_u}
    for state in range(n_states):
        for column in sorted(free[state]):
            highest = layers[state] + 1
            candidates = [
                destination
                for destination in range(n_states)
                if layers[destination] <= highest
                # No shortcut into either designated state from below layer depth-1.
                and (destination not in designated or layers[state] >= depth - 1)
            ]
            delta[state][column] = rng.choice(candidates)
        free[state].clear()

    built = tuple(tuple(row) for row in delta)  # type: ignore[arg-type]
    probe = DFA(
        n_states=n_states,
        alphabet=alphabet,
        start=0,
        accepting=frozenset(),
        delta=built,
    )
    distances = access_distances(probe)
    if any(distance is None for distance in distances):
        return None
    if distances[target_t] != depth or distances[target_u] != depth:
        return None

    return TwoTargetGraph(
        delta=built,
        target_t=target_t,
        target_u=target_u,
        layers=tuple(layers),
        prefix_word=prefix_word,
        branch_symbols=(symbol_to_t, symbol_to_u),
    )


def neq_depth_pair_v2(
    rng: random.Random,
    n_states: int,
    depth: int,
    alphabet: tuple[str, ...],
    accepting_count: int,
) -> Draw:
    """A non-equivalent pair at exactly ``depth`` with **equal** accepting counts.

    ``A`` and ``B`` share every transition and every acceptance decision except on the two
    designated states, whose statuses are swapped. Both machines therefore accept
    ``accepting_count`` states, and the accepting-count cue that decided v1 pairs carries no
    information at all.

    A word distinguishes the pair exactly when it ends at ``t`` or ``u``, both of which sit at
    distance ``depth``, so the shortest distinguishing word has length ``depth``. The oracle
    still confirms it before the pair is accepted (rule 1).
    """
    if not 1 <= accepting_count <= n_states - 1:
        return Draw.rejected("accepting_count_out_of_range")

    graph = layered_two_target_graph(rng, n_states, depth, alphabet)
    if graph is None:
        return Draw.rejected("graph_construction_failed")

    # The shared accepting set excludes both designated states; each machine adds one of
    # them, which is what keeps the two counts equal.
    others = [s for s in range(n_states) if s not in (graph.target_t, graph.target_u)]
    if accepting_count - 1 > len(others):
        return Draw.rejected("accepting_count_out_of_range")
    shared = frozenset(rng.sample(others, accepting_count - 1))

    dfa_a = DFA.build(
        n_states=n_states,
        alphabet=alphabet,
        start=0,
        accepting=shared | {graph.target_t},
        delta=graph.delta,
    )
    dfa_b = DFA.build(
        n_states=n_states,
        alphabet=alphabet,
        start=0,
        accepting=shared | {graph.target_u},
        delta=graph.delta,
    )

    if len(dfa_a.accepting) != len(dfa_b.accepting):  # pragma: no cover - structural
        return Draw.rejected("accepting_count_mismatch")
    if (dfa_a.start in dfa_a.accepting) != (dfa_b.start in dfa_b.accepting):
        # Impossible while depth >= 1 keeps the start off both designated states, but an
        # epsilon cue would silently collapse the stratum to depth 0, so never ship one.
        return Draw.rejected("epsilon_cue")
    return Draw.ok(dfa_a, dfa_b)


# --------------------------------------------------------------------------- equivalent pairs


def random_complete_dfa_with_count(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    accepting_count: int,
    max_attempts: int = 500,
) -> DFA | None:
    """A random complete, fully reachable DFA accepting exactly ``accepting_count`` states.

    The accepting set is drawn as a fixed-size sample rather than by independent coin flips,
    because the count is the very thing v2 holds constant across families.
    """
    if not 0 <= accepting_count <= n_states:
        return None
    for _ in range(max_attempts):
        delta = tuple(
            tuple(rng.randrange(n_states) for _ in alphabet) for _ in range(n_states)
        )
        accepting = frozenset(rng.sample(range(n_states), accepting_count))
        dfa = DFA.build(
            n_states=n_states,
            alphabet=alphabet,
            start=0,
            accepting=accepting,
            delta=delta,
        )
        if len(reachable(dfa)) == n_states:
            return dfa
    return None


def eq_iso_pair_v2(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    accepting_count: int,
) -> Draw:
    """An equivalent pair related by a state permutation, with the frozen accepting count.

    A permutation preserves the count, so only ``A`` needs constraining.
    """
    dfa_a = random_complete_dfa_with_count(rng, n_states, alphabet, accepting_count)
    if dfa_a is None:
        return Draw.rejected("core_sample_failed")
    permutation = list(range(n_states))
    rng.shuffle(permutation)
    return Draw.ok(dfa_a, dfa_a.relabel(tuple(permutation)))


def expand_to_with_count(
    rng: random.Random,
    dfa: DFA,
    n_states: int,
    accepting_count: int,
    max_attempts: int = 400,
) -> DFA | None:
    """Clone states up to ``n_states`` so the result accepts exactly ``accepting_count``.

    A clone copies its source's acceptance, so the final count is the core's count plus the
    number of *accepting* states cloned. Rather than expanding at random and rejecting the
    misses, the schedule is decided up front -- how many clone steps must copy an accepting
    state -- and each step then draws only from states of the required status. Rejection
    sampling would work but wastes most draws once the count is pinned.
    """
    total_clones = n_states - dfa.n_states
    if total_clones < 0:
        return None
    needed_accepting = accepting_count - len(dfa.accepting)
    if not 0 <= needed_accepting <= total_clones:
        return None

    schedule = [True] * needed_accepting + [False] * (total_clones - needed_accepting)
    rng.shuffle(schedule)

    current = dfa
    for want_accepting in schedule:
        candidates = [
            state
            for state in range(current.n_states)
            if (state in current.accepting) == want_accepting
        ]
        rng.shuffle(candidates)
        for attempt, state in enumerate(candidates):
            if attempt >= max_attempts:
                return None
            expanded = clone_state(rng, current, state)
            if expanded is not None:
                current = expanded
                break
        else:
            return None
    if len(current.accepting) != accepting_count or current.n_states != n_states:
        return None  # pragma: no cover - the schedule makes both hold
    return current


def eq_expand_pair_v2(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    core_states: tuple[int, int],
    accepting_count: int,
    require_non_isomorphic: bool = True,
) -> Draw:
    """An equivalent, non-isomorphic pair whose two sides both hit the frozen count.

    The core is sampled *at* a size in range and kept only if it is still in range after
    minimization -- a padded core is a smaller machine in disguise, and the size is meant to
    mean something. Both expansions then run independent clone schedules that land on
    ``accepting_count``.
    """
    low, high = core_states
    core_size = rng.randint(low, min(high, n_states))
    # The core's own accepting count must leave room to reach the target by cloning: it can
    # only grow, and only by at most one per clone step.
    max_core_accepting = min(accepting_count, core_size)
    min_core_accepting = max(0, accepting_count - (n_states - core_size))
    if min_core_accepting > max_core_accepting:
        return Draw.rejected("accepting_count_unreachable")
    core_accepting = rng.randint(min_core_accepting, max_core_accepting)

    seed_dfa = random_complete_dfa_with_count(rng, core_size, alphabet, core_accepting)
    if seed_dfa is None:
        return Draw.rejected("core_sample_failed")

    core = minimize(seed_dfa)
    if not low <= core.n_states <= high:
        return Draw.rejected("core_not_minimal")
    if core.n_states > n_states:
        return Draw.rejected("core_larger_than_target")

    dfa_a = expand_to_with_count(rng, core, n_states, accepting_count)
    if dfa_a is None:
        return Draw.rejected("expansion_stalled")
    dfa_b = expand_to_with_count(rng, core, n_states, accepting_count)
    if dfa_b is None:
        return Draw.rejected("expansion_stalled")

    if len(dfa_a.accepting) != accepting_count or len(dfa_b.accepting) != accepting_count:
        return Draw.rejected("accepting_count_mismatch")  # pragma: no cover
    if require_non_isomorphic and canonical_form(dfa_a) == canonical_form(dfa_b):
        return Draw.rejected("isomorphic_expansions")
    return Draw.ok(dfa_a, dfa_b)
