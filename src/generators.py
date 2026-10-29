"""Controlled generators for the three base-pair families (§5.2-5.5).

Every function here *requests* a property; nothing here assigns a label. The oracle decides
whether a candidate is equivalent and at what depth, and :mod:`src.generate` rejects any
candidate whose oracle verdict does not match the request (research rule 1, §7.3).

Each builder takes an explicit ``random.Random`` and returns a :class:`Draw`: either a pair,
or a rejection carrying the reason it failed. Rejection is an ordinary, countable outcome
rather than an exception, because §Step 4 requires rejection counts by reason -- a stratum
that is unusually hard to sample is itself a finding.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from src.dfa import DFA, canonical_form
from src.minimize import minimize
from src.simulate import access_distances, reachable

__all__ = [
    "Draw",
    "LayeredGraph",
    "clone_state",
    "eq_expand_pair",
    "eq_iso_pair",
    "expand_to",
    "layered_target_depth_graph",
    "neq_depth_pair",
    "permute_independently",
    "random_complete_dfa",
]


@dataclass(frozen=True, slots=True)
class Draw:
    """The outcome of one generation attempt.

    Rejections carry a reason because §Step 4 requires rejection *counts by reason*: a
    stratum that is unusually hard to sample is evidence about the design, and a single
    undifferentiated "failed" tally would hide which step is actually rejecting.
    """

    pair: tuple[DFA, DFA] | None
    reason: str | None

    @classmethod
    def ok(cls, dfa_a: DFA, dfa_b: DFA) -> Draw:
        return cls(pair=(dfa_a, dfa_b), reason=None)

    @classmethod
    def rejected(cls, reason: str) -> Draw:
        return cls(pair=None, reason=reason)

    @property
    def accepted(self) -> bool:
        return self.pair is not None


@dataclass(frozen=True, slots=True)
class LayeredGraph:
    """A transition structure with a designated state at a known BFS distance.

    Attributes:
        delta: The transition table, ``delta[state][symbol_index]``.
        target: The designated state, at shortest access distance ``depth`` from state 0.
        layers: Intended BFS layer of each state, which the construction makes equal to the
            true shortest access distance.
        target_word: The backbone word, of length ``depth``, that spells a shortest path
            from the start to ``target``.
    """

    delta: tuple[tuple[int, ...], ...]
    target: int
    layers: tuple[int, ...]
    target_word: str


# --------------------------------------------------------------------------- depth control


def layered_target_depth_graph(
    rng: random.Random,
    n_states: int,
    depth: int,
    alphabet: tuple[str, ...],
) -> LayeredGraph | None:
    """Build a complete, fully reachable graph whose target state sits at exactly ``depth``.

    Implements the layered backbone of §5.4. Uniformly random complete DFAs almost never
    contain a state as deep as 9-12, so the depth has to be constructed rather than waited
    for.

    The construction maintains one invariant: **every edge leads to a state whose layer is
    at most one greater than its source's**. Two consequences follow, and together they pin
    the target's distance exactly:

    - No state can be reached faster than its layer, so ``dist(v) >= layer(v)``. In
      particular nothing reaches ``target`` in fewer than ``depth`` steps.
    - Every state is given an incoming edge from the layer below it, so
      ``dist(v) <= layer(v)``.

    The start is always state 0 and the target is always state ``depth``; callers permute
    the numbering afterwards (§5.3 step 6).

    Returns:
        A :class:`LayeredGraph`, or ``None`` if the draw ran out of outgoing edges to keep
        every state reachable -- a rejection the caller should count and retry.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if n_states < depth + 1:
        raise ValueError(
            f"{n_states} states cannot hold a depth-{depth} backbone; need {depth + 1}"
        )

    n_symbols = len(alphabet)
    symbol_index = {symbol: i for i, symbol in enumerate(alphabet)}
    target = depth

    # Backbone: states 0..depth in a chain spelling target_word.
    target_word = "".join(rng.choice(alphabet) for _ in range(depth))
    delta: list[list[int | None]] = [[None] * n_symbols for _ in range(n_states)]
    free: list[set[int]] = [set(range(n_symbols)) for _ in range(n_states)]
    for i, symbol in enumerate(target_word):
        column = symbol_index[symbol]
        delta[i][column] = i + 1
        free[i].discard(column)

    layers: list[int] = [-1] * n_states
    states_by_layer: dict[int, list[int]] = defaultdict(list)
    for state in range(depth + 1):
        layers[state] = state
        states_by_layer[state].append(state)
    max_layer = depth

    # Attach the remaining states as side branches, each hung off a state one layer up the
    # tree. Only layers that still have a spare outgoing edge are eligible, which keeps the
    # draw from painting itself into a corner.
    for state in range(depth + 1, n_states):
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

    # Fill every remaining slot, respecting the layer invariant. State 0 is always a legal
    # destination, so no slot can be left unfillable.
    for state in range(n_states):
        for column in sorted(free[state]):
            highest = layers[state] + 1
            candidates = [
                destination
                for destination in range(n_states)
                if layers[destination] <= highest
                # Stated separately in §5.4 though implied by the layer invariant: nothing
                # below layer depth-1 may point at the target.
                and (destination != target or layers[state] >= depth - 1)
            ]
            delta[state][column] = rng.choice(candidates)
        free[state].clear()

    built = tuple(tuple(row) for row in delta)  # type: ignore[arg-type]
    graph = LayeredGraph(
        delta=built, target=target, layers=tuple(layers), target_word=target_word
    )

    # §5.4 mandates the check even though the invariant makes it redundant. It is cheap and
    # it is the only thing standing between a construction bug and a mislabelled stratum.
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
    if distances[target] != depth:
        return None
    return graph


def neq_depth_pair(
    rng: random.Random,
    n_states: int,
    depth: int,
    alphabet: tuple[str, ...],
    accept_probability: float = 0.4,
) -> Draw:
    """A non-equivalent pair whose shortest distinguishing word has length ``depth`` (§5.3).

    A and B share a transition table and differ only in whether the target state accepts.
    A word therefore distinguishes them exactly when it *ends at the target*, so the set of
    distinguishing words is the set of access words of the target and the shortest one has
    length ``dist(target) == depth``.

    That makes the depth exact by construction rather than by luck, which is the whole point
    of the layered backbone. The oracle still has the final say (§7.3); it simply should
    never disagree, and :mod:`src.generate` counts it as a rejection if it ever does.

    Because ``depth >= 1`` the target is never the start state, so the two machines always
    agree on the empty word -- QA gate 3's "no epsilon leak" holds by construction.
    """
    graph = layered_target_depth_graph(rng, n_states, depth, alphabet)
    if graph is None:
        return Draw.rejected("graph_construction_failed")

    accepting = frozenset(
        state for state in range(n_states) if rng.random() < accept_probability
    )
    dfa_a = DFA.build(
        n_states=n_states,
        alphabet=alphabet,
        start=0,
        accepting=accepting,
        delta=graph.delta,
    )
    dfa_b = dfa_a.toggled(graph.target)

    if (dfa_a.start in dfa_a.accepting) != (dfa_b.start in dfa_b.accepting):
        # Unreachable given depth >= 1, but never ship an epsilon cue.
        return Draw.rejected("epsilon_cue")
    return Draw.ok(dfa_a, dfa_b)


# --------------------------------------------------------------------------- equivalent pairs


def random_complete_dfa(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    accept_probability: float = 0.4,
    max_attempts: int = 500,
) -> DFA | None:
    """A uniformly random complete DFA in which every state is reachable.

    Reachability is obtained by rejection, not by construction, so the transition table
    stays uniform -- a constructive fix would bias the graph statistics that §13 relies on
    to rule out depth confounding.
    """
    for _ in range(max_attempts):
        delta = tuple(
            tuple(rng.randrange(n_states) for _ in alphabet) for _ in range(n_states)
        )
        accepting = frozenset(
            state for state in range(n_states) if rng.random() < accept_probability
        )
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


def eq_iso_pair(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    accept_probability: float = 0.4,
) -> Draw:
    """An equivalent pair related by a state permutation (§5.2, EQ_ISO).

    The easiest equivalent case by design: B is A with its states renamed, so the two
    machines are isomorphic and a solver that matches graphs succeeds without reasoning
    about languages. EQ_EXPAND supplies the equivalent case that graph matching cannot
    shortcut.
    """
    dfa_a = random_complete_dfa(rng, n_states, alphabet, accept_probability)
    if dfa_a is None:
        return Draw.rejected("core_sample_failed")
    permutation = list(range(n_states))
    rng.shuffle(permutation)
    return Draw.ok(dfa_a, dfa_a.relabel(tuple(permutation)))


def clone_state(
    rng: random.Random, dfa: DFA, state: int
) -> DFA | None:
    """Split ``state`` into two behaviourally identical states (§5.5).

    The clone is appended as the highest-numbered state with the same accepting status and
    the same outgoing transitions as the original. One incoming edge is then redirected to
    it. Original and clone have the same right language, so no word changes its acceptance
    and ``L`` is preserved exactly.

    Returns:
        The expanded DFA, or ``None`` if no redirect leaves every state reachable -- which
        happens when the only incoming edge is the one that keeps ``state`` alive.
    """
    clone = dfa.n_states
    rows = [list(row) for row in dfa.delta]
    rows.append(list(dfa.delta[state]))  # identical outgoing behaviour

    incoming = [
        (source, column)
        for source in range(dfa.n_states)
        for column, destination in enumerate(dfa.delta[source])
        if destination == state
    ]
    rng.shuffle(incoming)

    for source, column in incoming:
        candidate_rows = [list(row) for row in rows]
        candidate_rows[source][column] = clone
        candidate = DFA.build(
            n_states=dfa.n_states + 1,
            alphabet=dfa.alphabet,
            start=dfa.start,
            accepting=(
                sorted(dfa.accepting) + [clone]
                if state in dfa.accepting
                else sorted(dfa.accepting)
            ),
            delta=candidate_rows,
        )
        if len(reachable(candidate)) == candidate.n_states:
            return candidate
    return None


def expand_to(
    rng: random.Random, dfa: DFA, n_states: int, max_attempts: int = 200
) -> DFA | None:
    """Clone states until the automaton has ``n_states``, preserving the language.

    Each step picks a state at random and clones it; a state whose only incoming edge is
    load-bearing is simply passed over.
    """
    current = dfa
    attempts = 0
    while current.n_states < n_states:
        attempts += 1
        if attempts > max_attempts:
            return None
        expanded = clone_state(rng, current, rng.randrange(current.n_states))
        if expanded is not None:
            current = expanded
    return current


def eq_expand_pair(
    rng: random.Random,
    n_states: int,
    alphabet: tuple[str, ...],
    core_states: tuple[int, int],
    accept_probability: float = 0.4,
    require_non_isomorphic: bool = True,
) -> Draw:
    """An equivalent pair that is not merely a renaming (§5.5, EQ_EXPAND).

    Sample a core DFA *at* a size drawn from the configured range, minimize it, and keep it
    only if the minimal machine is still in range. Minimality is what makes the core's size
    meaningful -- a padded 12-state machine is really a smaller one in disguise -- and
    re-checking after minimization is what enforces it.

    Sampling at the target size matters for throughput. Drawing a 14-state machine and
    hoping it minimizes into 10-12 lands in range about 4% of the time, because a random
    complete DFA is nearly always already minimal; drawing at 10-12 lands in range 79-100%
    of the time. Both are correct, since the oracle gates the result either way, but the
    first wastes most of its draws and would bury the informative rejection reasons under
    noise.

    The core is then expanded twice along independent clone sequences. Both expansions
    recognize the core's language, hence each other's, but they distribute the redundancy
    differently -- so aligning them requires reasoning about behaviour, not about shape.
    """
    low, high = core_states
    core_size = rng.randint(low, min(high, n_states))
    seed_dfa = random_complete_dfa(rng, core_size, alphabet, accept_probability)
    if seed_dfa is None:
        return Draw.rejected("core_sample_failed")

    core = minimize(seed_dfa)
    if not low <= core.n_states <= high:
        return Draw.rejected("core_not_minimal")
    if core.n_states > n_states:
        return Draw.rejected("core_larger_than_target")

    dfa_a = expand_to(rng, core, n_states)
    if dfa_a is None:
        return Draw.rejected("expansion_stalled")
    dfa_b = expand_to(rng, core, n_states)
    if dfa_b is None:
        return Draw.rejected("expansion_stalled")

    if require_non_isomorphic and canonical_form(dfa_a) == canonical_form(dfa_b):
        return Draw.rejected("isomorphic_expansions")
    return Draw.ok(dfa_a, dfa_b)


# --------------------------------------------------------------------------- presentation


def permute_independently(
    rng: random.Random, dfa_a: DFA, dfa_b: DFA
) -> tuple[DFA, DFA]:
    """Renumber the two machines under independent random permutations (§5.3 step 6).

    Keeps the differing state of a NEQ_DEPTH pair from sitting at the same index in both
    tables. Renaming states cannot change a language, so this is semantics-preserving and
    the oracle verdict is unaffected.

    Note that whether this survives into what the model actually sees is a *rendering*
    decision, not a generation one: a rendering that relabels by BFS order recomputes the
    same numbering for both machines and cancels the permutation out. See the R0 note in
    protocol §5.1.
    """
    permutation_a = list(range(dfa_a.n_states))
    permutation_b = list(range(dfa_b.n_states))
    rng.shuffle(permutation_a)
    rng.shuffle(permutation_b)
    return dfa_a.relabel(tuple(permutation_a)), dfa_b.relabel(tuple(permutation_b))
