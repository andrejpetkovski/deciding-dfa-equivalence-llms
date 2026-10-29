"""Surface serialization of a DFA: the R0 and R1 representation conditions (§5.6).

This module produces the only thing a model ever sees. Nothing here may embed a `pair_id`, a
family, a depth, a seed, or the R0-to-R1 label bijection (research rule 8) -- the bijection
is returned to the caller as data and stored in a separate manifest, never in the text.

**R0 canonical.** Sequential labels ``q00, q01, …`` assigned in the DFA's *stored* order, with
the initial state's rows first and the rest in lexical label order.

The stored order, not a recomputed BFS order, per decision D1 in ``docs/freeze_log.md``. §5.6
specifies BFS order, but that recomputes the same numbering regardless of the independent
permutations §5.3 step 6 requires, so the two cancel: `EQ_ISO` pairs would render as identical
text and `NEQ_DEPTH` pairs would share a transition table. See protocol §5.0.

**R1 obfuscated.** The *same* label vocabulary as R0, re-assigned to states by a random
derangement, plus a uniform random permutation of the transition rows. Semantically identical
to R0 by construction, so any change in a model's answer between the two is a
representation-invariance violation (H3).

R1 used pronounceable nonces (``kub``, ``daf``) until 2026-09-08. Measurement killed that:
gate 8 came back at R1/R0 = 0.709-0.783 across all three models against a +-10% tolerance,
because a nonce costs ~1 token while ``q06`` costs ~2.6, times 128 label occurrences per
prompt. R0 and R1 were differing in *input length* by a quarter as well as in presentation,
so an H3 effect would have been partly a context-length effect. Sharing one vocabulary makes
the two renderings token-identical **by construction** rather than by measurement -- the same
multiset of label strings appears in both, only the assignment and the row order change.
See ``docs/freeze_log.md`` D5.4.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from src.dfa import DFA, DFAValidationError

__all__ = [
    "ParsedRendering",
    "Rendering",
    "derive_rng",
    "nonce_labels",
    "permuted_sequential_labels",
    "parse_rendering",
    "render",
    "render_R0",
    "render_R1",
    "sequential_labels",
]

# Consonant-vowel alternation gives pronounceable, clearly non-semantic labels.
_CONSONANTS = "bdfgjklmnprstvz"
_VOWELS = "aeiou"

# Random three-letter strings occasionally spell something offensive or loaded. With 5,600
# labels in the main set that is a near-certainty, and these strings end up in a published
# artifact, so the small set below is excluded at generation time. Filtering shrinks the label
# space by a negligible fraction and cannot bias the experiment: labels carry no meaning.
_BLOCKLIST = frozenset(
    [
        "ass", "fag", "fuk", "gay", "jap", "jew", "kkk", "nig",
        "pis", "rap", "sex", "tit", "vag", "wog", "gook", "kike", "spic",
    ]
)


@dataclass(frozen=True, slots=True)
class Rendering:
    """One serialized DFA, plus the label mapping needed to grade a model's trace.

    Attributes:
        name: ``"A"`` or ``"B"``, matching the prompt template's placeholder.
        condition: ``"R0"`` or ``"R1"``.
        text: Exactly what goes into the prompt.
        labels: ``labels[state]`` is the surface label of that internal state. This is the
            bijection: it must reach the grader but never the prompt.
    """

    name: str
    condition: str
    text: str
    labels: tuple[str, ...]

    @property
    def label_to_state(self) -> dict[str, int]:
        """Inverse of :attr:`labels`, for mapping a model's trace back to internal ids."""
        return {label: state for state, label in enumerate(self.labels)}

    @property
    def text_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedRendering:
    """A rendering read back into an automaton, for QA gate 6."""

    name: str
    dfa: DFA
    labels: tuple[str, ...]


def derive_rng(*parts: object) -> random.Random:
    """A ``random.Random`` seeded deterministically from the given parts.

    Seeds through a SHA-256 of the parts rather than Python's ``hash()``, which is salted per
    process and would make renderings irreproducible across runs (research rule 7).
    """
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16))


# --------------------------------------------------------------------------- labels


def _label_width(n_states: int) -> int:
    """Digits in a sequential label, at least two so ``q00`` reads as a fixed-width id."""
    return max(2, len(str(n_states - 1)))


def sequential_labels(n_states: int) -> tuple[str, ...]:
    """R0 labels ``q00, q01, …`` in stored-state order."""
    width = _label_width(n_states)
    return tuple(f"q{state:0{width}d}" for state in range(n_states))


def permuted_sequential_labels(rng: random.Random, n_states: int) -> tuple[str, ...]:
    """R1 labels: R0's own vocabulary, re-assigned to states by a random derangement.

    Returns the same multiset of strings :func:`sequential_labels` returns, so the two
    renderings cannot differ systematically in token count -- which is what gate 8 requires
    and what nonce labels failed to deliver.

    A **derangement**, not just any permutation: every state must carry a different surface
    label than it does in R0. The nonce scheme gave that property for free, and losing it
    would let a pair render with some states labelled identically across conditions, weakening
    the manipulation for that pair. Rejection sampling keeps the draw uniform over
    derangements and stays deterministic under a seeded rng (rule 7).

    Raises:
        ValueError: for ``n_states < 2``, where no derangement exists.
    """
    if n_states < 2:
        raise ValueError(f"no derangement exists for {n_states} state(s)")
    vocabulary = sequential_labels(n_states)
    states = list(range(n_states))
    for _ in range(1000):
        permutation = states[:]
        rng.shuffle(permutation)
        if all(permutation[state] != state for state in states):
            return tuple(vocabulary[permutation[state]] for state in states)
    raise RuntimeError(  # pragma: no cover - acceptance rate is ~1/e
        f"could not draw a derangement of {n_states} labels"
    )


def nonce_labels(rng: random.Random, n_states: int) -> tuple[str, ...]:
    """R1 labels: distinct pronounceable nonces, length-matched to the R0 labels.

    Matching the length is what keeps QA gate 8's ±10% token-balance tolerance attainable:
    a rendering whose labels are systematically longer would differ from R0 in prompt size
    as well as in presentation, confounding the H3 comparison.
    """
    length = 1 + _label_width(n_states)  # 'q' + digits
    chosen: list[str] = []
    seen: set[str] = set()
    guard = 0
    while len(chosen) < n_states:
        guard += 1
        if guard > 10000:  # pragma: no cover - the label space is orders larger
            raise RuntimeError(f"could not draw {n_states} distinct nonce labels")
        label = "".join(
            rng.choice(_CONSONANTS if position % 2 == 0 else _VOWELS)
            for position in range(length)
        )
        if label in seen or label in _BLOCKLIST:
            continue
        seen.add(label)
        chosen.append(label)
    return tuple(chosen)


# --------------------------------------------------------------------------- rendering


def _transition_rows(dfa: DFA, labels: tuple[str, ...]) -> list[str]:
    """One ``<state>, <symbol> -> <state>`` line per transition, in stored-state order."""
    return [
        f"{labels[state]}, {symbol} -> {labels[dfa.delta[state][index]]}"
        for state in range(dfa.n_states)
        for index, symbol in enumerate(dfa.alphabet)
    ]


def _assemble(
    name: str,
    dfa: DFA,
    labels: tuple[str, ...],
    rows: list[str],
    accepting_order: list[int],
) -> str:
    """Build the serialized block. R0 and R1 differ only in labels and row order."""
    accepting = ", ".join(labels[state] for state in accepting_order)
    return "\n".join(
        [
            f"DFA {name}",
            f"alphabet: {', '.join(dfa.alphabet)}",
            f"initial: {labels[dfa.start]}",
            f"accepting: {accepting}" if accepting else "accepting: (none)",
            "transitions:",
            *rows,
        ]
    )


def render_R0(dfa: DFA, name: str) -> Rendering:
    """Serialize under the canonical condition (§5.6, as amended by decision D1).

    Labels run ``q00, q01, …`` over the stored states. Rows put the initial state first and
    then the remaining states in lexical label order, which for these labels is stored order.
    """
    labels = sequential_labels(dfa.n_states)
    order = [dfa.start] + sorted(
        (s for s in range(dfa.n_states) if s != dfa.start), key=lambda s: labels[s]
    )
    rows = [
        f"{labels[state]}, {symbol} -> {labels[dfa.delta[state][index]]}"
        for state in order
        for index, symbol in enumerate(dfa.alphabet)
    ]
    accepting_order = sorted(dfa.accepting, key=lambda s: labels[s])
    return Rendering(
        name=name,
        condition="R0",
        text=_assemble(name, dfa, labels, rows, accepting_order),
        labels=labels,
    )


#: Label schemes R1 may use. ``nonce_cvc`` is retained solely so the legacy v1 pilot stays
#: byte-reproducible (D2); ``shared_vocabulary_deranged`` is the v2 scheme (D5.4). The config
#: names one under ``representations[].labels`` -- the renderer never picks.
R1_LABEL_SCHEMES = ("shared_vocabulary_deranged", "nonce_cvc")


def r1_labels(scheme: str, rng: random.Random, n_states: int) -> tuple[str, ...]:
    """Draw R1's labels under a named scheme.

    Raises:
        ValueError: for an unknown scheme. Defaulting would let a config typo silently change
            the representation manipulation, which is the thing H3 measures.
    """
    if scheme == "shared_vocabulary_deranged":
        return permuted_sequential_labels(rng, n_states)
    if scheme == "nonce_cvc":
        return nonce_labels(rng, n_states)
    raise ValueError(
        f"unknown R1 label scheme {scheme!r}; known schemes are {list(R1_LABEL_SCHEMES)}"
    )


def render_R1(
    dfa: DFA,
    name: str,
    rng: random.Random,
    scheme: str = "shared_vocabulary_deranged",
) -> Rendering:
    """Serialize under the obfuscated condition: relabelled states, shuffled rows (§5.6).

    Under the default scheme the labels are R0's own vocabulary re-assigned by a random
    derangement, so the label-to-state mapping is arbitrary while the multiset of label
    strings is identical to R0's. Every transition row is then shuffled uniformly, so rows
    for one state no longer sit together -- that is the presentation change H3 tests. The
    accepting list is shuffled too; it is part of the same predeclared row-order
    transformation (protocol §5.1).

    Note that R0's numbering is already arbitrary: D1 assigns labels in *stored* order, not
    BFS order, so ``q00`` is not the initial state and the digits track nothing a model could
    infer. Permuting that vocabulary therefore does not introduce a misleading ordinal cue
    that was not already present; what it removes is the correlation between a label and its
    row position, which R0 has and R1 does not.

    Args:
        scheme: which R1 label scheme to use. ``nonce_cvc`` reproduces the pre-D5.4
            renderings and exists so the legacy v1 pilot stays byte-identical (D2); it must
            not be used for new datasets, since it is what failed gate 8.
    """
    labels = r1_labels(scheme, rng, dfa.n_states)
    rows = _transition_rows(dfa, labels)
    rng.shuffle(rows)
    accepting_order = sorted(dfa.accepting)
    rng.shuffle(accepting_order)
    return Rendering(
        name=name,
        condition="R1",
        text=_assemble(name, dfa, labels, rows, accepting_order),
        labels=labels,
    )


def render(
    dfa: DFA,
    name: str,
    condition: str,
    rng: random.Random | None = None,
    r1_scheme: str = "shared_vocabulary_deranged",
) -> Rendering:
    """Dispatch to :func:`render_R0` or :func:`render_R1` by condition id."""
    if condition == "R0":
        return render_R0(dfa, name)
    if condition == "R1":
        if rng is None:
            raise ValueError("R1 needs an rng; the relabelling and row order are random")
        return render_R1(dfa, name, rng, r1_scheme)
    raise ValueError(f"unknown representation condition {condition!r}")


# --------------------------------------------------------------------------- parsing


def parse_rendering(text: str) -> ParsedRendering:
    """Read a rendering back into a DFA -- the inverse used by QA gate 6.

    States are numbered by sorted label, which is arbitrary but deterministic. Gate 6 compares
    canonical forms, so the numbering does not matter; what matters is that a round trip
    recovers the same automaton up to isomorphism.

    Raises:
        DFAValidationError: on a malformed block, a duplicate transition, or a transition
            table that is not total.
    """
    header: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    name = ""
    in_transitions = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("DFA "):
            name = line[len("DFA ") :].strip()
            continue
        if line == "transitions:":
            in_transitions = True
            continue
        if not in_transitions:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
            continue
        source, _, rest = line.partition(",")
        symbol, arrow, destination = rest.partition("->")
        if not arrow:
            raise DFAValidationError(f"malformed transition line: {line!r}")
        rows.append((source.strip(), symbol.strip(), destination.strip()))

    for required in ("alphabet", "initial", "accepting"):
        if required not in header:
            raise DFAValidationError(f"rendering is missing the {required!r} line")

    alphabet = tuple(s.strip() for s in header["alphabet"].split(","))
    accepting_text = header["accepting"]
    accepting_labels = (
        []
        if accepting_text == "(none)"
        else [s.strip() for s in accepting_text.split(",") if s.strip()]
    )

    labels = sorted({row[0] for row in rows} | {row[2] for row in rows})
    index = {label: i for i, label in enumerate(labels)}
    if header["initial"] not in index:
        raise DFAValidationError(
            f"initial state {header['initial']!r} never appears in a transition"
        )

    delta: list[list[int | None]] = [[None] * len(alphabet) for _ in labels]
    for source, symbol, destination in rows:
        if symbol not in alphabet:
            raise DFAValidationError(f"symbol {symbol!r} is not in alphabet {alphabet!r}")
        if destination not in index:
            raise DFAValidationError(f"unknown destination state {destination!r}")
        column = alphabet.index(symbol)
        if delta[index[source]][column] is not None:
            raise DFAValidationError(
                f"duplicate transition for ({source!r}, {symbol!r})"
            )
        delta[index[source]][column] = index[destination]

    for state, row in enumerate(delta):
        for column, destination in enumerate(row):
            if destination is None:
                raise DFAValidationError(
                    f"no transition for ({labels[state]!r}, {alphabet[column]!r})"
                )

    dfa = DFA.build(
        n_states=len(labels),
        alphabet=alphabet,
        start=index[header["initial"]],
        accepting=[index[label] for label in accepting_labels],
        delta=delta,  # type: ignore[arg-type]
    )
    return ParsedRendering(name=name, dfa=dfa, labels=tuple(labels))
