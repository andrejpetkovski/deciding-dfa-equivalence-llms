"""Strict parsing and deterministic grading of one model response (Step 10, §8.5, D3).

Two stages, deliberately separated:

1. :func:`strict_parse` decides whether the final answer *is* a well-formed response. It is
   unforgiving by design (§8.5): a code fence, a word of prose, a missing or extra key, a
   wrong type, or a symbol outside the alphabet all make the response a parse failure, and a
   parse failure counts as wrong for the primary metric (rule 5).
2. :func:`grade` decides whether a well-formed response is *correct*, against the oracle.

Nothing here repairs, strips, or searches. A forgiving parser exists as
:func:`forgiving_parse`, but its result is labelled sensitivity analysis and never replaces
the strict verdict -- §8.5 is explicit that the strict result is primary.

**Outcome definitions (D3).** H1's and H2's primary outcome is
:attr:`Grade.witness_success`: strict parse **and** predicts non-equivalent **and** supplies a
word accepted by exactly one machine. Three consequences follow, and each is a deliberate
choice recorded in ``docs/experiment_revision.md``:

- A *valid but non-shortest* witness is a success. The prompt asks for **a** witness, not a
  shortest one, so scoring shortestness as failure would grade against an instruction that was
  never given. Shortestness is recorded descriptively.
- An *incorrect trace* does not change witness success. Requiring a valid trace would make the
  outcome mean something different under P1 than under P0, which is precisely the comparison
  H2 is trying to make. Trace validity is a separate P1 outcome.
- A *transport failure or truncation* is a failure of the scheduled cell, not an absent row.
  Dropping it would shrink the denominator and silently inflate the rate.

Pure module: no file I/O, no network, no logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.dfa import DFA, DFAValidationError
from src.simulate import accepts, first_divergence, trace, validate_word

__all__ = [
    "Grade",
    "ParseOutcome",
    "ParsedAnswer",
    "forgiving_parse",
    "grade",
    "strict_parse",
]

#: Exact key sets each prompt condition must produce. Anything else is a parse failure.
_REQUIRED_KEYS = {
    "P0": frozenset({"equivalent", "witness"}),
    "P1": frozenset({"equivalent", "witness", "trace_A", "trace_B"}),
}


class ParseOutcome:
    """Why a response did or did not parse. Values are stable and appear in analysis rows."""

    OK = "ok"
    NOT_JSON = "not_json"
    NOT_AN_OBJECT = "not_an_object"
    WRONG_KEYS = "wrong_keys"
    WRONG_TYPE = "wrong_type"
    ILLEGAL_SYMBOL = "illegal_symbol"
    INCONSISTENT_FIELDS = "inconsistent_fields"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    """The result of parsing one final answer."""

    ok: bool
    outcome: str
    equivalent: bool | None = None
    witness: str | None = None
    trace_a: tuple[str, ...] | None = None
    trace_b: tuple[str, ...] | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Grade:
    """Every outcome scored for one scheduled cell.

    Attributes:
        parse_valid: The strict parser accepted the final answer.
        decision_correct: The Boolean matches the oracle. ``False`` when unparsed.
        witness_success: **H1/H2 primary outcome.** Strict parse, predicts non-equivalent, and
            the witness is accepted by exactly one machine. Only defined on ground-truth
            non-equivalent cells; ``None`` elsewhere.
        witness_valid: The proposed witness satisfies XOR acceptance, given the model predicted
            non-equivalent and parsed. This is §10.1's *conditional* witness validity, whose
            denominator differs from :attr:`witness_success`.
        witness_shortest: Descriptive only: the valid witness has length equal to the oracle
            depth. Never part of success (D3).
        trace_valid: P1 only. Both traces equal the deterministic simulations.
        first_wrong_step: Index of the first incorrect trace state, for the error taxonomy.
    """

    parse_valid: bool
    parse_outcome: str
    decision_correct: bool
    witness_success: bool | None
    witness_valid: bool | None
    witness_shortest: bool | None
    trace_valid: bool | None
    first_wrong_step: int | None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_valid": self.parse_valid,
            "parse_outcome": self.parse_outcome,
            "decision_correct": self.decision_correct,
            "witness_success": self.witness_success,
            "witness_valid": self.witness_valid,
            "witness_shortest": self.witness_shortest,
            "trace_valid": self.trace_valid,
            "first_wrong_step": self.first_wrong_step,
            **self.detail,
        }


# --------------------------------------------------------------------------- parsing


def strict_parse(text: str | None, condition: str, alphabet: tuple[str, ...]) -> ParsedAnswer:
    """Parse a final answer under the frozen strict contract (§8.5).

    Accepts exactly one JSON object with exactly the condition's keys and nothing else.
    Surrounding whitespace is tolerated because it carries no content; a code fence, a prefix,
    or a trailing sentence is not.

    Args:
        text: The final-answer channel only. Separating a reasoning channel is the wire-format
            adapter's job and must happen before this is called -- reasoning text is not
            answer prose (D4).
        condition: ``"P0"`` or ``"P1"``.
        alphabet: The legal witness symbols.
    """
    if condition not in _REQUIRED_KEYS:
        raise ValueError(f"unknown prompt condition {condition!r}")
    if text is None or not text.strip():
        return ParsedAnswer(False, ParseOutcome.EMPTY)

    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return ParsedAnswer(False, ParseOutcome.NOT_JSON, detail=str(error))

    if not isinstance(payload, dict):
        return ParsedAnswer(
            False, ParseOutcome.NOT_AN_OBJECT, detail=type(payload).__name__
        )

    required = _REQUIRED_KEYS[condition]
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        return ParsedAnswer(
            False,
            ParseOutcome.WRONG_KEYS,
            detail=f"missing={missing} extra={extra}",
        )

    equivalent = payload["equivalent"]
    if not isinstance(equivalent, bool):
        return ParsedAnswer(
            False, ParseOutcome.WRONG_TYPE, detail=f"equivalent={equivalent!r}"
        )

    witness = payload["witness"]
    if witness is not None and not isinstance(witness, str):
        return ParsedAnswer(False, ParseOutcome.WRONG_TYPE, detail=f"witness={witness!r}")
    if isinstance(witness, str):
        illegal = [symbol for symbol in witness if symbol not in alphabet]
        if illegal:
            return ParsedAnswer(
                False, ParseOutcome.ILLEGAL_SYMBOL, detail=f"symbols={sorted(set(illegal))}"
            )

    traces: dict[str, tuple[str, ...] | None] = {"trace_A": None, "trace_B": None}
    if condition == "P1":
        for key in ("trace_A", "trace_B"):
            value = payload[key]
            if value is None:
                continue
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return ParsedAnswer(
                    False, ParseOutcome.WRONG_TYPE, detail=f"{key}={value!r}"
                )
            traces[key] = tuple(value)

    # Cross-field consistency. The schema cannot express these, but a response asserting
    # non-equivalence with no witness is not a well-formed certificate.
    if equivalent and witness is not None:
        return ParsedAnswer(
            False, ParseOutcome.INCONSISTENT_FIELDS, detail="equivalent=true with a witness"
        )
    if not equivalent and witness is None:
        return ParsedAnswer(
            False, ParseOutcome.INCONSISTENT_FIELDS, detail="equivalent=false without a witness"
        )
    if condition == "P1":
        if equivalent and any(traces[key] is not None for key in traces):
            return ParsedAnswer(
                False, ParseOutcome.INCONSISTENT_FIELDS, detail="equivalent=true with a trace"
            )
        if not equivalent and any(traces[key] is None for key in traces):
            return ParsedAnswer(
                False,
                ParseOutcome.INCONSISTENT_FIELDS,
                detail="equivalent=false without both traces",
            )

    return ParsedAnswer(
        True,
        ParseOutcome.OK,
        equivalent=equivalent,
        witness=witness,
        trace_a=traces["trace_A"],
        trace_b=traces["trace_B"],
    )


def forgiving_parse(
    text: str | None, condition: str, alphabet: tuple[str, ...]
) -> ParsedAnswer:
    """Strip at most one surrounding code fence, then parse strictly (§8.5).

    **Sensitivity analysis only.** Its result must be reported beside the strict one and never
    substituted for it: a study whose headline number depends on how forgiving its parser is
    has measured its parser. The single fence is the only concession, and nothing else about
    the response is altered.
    """
    if text is None:
        return strict_parse(text, condition, alphabet)
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    return strict_parse(stripped, condition, alphabet)


# --------------------------------------------------------------------------- grading


def _map_trace(
    reported: tuple[str, ...] | None, labels: tuple[str, ...]
) -> tuple[int, ...] | None:
    """Map a trace of surface labels back to internal state ids.

    Returns ``None`` if any label is unknown, which is a wrong trace rather than a crash: the
    model may invent a state name, and that is an observation.
    """
    if reported is None:
        return None
    index = {label: state for state, label in enumerate(labels)}
    mapped: list[int] = []
    for label in reported:
        if label not in index:
            return None
        mapped.append(index[label])
    return tuple(mapped)


def grade(
    parsed: ParsedAnswer,
    dfa_a: DFA,
    dfa_b: DFA,
    oracle_equivalent: bool,
    oracle_depth: int | None,
    labels_a: tuple[str, ...],
    labels_b: tuple[str, ...],
) -> Grade:
    """Score one parsed response against the oracle.

    Args:
        labels_a: The rendering's state-to-label mapping for machine A, from the manifest.
            Traces arrive as surface labels and must come back through the *same* bijection
            the model was shown; QA gate 6 verifies it is the right one.
    """
    if not parsed.ok:
        # A parse failure is wrong for the decision, and is a failed witness attempt on a
        # non-equivalent cell rather than a missing row (rule 5, D3).
        return Grade(
            parse_valid=False,
            parse_outcome=parsed.outcome,
            decision_correct=False,
            witness_success=False if not oracle_equivalent else None,
            witness_valid=None,
            witness_shortest=None,
            trace_valid=None,
            first_wrong_step=None,
            detail={"parse_detail": parsed.detail},
        )

    decision_correct = parsed.equivalent == oracle_equivalent

    witness_valid: bool | None = None
    witness_shortest: bool | None = None
    detail: dict[str, Any] = {}

    if parsed.equivalent is False and parsed.witness is not None:
        try:
            validate_word(dfa_a, parsed.witness)
            validate_word(dfa_b, parsed.witness)
            witness_valid = accepts(dfa_a, parsed.witness) != accepts(dfa_b, parsed.witness)
        except DFAValidationError as error:
            # The strict parser already rejects out-of-alphabet symbols, so this is a guard
            # against a caller that skipped it rather than an expected path.
            witness_valid = False
            detail["witness_error"] = str(error)
        if witness_valid and oracle_depth is not None:
            witness_shortest = len(parsed.witness) == oracle_depth
        detail["witness_length"] = len(parsed.witness)

    # H1/H2 primary outcome, defined only where the ground truth is non-equivalent.
    witness_success: bool | None = None
    if not oracle_equivalent:
        witness_success = bool(
            parsed.equivalent is False and witness_valid is True
        )

    trace_valid: bool | None = None
    first_wrong_step: int | None = None
    if parsed.trace_a is not None or parsed.trace_b is not None:
        witness = parsed.witness or ""
        expected = {"A": trace(dfa_a, witness), "B": trace(dfa_b, witness)}
        reported = {
            "A": _map_trace(parsed.trace_a, labels_a),
            "B": _map_trace(parsed.trace_b, labels_b),
        }
        divergences: list[int] = []
        trace_valid = True
        for name in ("A", "B"):
            if reported[name] is None:
                trace_valid = False
                detail[f"trace_{name}_unmapped"] = True
                continue
            if reported[name] != expected[name]:
                trace_valid = False
                position = first_divergence(expected[name], reported[name])
                if position is not None:
                    divergences.append(position)
        if divergences:
            first_wrong_step = min(divergences)

    return Grade(
        parse_valid=True,
        parse_outcome=ParseOutcome.OK,
        decision_correct=decision_correct,
        witness_success=witness_success,
        witness_valid=witness_valid,
        witness_shortest=witness_shortest,
        trace_valid=trace_valid,
        first_wrong_step=first_wrong_step,
        detail=detail,
    )
