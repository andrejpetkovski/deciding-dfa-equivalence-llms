"""Frozen wire-format adapters: raw response in, final-answer text out.

This is the narrowest module in the project on purpose. It decides **which field is the
answer**, and nothing else. It does not repair JSON, strip fences, remove explanations, or
search prose for a verdict -- ``src/grading.py`` scores whatever comes out of here, and any
cleanup done at this layer would be invisible to the strict parser and would inflate the
primary metric.

The graded channel has never changed: ``choices[0].message.content``, verbatim. Only the
*retained* reasoning lookup has grown as new providers surfaced new field names.

The one substantive decision it encodes is the answer/reasoning split. Both models verified
on 2026-09-07 return the final answer in ``choices[0].message.content`` and the chain of
thought in ``message.reasoning_content`` (mirrored under
``provider_specific_fields.reasoning`` and ``.reasoning_details[].text``). Reasoning is
retained separately and **never** graded: it is not answer prose, and reading it for a
verdict the answer field did not contain would score a different channel under P0 than
under P1.

Adapters are versioned and named in the frozen config so a change is a freeze event rather
than an edit. ``docs/model_manifest.md`` records which adapter each model uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ADAPTERS",
    "DEFAULT_ADAPTER",
    "ExtractedAnswer",
    "extract",
]

#: Every model verified so far speaks this shape; the registry exists so a model that does
#: not can be added without touching the ones that do.
DEFAULT_ADAPTER = "openai_chat_content_v1"


@dataclass(frozen=True, slots=True)
class ExtractedAnswer:
    """The graded channel, the retained channel, and why the first might be unusable."""

    adapter: str
    final_text: str | None
    reasoning_text: str | None
    finish_reason: str | None
    truncated: bool
    #: Set when no answer text could be located. The grader turns this into a parse failure
    #: (rule 5) -- it is an observed outcome, not an error to route around.
    problem: str | None = None

    @property
    def usable(self) -> bool:
        return self.final_text is not None and self.final_text.strip() != ""


def _first_choice(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def _reasoning_of(message: dict[str, Any]) -> str | None:
    """Collect the reasoning channel from the documented fields, in priority order.

    Retained for the record only. ``usage.reasoning_tokens`` is not consulted: the probe saw
    ``mimo-v2.5-free`` report ``reasoning_tokens: 0`` while emitting a full reasoning block,
    so the count is unreliable and the text is the evidence.
    """
    direct = message.get("reasoning_content")
    if isinstance(direct, str) and direct != "":
        return direct

    # OpenRouter puts these directly on the message rather than under
    # provider_specific_fields (verified 2026-09-09 against deepseek-v4-flash-0731: a 3,741
    # character `reasoning` string alongside a 37 character `content`). Missing them did not
    # corrupt any grade -- the graded channel is still message.content, verbatim -- but it
    # silently dropped the reasoning from the record and reported reasoning_present=False
    # where reasoning plainly existed.
    plain = message.get("reasoning")
    if isinstance(plain, str) and plain != "":
        return plain
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = [
            item["text"]
            for item in details
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return "\n".join(parts)

    fields = message.get("provider_specific_fields")
    if isinstance(fields, dict):
        mirrored = fields.get("reasoning")
        if isinstance(mirrored, str) and mirrored != "":
            return mirrored
        details = fields.get("reasoning_details")
        if isinstance(details, list):
            parts = [
                item["text"]
                for item in details
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
    return None


def _openai_chat_content_v1(body: Any) -> ExtractedAnswer:
    """OpenAI-compatible chat completion: the answer is ``message.content``, verbatim.

    ``content`` is returned exactly as received -- not stripped, not unwrapped. The strict
    parser is entitled to reject a fenced or prefixed answer, and that rejection is a
    result about the model, so trimming here would erase it.
    """
    choice = _first_choice(body)
    if choice is None:
        detail = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            detail = str(body["error"].get("message"))
        return ExtractedAnswer(
            adapter=DEFAULT_ADAPTER,
            final_text=None,
            reasoning_text=None,
            finish_reason=None,
            truncated=False,
            problem=(
                f"no choices in response body ({detail})"
                if detail
                else "no choices in response body"
            ),
        )

    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    finish_reason = choice.get("finish_reason")
    finish_reason = str(finish_reason) if finish_reason is not None else None
    truncated = finish_reason == "length"
    reasoning = _reasoning_of(message)

    if content is None:
        # A reasoning model can spend its whole output cap thinking and return no answer.
        # That is a real, reportable failure mode; the reasoning text is kept but is not
        # promoted into the answer channel.
        problem = "message.content is null"
        if truncated:
            problem += " (finish_reason=length: output cap reached)"
        elif reasoning is not None:
            problem += " (reasoning present but not gradeable)"
        return ExtractedAnswer(
            adapter=DEFAULT_ADAPTER,
            final_text=None,
            reasoning_text=reasoning,
            finish_reason=finish_reason,
            truncated=truncated,
            problem=problem,
        )

    if not isinstance(content, str):
        return ExtractedAnswer(
            adapter=DEFAULT_ADAPTER,
            final_text=None,
            reasoning_text=reasoning,
            finish_reason=finish_reason,
            truncated=truncated,
            problem=f"message.content is {type(content).__name__}, not a string",
        )

    return ExtractedAnswer(
        adapter=DEFAULT_ADAPTER,
        final_text=content,
        reasoning_text=reasoning,
        finish_reason=finish_reason,
        truncated=truncated,
        problem="empty answer text" if content.strip() == "" else None,
    )


#: Name -> adapter. Frozen at the freeze tag; adding an entry is a protocol change.
ADAPTERS = {DEFAULT_ADAPTER: _openai_chat_content_v1}


def extract(body: Any, adapter: str = DEFAULT_ADAPTER) -> ExtractedAnswer:
    """Apply a named adapter.

    Raises:
        KeyError: for an unknown adapter name. Falling back to a default would let a
            config typo silently change which field is graded.
    """
    if adapter not in ADAPTERS:
        raise KeyError(
            f"unknown final-answer adapter {adapter!r}; "
            f"frozen adapters are {sorted(ADAPTERS)}"
        )
    return ADAPTERS[adapter](body)
