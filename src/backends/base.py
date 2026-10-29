"""The transport-neutral contract every backend implements (§Step 9).

One *scheduled cell* -- a (pair, representation, prompt, model, sample) tuple -- may cost
several *attempts*. This module fixes the vocabulary that keeps those two apart, because
conflating them is how retries silently inflate a sample size.

Three research rules are enforced here rather than left to each backend:

- **Retry transport, never content** (rule 4). :func:`classify_failure` is the single place
  that decides what may be retried, and it can only ever see a status code or an exception --
  never the answer. A backend that wanted to retry a malformed answer would have to route
  around this module to do it.
- **Raw payloads survive** (rule 3). :class:`RawResult` carries the decoded body verbatim,
  including error bodies and non-JSON text. Nothing here parses an answer.
- **Credentials never reach a log.** :func:`redact_headers` is applied on the way in, so a
  ``RawResult`` cannot hold an authorization header even if a caller forgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.io_utils import canonical_json, sha256_hex

__all__ = [
    "PERMANENT",
    "RETRYABLE_STATUS",
    "Backend",
    "ChatRequest",
    "RawResult",
    "classify_failure",
    "redact_headers",
    "MODEL_OUTCOME_FINISH_REASONS",
    "upstream_failure_code",
]

#: Statuses worth another attempt: rate limiting, and the gateway's own unavailability.
#: The 2026-09-07 probe saw 503/504 "Endpoint is unavailable" from an upstream outage that
#: recovered minutes later, which is exactly the case this exists for.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Statuses that will not change by asking again. 403 is here because the gateway returns it
#: for a blocked model (`deepseek-v4-flash-free`): retrying would burn the ceiling on a
#: request that can never succeed.
PERMANENT = frozenset({400, 401, 403, 404, 405, 413, 422})

#: Finish reasons that mean the model genuinely produced an outcome, so an empty answer
#: alongside one is a *result* about the model and must never be retried (rule 4).
#: "length" is cap exhaustion, "stop" a real completion, "content_filter" a refusal by the
#: provider's filter, "tool_calls" a structured reply. Anything else -- "error", null, or a
#: value not yet seen -- is treated as a transport fault.
MODEL_OUTCOME_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})

_REDACTED_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "proxy-authorization"})


def redact_headers(headers: Any) -> dict[str, str]:
    """Copy headers, replacing credential-bearing values with a marker.

    Applied to request *and* response headers. Response headers are the study's main
    identity evidence (``x-litellm-model-id``, rate-limit counters), so this redacts by
    name rather than by pattern -- a pattern wide enough to catch a key also catches the
    64-hex deployment id the record exists to keep.
    """
    out: dict[str, str] = {}
    for key, value in dict(headers).items():
        low = str(key).lower()
        if low == "authorization":
            out[str(key)] = "Bearer <redacted>"
        elif low in _REDACTED_HEADERS:
            out[str(key)] = "<redacted>"
        else:
            out[str(key)] = str(value)
    return out


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One outbound chat request, as the frozen decoding policy defines it.

    ``extra`` holds per-model parameters the frozen policy declares. It is deliberately not
    a free-for-all: the runner passes what the config froze, identically across a model's R
    and P conditions, so a parameter cannot differ between the arms being compared.
    """

    model: str
    messages: tuple[dict[str, str], ...]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """The JSON body, omitting every unset control.

        Omission matters: sending ``"seed": null`` asserts a control the gateway may reject
        outright, and a rejected request is not the same experiment as an unsent parameter.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in self.messages],
        }
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("seed", self.seed),
        ):
            if value is not None:
                body[name] = value
        body.update(self.extra)
        return body

    @property
    def sha256(self) -> str:
        """Digest of the exact outbound body.

        This is the hash the protocol always requires. It is *not* the post-template input
        the model saw -- the gateway wraps these messages in an unseen chat template
        (``mimo-v2.5-free`` reported 254 prompt tokens for a six-token message). That second
        hash stays null with a reason; see ``docs/cluster.md``.
        """
        return sha256_hex(canonical_json(self.payload()))


@dataclass(frozen=True, slots=True)
class RawResult:
    """The complete record of one attempt, before anything is parsed.

    ``body`` is whatever came back -- a completion, an error object, or
    ``{"_non_json_text": ...}`` for a response that was not JSON at all. It is never
    normalised, repaired, or dropped.
    """

    ok: bool
    attempt: int
    requested_model: str
    request_sha256: str
    started_at_utc: str
    latency_ms: float
    status: int | None = None
    body: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    failure_kind: str | None = None
    failure_detail: str | None = None
    retryable: bool = False

    @property
    def returned_model(self) -> str | None:
        """The model id the gateway reports, which may differ from the alias requested."""
        if isinstance(self.body, dict):
            value = self.body.get("model")
            return str(value) if value is not None else None
        return None

    @property
    def response_id(self) -> str | None:
        if isinstance(self.body, dict):
            value = self.body.get("id")
            return str(value) if value is not None else None
        return None

    @property
    def finish_reason(self) -> str | None:
        choices = (self.body or {}).get("choices") if isinstance(self.body, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            value = choices[0].get("finish_reason")
            return str(value) if value is not None else None
        return None

    @property
    def usage(self) -> dict[str, Any] | None:
        if isinstance(self.body, dict) and isinstance(self.body.get("usage"), dict):
            return dict(self.body["usage"])
        return None

    def identity_headers(self) -> dict[str, str]:
        """The subset of response headers that identifies the deployment and its limits.

        Pulled out so the per-attempt record keeps drift evidence in a stable place while
        the full header set is retained alongside it.
        """
        wanted = (
            "x-litellm-model-id",
            "x-litellm-model-group",
            "x-litellm-model-api-base",
            "x-litellm-version",
            "x-litellm-call-id",
            "x-litellm-attempted-retries",
            "x-litellm-attempted-fallbacks",
            "x-litellm-response-cost",
            "x-litellm-response-duration-ms",
            "x-ratelimit-api_key-limit-max_parallel_requests",
            "x-ratelimit-team-remaining-requests",
            "x-ratelimit-team-remaining-tokens",
        )
        lowered = {key.lower(): value for key, value in self.response_headers.items()}
        return {name: lowered[name] for name in wanted if name in lowered}


def classify_failure(status: int | None, exception: BaseException | None) -> tuple[str, bool]:
    """Name a failure and say whether another attempt is permitted.

    Returns ``(kind, retryable)``. The signature is the enforcement: this function is given
    a status code and an exception and nothing else, so no amount of pressure to "just try
    once more on a bad answer" can be expressed through it (rule 4).
    """
    if exception is not None:
        name = type(exception).__name__
        if "Timeout" in name:
            return f"transport_timeout:{name}", True
        return f"transport_error:{name}", True
    if status is None:
        return "transport_error:no_status", True
    if status in RETRYABLE_STATUS:
        return ("rate_limited" if status == 429 else f"server_unavailable:{status}"), True
    if status in PERMANENT:
        return f"permanent_http:{status}", False
    if 200 <= status < 300:
        return "ok", False
    return f"http_error:{status}", False


def upstream_failure_code(body: Any) -> int | None:
    """Return the transport code hidden inside a 2xx body, or ``None``.

    Some gateways answer ``200 OK`` for an upstream failure and put the real code in the
    body. OpenRouter returns ``{"error": {"code": 502, "message": "Provider returned an
    empty response"}}`` with no ``choices`` at all; the 2026-09-09 pilot recorded 20 of
    these for ``xiaomi/mimo-v2.5`` on GMICloud, plus 19 completions that stopped with no
    content and no finish reason. Every one was written as ``ok`` and then graded as a
    wrong answer, because classification looked at the status code alone. That is 41% of
    one model's cells: an infrastructure failure scored as the model failing at DFAs.

    **Why this is not "retry on content" (rule 4).** Both branches are reachable only when
    there is no answer to read:

    - no ``choices`` at all: the gateway is reporting its own failure rather than relaying
      a completion, so there is no model output in the response;
    - a choice whose ``content`` is null *and* whose ``finish_reason`` is null: a
      non-streamed completion always carries a finish reason, so a null one means the
      generation stopped without ever finishing.

    The moment a ``content`` string exists -- however malformed, fenced, empty or wrong --
    this returns ``None`` and that answer stands as an observed outcome. A non-null
    ``finish_reason`` does the same: ``length`` is cap exhaustion and ``stop`` is a real
    answer, and both are results, not transport faults. So the function cannot be used to
    ask a model again for a better answer, which is the thing rule 4 forbids.

    Unknown body shapes return ``None``. Guessing that an unrecognised 200 is broken would
    invent retries the evidence does not support.
    """
    if not isinstance(body, dict):
        return None

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            # Honour the gateway's own code so a permanent one (400/403) is not retried
            # forever; fall back to 502 when the envelope carries no usable code.
            return code if isinstance(code, int) else 502
        if body.get("_non_json_text"):
            return 502
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    message = message if isinstance(message, dict) else {}
    if message.get("content") is not None:
        return None
    # No content. Whether that is a *result* or a *fault* is decided by the finish reason,
    # and only an explicit model-outcome reason makes it a result. The first version of this
    # check asked `finish_reason is not None`, which silently blessed "error" -- 4 cells of
    # the main run came back finish_reason="error", native_finish_reason=None, content=null
    # and were graded as wrong answers (D12). An allowlist fails closed instead: a reason
    # nobody has seen before is treated as a fault and retried, not scored against the model.
    if first.get("finish_reason") in MODEL_OUTCOME_FINISH_REASONS:
        return None
    return 502


class Backend(Protocol):
    """What the runner needs from a transport. One attempt in, one record out.

    A backend never retries and never sleeps: bounded retry, backoff and concurrency belong
    to the runner, where the attempt ceiling is enforced and every attempt is logged.
    """

    name: str

    def complete(self, request: ChatRequest, attempt: int) -> RawResult:
        """Perform a single attempt, returning a record even when it failed."""
        ...
