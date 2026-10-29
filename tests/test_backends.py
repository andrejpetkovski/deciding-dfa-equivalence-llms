"""Backend contract, wire-format adapter, and HTTP client (M7).

The response fixtures in ``tests/fixtures/gateway/`` are **real bodies** copied out of the
committed probe reports, not hand-written approximations. That matters: the whole point of
the adapter is that it matches what this gateway actually emits, and a fixture invented to
match the code would test nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.backends.adapters import DEFAULT_ADAPTER, extract
from src.backends.base import (
    MODEL_OUTCOME_FINISH_REASONS,
    ChatRequest,
    classify_failure,
    redact_headers,
    upstream_failure_code,
)
from src.backends.gateway_http import GatewayHTTPBackend, load_env_file, resolve_credentials

FIXTURES = Path(__file__).parent / "fixtures" / "gateway"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- adapters


@pytest.mark.parametrize("name", ["nemotron_ok.json", "mimo_ok.json"])
def test_adapter_reads_content_and_separates_reasoning(name: str) -> None:
    body = fixture(name)["body"]
    answer = extract(body)

    assert answer.final_text == "ok"
    assert answer.finish_reason == "stop"
    assert answer.usable
    assert answer.problem is None
    # Reasoning is present on both models and must not be part of the graded channel.
    assert answer.reasoning_text is not None
    assert answer.reasoning_text != answer.final_text


def test_reasoning_is_never_promoted_into_the_answer() -> None:
    """A model that thinks and then says nothing is a failure, not a reasoning transcript."""
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": None,
                    "reasoning_content": '{"equivalent": false, "witness": "01"}',
                },
            }
        ]
    }
    answer = extract(body)

    assert answer.final_text is None
    assert not answer.usable
    assert answer.truncated
    assert "output cap" in (answer.problem or "")
    # The text is retained for the record, just not graded.
    assert answer.reasoning_text is not None


def test_adapter_does_not_strip_a_code_fence() -> None:
    """Unwrapping here would silently convert a strict-parse failure into a success."""
    fenced = '```json\n{"equivalent": true, "witness": null}\n```'
    body = {"choices": [{"finish_reason": "stop", "message": {"content": fenced}}]}

    assert extract(body).final_text == fenced


def test_adapter_falls_back_through_provider_specific_fields() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "{}",
                    "provider_specific_fields": {
                        "reasoning_details": [
                            {"text": "first"},
                            {"text": "second"},
                        ]
                    },
                },
            }
        ]
    }
    assert extract(body).reasoning_text == "first\nsecond"


def test_adapter_reports_an_error_body_rather_than_raising() -> None:
    answer = extract(fixture("deepseek_blocked.json")["body"])

    assert answer.final_text is None
    assert "Model is blocked" in (answer.problem or "")


def test_adapter_reports_a_non_string_content() -> None:
    body = {"choices": [{"message": {"content": {"text": "ok"}}}]}
    assert "not a string" in (extract(body).problem or "")


def test_the_adapter_stamps_its_own_name_on_every_result() -> None:
    """The graded row records which adapter produced the text it scored."""
    answer = extract(fixture("mimo_ok.json")["body"], DEFAULT_ADAPTER)
    assert answer.adapter == DEFAULT_ADAPTER == "openai_chat_content_v1"


def test_unknown_adapter_raises_rather_than_defaulting() -> None:
    with pytest.raises(KeyError, match="unknown final-answer adapter"):
        extract({"choices": []}, "make_it_work_v9")


# --------------------------------------------------------------------------- failures


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (200, False),  # ok, nothing to retry
        (429, True),
        (503, True),
        (504, True),
        (403, False),  # a blocked model never unblocks by asking again
        (400, False),
        (401, False),
    ],
)
def test_classify_failure_retryability(status: int, retryable: bool) -> None:
    _, actual = classify_failure(status, None)
    assert actual is retryable


def test_a_timeout_is_retryable_and_named() -> None:
    kind, retryable = classify_failure(None, httpx.ReadTimeout("slow"))
    assert retryable
    assert "timeout" in kind.lower()


def test_blocked_model_is_permanent() -> None:
    """The gateway's 403 for deepseek must not consume five attempts."""
    status = fixture("deepseek_blocked.json")["status"]
    kind, retryable = classify_failure(status, None)
    assert status == 403
    assert not retryable
    assert kind == "permanent_http:403"


def test_gateway_unavailable_is_retryable() -> None:
    """The 2026-09-07 outage recovered on its own; that is what backoff is for."""
    payload = fixture("unavailable.json")
    _, retryable = classify_failure(payload["status"], None)
    assert retryable


# --------------------------------------------------------------------------- redaction


def test_redaction_drops_credentials_and_keeps_identity() -> None:
    headers = {
        "Authorization": "Bearer sk-secret-value",
        "Set-Cookie": "session=abc",
        "x-litellm-model-id": "94dc785d" + "0" * 56,
        "x-litellm-version": "1.93.0",
    }
    redacted = redact_headers(headers)

    assert "sk-secret-value" not in json.dumps(redacted)
    assert redacted["Set-Cookie"] == "<redacted>"
    # The deployment id is the study's only identity anchor; redacting it would be a bug.
    assert redacted["x-litellm-model-id"] == "94dc785d" + "0" * 56
    assert redacted["x-litellm-version"] == "1.93.0"


# --------------------------------------------------------------------------- requests


def test_payload_omits_unset_controls() -> None:
    request = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},))
    payload = request.payload()

    assert set(payload) == {"model", "messages"}
    assert "seed" not in payload  # asserting a control the gateway may reject


def test_payload_includes_set_controls_and_extras() -> None:
    request = ChatRequest(
        model="m",
        messages=({"role": "user", "content": "hi"},),
        max_tokens=64,
        temperature=0,
        extra={"reasoning_effort": "low"},
    )
    payload = request.payload()

    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0
    assert payload["reasoning_effort"] == "low"


def test_request_hash_is_stable_and_order_independent() -> None:
    a = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},), temperature=0)
    b = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},), temperature=0)
    c = ChatRequest(model="m", messages=({"role": "user", "content": "hj"},), temperature=0)

    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


# --------------------------------------------------------------------------- HTTP client


def _backend(handler) -> GatewayHTTPBackend:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return GatewayHTTPBackend("https://example.invalid/v1", "sk-test-key", client=client)


def test_successful_call_records_identity_and_usage() -> None:
    payload = fixture("nemotron_ok.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"].startswith("Bearer ")
        return httpx.Response(200, json=payload["body"], headers=payload["response_headers"])

    backend = _backend(handler)
    result = backend.complete(
        ChatRequest(model="nemotron-3-ultra-free", messages=({"role": "user", "content": "x"},))
    )

    assert result.ok
    assert result.returned_model == "nemotron-3-ultra-free"
    assert result.finish_reason == "stop"
    assert result.usage is not None and result.usage["prompt_tokens"] == 23
    assert result.response_id is not None and result.response_id.startswith("gen-")
    assert "x-litellm-model-group" in result.identity_headers()


def test_the_api_key_never_reaches_a_record() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []}, headers={"x-litellm-version": "1.93.0"})

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )
    assert "sk-test-key" not in json.dumps(result.response_headers)


def test_a_non_json_body_is_kept_rather_than_discarded() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Caddy error</html>")

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )

    assert not result.ok
    assert result.retryable
    assert "Caddy" in result.body["_non_json_text"]


def test_a_transport_exception_becomes_a_record_not_a_crash() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )

    assert not result.ok
    assert result.retryable
    assert result.status is None
    assert "ConnectError" in (result.failure_detail or "")


def test_backend_never_retries_on_its_own() -> None:
    """Retry policy and the attempt ceiling live in the runner; a backend that retried
    internally would spend the ceiling without logging the attempts."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json=fixture("unavailable.json")["body"])

    _backend(handler).complete(ChatRequest(model="m", messages=({"role": "user", "content": "x"},)))
    assert calls == 1


# --------------------------------------------------------------------------- credentials


def test_env_file_is_parsed_without_exposing_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('FINKI_API_KEY="sk-abc"\n# comment\nFINKI_API_BASE_URL=https://x/v1\n',
                   encoding="utf-8")
    values = load_env_file(env)

    assert values["FINKI_API_KEY"] == "sk-abc"
    assert values["FINKI_API_BASE_URL"] == "https://x/v1"


def test_missing_credentials_name_the_variables_not_the_values(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as error:
        resolve_credentials(
            tmp_path / "absent.env",
            base_url_var="NO_SUCH_BASE_URL_VAR",
            key_var="NO_SUCH_KEY_VAR",
        )
    message = str(error.value)
    assert "NO_SUCH_KEY_VAR" in message
    assert "Never paste the key" in message


def test_the_adapter_reads_openrouters_reasoning_fields() -> None:
    """OpenRouter puts reasoning on the message, not under provider_specific_fields.

    Verified 2026-09-09 against deepseek-v4-flash-0731: a 3,741-character `reasoning` beside
    a 37-character `content`. Missing it corrupted no grade -- the graded channel is still
    message.content, verbatim -- but it dropped the reasoning from the record and reported
    reasoning_present=False where reasoning plainly existed.
    """
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"equivalent": false, "witness": "1"}',
                    "reasoning": "We need answer only JSON. Need determine equivalence...",
                    "reasoning_details": [{"format": "unknown", "text": "..."}],
                },
            }
        ]
    }
    answer = extract(body)

    assert answer.final_text == '{"equivalent": false, "witness": "1"}'
    assert answer.reasoning_text is not None
    assert "determine equivalence" in answer.reasoning_text


def test_reasoning_details_are_used_when_the_plain_field_is_absent() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "{}",
                    "reasoning_details": [{"text": "first"}, {"text": "second"}],
                }
            }
        ]
    }
    assert extract(body).reasoning_text == "first\nsecond"


# ------------------------------------------------- 200-wrapped upstream failures (D9)
#
# OpenRouter answers 200 OK for an upstream provider failure and puts the real code in the
# body. The 2026-09-09 pilot recorded 39 of these for xiaomi/mimo-v2.5 on GMICloud -- 41% of
# that model's cells -- every one written as `ok` and then graded as a wrong DFA answer.
# The tests below fix both halves of the contract: such a body is a transport failure, and
# a body that actually contains an answer is never one, however bad the answer is.


def test_a_200_wrapped_error_envelope_is_a_retryable_transport_failure() -> None:
    """The exact body the pilot saw 20 times."""
    body = {
        "error": {"code": 502, "message": "Provider returned an empty response"},
        "id": "gen-1788955816-TzPcW1pL4DiPEM5vO5MD",
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )

    assert not result.ok
    assert result.retryable
    assert result.status == 200, "the real status is still recorded verbatim (rule 3)"
    assert result.body == body, "the envelope is kept, not normalised away"


def test_a_completion_that_stopped_with_no_finish_reason_is_a_transport_failure() -> None:
    """The other 19: content null, finish_reason null, reasoning tokens burned.

    A non-streamed completion always carries a finish reason, so a null one means the
    generation never finished.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "native_finish_reason": None,
                        "message": {"role": "assistant", "content": None, "reasoning": "hm"},
                    }
                ],
                "usage": {"completion_tokens": 150},
            },
        )

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )

    assert not result.ok
    assert result.retryable


def test_cap_exhaustion_stays_an_observed_failure_and_is_never_retried() -> None:
    """finish_reason=length is a result about the model, not a transport fault.

    Retrying it would quietly resample until the model happened to fit inside the cap,
    which is exactly the bias rule 4 exists to prevent.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": None, "reasoning": "..."},
                    }
                ]
            },
        )

    result = _backend(handler).complete(
        ChatRequest(model="m", messages=({"role": "user", "content": "x"},))
    )

    assert result.ok, "a truncated generation is an answer-channel outcome, not a retry"
    assert upstream_failure_code(result.body) is None


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "I think they are equivalent, honestly",
        '```json {"equivalent": true} ```',
        "{ not valid json",
    ],
    ids=["empty", "whitespace", "prose", "fenced", "malformed"],
)
def test_any_content_string_blocks_a_retry_however_bad_it_is(content: str) -> None:
    """Rule 4's guard. A wrong answer is an observed failure, not a transport fault.

    If any of these were retryable, the runner could ask again until it liked the answer,
    and the strict-parse rate -- the primary metric -- would be inflated by resampling.
    """
    body = {
        "choices": [
            {"index": 0, "finish_reason": None, "message": {"content": content}}
        ]
    }
    assert upstream_failure_code(body) is None


def test_an_error_envelopes_own_code_decides_retryability() -> None:
    """A permanent upstream code must not be retried forever."""
    assert upstream_failure_code({"error": {"code": 502}}) == 502
    assert upstream_failure_code({"error": {"code": 400}}) == 400
    _, retryable = classify_failure(upstream_failure_code({"error": {"code": 400}}), None)
    assert not retryable
    _, retryable = classify_failure(upstream_failure_code({"error": {"code": 502}}), None)
    assert retryable


def test_unknown_shapes_are_left_alone() -> None:
    """Guessing that an unrecognised 200 is broken would invent unevidenced retries."""
    assert upstream_failure_code({"choices": []}) is None
    assert upstream_failure_code({"something": "else"}) is None
    assert upstream_failure_code(None) is None
    assert upstream_failure_code("text") is None


def test_a_healthy_completion_is_untouched() -> None:
    body = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "{}"}}]}
    assert upstream_failure_code(body) is None


# ------------------------------------------------- finish_reason allowlist (D12)
#
# The first version of the D9 guard asked `finish_reason is not None`, reasoning about
# "length" and "stop" and never considering that a provider might send "error". Four cells
# of the main run came back finish_reason="error" with null content, were recorded ok, and
# were graded as wrong answers. An allowlist fails closed: an unrecognised reason is a
# fault, not a result.


@pytest.mark.parametrize("reason", ["error", None, "weird_new_reason", ""])
def test_no_content_with_a_non_outcome_finish_reason_is_a_fault(reason) -> None:
    body = {"choices": [{"finish_reason": reason, "message": {"content": None}}]}
    assert upstream_failure_code(body) == 502


@pytest.mark.parametrize("reason", sorted(MODEL_OUTCOME_FINISH_REASONS))
def test_no_content_with_a_real_outcome_reason_stays_an_observed_failure(reason) -> None:
    """Cap exhaustion and refusals are results about the model. Retrying them resamples."""
    body = {"choices": [{"finish_reason": reason, "message": {"content": None}}]}
    assert upstream_failure_code(body) is None


def test_an_error_finish_reason_with_content_is_still_an_answer() -> None:
    """The rule-4 guard outranks the allowlist: if there is text, it is the answer."""
    body = {"choices": [{"finish_reason": "error", "message": {"content": "{}"}}]}
    assert upstream_failure_code(body) is None
