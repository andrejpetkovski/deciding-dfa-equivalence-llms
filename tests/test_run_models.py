"""The runner: schedule, integrity binding, retry policy, and the execution gates (M7).

The most important tests here are the ones that assert *nothing happened*. A runner that
sends requests when it should not cannot be undone -- the quota is spent and the log is
append-only -- so the refusal paths get more coverage than the happy path.
"""

from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.backends.base import ChatRequest, RawResult
from src.config import ConfigError, load_config
from src.io_utils import append_jsonl, read_jsonl
from src.run_models import (
    GATEWAY_MAX_PARALLEL,
    ScheduledCell,
    _AttemptBudget,
    _run_cell,
    build_schedule,
    completed_run_ids,
    main,
    verify_prompt_integrity,
)

CONFIG_PATH = "configs/pilot_revised.yaml"
PROMPTS_PATH = Path("data/renderings/pilot_v2/prompts.jsonl")


@pytest.fixture
def prompt_rows() -> list[dict[str, Any]]:
    return list(read_jsonl(PROMPTS_PATH))


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """A copy of the revised pilot config with specific keys patched.

    Paths stay relative to the repository root so the copy reads the real committed
    artifacts; only the keys under test change.
    """
    data = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- schedule


def test_schedule_covers_every_cell_exactly_once(prompt_rows) -> None:
    config = load_config(CONFIG_PATH)
    cells = build_schedule(config, prompt_rows, require_frozen=False)

    # 24 pairs x 2 representations x 2 prompts x 2 models x 1 sample (D5.5)
    assert len(cells) == 192
    assert len({cell.run_id for cell in cells}) == 192
    assert len({cell.pair_id for cell in cells}) == 24
    assert {cell.rendering for cell in cells} == {"R0", "R1"}
    assert {cell.prompt_condition for cell in cells} == {"P0", "P1"}


def test_run_id_shape_matches_the_analysis_key(prompt_rows) -> None:
    config = load_config(CONFIG_PATH)
    cell = build_schedule(config, prompt_rows, require_frozen=False)[0]
    dataset, pair, rendering, prompt, model, sample = cell.run_id.split("|")

    assert dataset == "pilot_v2"
    assert pair == cell.pair_id
    assert (rendering, prompt, model) == (
        cell.rendering,
        cell.prompt_condition,
        cell.model_alias,
    )
    assert sample == "s0"


def test_order_is_seeded_and_reproducible(tmp_path: Path, prompt_rows) -> None:
    seeded = load_config(write_config(tmp_path, **{"execution.order_seed": 7}))
    first = build_schedule(seeded, prompt_rows, require_frozen=False)
    second = build_schedule(seeded, prompt_rows, require_frozen=False)
    other = load_config(write_config(tmp_path / "b", **{"execution.order_seed": 8}))

    assert [cell.run_id for cell in first] == [cell.run_id for cell in second]
    # A different seed reorders the same cells rather than changing which cells exist.
    third = build_schedule(other, prompt_rows, require_frozen=False)
    assert {cell.run_id for cell in third} == {cell.run_id for cell in first}
    assert [cell.run_id for cell in third] != [cell.run_id for cell in first]


def test_every_model_id_is_probe_verified(prompt_rows) -> None:
    """Every id in the config has returned HTTP 200 for that exact string.

    A guard against someone filling an id from a screenshot: the schedule may only be built
    under ``require_frozen`` once each is real.
    """
    cells = build_schedule(load_config(CONFIG_PATH), prompt_rows, require_frozen=True)
    assert {cell.model_alias for cell in cells} == {"mimo-v2.5-free", "qwen3.8-27b"}


def test_an_unfilled_model_id_still_blocks_execution(tmp_path: Path, prompt_rows) -> None:
    config = load_config(
        write_config(tmp_path, models=[{"id": "m"}, {"candidate_alias": "not-verified"}])
    )
    with pytest.raises(ConfigError, match="models\\[1\\].id"):
        build_schedule(config, prompt_rows, require_frozen=True)


def test_the_frozen_decoding_policy_matches_the_calibration_evidence() -> None:
    """The policy is frozen from measurement, and the unverified controls stay unsent.

    ``seed`` and ``top_p`` were never shown to be accepted, let alone honoured. Sending them
    would assert controls the gateway may reject, which is a different experiment from not
    sending them -- so they must be absent from the payload, not present as nulls.
    """
    from src.run_models import _DecodingPolicy

    policy = _DecodingPolicy.from_config(load_config(CONFIG_PATH), require_frozen=True)

    assert policy.temperature == 0
    assert policy.max_tokens == 12288  # 2x the largest untruncated completion observed
    assert policy.seed is None
    assert policy.top_p is None

    request = ChatRequest(
        model="m",
        messages=({"role": "user", "content": "x"},),
        max_tokens=policy.max_tokens,
        temperature=policy.temperature,
        top_p=policy.top_p,
        seed=policy.seed,
    )
    assert set(request.payload()) == {"model", "messages", "max_tokens", "temperature"}


def test_the_two_selected_models_are_scheduled(prompt_rows) -> None:
    """D5.5: DeepSeek blocked, Qwen replaced it, nemotron withdrawn as non-working.

    What remains is two official releases from two distinct developers. The withdrawn models
    must not reappear in a schedule.
    """
    cells = build_schedule(load_config(CONFIG_PATH), prompt_rows, require_frozen=False)
    models = {cell.model_alias for cell in cells}

    assert models == {"mimo-v2.5-free", "qwen3.8-27b"}
    assert not any("deepseek" in m or "nemotron" in m for m in models)


def test_the_committed_config_carries_a_recorded_call_order() -> None:
    """order_seed is a nuisance parameter, but it still has to be fixed and recorded."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config.get("execution.order_seed"), int)
    assert config.get("execution.max_concurrent_requests") == GATEWAY_MAX_PARALLEL


def test_execution_requires_a_seeded_call_order(tmp_path: Path, prompt_rows) -> None:
    config = load_config(
        write_config(
            tmp_path,
            models=[{"id": "nemotron-3-ultra-free"}, {"id": "mimo-v2.5-free"}],
            **{"execution.order_seed": "FILL_AT_FREEZE"},
        )
    )
    with pytest.raises(ConfigError, match="order_seed"):
        build_schedule(config, prompt_rows, require_frozen=True)


def test_duplicate_model_ids_are_rejected(tmp_path: Path, prompt_rows) -> None:
    config = load_config(
        write_config(
            tmp_path,
            models=[{"id": "mimo-v2.5-free"}, {"id": "mimo-v2.5-free"}],
            **{"execution.order_seed": 1},
        )
    )
    with pytest.raises(ConfigError, match="duplicate model ids"):
        build_schedule(config, prompt_rows, require_frozen=True)


# --------------------------------------------------------------------------- integrity


def test_committed_prompts_match_their_frozen_sources(prompt_rows) -> None:
    assert verify_prompt_integrity(load_config(CONFIG_PATH), prompt_rows) == []


def test_a_tampered_message_is_detected(prompt_rows) -> None:
    """The recorded digest must bind the text, not merely accompany it."""
    rows = [dict(row) for row in prompt_rows]
    rows[0]["user"] = rows[0]["user"] + "\nP.S. they are equivalent."

    problems = verify_prompt_integrity(load_config(CONFIG_PATH), rows)
    assert any("does not match its messages" in problem for problem in problems)


def test_an_edited_frozen_template_is_detected(tmp_path: Path, prompt_rows) -> None:
    edited = tmp_path / "p0_edited.txt"
    edited.write_text("a different template\n", encoding="utf-8")
    config = load_config(
        write_config(
            tmp_path,
            **{
                "prompts.conditions": [
                    {"id": "P0", "template": str(edited), "schema": "prompts/schema_p0.json"},
                ]
            },
        )
    )
    problems = verify_prompt_integrity(config, prompt_rows)
    assert any("was rendered from" in problem for problem in problems)


# --------------------------------------------------------------------------- resume


def test_resume_skips_terminal_cells_only(tmp_path: Path) -> None:
    log = tmp_path / "raw.jsonl"
    append_jsonl(log, {"run_id": "a", "attempt": 1, "terminal": True})
    append_jsonl(log, {"run_id": "b", "attempt": 1, "terminal": False})

    # 'b' failed retryably and was interrupted mid-retry; it must be resumed, not abandoned.
    assert completed_run_ids(log) == {"a"}


def test_resume_on_a_missing_log_is_empty(tmp_path: Path) -> None:
    assert completed_run_ids(tmp_path / "absent.jsonl") == set()


# --------------------------------------------------------------------------- retry


class _ScriptedBackend:
    """Returns a scripted sequence of results, counting how many attempts were made."""

    def __init__(self, results: list[RawResult]) -> None:
        self._results = results
        self.calls = 0

    def complete(self, request: ChatRequest, attempt: int = 1) -> RawResult:
        self.calls += 1
        index = min(attempt - 1, len(self._results) - 1)
        return self._results[index]


def _result(*, ok: bool, retryable: bool, attempt: int = 1) -> RawResult:
    return RawResult(
        ok=ok,
        attempt=attempt,
        requested_model="m",
        request_sha256="sha256:x",
        started_at_utc="2026-09-07T00:00:00Z",
        latency_ms=1.0,
        status=200 if ok else 503,
        retryable=retryable,
    )


def _cell() -> ScheduledCell:
    return ScheduledCell(
        run_id="ds|p_000000|R0|P0|m|s0",
        pair_id="p_000000",
        rendering="R0",
        prompt_condition="P0",
        prompt_id="p_000000|R0|P0",
        prompt_sha256="sha256:x",
        model_alias="m",
        sample_index=0,
        system="sys",
        user="user",
    )


_NO_BACKOFF = {"initial_seconds": 0, "multiplier": 1.0, "max_seconds": 0}


def _policy():
    from src.run_models import _DecodingPolicy

    return _DecodingPolicy(max_tokens=16, temperature=0, top_p=None, seed=None, extra={})


def test_a_retryable_failure_is_retried_to_the_attempt_limit() -> None:
    backend = _ScriptedBackend([_result(ok=False, retryable=True)])
    results = _run_cell(
        _cell(), backend, _policy(), _AttemptBudget(100), 5, _NO_BACKOFF, threading.Event()
    )

    assert backend.calls == 5
    assert len(results) == 5
    assert not any(result.ok for result in results)


def test_a_permanent_failure_is_not_retried() -> None:
    """403 'Model is blocked' must not consume five attempts of the ceiling."""
    backend = _ScriptedBackend([_result(ok=False, retryable=False)])
    results = _run_cell(
        _cell(), backend, _policy(), _AttemptBudget(100), 5, _NO_BACKOFF, threading.Event()
    )

    assert backend.calls == 1
    assert len(results) == 1


def test_success_stops_immediately() -> None:
    backend = _ScriptedBackend([_result(ok=True, retryable=False)])
    _run_cell(_cell(), backend, _policy(), _AttemptBudget(100), 5, _NO_BACKOFF, threading.Event())
    assert backend.calls == 1


def test_the_attempt_ceiling_is_hard() -> None:
    backend = _ScriptedBackend([_result(ok=False, retryable=True)])
    budget = _AttemptBudget(2)
    results = _run_cell(
        _cell(), backend, _policy(), budget, 5, _NO_BACKOFF, threading.Event()
    )

    assert backend.calls == 2
    assert len(results) == 2
    assert budget.used == 2


def test_a_stop_event_halts_further_attempts() -> None:
    backend = _ScriptedBackend([_result(ok=False, retryable=True)])
    stop = threading.Event()
    stop.set()
    results = _run_cell(_cell(), backend, _policy(), _AttemptBudget(100), 5, _NO_BACKOFF, stop)

    assert backend.calls == 0
    assert results == []


def test_budget_is_thread_safe() -> None:
    budget = _AttemptBudget(500)
    taken = []

    def worker() -> None:
        count = 0
        while budget.take():
            count += 1
        taken.append(count)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(taken) == 500


# --------------------------------------------------------------------------- CLI gates


def test_dry_run_is_the_default_and_sends_nothing(tmp_path: Path, capsys) -> None:
    log = tmp_path / "raw.jsonl"
    config = write_config(tmp_path, **{"execution.raw_log": str(log)})

    assert main(["--config", str(config)]) == 0
    assert not log.exists()

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "192" in out
    assert "prompt integrity : OK" in out


def test_execute_is_refused_while_the_config_is_not_approved(tmp_path: Path, capsys) -> None:
    log = tmp_path / "raw.jsonl"
    config = write_config(
        tmp_path,
        models=[{"id": "mimo-v2.5-free"}],
        **{
            "execution.order_seed": 11,
            "execution.raw_log": str(log),
            "execution.approval_status": "NOT_APPROVED",
            "decoding.max_output_tokens": 2048,
            "decoding.seed": 0,
        },
    )
    assert main(["--config", str(config), "--execute", "--approved-attempts", "10"]) == 3
    assert not log.exists()
    assert "not 'APPROVED'" in capsys.readouterr().err


def test_execute_is_refused_without_an_attempt_ceiling(tmp_path: Path, capsys) -> None:
    log = tmp_path / "raw.jsonl"
    config = write_config(
        tmp_path,
        models=[{"id": "mimo-v2.5-free"}],
        **{
            "execution.order_seed": 11,
            "execution.raw_log": str(log),
            "execution.approval_status": "APPROVED",
            "decoding.max_output_tokens": 2048,
            "decoding.seed": 0,
        },
    )
    assert main(["--config", str(config), "--execute"]) == 3
    assert not log.exists()
    assert "--approved-attempts" in capsys.readouterr().err


def test_execute_is_refused_when_the_decoding_policy_is_unfrozen(tmp_path: Path) -> None:
    """A placeholder anywhere in the decoding policy still stops the run."""
    config = write_config(
        tmp_path,
        models=[{"id": "mimo-v2.5-free"}],
        **{
            "execution.order_seed": 11,
            "execution.approval_status": "APPROVED",
            "decoding.max_output_tokens": "FILL_AT_FREEZE",
        },
    )
    assert main(["--config", str(config), "--execute", "--approved-attempts", "10"]) == 2


def test_execute_and_dry_run_together_is_an_error(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    assert main(["--config", str(config), "--execute", "--dry-run"]) == 2


def test_concurrency_never_exceeds_the_gateway_limit() -> None:
    assert GATEWAY_MAX_PARALLEL == 2


# --------------------------------------------------------- resume after an outage


def _terminal(
    run_id: str,
    *,
    ok: bool,
    kind: str | None = None,
    retryable: bool | None = None,
    at: str = "2026-09-08T13:00:00Z",
):
    """A terminal attempt as the runner actually writes one, `retryable` included."""
    return {
        "run_id": run_id,
        "attempt": 2,
        "attempt_of": 2,
        "terminal": True,
        "ok": ok,
        "failure_kind": kind,
        "retryable": (not ok) if retryable is None else retryable,
        "started_at_utc": at,
    }


def test_a_plain_resume_still_skips_transport_failures(tmp_path: Path) -> None:
    """Default behaviour is unchanged: a resume must not silently re-spend budget."""
    log = tmp_path / "raw.jsonl"
    append_jsonl(log, _terminal("answered", ok=True))
    append_jsonl(log, _terminal("outage", ok=False, kind="server_unavailable:504"))

    assert completed_run_ids(log) == {"answered", "outage"}


def test_retrying_transport_failures_re_offers_only_the_failed_cells(tmp_path: Path) -> None:
    """A cell that died against an outage carries no observation about the model.

    Skipping it forever would bake an infrastructure failure into the dataset as though it
    were a model failure -- which is what the 2026-09-08 pilot outage would have done.
    """
    log = tmp_path / "raw.jsonl"
    append_jsonl(log, _terminal("answered", ok=True))
    append_jsonl(log, _terminal("outage", ok=False, kind="server_unavailable:504"))

    done = completed_run_ids(log, retry_transport_failures=True)
    assert done == {"answered"}  # the answered cell is never re-sent


def test_a_cell_recorded_ok_but_holding_an_error_envelope_is_re_offered(
    tmp_path: Path,
) -> None:
    """The pilot's poisoned MiMo rows: written before a 200 was known to hide a 502.

    These were logged ``ok`` and then graded as wrong DFA answers. They contain no model
    output at all, so re-offering them is transport retry, not content retry.
    """
    log = tmp_path / "raw.jsonl"
    poisoned = _terminal("gmicloud_502", ok=True)
    poisoned["raw_body"] = {
        "error": {"code": 502, "message": "Provider returned an empty response"}
    }
    append_jsonl(log, poisoned)

    assert "gmicloud_502" not in completed_run_ids(log, retry_transport_failures=True)
    # ...but a plain resume still leaves it alone: re-spending budget stays opt-in.
    assert "gmicloud_502" in completed_run_ids(log)


def test_a_real_answer_recorded_ok_is_still_never_re_offered(tmp_path: Path) -> None:
    """The guard on the rule above. A body with content is final, however bad."""
    log = tmp_path / "raw.jsonl"
    answered = _terminal("answered", ok=True)
    answered["raw_body"] = {
        "choices": [{"finish_reason": "stop", "message": {"content": "nonsense"}}]
    }
    append_jsonl(log, answered)

    assert "answered" in completed_run_ids(log, retry_transport_failures=True)


def test_a_cell_that_answered_is_never_re_offered(tmp_path: Path) -> None:
    """Rule 4: retry transport, never content. A returned answer is final, good or bad."""
    log = tmp_path / "raw.jsonl"
    append_jsonl(log, _terminal("bad_answer", ok=True))

    assert "bad_answer" in completed_run_ids(log, retry_transport_failures=True)


def test_the_grader_takes_the_chronologically_last_attempt(tmp_path: Path) -> None:
    """A resumed run restarts attempt numbering, so ordering must use the clock.

    Without this, a cell retried after an outage has two "attempt 2" rows and which one wins
    depends on file order -- an outage 504 could silently outrank the answer that replaced it.
    """
    from src.parse_and_grade import _terminal_attempts

    rows = [
        {"run_id": "c", "attempt": 2, "ok": False, "started_at_utc": "2026-09-08T13:00:00Z"},
        {"run_id": "c", "attempt": 1, "ok": False, "started_at_utc": "2026-09-08T12:59:00Z"},
        # the resumed invocation, later in time but numbered from 1 again
        {"run_id": "c", "attempt": 1, "ok": True, "started_at_utc": "2026-09-09T09:00:00Z"},
    ]
    last = _terminal_attempts(rows)["c"]

    assert last["ok"] is True
    assert last["started_at_utc"] == "2026-09-09T09:00:00Z"
    assert last["_attempt_count"] == 3
    assert last["_transport_failures"] == 2


def test_models_filter_restricts_without_adding(tmp_path: Path, capsys) -> None:
    """A filter for skipping a model that is down, not a way to schedule a new one."""
    log = tmp_path / "raw.jsonl"
    config = write_config(tmp_path, **{"execution.raw_log": str(log)})

    assert main(["--config", str(config), "--models", "mimo-v2.5-free"]) == 0
    out = capsys.readouterr().out
    assert "scheduled    : 96 cells" in out
    assert "restricted to mimo-v2.5-free" in out
    assert not log.exists()


def test_models_filter_rejects_a_model_the_config_does_not_declare(
    tmp_path: Path, capsys
) -> None:
    """Otherwise a typo would put cells in the log that no approved schedule contains."""
    config = write_config(tmp_path)
    assert main(["--config", str(config), "--models", "gpt-9-turbo"]) == 2
    assert "does not declare" in capsys.readouterr().err


def test_the_filter_leaves_run_ids_untouched(prompt_rows) -> None:
    """Filtering must not change identities, or a later resume would not match up."""
    cells = build_schedule(load_config(CONFIG_PATH), prompt_rows, require_frozen=True)
    subset = [c for c in cells if c.model_alias == "qwen3.8-27b"]

    assert len(subset) == 96
    assert all(cell.run_id.endswith("|qwen3.8-27b|s0") for cell in subset)


# ------------------------------------------------------- surviving a long outage


def test_a_permanently_failed_cell_is_never_re_offered(tmp_path: Path) -> None:
    """A 403 fails identically forever; re-offering it would drain a long run's ceiling."""
    log = tmp_path / "raw.jsonl"
    append_jsonl(log, {"run_id": "blocked", "attempt": 1, "terminal": True, "ok": False,
                       "retryable": False, "failure_kind": "permanent_http:403",
                       "started_at_utc": "2026-09-08T13:00:00Z"})
    append_jsonl(log, {"run_id": "outage", "attempt": 2, "terminal": True, "ok": False,
                       "retryable": True, "failure_kind": "server_unavailable:504",
                       "started_at_utc": "2026-09-08T13:00:00Z"})

    done = completed_run_ids(log, retry_transport_failures=True)
    assert "blocked" in done  # permanent: leave it failed
    assert "outage" not in done  # retryable: offer it again


def test_storm_backoff_starts_only_after_a_streak() -> None:
    from src.run_models import _GatewayHealth

    health = _GatewayHealth(threshold=3, initial_seconds=10, max_seconds=100)
    for _ in range(3):
        health.record(ok=False)

    assert health.pause_seconds() == 10  # streak just reached the threshold
    health.record(ok=False)
    assert health.pause_seconds() == 20  # and grows while the outage lasts


def test_one_success_clears_the_storm() -> None:
    from src.run_models import _GatewayHealth

    health = _GatewayHealth(threshold=2, initial_seconds=10, max_seconds=100)
    for _ in range(5):
        health.record(ok=False)
    assert health.pause_seconds() > 0

    health.record(ok=True)
    assert health.pause_seconds() == 0.0


def test_storm_backoff_is_capped() -> None:
    from src.run_models import _GatewayHealth

    health = _GatewayHealth(threshold=1, initial_seconds=10, max_seconds=45)
    for _ in range(20):
        health.record(ok=False)
    assert health.pause_seconds() == 45


def test_a_paused_run_spends_no_attempts() -> None:
    """The pause happens before the budget is taken, so waiting out an outage is free."""
    from src.run_models import _GatewayHealth

    health = _GatewayHealth(threshold=1, initial_seconds=0.01, max_seconds=0.01)
    health.record(ok=False)
    health.record(ok=False)
    budget = _AttemptBudget(10)
    backend = _ScriptedBackend([_result(ok=False, retryable=True)])
    stop = threading.Event()

    _run_cell(_cell(), backend, _policy(), budget, 2, _NO_BACKOFF, stop, health)

    assert budget.used == backend.calls  # every pause preceded a real attempt, none wasted
    assert health.total_paused_seconds > 0


# ------------------------------------------------ execution environment is config (D6)


def test_the_backend_is_built_from_the_config_not_a_hard_coded_provider(tmp_path) -> None:
    """Switching providers must be a config change with an audit trail, not a code edit."""
    from src.run_models import backend_from_config

    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1\n"
        "OPENROUTER_API_KEY=sk-or-test\n",
        encoding="utf-8",
    )
    backend = backend_from_config(load_config("configs/pilot_openrouter.yaml"), str(env))

    assert backend.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    backend.close()


def test_the_finki_configs_still_resolve_to_finki(tmp_path) -> None:
    """The default must not shift under the FINKI configs when a new provider is added."""
    from src.run_models import backend_from_config

    env = tmp_path / ".env"
    env.write_text(
        "FINKI_API_BASE_URL=https://vllm.finki.ukim.mk/v1\nFINKI_API_KEY=sk-test\n",
        encoding="utf-8",
    )
    backend = backend_from_config(load_config(CONFIG_PATH), str(env))

    assert backend.endpoint.startswith("https://vllm.finki.ukim.mk/v1")
    backend.close()


def test_the_openrouter_config_shares_the_dataset_but_not_the_raw_log() -> None:
    """Same pairs and prompts, different deployment -- so responses must not be pooled."""
    finki = load_config(CONFIG_PATH)
    other = load_config("configs/pilot_openrouter.yaml")

    assert other.path_for("pairs_dir") == finki.path_for("pairs_dir")
    assert other.path_for("renderings_dir") == finki.path_for("renderings_dir")
    assert other.get("execution.raw_log") != finki.get("execution.raw_log")
    assert other.config_id != finki.config_id  # so run_ids differ too


def test_the_openrouter_config_is_calibrated_but_unapproved() -> None:
    """Calibration settled every wire-format and control value; approval is the last gate.

    The cap is 12,288 -- 2.3x the largest untruncated completion measured across all three
    models -- applied identically to every model and R/P condition (user decision,
    2026-09-09). Approval remains the one thing code cannot supply.
    """
    config = load_config("configs/pilot_openrouter.yaml")

    # approval_status is deliberately NOT asserted here: it flips legitimately when a run is
    # approved, and the refusal paths are covered by their own tests.
    assert config.get("decoding.max_output_tokens") == 12288
    assert config.get("decoding.seed") is None  # never verified, so never sent
    for model in config.require("models"):
        assert model["answer_adapter"] == "openai_chat_content_v1"
        # Pinned per model, on measured throughput -- see test_each_model_carries_its_own_pin.
        assert model["provider_pin"] in {"DeepInfra", "GMICloud"}
        assert "FILL" not in str(model["id"])


def test_all_three_openrouter_ids_are_probe_verified(prompt_rows) -> None:
    """Every id returned HTTP 200 for that exact string on the pinned provider."""
    cells = build_schedule(
        load_config("configs/pilot_openrouter.yaml"), prompt_rows, require_frozen=True
    )
    assert {c.model_alias for c in cells} == {
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-5.3-flash",
        "xiaomi/mimo-v2.5",
    }




def test_each_model_carries_its_own_provider_pin() -> None:
    """One provider for all three cost ~35 h: DeepInfra served mimo at 6.9 tok/s where
    GMICloud managed 74.2. Each model is still pinned to exactly ONE provider, which is what
    the protocol requires; only the same-host-for-all bonus is given up."""
    from src.run_models import provider_pins

    pins = provider_pins(load_config("configs/pilot_openrouter.yaml"))

    assert pins["xiaomi/mimo-v2.5"]["provider"]["order"] == ["GMICloud"]
    assert pins["deepseek/deepseek-v4-flash-0731"]["provider"]["order"] == ["DeepInfra"]
    assert all(p["provider"]["allow_fallbacks"] is False for p in pins.values())
    assert len(pins) == 3


def test_the_pin_reaches_the_payload_per_model() -> None:
    """A pin recorded but not sent would be documentation, not a control."""
    from src.run_models import _DecodingPolicy, provider_pins

    config = load_config("configs/pilot_openrouter.yaml")
    policy = _DecodingPolicy.from_config(config, require_frozen=False)
    pins = provider_pins(config)
    payload = ChatRequest(
        model="xiaomi/mimo-v2.5", messages=({"role": "user", "content": "x"},),
        extra={**policy.extra, **pins["xiaomi/mimo-v2.5"]},
    ).payload()

    assert payload["provider"] == {"order": ["GMICloud"], "allow_fallbacks": False}


# ------------------------------------------------- repeatability audit schedule (D14)


def _audit_config(**overrides):
    from src.config import Config, load_config

    spec = {
        "enabled": True,
        "fraction": 0.2,
        "stratified_by": ["family", "depth_bin"],
        "extra_samples": 2,
        "selection_seed": 109702936,
    }
    spec.update(overrides)
    base = load_config("configs/main_frozen.yaml")
    return Config(path=base.path, data={**base.data, "samples": {"main": 1,
                                                                 "repeatability": spec}})


def _fake_pairs(n_neq: int = 100, n_eq: int = 50):
    rows = []
    for i in range(n_neq):
        rows.append({"pair_id": f"n{i:04d}", "family": "NEQ_DEPTH",
                     "oracle_depth": 1 + (i % 12)})
    for i in range(n_eq):
        rows.append({"pair_id": f"i{i:04d}", "family": "EQ_ISO", "oracle_depth": None})
        rows.append({"pair_id": f"x{i:04d}", "family": "EQ_EXPAND", "oracle_depth": None})
    return rows


def test_audit_selection_is_deterministic_and_order_independent():
    """It must regenerate from the config alone, or the audit is not reproducible."""
    from src.run_models import audit_pair_ids

    config, pairs = _audit_config(), _fake_pairs()
    first = audit_pair_ids(config, pairs)
    assert first == audit_pair_ids(config, pairs)
    assert first == audit_pair_ids(config, list(reversed(pairs)))


def test_audit_selection_covers_every_stratum_proportionally():
    """A global 20% draw could miss a depth bin entirely; stratifying cannot."""
    from src.run_models import _depth_bin, audit_pair_ids

    pairs = _fake_pairs()
    index = {row["pair_id"]: row for row in pairs}
    chosen = audit_pair_ids(_audit_config(), pairs)

    counts = collections.Counter(
        (index[p]["family"], _depth_bin(index[p]["oracle_depth"])) for p in chosen
    )
    assert set(counts) == {
        ("NEQ_DEPTH", "1-2"), ("NEQ_DEPTH", "3-4"),
        ("NEQ_DEPTH", "5-8"), ("NEQ_DEPTH", "9-12"),
        ("EQ_ISO", None), ("EQ_EXPAND", None),
    }
    for stratum, taken in counts.items():
        total = sum(
            1 for r in pairs
            if (r["family"], _depth_bin(r["oracle_depth"])) == stratum
        )
        assert taken == round(total * 0.2)


def test_audit_is_off_unless_the_config_enables_it():
    from src.run_models import audit_pair_ids

    assert audit_pair_ids(_audit_config(enabled=False), _fake_pairs()) == []
    assert audit_pair_ids(_audit_config(fraction=0.0), _fake_pairs()) == []


def test_audit_requires_a_seed():
    from src.config import ConfigError
    from src.run_models import audit_pair_ids

    with pytest.raises(ConfigError, match="selection_seed"):
        audit_pair_ids(_audit_config(selection_seed="FILL_AT_FREEZE"), _fake_pairs())


def test_repeats_get_distinct_run_ids_and_only_for_audit_pairs():
    """Repeats must be addressable separately, and must not touch other pairs."""
    from src.run_models import audit_pair_ids, build_schedule

    config, pairs = _audit_config(), _fake_pairs()
    chosen = set(audit_pair_ids(config, pairs))
    rows = [
        {"pair_id": p["pair_id"], "rendering": "R0", "prompt_condition": "P0",
         "prompt_id": "P0", "prompt_sha256": "x", "user": "u", "system": "s"}
        for p in pairs
    ]
    cells = build_schedule(config, rows, require_frozen=False, pair_records=pairs)

    by_pair = collections.defaultdict(set)
    for cell in cells:
        by_pair[cell.pair_id].add(cell.sample_index)
    for pair_id, samples in by_pair.items():
        assert samples == ({0, 1, 2} if pair_id in chosen else {0}), pair_id
    assert len({cell.run_id for cell in cells}) == len(cells)


def test_a_schedule_without_pair_records_is_exactly_the_primary_one():
    """A caller holding only prompts still gets sample 0 and nothing else."""
    from src.run_models import build_schedule

    config, pairs = _audit_config(), _fake_pairs()
    rows = [
        {"pair_id": p["pair_id"], "rendering": "R0", "prompt_condition": "P0",
         "prompt_id": "P0", "prompt_sha256": "x", "user": "u", "system": "s"}
        for p in pairs
    ]
    cells = build_schedule(config, rows, require_frozen=False)
    assert {cell.sample_index for cell in cells} == {0}
