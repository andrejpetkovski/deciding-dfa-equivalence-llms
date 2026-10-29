"""Execute the scheduled model calls and append every attempt to the raw log (Step 9).

Usage::

    python -m src.run_models --config configs/pilot_revised.yaml            # dry run
    python -m src.run_models --config configs/pilot_revised.yaml \\
        --execute --approved-attempts 400 --yes

**Dry run is the default and executing is deliberately awkward.** Four independent things
must all be true before a single request leaves this machine: ``--execute`` is passed, an
explicit ``--approved-attempts`` ceiling is given, the config's ``execution.approval_status``
reads ``APPROVED``, and the operator confirms the printed estimate. Any one of them missing
stops the run. Gateway calls spend a university quota that nobody here can refund, and a run
started by accident cannot be taken back out of an append-only log.

Three invariants shape the rest of the module:

- **Cells and attempts are counted separately** (§Step 9). One scheduled cell may cost
  several attempts; the ceiling and the log are in attempts, the sample size is in cells.
  Conflating them lets retries inflate n.
- **Retry transport only** (rule 4). The retry decision reads ``RawResult.retryable``, which
  is derived from a status code alone -- the answer is never available at that point.
- **Append-only, resumable** (rule 3). Every attempt is written as it completes. A resumed
  run re-reads the log and skips cells that already reached a terminal outcome; it never
  rewrites a line.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.backends.base import ChatRequest, RawResult, upstream_failure_code
from src.backends.gateway_http import GatewayHTTPBackend
from src.config import PLACEHOLDERS, Config, ConfigError, load_config
from src.io_utils import (
    append_jsonl,
    canonical_json,
    read_jsonl,
    sha256_hex,
    sha256_of_file,
)

__all__ = [
    "ScheduledCell",
    "audit_pair_ids",
    "build_schedule",
    "completed_run_ids",
    "main",
]

#: FINKI's per-key concurrency, observed as
#: ``x-ratelimit-api_key-limit-max_parallel_requests: 2`` on 2026-09-07. Exceeding it does
#: not go faster, it just converts requests into 429s that consume the attempt ceiling.
#: Used only as the default when a config does not state its provider's limit -- OpenRouter,
#: for instance, allows far more, and pinning every provider to 2 would be a FINKI artefact
#: rather than a measurement.
GATEWAY_MAX_PARALLEL = 2

#: Why the post-template input hash is null for this gateway. Recorded on every attempt so
#: the limitation travels with the data rather than living only in the docs (§5.1).
POST_TEMPLATE_HASH_REASON = (
    "gateway applies an unseen server-side chat template "
    "(x-litellm-model-api-base=opencode-adapter); exact post-template input not exposed"
)

#: Rough characters per token, used only to put an order of magnitude on the dry-run
#: estimate. It is **not** a measurement and must never be used for QA gate 8, which
#: requires real per-model tokenization (``docs/cluster.md``).
_CHARS_PER_TOKEN_ESTIMATE = 4.0


@dataclass(frozen=True, slots=True)
class ScheduledCell:
    """One (pair, representation, prompt, model, sample) unit of the design."""

    run_id: str
    pair_id: str
    rendering: str
    prompt_condition: str
    prompt_id: str
    prompt_sha256: str
    model_alias: str
    sample_index: int
    system: str | None
    user: str

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        parts = []
        if self.system is not None:
            parts.append({"role": "system", "content": self.system})
        parts.append({"role": "user", "content": self.user})
        return tuple(parts)

    @property
    def input_characters(self) -> int:
        return sum(len(message["content"]) for message in self.messages)


# --------------------------------------------------------------------------- schedule


def backend_from_config(config: Config, env_path: str) -> GatewayHTTPBackend:
    """Build the backend the *config* names, not a hard-coded provider.

    The execution environment is a recorded decision (D6), so which credentials and which
    endpoint a run uses is config, with an audit trail. Defaults keep the FINKI configs
    working unchanged.
    """
    return GatewayHTTPBackend.from_env(
        env_path,
        timeout_s=float(config.get("execution.request_timeout_seconds", 900)),
        chat_path=str(config.get("gateway.chat_path") or "/chat/completions"),
        base_url_var=str(config.get("gateway.api_base_url_env") or "FINKI_API_BASE_URL"),
        key_var=str(config.get("gateway.api_key_env") or "FINKI_API_KEY"),
    )


def _model_aliases(config: Config, *, require_frozen: bool) -> list[str]:
    """The exact request ids to send.

    ``models[].id`` is the frozen request id; ``candidate_alias`` is the dashboard label the
    probe happened to verify. A dry run may fall back to the alias so the schedule can be
    inspected before the freeze, but execution requires the frozen id -- a label that
    happened to work in a probe is not a recorded selection.
    """
    entries = config.require("models")
    aliases: list[str] = []
    for index, entry in enumerate(entries):
        frozen = entry.get("id")
        if isinstance(frozen, str) and frozen not in PLACEHOLDERS:
            aliases.append(frozen)
            continue
        if require_frozen:
            raise ConfigError(
                f"{config.path}: models[{index}].id is {frozen!r}. Fill in the verified "
                f"request id (the probe confirmed 'nemotron-3-ultra-free' and "
                f"'mimo-v2.5-free') and record the selection in docs/freeze_log.md before "
                f"executing."
            )
        alias = entry.get("candidate_alias")
        if not isinstance(alias, str) or alias in PLACEHOLDERS:
            raise ConfigError(f"{config.path}: models[{index}] has neither id nor alias")
        aliases.append(alias)
    if len(set(aliases)) != len(aliases):
        raise ConfigError(f"{config.path}: duplicate model ids in the schedule: {aliases}")
    return aliases


def audit_pair_ids(config: Config, pair_records: list[dict[str, Any]]) -> list[str]:
    """The pairs that get repeat samples, chosen exactly as the frozen config declares.

    ``samples.repeatability`` was frozen with ``fraction: 0.2``, ``stratified_by:
    [family, depth_bin]``, ``extra_samples: 2`` and its own ``selection_seed``. This
    implements that and nothing else: the selection is a seeded draw **within each stratum**,
    so the audit covers every family and every depth bin in proportion rather than landing
    wherever a global shuffle happened to fall.

    Repeats are an **operational repeatability audit** (§10.5). They are not extra
    observations, and ``analysis.metrics.primary_rows`` drops every ``sample_index > 0``
    before any primary rate is computed. Reading ``family`` and ``oracle_depth`` here decides
    *which pair to call again* and never touches prompt content, so rule 8 is untouched.

    Returns an empty list when the audit is disabled or the fraction rounds every stratum
    to zero.
    """
    spec = config.get("samples.repeatability") or {}
    if not spec.get("enabled"):
        return []
    fraction = float(spec.get("fraction") or 0.0)
    if fraction <= 0:
        return []
    seed = spec.get("selection_seed")
    if not isinstance(seed, int):
        raise ConfigError(
            f"{config.path}: samples.repeatability.selection_seed is {seed!r}. The audit "
            f"subset must be seeded, or it cannot be regenerated from the config."
        )

    strata: dict[tuple[str, str | None], list[str]] = {}
    for record in pair_records:
        depth = record.get("oracle_depth")
        stratum = (str(record.get("family")), _depth_bin(depth))
        strata.setdefault(stratum, []).append(str(record["pair_id"]))

    chosen: list[str] = []
    for stratum in sorted(strata, key=lambda k: (k[0], k[1] or "")):
        members = sorted(strata[stratum])
        take = int(round(len(members) * fraction))
        if take <= 0:
            continue
        # A per-stratum RNG derived from the frozen seed and the stratum name: adding a
        # stratum later cannot reshuffle the pairs already drawn for the others.
        rng = random.Random(f"{seed}|{stratum[0]}|{stratum[1]}")
        chosen.extend(rng.sample(members, take))
    return sorted(chosen)


def _depth_bin(depth: Any) -> str | None:
    """Depth stratum label, or ``None`` for the equivalent families, which have no depth."""
    if depth is None:
        return None
    for label, low, high in (("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9-12", 9, 12)):
        if low <= int(depth) <= high:
            return label
    return None


def build_schedule(
    config: Config,
    prompt_rows: list[dict[str, Any]],
    *,
    require_frozen: bool,
    dataset_id: str | None = None,
    pair_records: list[dict[str, Any]] | None = None,
) -> list[ScheduledCell]:
    """Every cell the design calls for, in a seeded random order.

    Ordering is randomised so that a run interrupted part-way -- by a quota, an outage, or a
    stop -- leaves a sample that is not systematically one model, one depth stratum, or one
    prompt condition (§Step 9).
    """
    dataset = dataset_id or config.config_id
    samples = int(config.get("samples.main", 1))
    aliases = _model_aliases(config, require_frozen=require_frozen)

    # Repeat samples for the frozen audit subset. Empty unless pair records are supplied,
    # so a caller that only has prompts still gets exactly the primary schedule.
    audit = set(audit_pair_ids(config, pair_records or []))
    extra = int(((config.get("samples.repeatability") or {}).get("extra_samples")) or 0)

    cells: list[ScheduledCell] = []
    for row in prompt_rows:
        for alias in aliases:
            repeats = extra if (audit and row["pair_id"] in audit) else 0
            for sample in range(samples + repeats):
                cells.append(
                    ScheduledCell(
                        run_id=(
                            f"{dataset}|{row['pair_id']}|{row['rendering']}|"
                            f"{row['prompt_condition']}|{alias}|s{sample}"
                        ),
                        pair_id=row["pair_id"],
                        rendering=row["rendering"],
                        prompt_condition=row["prompt_condition"],
                        prompt_id=row["prompt_id"],
                        prompt_sha256=row["prompt_sha256"],
                        model_alias=alias,
                        sample_index=sample,
                        system=row.get("system"),
                        user=row["user"],
                    )
                )

    order_seed = config.get("execution.order_seed")
    if isinstance(order_seed, int):
        random.Random(order_seed).shuffle(cells)
    elif require_frozen:
        raise ConfigError(
            f"{config.path}: execution.order_seed is {order_seed!r}. Call order must be "
            f"seeded so an interrupted run is not systematically biased."
        )
    else:
        cells.sort(key=lambda cell: cell.run_id)
    return cells


def verify_prompt_integrity(config: Config, prompt_rows: list[dict[str, Any]]) -> list[str]:
    """Check that the prompts about to be sent are the ones that were rendered and frozen.

    Returns a list of problems, empty when everything matches. Three separate bindings:
    each row's recorded digest must match the messages it actually carries; the frozen
    system and template files on disk must still hash to what the rendering recorded; and
    no row may be missing either.
    """
    problems: list[str] = []
    system_hashes: set[str] = set()
    template_hashes: dict[str, set[str]] = {}

    for row in prompt_rows:
        messages = []
        if row.get("system") is not None:
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["user"]})
        actual = sha256_hex(canonical_json(messages))
        if actual != row.get("prompt_sha256"):
            problems.append(
                f"{row.get('prompt_id')}: recorded prompt_sha256 {row.get('prompt_sha256')} "
                f"does not match its messages ({actual})"
            )
        if row.get("system_sha256"):
            system_hashes.add(row["system_sha256"])
        if row.get("template_sha256"):
            template_hashes.setdefault(row["prompt_condition"], set()).add(
                row["template_sha256"]
            )

    system_path = Path(str(config.require("prompts.system")))
    if system_path.is_file():
        on_disk = sha256_of_file(system_path)
        if system_hashes and on_disk not in system_hashes:
            problems.append(
                f"{system_path} hashes to {on_disk} but the renderings were built from "
                f"{sorted(system_hashes)}. Re-run render_prompts or restore the frozen text."
            )
    else:
        problems.append(f"{system_path} is missing")

    for condition in config.require("prompts.conditions"):
        path = Path(str(condition["template"]))
        recorded = template_hashes.get(condition["id"], set())
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif recorded and sha256_of_file(path) not in recorded:
            problems.append(
                f"{path} hashes to {sha256_of_file(path)} but condition "
                f"{condition['id']} was rendered from {sorted(recorded)}"
            )
    return problems


def completed_run_ids(raw_log: Path, *, retry_transport_failures: bool = False) -> set[str]:
    """Run ids that already reached a terminal outcome in the append-only log.

    A cell whose last recorded attempt was retryable and non-terminal is *not* returned, so a
    run interrupted mid-retry resumes it rather than abandoning it.

    ``retry_transport_failures`` widens that. A cell that exhausted its attempts against a
    gateway outage is "terminal" only in the sense that *this invocation* gave up on it; it
    carries no observation about the model. Skipping it forever would bake an infrastructure
    outage into the dataset as if it were a model failure. With the flag set, such cells are
    offered again -- which is still "retry transport, never content" (rule 4), because the
    decision reads the recorded failure kind and never the answer.

    It also re-offers cells that were recorded ``ok`` but whose body is an upstream failure
    envelope rather than a completion. Those were written before the runner learned that a
    200 can hide a 502 (see :func:`~src.backends.base.upstream_failure_code`); they carry no
    model output, so leaving them would bake 39 GMICloud failures into the pilot as if
    ``xiaomi/mimo-v2.5`` had answered badly.

    It is **off by default**: a plain resume must not silently re-spend budget on cells the
    operator may have intended to leave failed.
    """
    if not raw_log.is_file():
        return set()
    terminal: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(raw_log):
        if record.get("terminal") and record.get("run_id"):
            terminal[str(record["run_id"])] = record
    if not retry_transport_failures:
        return set(terminal)
    # Re-offer only cells whose terminal failure was *retryable*. A permanent one -- a 403 for
    # a blocked model, a 400 for a malformed request -- will fail identically forever, and in
    # a long grinding run it would consume the ceiling one hopeless attempt at a time.
    return {
        run_id
        for run_id, record in terminal.items()
        if (record.get("ok") and upstream_failure_code(record.get("raw_body")) is None)
        or (not record.get("ok") and not record.get("retryable"))
    }


# --------------------------------------------------------------------------- execution


class _GatewayHealth:
    """Global backoff across cells, distinct from the per-cell retry backoff.

    Per-cell backoff assumes the *cell* is unlucky. The 2026-09-08 pilot showed the other
    failure mode: the gateway itself goes down for tens of minutes and every cell fails in
    turn, at ~50 s per 504. Hammering through that converts an approved ceiling into error
    pages at about seven attempts per usable cell.

    So consecutive failures across the whole run trigger a shared pause that grows while the
    outage lasts and resets the moment anything succeeds. The run keeps its place, spends far
    fewer attempts, and comes back when the gateway does -- which is what "keep using it, even
    if it takes a long time" actually requires.
    """

    def __init__(
        self,
        threshold: int = 6,
        initial_seconds: float = 30.0,
        max_seconds: float = 600.0,
    ) -> None:
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self.threshold = threshold
        self.initial_seconds = initial_seconds
        self.max_seconds = max_seconds
        self.total_paused_seconds = 0.0

    def record(self, ok: bool) -> None:
        with self._lock:
            self._consecutive_failures = 0 if ok else self._consecutive_failures + 1

    def pause_seconds(self) -> float:
        """How long to wait before the next attempt, given the current failure streak."""
        with self._lock:
            over = self._consecutive_failures - self.threshold
        if over < 0:
            return 0.0
        return min(self.initial_seconds * (2**over), self.max_seconds)

    def wait_if_storming(self, stop: threading.Event) -> float:
        """Sleep out a storm, in short slices so a stop request is honoured promptly."""
        seconds = self.pause_seconds()
        if seconds <= 0:
            return 0.0
        remaining = seconds
        while remaining > 0 and not stop.is_set():
            slice_seconds = min(5.0, remaining)
            time.sleep(slice_seconds)
            remaining -= slice_seconds
        with self._lock:
            self.total_paused_seconds += seconds - max(remaining, 0.0)
        return seconds


class _AttemptBudget:
    """A shared, hard attempt ceiling. The only thing several worker threads contend on."""

    def __init__(self, ceiling: int) -> None:
        self._lock = threading.Lock()
        self._remaining = ceiling
        self.ceiling = ceiling

    def take(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self.ceiling - self._remaining


@dataclass(frozen=True, slots=True)
class _DecodingPolicy:
    """The frozen per-model sampling policy, identical across a model's R and P arms."""

    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None
    extra: dict[str, Any]

    @classmethod
    def from_config(cls, config: Config, *, require_frozen: bool) -> _DecodingPolicy:
        def value(key: str) -> Any:
            raw = config.get(f"decoding.{key}")
            if isinstance(raw, str) and raw in PLACEHOLDERS:
                if require_frozen:
                    raise ConfigError(
                        f"{config.path}: decoding.{key} is {raw!r}. Freeze the verified "
                        f"policy before executing; the probe confirmed only that "
                        f"temperature=0 is accepted, not that it is effective."
                    )
                return None
            return raw

        return cls(
            max_tokens=value("max_output_tokens"),
            temperature=value("temperature"),
            top_p=value("top_p"),
            seed=value("seed"),
            extra=dict(config.get("decoding.extra_body") or {}),
        )


def _attempt_record(
    cell: ScheduledCell,
    result: RawResult,
    *,
    config: Config,
    config_sha256: str,
    dataset_id: str,
    endpoint: str,
    policy: _DecodingPolicy,
    attempt_of: int,
    terminal: bool,
) -> dict[str, Any]:
    """One append-only line.

    The outbound *messages* are not duplicated here: they are already in
    ``renderings/prompts.jsonl`` and are bound to this line by ``prompt_sha256`` and
    ``request_sha256``. The decoding parameters are recorded per attempt, because those are
    what could differ between attempts and must be shown not to.
    """
    return {
        "run_id": cell.run_id,
        "dataset_id": dataset_id,
        "config_id": config.config_id,
        "config_sha256": config_sha256,
        "pair_id": cell.pair_id,
        "rendering": cell.rendering,
        "prompt_condition": cell.prompt_condition,
        "prompt_id": cell.prompt_id,
        "prompt_sha256": cell.prompt_sha256,
        "sample_index": cell.sample_index,
        "attempt": result.attempt,
        "attempt_of": attempt_of,
        "terminal": terminal,
        "endpoint": endpoint,
        "requested_model": result.requested_model,
        "returned_model": result.returned_model,
        "request_sha256": result.request_sha256,
        "decoding": {
            "max_tokens": policy.max_tokens,
            "temperature": policy.temperature,
            "top_p": policy.top_p,
            "seed": policy.seed,
            "extra": policy.extra,
        },
        "post_template_input_sha256": None,
        "post_template_input_sha256_reason": POST_TEMPLATE_HASH_REASON,
        "started_at_utc": result.started_at_utc,
        "latency_ms": round(result.latency_ms, 3),
        "status": result.status,
        "ok": result.ok,
        "failure_kind": result.failure_kind,
        "failure_detail": result.failure_detail,
        "retryable": result.retryable,
        "response_id": result.response_id,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "identity_headers": result.identity_headers(),
        "response_headers": result.response_headers,
        "raw_body": result.body,
    }


def provider_pins(config: Config) -> dict[str, dict[str, Any]]:
    """Per-model ``provider`` blocks, keyed by request id.

    One provider for every model held the serving stack constant, which is desirable -- but on
    2026-09-09 it cost about 35 hours: DeepInfra served ``xiaomi/mimo-v2.5`` at 6.9 tok/s where
    GMICloud managed 74.2. Each model is still pinned to exactly ONE provider for the whole
    study, which is what the protocol actually requires; only the "same host for all three"
    bonus is given up, and that is recorded rather than assumed away.
    """
    pins: dict[str, dict[str, Any]] = {}
    for entry in config.require("models"):
        model = entry.get("id") or entry.get("candidate_alias")
        pin = entry.get("provider_pin")
        if isinstance(model, str) and isinstance(pin, str) and pin not in PLACEHOLDERS:
            pins[model] = {"provider": {"order": [pin], "allow_fallbacks": False}}
    return pins


def _run_cell(
    cell: ScheduledCell,
    backend: GatewayHTTPBackend,
    policy: _DecodingPolicy,
    budget: _AttemptBudget,
    max_attempts: int,
    backoff: dict[str, Any],
    stop: threading.Event,
    health: _GatewayHealth | None = None,
    extra: dict[str, Any] | None = None,
) -> list[RawResult]:
    """All attempts for one cell, stopping at the first success or a permanent failure."""
    request = ChatRequest(
        model=cell.model_alias,
        messages=cell.messages,
        max_tokens=policy.max_tokens,
        temperature=policy.temperature,
        top_p=policy.top_p,
        seed=policy.seed,
        extra={**policy.extra, **(extra or {})},
    )
    results: list[RawResult] = []
    delay = float(backoff.get("initial_seconds", 2))

    for attempt in range(1, max_attempts + 1):
        if stop.is_set():
            break
        if health is not None:
            # Wait out a gateway-wide storm before spending an attempt on it. Checked before
            # the budget is taken, so a paused run consumes no ceiling.
            health.wait_if_storming(stop)
            if stop.is_set():
                break
        if not budget.take():
            break
        result = backend.complete(request, attempt=attempt)
        results.append(result)
        if health is not None:
            health.record(result.ok)
        if result.ok or not result.retryable:
            break
        if attempt < max_attempts:
            # Sleeping in the worker keeps the gateway's per-key concurrency honoured: this
            # slot stays occupied rather than being handed to another cell mid-backoff.
            time.sleep(min(delay, float(backoff.get("max_seconds", 60))))
            delay *= float(backoff.get("multiplier", 2.0))
    return results


# --------------------------------------------------------------------------- estimate


def _estimate(
    cells: list[ScheduledCell], policy: _DecodingPolicy, max_attempts: int
) -> dict[str, Any]:
    """A worst-case picture, printed before anything is sent."""
    by_model: dict[str, int] = {}
    characters = 0
    for cell in cells:
        by_model[cell.model_alias] = by_model.get(cell.model_alias, 0) + 1
        characters += cell.input_characters
    est_input = int(characters / _CHARS_PER_TOKEN_ESTIMATE)
    cap = policy.max_tokens
    return {
        "cells": len(cells),
        "cells_by_model": by_model,
        "max_attempts_per_cell": max_attempts,
        "worst_case_attempts": len(cells) * max_attempts,
        "input_characters": characters,
        "estimated_input_tokens": est_input,
        "estimated_input_tokens_note": (
            "~chars/4 client-side ESTIMATE; excludes the gateway's unseen server template "
            "(mimo-v2.5-free added ~230 tokens to a 6-token message). Not usable for QA "
            "gate 8."
        ),
        "worst_case_output_tokens": (
            len(cells) * max_attempts * int(cap) if isinstance(cap, int) else None
        ),
    }


def _print_estimate(estimate: dict[str, Any], ceiling: int | None) -> None:
    print("\nCall estimate")
    print(f"  scheduled cells          : {estimate['cells']}")
    for model, count in sorted(estimate["cells_by_model"].items()):
        print(f"    {model:<32} {count}")
    print(f"  max attempts per cell    : {estimate['max_attempts_per_cell']}")
    print(f"  worst-case attempts      : {estimate['worst_case_attempts']}")
    print(f"  input characters         : {estimate['input_characters']:,}")
    print(f"  estimated input tokens   : ~{estimate['estimated_input_tokens']:,}")
    print(f"      {estimate['estimated_input_tokens_note']}")
    worst_out = estimate["worst_case_output_tokens"]
    print(
        "  worst-case output tokens : "
        + (f"{worst_out:,}" if worst_out is not None else "unknown (max_output_tokens unset)")
    )
    if ceiling is not None:
        print(f"  approved attempt ceiling : {ceiling}")
        if ceiling < estimate["worst_case_attempts"]:
            print(
                "      NOTE: below the worst case. The run will stop at the ceiling with "
                "cells unattempted; resume after raising it."
            )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.run_models",
        description="Execute scheduled model calls (dry run unless --execute is given).",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually send requests; requires --approved-attempts and an APPROVED config",
    )
    parser.add_argument("--dry-run", action="store_true", help="explicit no-op (the default)")
    parser.add_argument(
        "--approved-attempts",
        type=int,
        help="hard ceiling on transport attempts for this invocation (docs/budget.md)",
    )
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    parser.add_argument("--limit", type=int, help="run at most this many cells (a smoke test)")
    parser.add_argument(
        "--models",
        nargs="+",
        help="restrict this invocation to a subset of the config's models, e.g. to skip one "
        "that is currently down. Never adds a model the config does not declare. Run ids are "
        "unchanged, so the remaining models are picked up by a later resume.",
    )
    parser.add_argument(
        "--retry-transport-failures",
        action="store_true",
        help="also re-attempt cells whose terminal outcome was a transport failure, e.g. "
        "after a gateway outage. Never re-attempts a cell that returned an answer.",
    )
    parser.add_argument("--env", default=".env", help="path to the credentials file")
    args = parser.parse_args(argv)

    if args.execute and args.dry_run:
        print("--execute and --dry-run are mutually exclusive.", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        renderings_dir = config.path_for("renderings_dir")
        prompts_path = renderings_dir / "prompts.jsonl"
        if not prompts_path.is_file():
            print(
                f"{prompts_path} not found. Run python -m src.render_prompts first.",
                file=sys.stderr,
            )
            return 1
        prompt_rows = list(read_jsonl(prompts_path))
        # Pair records carry family and oracle_depth, which the frozen repeatability spec
        # stratifies on. They never reach a prompt -- only the choice of which pair to
        # sample again (rule 8).
        pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
        pair_records = list(read_jsonl(pairs_path)) if pairs_path.is_file() else []
        cells = build_schedule(
            config, prompt_rows, require_frozen=args.execute, pair_records=pair_records
        )
        policy = _DecodingPolicy.from_config(config, require_frozen=args.execute)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    if args.models:
        # A filter, never an addition: running a model the config does not declare would put
        # cells in the log that no approved schedule contains.
        declared = {cell.model_alias for cell in cells}
        unknown = sorted(set(args.models) - declared)
        if unknown:
            print(
                f"--models names {unknown} which the config does not declare "
                f"(declared: {sorted(declared)}).",
                file=sys.stderr,
            )
            return 2
        cells = [cell for cell in cells if cell.model_alias in set(args.models)]

    raw_log = Path(str(config.require("execution.raw_log")))
    done = completed_run_ids(
        raw_log, retry_transport_failures=args.retry_transport_failures
    )
    pending = [cell for cell in cells if cell.run_id not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    max_attempts = int(config.get("execution.max_transport_attempts", 5))
    estimate = _estimate(pending, policy, max_attempts)

    print(f"config       : {config.path} ({config.config_id})")
    print(f"prompts      : {prompts_path} ({len(prompt_rows)} rows)")
    print(f"raw log      : {raw_log}")
    print(f"scheduled    : {len(cells)} cells; already terminal: {len(done)}; "
          f"pending: {len(pending)}")
    if args.models:
        print(f"               (restricted to {', '.join(sorted(args.models))}; the rest of "
              f"the schedule is left for a later resume)")
    if args.retry_transport_failures:
        print("               (transport-failed cells are being re-offered; cells that "
              "returned an answer are not)")

    problems = verify_prompt_integrity(config, prompt_rows)
    if problems:
        print("\nPrompt integrity FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        if args.execute:
            print(
                "\nRefusing to execute against prompts that do not match the frozen text.",
                file=sys.stderr,
            )
            return 1
    else:
        print("prompt integrity : OK (messages, system and templates all match)")

    _print_estimate(estimate, args.approved_attempts)

    if not args.execute:
        print("\nDRY RUN — nothing was sent.")
        approval = str(config.get("execution.approval_status", "NOT_APPROVED"))
        if approval == "APPROVED":
            print("The config is APPROVED. To execute, add: --execute --approved-attempts <N>")
        else:
            print(
                f"execution.approval_status is {approval!r}. Record the approval in "
                f"docs/budget.md, set it to APPROVED, then pass "
                f"--execute --approved-attempts <N>."
            )
        return 0

    # ---------------------------------------------------------------- execution gates
    approval = str(config.get("execution.approval_status", "NOT_APPROVED"))
    if approval != "APPROVED":
        print(
            f"\nexecution.approval_status is {approval!r}, not 'APPROVED'. "
            f"Record the approval in docs/budget.md and the config first.",
            file=sys.stderr,
        )
        return 3
    if args.approved_attempts is None or args.approved_attempts <= 0:
        print("\n--approved-attempts <N> is required with --execute.", file=sys.stderr)
        return 3
    if not pending:
        print("\nNothing pending; the schedule is already complete.")
        return 0

    if not args.yes:
        portal = config.get("gateway.portal_url")
        print(f"\nSend up to {args.approved_attempts} attempts to {portal}?")
        try:
            answer = input("Type 'yes' to proceed: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("Aborted; nothing was sent.")
            return 0

    try:
        backend = backend_from_config(config, args.env)
    except RuntimeError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2

    # The ceiling is the provider's, and the provider is a config decision. A config that
    # states its own limit is trusted; one that does not falls back to FINKI's measured 2.
    stated = config.get("gateway.max_parallel_requests")
    limit = stated if isinstance(stated, int) and stated > 0 else GATEWAY_MAX_PARALLEL
    configured = config.get("execution.max_concurrent_requests")
    concurrency = limit
    if isinstance(configured, int) and 0 < configured <= limit:
        concurrency = configured
    elif isinstance(configured, int) and configured > limit:
        print(
            f"execution.max_concurrent_requests={configured} exceeds this provider's stated "
            f"limit of {limit}; using {limit}."
        )

    pins = provider_pins(config)
    if pins:
        print("provider pins  : " + ", ".join(
            f"{m} -> {p['provider']['order'][0]}" for m, p in sorted(pins.items())))
    budget = _AttemptBudget(args.approved_attempts)
    health = _GatewayHealth()
    stop = threading.Event()
    config_sha = sha256_of_file(config.path)
    dataset_id = config.config_id
    written = 0
    succeeded = 0

    print(f"\nExecuting: {len(pending)} cells, concurrency {concurrency}, "
          f"ceiling {args.approved_attempts} attempts.")
    try:
        with backend, ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _run_cell,
                    cell,
                    backend,
                    policy,
                    budget,
                    max_attempts,
                    dict(config.get("execution.backoff") or {}),
                    stop,
                    health,
                    pins.get(cell.model_alias),
                ): cell
                for cell in pending
            }
            for future in as_completed(futures):
                cell = futures[future]
                results = future.result()
                for index, result in enumerate(results):
                    append_jsonl(
                        raw_log,
                        _attempt_record(
                            cell,
                            result,
                            config=config,
                            config_sha256=config_sha,
                            dataset_id=dataset_id,
                            endpoint=backend.endpoint,
                            policy=policy,
                            attempt_of=max_attempts,
                            terminal=index == len(results) - 1
                            and (result.ok or not result.retryable
                                 or result.attempt >= max_attempts),
                        ),
                    )
                    written += 1
                if results and results[-1].ok:
                    succeeded += 1
                if budget.used >= budget.ceiling:
                    stop.set()
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted; attempts already made are in the log.", file=sys.stderr)

    print(f"\nattempts written : {written} (ceiling {budget.ceiling}, used {budget.used})")
    if health.total_paused_seconds:
        print(
            f"storm pauses     : {health.total_paused_seconds / 60:.0f} min waited rather "
            f"than spent on a gateway that was down"
        )
    print(f"cells succeeded  : {succeeded} / {len(pending)}")
    print(f"raw log          : {raw_log}")
    if stop.is_set():
        print("Stopped at the attempt ceiling. Re-run to resume the remaining cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
