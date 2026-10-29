"""Assembly of the exact prompt text sent to a model (§8).

Loads the frozen system message, prompt templates, and JSON schemas, and substitutes the two
DFA renderings into a template. Nothing here decides anything about the experiment; it exists
so that the bytes a model receives are produced in one place, hashed, and checkable against
the freeze record.

Two rules hold the whole module together:

- **Substitute with :meth:`str.replace`, never :meth:`str.format` or an f-string.** The
  templates contain literal JSON braces such as ``{"equivalent": true|false}``; ``format``
  would read those as fields and either raise or silently corrupt the frozen prompt text.
- **The prompt carries the machines and nothing else.** No ``pair_id``, family, seed, depth,
  oracle verdict, or witness may appear (research rule 8). :func:`assert_no_leak` enforces
  this on every assembled prompt, not just in a test.

Pure module apart from reading the frozen prompt files named by the config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PLACEHOLDERS, Config, ConfigError
from src.io_utils import canonical_json, read_text, sha256_hex

__all__ = [
    "PLACEHOLDER_A",
    "PLACEHOLDER_B",
    "Prompt",
    "PromptAssets",
    "PromptCondition",
    "assert_no_leak",
    "build_messages",
    "build_user_message",
    "load_prompt_assets",
]

PLACEHOLDER_A = "{DFA_A_RENDERING}"
PLACEHOLDER_B = "{DFA_B_RENDERING}"

#: Words that must never reach a prompt. ``witness`` is deliberately absent: both templates
#: legitimately use it to name the output field, so flagging it would fire on frozen text.
#: The witness *value* is checked separately, and by value rather than by name.
_FORBIDDEN_WORDS = ("pair_id", "oracle", "family", "seed", "depth", "canonical_hash")

#: A shorter witness would collide with ordinary alphabet symbols in the transition table.
_MIN_CHECKABLE_WITNESS = 3


@dataclass(frozen=True, slots=True)
class Prompt:
    """One assembled prompt: a system message and a user message."""

    system: str | None
    user: str

    @property
    def messages(self) -> list[dict[str, str]]:
        """Chat messages in the order a provider expects them."""
        parts = []
        if self.system is not None:
            parts.append({"role": "system", "content": self.system})
        parts.append({"role": "user", "content": self.user})
        return parts

    @property
    def sha256(self) -> str:
        """Digest over the assembled messages.

        This identifies the prompt *this code* produced. It is not the digest of what the
        model finally saw: a chat template is applied server-side and wraps these messages in
        model-specific role markers. The runner records that second hash separately
        (``docs/cluster.md`` §5.1).
        """
        return sha256_hex(canonical_json(self.messages))


@dataclass(frozen=True, slots=True)
class PromptCondition:
    """A frozen prompt condition: its template text and its strict response schema."""

    id: str
    name: str
    template: str
    template_path: str
    template_sha256: str
    schema: dict[str, Any]
    schema_path: str


@dataclass(frozen=True, slots=True)
class PromptAssets:
    """Everything the frozen config declares about prompting."""

    system: str
    system_path: str
    system_sha256: str
    conditions: tuple[PromptCondition, ...]
    system_placement: str

    def condition(self, condition_id: str) -> PromptCondition:
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        raise KeyError(f"no prompt condition {condition_id!r}")


# --------------------------------------------------------------------------- loading


def _verify_declared_hash(
    config: Config, key: str, path: str, actual: str
) -> None:
    """Check a file against the hash the config froze for it, if one is recorded.

    §14.2 requires the prompt text to be frozen *and hashed*. Recording a hash and never
    checking it would make the freeze decorative: this is what actually detects a prompt
    edited after the tag.
    """
    declared = config.get(key)
    if declared is None or str(declared) in PLACEHOLDERS:
        return
    if str(declared) != actual:
        raise ConfigError(
            f"{path} does not match the hash frozen in {config.path} under {key!r}.\n"
            f"  frozen: {declared}\n"
            f"  actual: {actual}\n"
            f"The prompt text is frozen; changing it requires a new freeze tag and a note "
            f"in docs/freeze_log.md."
        )


def load_prompt_assets(config: Config) -> PromptAssets:
    """Read the system message, templates, and schemas named by the config.

    Raises:
        ConfigError: if a file is missing, a template lacks a placeholder, or a file's
            content no longer matches a hash frozen in the config.
    """
    system_path = str(config.require("prompts.system"))
    if not Path(system_path).is_file():
        raise ConfigError(f"system prompt not found: {system_path}")
    system = read_text(system_path)
    system_sha = sha256_hex(system)
    _verify_declared_hash(config, "prompts.system_sha256", system_path, system_sha)

    declared = config.require("prompts.conditions")
    if not isinstance(declared, list) or not declared:
        raise ConfigError(f"{config.path}: prompts.conditions must be a non-empty list")

    conditions: list[PromptCondition] = []
    for entry in declared:
        template_path = str(entry["template"])
        schema_path = str(entry["schema"])
        for path in (template_path, schema_path):
            if not Path(path).is_file():
                raise ConfigError(f"prompt asset not found: {path}")

        template = read_text(template_path)
        for placeholder in (PLACEHOLDER_A, PLACEHOLDER_B):
            if placeholder not in template:
                raise ConfigError(
                    f"{template_path} is missing the {placeholder} placeholder"
                )
        template_sha = sha256_hex(template)
        # Conditions are a list, so the dotted lookup _verify_declared_hash uses cannot
        # address them; the frozen hash is checked directly off the entry instead.
        frozen = str(entry.get("template_sha256", ""))
        if frozen not in ("", *PLACEHOLDERS) and frozen != template_sha:
            raise ConfigError(
                f"{template_path} does not match its frozen template_sha256 in "
                f"{config.path}.\n  frozen: {frozen}\n  actual: {template_sha}\n"
                f"The prompt text is frozen; changing it requires a new freeze tag and a "
                f"note in docs/freeze_log.md."
            )

        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)

        conditions.append(
            PromptCondition(
                id=str(entry["id"]),
                name=str(entry.get("name", entry["id"])),
                template=template,
                template_path=template_path,
                template_sha256=template_sha,
                schema=schema,
                schema_path=schema_path,
            )
        )

    placement = str(config.get("prompts.system_placement", "system_role"))
    if placement not in ("system_role", "prepend_to_user"):
        raise ConfigError(
            f"{config.path}: prompts.system_placement must be 'system_role' or "
            f"'prepend_to_user', got {placement!r}"
        )

    return PromptAssets(
        system=system,
        system_path=system_path,
        system_sha256=system_sha,
        conditions=tuple(conditions),
        system_placement=placement,
    )


# --------------------------------------------------------------------------- assembly


def build_user_message(template: str, rendering_a: str, rendering_b: str) -> str:
    """Substitute the two renderings into a template.

    Uses :meth:`str.replace` rather than :meth:`str.format`, because the templates contain
    literal JSON braces that ``format`` would treat as replacement fields.

    Raises:
        ValueError: if a placeholder is missing, or if one survives substitution -- either
            would mean a model was about to be shown a template rather than a problem.
    """
    for placeholder in (PLACEHOLDER_A, PLACEHOLDER_B):
        if placeholder not in template:
            raise ValueError(f"template is missing the {placeholder} placeholder")

    user = template.replace(PLACEHOLDER_A, rendering_a).replace(
        PLACEHOLDER_B, rendering_b
    )
    for placeholder in (PLACEHOLDER_A, PLACEHOLDER_B):
        if placeholder in user:
            raise ValueError(f"{placeholder} survived substitution")
    return user


def build_messages(
    system: str,
    template: str,
    rendering_a: str,
    rendering_b: str,
    system_placement: str = "system_role",
) -> Prompt:
    """Assemble one prompt from the frozen system message, template, and renderings.

    Args:
        system_placement: ``"system_role"`` sends the system message as its own turn;
            ``"prepend_to_user"`` folds it into the user turn instead. Some open-weight chat
            templates reject a system role, but the choice is a **frozen, study-wide**
            setting rather than a per-model fallback: §8.4 requires the interface to be
            identical across conditions and models, so silently folding for one model would
            introduce an interface confound rather than fix one.
    """
    user = build_user_message(template, rendering_a, rendering_b)
    if system_placement == "prepend_to_user":
        return Prompt(system=None, user=f"{system.rstrip()}\n\n{user}")
    if system_placement == "system_role":
        return Prompt(system=system, user=user)
    raise ValueError(f"unknown system_placement {system_placement!r}")


# --------------------------------------------------------------------------- leak check


def assert_no_leak(prompt: Prompt, record: dict[str, Any]) -> None:
    """Fail loudly if experiment metadata reached the text a model will read.

    The failure this guards against would not announce itself: a prompt that names its
    stratum, or contains the answer, still parses and still grades, and would silently
    invalidate every number in the paper (research rule 8).

    Raises:
        ValueError: naming what leaked.
    """
    text = "\n".join(message["content"] for message in prompt.messages)
    lowered = text.lower()

    for word in _FORBIDDEN_WORDS:
        if word in lowered:
            raise ValueError(f"prompt contains the forbidden word {word!r}")

    pair_id = str(record.get("pair_id", ""))
    if pair_id and pair_id.lower() in lowered:
        raise ValueError(f"prompt contains its pair_id {pair_id!r}")

    family = str(record.get("family", ""))
    if family and family.lower() in lowered:
        raise ValueError(f"prompt contains its family {family!r}")

    seed = record.get("seed")
    if seed is not None and str(seed) in text:
        raise ValueError("prompt contains its generation seed")

    witness = record.get("oracle_witness")
    if (
        isinstance(witness, str)
        and len(witness) >= _MIN_CHECKABLE_WITNESS
        and witness in text
    ):
        raise ValueError("prompt contains the oracle witness")
