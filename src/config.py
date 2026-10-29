"""Loading and validating a frozen experiment configuration.

A config is the experiment's contract. Reading one is deliberately strict: a missing key or
a leftover ``FILL_AT_FREEZE`` placeholder aborts the command rather than being defaulted
away, because a silently defaulted value would not appear in the freeze record and could not
be reconstructed afterwards (§14.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ConfigError", "PLACEHOLDERS", "load_config"]

#: Sentinels the templates use for values that must be supplied before a run.
#: Markers for a value that is deliberately not yet set. Every guard in the codebase treats
#: these as "absent" rather than as data, so a placeholder can never be sent to a model or
#: written into a result. ``FILL_AFTER_PROBE`` is distinct in meaning -- it marks a value that
#: must be *measured* against a live endpoint, not merely decided -- but identical in effect.
PLACEHOLDERS = ("FILL_AT_FREEZE", "FILL_FROM_CLUSTER", "FILL_AFTER_PROBE", "TBD")


class ConfigError(ValueError):
    """A configuration is missing, malformed, or still carries an unfilled placeholder."""


@dataclass(frozen=True, slots=True)
class Config:
    """A loaded configuration, addressed by dotted key."""

    path: Path
    data: dict[str, Any]

    @property
    def config_id(self) -> str:
        return str(self.get("config_id", "unnamed"))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up ``a.b.c``, returning ``default`` if any level is absent."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Look up ``a.b.c``, raising if it is absent or still a placeholder.

        Raises:
            ConfigError: naming the key and the config file, so the message says what to
                edit rather than merely that something was ``None``.
        """
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise ConfigError(f"{self.path}: missing required key {dotted_key!r}")
        if isinstance(value, str) and value in PLACEHOLDERS:
            raise ConfigError(
                f"{self.path}: {dotted_key!r} is still {value!r}. "
                f"Fill it in and record the decision in docs/freeze_log.md."
            )
        return value

    def require_no_placeholders(self, dotted_key: str) -> Any:
        """Like :meth:`require`, but also rejects placeholders nested anywhere inside."""
        value = self.require(dotted_key)
        offenders = sorted(_find_placeholders(value, dotted_key))
        if offenders:
            raise ConfigError(
                f"{self.path}: unfilled placeholders under {dotted_key!r}: "
                + ", ".join(offenders)
            )
        return value

    @property
    def alphabet(self) -> tuple[str, ...]:
        """The input alphabet, as an ordered tuple of single-character symbols."""
        symbols = self.require("alphabet")
        if not isinstance(symbols, list) or not symbols:
            raise ConfigError(f"{self.path}: 'alphabet' must be a non-empty list")
        if any(not isinstance(s, str) or len(s) != 1 for s in symbols):
            raise ConfigError(
                f"{self.path}: every alphabet symbol must be a single character, "
                f"got {symbols!r}"
            )
        if len(set(symbols)) != len(symbols):
            raise ConfigError(f"{self.path}: 'alphabet' has duplicate symbols: {symbols!r}")
        return tuple(symbols)

    def path_for(self, name: str) -> Path:
        """A directory from the ``paths`` section, relative to the repository root."""
        return Path(str(self.require(f"paths.{name}")))


def _find_placeholders(node: Any, prefix: str) -> list[str]:
    """Collect dotted paths of every placeholder string inside ``node``."""
    if isinstance(node, str):
        return [prefix] if node in PLACEHOLDERS else []
    if isinstance(node, dict):
        return [
            found
            for key, value in node.items()
            for found in _find_placeholders(value, f"{prefix}.{key}")
        ]
    if isinstance(node, list):
        return [
            found
            for index, value in enumerate(node)
            for found in _find_placeholders(value, f"{prefix}[{index}]")
        ]
    return []


def load_config(path: str | Path) -> Config:
    """Read and parse a YAML configuration.

    Raises:
        ConfigError: if the file is missing or does not parse to a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"no such config file: {p}")
    with open(p, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a YAML mapping at the top level")
    return Config(path=p, data=data)
