"""Resolve self-coding allow/deny lists from environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Iterable, Tuple
import logging
import os

logger = logging.getLogger(__name__)


def _parse_names(raw: str | None) -> FrozenSet[str]:
    """Return a normalised set of bot names parsed from *raw*."""

    if raw is None:
        return frozenset()
    entries = {
        item.strip()
        for item in raw.split(",")
        if item and item.strip()
    }
    return frozenset(entries)


def _parse_allowlist(raw: str | None) -> FrozenSet[str] | None:
    """Parse the allowlist env var, returning ``None`` when unconstrained."""

    names = _parse_names(raw)
    if not names or "*" in names:
        return None
    return names


@dataclass(frozen=True)
class SelfCodingPolicy:
    """Configuration describing which bots may participate in self-coding."""

    allowlist: FrozenSet[str] | None
    denylist: FrozenSet[str]

    def is_enabled(self, name: str) -> bool:
        """Return ``True`` when *name* is allowed to self-code."""

        if name in self.denylist:
            return False
        if self.allowlist is not None and name not in self.allowlist:
            return False
        return True


@dataclass(frozen=True)
class PolicySummary:
    """Snapshot of the current self-coding coverage for a set of bots."""

    allowlist: Tuple[str, ...] | None
    denylist: Tuple[str, ...]
    enabled: Tuple[str, ...]
    disabled: Tuple[str, ...]


@lru_cache(maxsize=1)
def get_self_coding_policy() -> SelfCodingPolicy:
    """Return the cached self-coding policy derived from environment state."""

    allowlist = _parse_allowlist(os.environ.get("MENACE_SELF_CODING_ALLOWLIST"))
    denylist = _parse_names(os.environ.get("MENACE_SELF_CODING_DENYLIST"))
    return SelfCodingPolicy(allowlist=allowlist, denylist=denylist)


def summarise_policy(bots: Iterable[str] | None = None) -> PolicySummary:
    """Return a summary describing how the policy applies to *bots*."""

    policy = get_self_coding_policy()
    known = set(bots or ())
    enabled = {name for name in known if policy.is_enabled(name)}
    disabled = {name for name in known if not policy.is_enabled(name)}
    if policy.denylist:
        disabled.update(policy.denylist)
    allowlist = None
    if policy.allowlist is not None:
        allowlist = tuple(sorted(policy.allowlist))
    denylist = tuple(sorted(policy.denylist))
    return PolicySummary(
        allowlist=allowlist,
        denylist=denylist,
        enabled=tuple(sorted(enabled)),
        disabled=tuple(sorted(disabled)),
    )


def log_policy_state(*, bots: Iterable[str] | None = None) -> None:
    """Emit log messages describing which bots can self-code."""

    summary = summarise_policy(bots)
    allow_desc = "*" if summary.allowlist is None else ", ".join(summary.allowlist)
    deny_desc = ", ".join(summary.denylist) if summary.denylist else "<empty>"
    logger.info(
        "Self-coding policy resolved (allowlist=%s, denylist=%s)",
        allow_desc,
        deny_desc,
    )
    for name in summary.enabled:
        logger.info("self-coding enabled for %s", name)
    for name in summary.disabled:
        if name in summary.enabled:
            continue
        logger.info("self-coding disabled for %s", name)
