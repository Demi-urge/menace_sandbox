"""Resolve self-coding allow/deny lists from environment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Iterable, Mapping, Tuple
import logging
import os

from menace_sandbox.stabilization.roi import evaluate_roi_delta_policy

logger = logging.getLogger(__name__)


_DEFAULT_SELF_CODING_UNSAFE_PATHS: Tuple[str, ...] = (
    "reward_dispatcher.py",
    "reward_sanity_checker.py",
    "kpi_reward_core.py",
    "mvp_evaluator.py",
    "kpi_editing_detector.py",
    "menace/core/evaluator.py",
    "billing/billing_ledger.py",
    "billing/stripe_ledger.py",
    "payout",
    "payouts",
    "ledger",
)


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


def _parse_decimal(raw: str | None, default: Decimal) -> Decimal:
    if raw is None:
        return default
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return default


def _parse_paths(raw: str | None) -> Tuple[Path, ...]:
    if raw is None:
        return tuple()
    paths = []
    for item in raw.split(","):
        value = item.strip()
        if value:
            paths.append(Path(value).expanduser())
    return tuple(paths)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _as_repo_relative(path: Path, *, repo_root: Path) -> str | None:
    candidate = path if path.is_absolute() else (repo_root / path)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return None
    normalised = relative.as_posix().strip("/")
    if normalised in {"", "."}:
        return ""
    return normalised


def _rule_matches_target(*, target: str, rule: str) -> bool:
    cleaned_rule = rule.strip("/")
    if not cleaned_rule:
        return True
    return target == cleaned_rule or target.startswith(f"{cleaned_rule}/")


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


@dataclass(frozen=True)
class PatchPromotionPolicy:
    """Policy describing when self-debug patches can be promoted."""

    min_roi_delta: Decimal
    safe_roots: Tuple[Path, ...]
    deny_roots: Tuple[Path, ...]
    repo_root: Path | None = None

    def is_safe_target(self, path: Path) -> bool:
        repo_root = self.repo_root.resolve() if self.repo_root is not None else Path.cwd().resolve()
        target = path if path.is_absolute() else (repo_root / path)
        relative_target = _as_repo_relative(target, repo_root=repo_root)
        if relative_target is None:
            return False

        safe_rules = tuple(
            rule
            for rule in (_as_repo_relative(root, repo_root=repo_root) for root in self.safe_roots)
            if rule is not None
        )
        if not safe_rules:
            return False
        if not any(
            _rule_matches_target(target=relative_target, rule=rule)
            for rule in safe_rules
        ):
            return False

        deny_rules = tuple(
            rule
            for rule in (_as_repo_relative(root, repo_root=repo_root) for root in self.deny_roots)
            if rule is not None
        )
        if any(
            _rule_matches_target(target=relative_target, rule=rule)
            for rule in deny_rules
        ):
            return False
        return True


@dataclass(frozen=True)
class PatchPromotionDecision:
    allowed: bool
    reasons: Tuple[str, ...]
    roi_delta_total: Decimal | None


def get_patch_promotion_policy(repo_root: Path | None = None) -> PatchPromotionPolicy:
    min_delta = _parse_decimal(
        os.environ.get("MENACE_SELF_CODING_MIN_ROI_DELTA"), Decimal("0")
    )
    safe_roots = _parse_paths(os.environ.get("MENACE_SELF_CODING_SAFE_PATHS"))
    unsafe_raw = os.environ.get("MENACE_SELF_CODING_UNSAFE_PATHS")
    if unsafe_raw is None:
        deny_roots = tuple(Path(path) for path in _DEFAULT_SELF_CODING_UNSAFE_PATHS)
    else:
        deny_roots = _parse_paths(unsafe_raw)
    resolved_repo_root = repo_root.resolve() if repo_root is not None else None
    if not safe_roots and resolved_repo_root is not None:
        safe_roots = (resolved_repo_root,)
    return PatchPromotionPolicy(
        min_roi_delta=min_delta,
        safe_roots=safe_roots,
        deny_roots=deny_roots,
        repo_root=resolved_repo_root,
    )


def evaluate_patch_promotion(
    *,
    policy: PatchPromotionPolicy,
    roi_delta: Mapping[str, object] | None,
    patch_validation: Mapping[str, object] | None,
    source_path: Path,
) -> PatchPromotionDecision:
    reasons: list[str] = []
    roi_result = evaluate_roi_delta_policy(roi_delta, min_delta=policy.min_roi_delta)
    if not roi_result.ok:
        reasons.append(roi_result.reason or "roi_delta_invalid")
    if not patch_validation or not patch_validation.get("valid"):
        reasons.append("invalid_patch")
    if not policy.is_safe_target(source_path):
        reasons.append("unsafe_target")
    return PatchPromotionDecision(
        allowed=not reasons,
        reasons=tuple(reasons),
        roi_delta_total=roi_result.total,
    )
