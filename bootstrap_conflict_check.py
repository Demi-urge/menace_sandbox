"""Preflight guardrails for sandbox bootstrap concurrency and watchers."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict
from typing import Iterable, Tuple

BOOTSTRAP_PROCESS_MARKERS: tuple[str, ...] = (
    "start_autonomous_sandbox.py",
    "run_autonomous.py",
    "autonomous_bootstrap",
    "sandbox_runner.py",
)

WATCHER_NAMES: tuple[str, ...] = (
    "watchman",
    "chokidar",
    "fswatch",
    "watchmedo",
    "watchexec",
    "entr",
    "inotifywait",
    "inotifywatch",
)

PROTECTED_PATH_TOKENS: tuple[str, ...] = (
    "sandbox_data",
    "checkpoint",
    "checkpoints",
    "venv",
    ".venv",
    "virtualenv",
)


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enforce_timeout_floor_envs(logger: logging.Logger | None = None) -> Dict[str, float]:
    """Clamp bootstrap timeout environment variables to escalated floors.

    Legacy service units and ".env" files occasionally inject overly short
    timeout ceilings.  To keep the hydra-style bootstrap budgets consistent,
    fetch the persisted timeout floors from :mod:`bootstrap_timeout_policy`
    and overwrite any lower environment values before orchestration begins.
    """

    try:
        from bootstrap_timeout_policy import load_escalated_timeout_floors
    except Exception as exc:  # pragma: no cover - optional dependency during linting
        if logger:
            logger.warning(
                "bootstrap_timeout_policy unavailable; skipping timeout floor enforcement: %s",
                exc,
            )
        return {}

    floors = load_escalated_timeout_floors()
    applied: Dict[str, float] = {}

    for env_var, floor in floors.items():
        raw_value = os.getenv(env_var)
        parsed = _parse_float(raw_value)

        if parsed is None or parsed < floor:
            if parsed is not None and logger:
                logger.warning(
                    "%s below escalated floor (requested=%.1fs, floor=%.1fs); overriding",
                    env_var,
                    parsed,
                    floor,
                )
            elif parsed is None and raw_value is not None and logger:
                logger.warning(
                    "%s is not a valid float (%r); enforcing escalated floor %.1fs",
                    env_var,
                    raw_value,
                    floor,
                )
            os.environ[env_var] = str(floor)
            applied[env_var] = floor
        else:
            applied[env_var] = parsed

    if logger:
        logger.info(
            "bootstrap timeout floors enforced",
            extra={"floors": floors, "applied": applied},
        )

    return applied


def _iter_processes(logger: logging.Logger | None) -> Iterable[Tuple[int, str, str]]:
    try:
        import psutil  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger:
            logger.warning("psutil unavailable; skipping conflict scan: %s", exc)
        return []

    current_pid = os.getpid()
    processes: list[Tuple[int, str, str]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info.get("pid", -1))
            if pid == current_pid:
                continue
            name = str(proc.info.get("name", "") or "").strip()
            cmdline_list = proc.info.get("cmdline") or []
            cmdline = " ".join(str(part) for part in cmdline_list if part)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover - race conditions
            continue
        except Exception:  # pragma: no cover - defensive fallback
            continue

        processes.append((pid, name, cmdline))

    return processes


def detect_conflicts(logger: logging.Logger | None = None) -> tuple[list[str], list[str]]:
    """Return bootstrap and watcher processes that conflict with startup."""

    bootstraps: list[str] = []
    watchers: list[str] = []

    for pid, name, cmdline in _iter_processes(logger):
        combined = f"{name} {cmdline}".strip()
        normalized = combined.lower()

        if any(marker in normalized for marker in BOOTSTRAP_PROCESS_MARKERS):
            bootstraps.append(f"pid {pid}: {combined}")

        watcher_match = any(marker in normalized for marker in WATCHER_NAMES)
        protected_scope = any(token in normalized for token in PROTECTED_PATH_TOKENS)
        if watcher_match and protected_scope:
            watchers.append(f"pid {pid}: {combined}")

    return bootstraps, watchers


def enforce_conflict_free_environment(
    logger: logging.Logger | None = None, *, abort_on_conflict: bool = True
) -> None:
    """Warn or abort when conflicting bootstrap/watchers are active."""

    bootstraps, watchers = detect_conflicts(logger)

    if not bootstraps and not watchers:
        if logger:
            logger.info("No conflicting bootstraps or broad watchers detected.")
        return

    segments: list[str] = []
    if bootstraps:
        segments.append(
            "conflicting sandbox bootstraps: " + "; ".join(sorted(bootstraps))
        )
    if watchers:
        segments.append(
            "filesystem watchers touch sandbox_data/checkpoints/venv: "
            + "; ".join(sorted(watchers))
        )

    message = "; ".join(segments)
    if logger:
        logger.warning(message)

    if abort_on_conflict:
        raise RuntimeError(message)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Log conflicts but exit successfully instead of aborting the bootstrap.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("bootstrap_conflict_check")

    try:
        enforce_conflict_free_environment(
            logger=logger, abort_on_conflict=not args.warn_only
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
