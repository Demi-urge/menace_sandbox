"""Retry failed tests by iteratively validating and applying quick-fix patches.

This module executes a pared-down version of the self-coding bootstrap flow: it
validates the prospective patch with :func:`quick_fix.validate_patch`, applies
it via :func:`quick_fix.apply_validated_patch`, and only then retries the
failing test. Using the same validate/apply sequence as
``SelfCodingManager.run_post_patch_cycle`` surfaces schema or flag issues before
pytest runs, which keeps the feedback loop fast while ensuring patches are
properly vetted.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from menace_sandbox.context_builder import create_context_builder
from menace_sandbox.quick_fix_engine import quick_fix


class RepairLoopError(RuntimeError):
    """Raised when the repair loop cannot make forward progress."""


def _pick_first_diagnostic(results: Any) -> Mapping[str, Any]:
    diagnostics = getattr(results, "diagnostics", None) or []
    if not diagnostics:
        raise RepairLoopError("No diagnostics available to repair")

    diag = diagnostics[0]
    if not isinstance(diag, Mapping):
        raise RepairLoopError("Diagnostic entries must be mapping-like objects")

    missing_keys = {"file", "test_name", "error_summary"} - set(diag)
    if missing_keys:
        raise RepairLoopError(
            f"Diagnostic record missing required keys: {', '.join(sorted(missing_keys))}"
        )

    return diag


def _ensure_provenance(builder: Any) -> str:
    provenance = getattr(builder, "provenance_token", None)
    if not provenance:
        raise RepairLoopError("ContextBuilder did not expose a provenance token")
    return provenance


def _resolve_manager(service: Any, manager: Any | None) -> Any:
    manager = manager or getattr(service, "manager", None)
    if manager is None:
        raise RepairLoopError(
            "A SelfCodingManager instance is required for quick_fix validation"
        )
    return manager


def _discover_repo_root(target_path: Path, override: str | Path | None) -> Path:
    """Resolve the repository root used for quick-fix validation.

    ``run_post_patch_cycle`` performs its validation in a temporary clone rooted at
    the repository root. Mirroring that behaviour here avoids applying patches in
    an unrelated working directory (for example when the target file lives under
    a nested package directory) and ensures ``git checkout`` reversions operate on
    the correct worktree.
    """

    if override:
        return Path(override).resolve()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(target_path.parent),
            check=True,
            capture_output=True,
            text=True,
        )
        root = result.stdout.strip()
        if root:
            return Path(root).resolve()
    except Exception:
        # Fall back to the target file's parent directory when git metadata is
        # unavailable (for example in ephemeral tmp dirs used by tests).
        pass

    return target_path.parent.resolve()


def run_repair_loop(
    results: Any,
    service: Any,
    *,
    repo_root: str | Path | None = None,
    repair_limit: int = 3,
    manager: Any | None = None,
) -> Any:
    """Attempt to self-repair the first failing diagnostic.

    Parameters
    ----------
    results:
        A test results object exposing a ``diagnostics`` attribute with entries
        containing ``file``, ``test_name``, and ``error_summary`` keys.
    service:
        Test runner exposing ``run_once(pytest_args=[...])`` and optionally a
        ``manager`` attribute for quick-fix context.
    repo_root:
        Optional repository root. When omitted the module's parent directory is
        used.
    repair_limit:
        Maximum number of repair attempts before giving up.
    manager:
        Optional ``SelfCodingManager`` instance to supply quick-fix context.
    """

    diag = _pick_first_diagnostic(results)
    target_path = Path(diag["file"]).resolve()
    root = _discover_repo_root(target_path, repo_root)
    description = f"Fix: {diag['error_summary']} in {target_path.name}"

    context_builder = create_context_builder(repo_root=root)
    provenance = _ensure_provenance(context_builder)
    manager = _resolve_manager(service, manager)
    if hasattr(manager, "refresh_quick_fix_context"):
        manager.refresh_quick_fix_context()

    base_context: dict[str, Any] = {
        "description": description,
        "test_name": diag["test_name"],
        "error_summary": diag["error_summary"],
        "target_module": str(target_path),
    }

    for attempt in range(1, repair_limit + 1):
        print(f"\n🔁 Repair attempt {attempt}...")

        try:
            valid, validation_flags = quick_fix.validate_patch(
                module_path=str(target_path),
                description=description,
                repo_root=str(root),
                provenance_token=provenance,
                manager=manager,
                context_builder=context_builder,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper around engine failures
            raise RepairLoopError(
                f"Repair validation raised an exception on attempt {attempt}: {exc}"
            ) from exc

        flags = list(validation_flags or [])
        if not valid or flags:
            raise RepairLoopError(
                f"Repair validation failed (attempt {attempt}): {flags}"
            )

        context_meta = dict(base_context, repair_attempt=attempt)
        try:
            passed, _patch_id, apply_flags = quick_fix.apply_validated_patch(
                module_path=str(target_path),
                description=description,
                flags=flags,
                context_meta=context_meta,
                repo_root=str(root),
                provenance_token=provenance,
                manager=manager,
                context_builder=context_builder,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper around engine failures
            raise RepairLoopError(
                f"Repair application raised an exception on attempt {attempt}: {exc}"
            ) from exc

        apply_flags = list(apply_flags or [])
        if not passed or apply_flags:
            raise RepairLoopError(
                f"Patch application failed (attempt {attempt}): {apply_flags}"
            )

        results, _ = service.run_once(pytest_args=["-k", diag["test_name"]])
        if getattr(results, "fail_count", None) == 0:
            print("✅ Tests passed after repair.")
            return results

    raise RepairLoopError(
        f"❌ Repair loop exhausted after {repair_limit} attempts without passing tests."
    )


__all__ = ["run_repair_loop", "RepairLoopError", "_discover_repo_root"]

