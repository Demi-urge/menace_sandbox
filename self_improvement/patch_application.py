from __future__ import annotations

"""Patch application helpers for the self-improvement engine.

The self-improvement engine ultimately delegates patch creation to the
``patch_generation`` module.  Exposing the helper via this dedicated module
keeps the public interface lightweight and focused on applying patches rather
than the underlying generation details.
"""

from pathlib import Path
import logging
import subprocess
import tempfile
import shutil
import sys

from .patch_generation import generate_patch
from .utils import _load_callable, _call_with_retries
from menace_sandbox.context_builder import create_context_builder

try:  # pragma: no cover - prefer lightweight stub when available
    from quick_fix_engine import quick_fix  # type: ignore
except Exception:  # pragma: no cover - fallback to package import
    from menace_sandbox.quick_fix_engine import quick_fix

from menace_sandbox.sandbox_settings import SandboxSettings

try:  # pragma: no cover - fallback for flat layout
    from menace_sandbox.dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback
    from dynamic_path_router import resolve_path  # type: ignore

_settings = SandboxSettings()


def apply_patch(
    patch_id: int,
    repo_path: str | Path,
    *,
    retries: int = _settings.patch_retries,
    delay: float = _settings.patch_retry_delay,
    sign: bool = False,
) -> tuple[str, str]:
    """Fetch and apply patch ``patch_id`` to ``repo_path``.

    Returns a tuple of the commit hash created from the patch and the textual
    diff that was applied.  Raises ``RuntimeError`` if fetching or applying the
    patch fails or if the repository worktree is dirty.
    """

    logger = logging.getLogger(__name__)
    inline_fetch = getattr(sys.modules.get("quick_fix_engine"), "fetch_patch", None)
    if callable(inline_fetch):
        fetch = inline_fetch
    else:
        try:
            fetch = _load_callable("quick_fix_engine", "fetch_patch")
        except RuntimeError as exc:  # pragma: no cover - best effort logging
            logger.error("quick_fix_engine missing", exc_info=exc)
            raise RuntimeError(
                "quick_fix_engine is required for patch application. Install it via `pip install quick_fix_engine`."
            ) from exc
    try:
        patch_data = _call_with_retries(
            fetch, patch_id, retries=retries, delay=delay
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - best effort logging
        logger.error("quick_fix_engine failed", exc_info=exc)
        raise RuntimeError("quick_fix_engine failed to fetch patch") from exc
    if not patch_data:
        logger.error("quick_fix_engine returned no patch data")
        raise RuntimeError("quick_fix_engine did not return patch data")
    repo_path = resolve_path(str(repo_path))
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise RuntimeError("worktree has uncommitted changes")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    tmpdir = Path(
        tempfile.mkdtemp(prefix="apply_patch_", dir=str(repo_path.parent))
    )
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(tmpdir), head],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True,
        )
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch_data,
            text=True,
            capture_output=True,
            cwd=str(tmpdir),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"patch application failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        subprocess.run(["git", "add", "-A"], cwd=str(tmpdir), check=True)
        commit_cmd = ["git", "commit", "-m", f"Apply patch {patch_id}"]
        if sign:
            commit_cmd.append("-S")
        commit_proc = subprocess.run(
            commit_cmd, cwd=str(tmpdir), capture_output=True, text=True
        )
        if commit_proc.returncode != 0:
            raise RuntimeError(
                f"git commit failed: {commit_proc.stderr.strip() or commit_proc.stdout.strip()}"
            )
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        reset_proc = subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if reset_proc.returncode != 0:
            raise RuntimeError(
                f"failed to update worktree: {reset_proc.stderr.strip() or reset_proc.stdout.strip()}"
            )
        return commit_hash, patch_data
    finally:
        subprocess.run(
            ["git", "worktree", "remove", str(tmpdir), "--force"],
            cwd=str(repo_path),
            capture_output=True,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)


def validate_patch_with_context(
    module_path: str | Path,
    description: str,
    *,
    repo_root: str | Path,
    manager: object,
    builder: object | None = None,
    context_meta: dict | None = None,
    provenance_token: str | None = None,
    patch_logger: object | None = None,
) -> dict[str, object]:
    """Validate and apply a patch using the quick-fix pipeline.

    The call mirrors the bootstrap sequence used by
    :meth:`self_coding_manager.SelfCodingManager.run_post_patch_cycle`: first
    ``quick_fix.validate_patch`` is invoked against the target module and
    description, then ``quick_fix.apply_validated_patch`` is executed with the
    returned flags and context metadata. Surfacing validation or application
    flags early prevents expensive self-test runs when the patch is malformed
    or missing required schema fields. When a prepared context builder is
    supplied it will be reused to retain any injected retrievers or caches.
    """

    repo_root_path = Path(repo_root).resolve()
    module_path = resolve_path(str(module_path))
    if not module_path.is_absolute():
        module_path = (repo_root_path / module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"module path not found: {module_path}")

    builder = builder or create_context_builder(repo_root=repo_root_path)
    provenance = provenance_token or getattr(builder, "provenance_token", None)
    if not provenance:
        raise RuntimeError("ContextBuilder did not expose a provenance token")

    validation_result = quick_fix.validate_patch(
        module_path=str(module_path),
        description=description,
        repo_root=str(repo_root_path),
        provenance_token=provenance,
        manager=manager,
        context_builder=builder,
        patch_logger=patch_logger,
    )
    try:
        valid, flags, *extras = validation_result
    except Exception as exc:  # pragma: no cover - defensive against schema drift
        raise RuntimeError(
            "quick_fix.validate_patch returned unexpected response format"
        ) from exc
    validation_context = extras[0] if extras else None
    if not valid or flags:
        raise RuntimeError(
            f"Quick-fix validation failed with flags: {sorted(flags)}"
        )

    merged_context = {
        "description": description,
        "target_module": str(module_path),
    }
    if validation_context:
        merged_context.update(validation_context)
    if context_meta:
        merged_context.update(context_meta)

    passed, patch_id, apply_flags = quick_fix.apply_validated_patch(
        module_path=str(module_path),
        description=description,
        flags=list(flags),
        context_meta=merged_context,
        repo_root=str(repo_root_path),
        provenance_token=provenance,
        manager=manager,
        context_builder=builder,
        patch_logger=patch_logger,
    )
    if not passed or apply_flags:
        raise RuntimeError(
            f"Quick-fix application failed with flags: {sorted(apply_flags)}"
        )

    return {
        "validation_flags": list(flags),
        "apply_flags": list(apply_flags),
        "patch_id": patch_id,
        "passed": bool(passed),
    }


__all__ = ["generate_patch", "apply_patch", "validate_patch_with_context"]
