"""Convenience wrapper around :mod:`quick_fix_engine` patch validation.

Running this module from the repository root now behaves like a small CLI:

```
python patch_application.py --module path/to/file.py --description "Fix bug"
```

The script wires up the shared :func:`context_builder.create_context_builder`
helper and forwards requests to :func:`quick_fix_engine.validate_patch`. When
validation succeeds it immediately applies the validated patch, mirroring the
behaviour used by the self-coding tooling while keeping the entry point small
and explicit.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


def _add_repo_to_syspath(repo_root: Path) -> None:
    """Ensure the repository parent is on ``sys.path`` for package imports."""

    parent = str(repo_root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Validate and apply a quick-fix patch")
    parser.add_argument(
        "--module",
        dest="module_path",
        required=True,
        help="Target module path (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Human-friendly description of the change",
    )
    return parser.parse_args()


def _resolve_module_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"module path does not exist: {path}")
    return path


def _parse_validation_result(result: Any) -> tuple[bool, list[str], dict[str, Any]]:
    """Normalize quick-fix validation output.

    ``quick_fix.validate_patch`` historically returned a ``(valid, flags)`` tuple
    but newer variants may append a context mapping. This helper accepts either
    shape and always returns a ``(valid, flags, context)`` triple, raising
    :class:`RuntimeError` when the payload cannot be interpreted.
    """

    try:
        valid, flags, *rest = result
    except Exception as exc:  # pragma: no cover - defensive against schema drift
        raise RuntimeError(
            "quick_fix.validate_patch returned unexpected response format"
        ) from exc

    context = rest[0] if rest else {}
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise RuntimeError("Validation context must be mapping-like")

    return bool(valid), list(flags or []), dict(context)


def _parse_apply_result(result: Any) -> tuple[bool, Any, list[str]]:
    """Normalize quick-fix application output and surface schema mismatches."""

    try:
        passed, patch_id, flags = result
    except Exception as exc:  # pragma: no cover - defensive against schema drift
        raise RuntimeError(
            "quick_fix.apply_validated_patch returned unexpected response format"
        ) from exc

    return bool(passed), patch_id, list(flags or [])


def _validate_and_apply(
    module_path: Path,
    description: str,
    repo_root: Path,
    *,
    manager: object,
    builder: object | None = None,
    context_meta: dict[str, object] | None = None,
) -> tuple[bool, list[str]]:
    """Run quick-fix validation and apply the patch when safe.

    This helper mirrors the bootstrap flow used by
    ``SelfCodingManager.run_post_patch_cycle``: it validates the prospective
    patch via ``quick_fix.validate_patch`` using the active manager and then
    immediately calls ``quick_fix.apply_validated_patch`` with the returned
    flags and any validation context. Failing fast on validation or application
    flags keeps subsequent self-tests from running with a malformed patch.
    """

    from menace_sandbox.context_builder import create_context_builder
    from menace_sandbox.quick_fix_engine import quick_fix

    if manager is None:
        raise RuntimeError(
            "A SelfCodingManager instance is required for quick-fix validation"
        )

    builder = builder or getattr(manager, "context_builder", None)
    builder = builder or create_context_builder(repo_root=repo_root)
    provenance = getattr(builder, "provenance_token", None)
    if not provenance:
        raise RuntimeError("ContextBuilder did not expose a provenance token")

    validation_result = quick_fix.validate_patch(
        module_path=str(module_path),
        description=description,
        repo_root=repo_root,
        provenance_token=provenance,
        manager=manager,
        context_builder=builder,
    )
    valid, flags, validation_context = _parse_validation_result(validation_result)

    merged_context = {
        "description": description,
        "target_module": str(module_path),
    }
    if validation_context:
        merged_context.update(validation_context)
    if context_meta:
        merged_context.update(context_meta)

    passed, _patch_id, apply_flags = _parse_apply_result(
        quick_fix.apply_validated_patch(
            module_path=str(module_path),
            description=description,
            flags=flags,
            context_meta=merged_context,
            repo_root=repo_root,
            provenance_token=provenance,
            manager=manager,
            context_builder=builder,
        )
    )

    if not valid or flags:
        raise RuntimeError(
            f"Quick-fix validation failed with flags: {sorted(flags)}"
        )

    if not passed or apply_flags:
        raise RuntimeError(
            f"Quick-fix application failed with flags: {sorted(apply_flags)}"
        )

    return valid, flags


def main() -> None:
    repo_root = Path(__file__).parent.resolve()
    _add_repo_to_syspath(repo_root)
    args = _parse_args()
    module_path = _resolve_module_path(repo_root, args.module_path)

    # ``patch_application`` is primarily intended for use inside an existing
    # self-coding sandbox where a ``SelfCodingManager`` has already been
    # instantiated. Without that manager quick-fix validation will return a
    # ``missing_context`` flag. The CLI therefore refuses to proceed when a
    # manager is not available rather than attempting a best-effort stub.
    manager = getattr(sys.modules.get("__main__"), "manager", None)
    if manager is None:
        print(
            "Patch application requires an active SelfCodingManager. "
            "Run this module from within the bootstrap process or use the "
            "`menace_cli patch` entrypoint to construct the manager first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        valid, flags = _validate_and_apply(
            module_path,
            args.description,
            repo_root,
            manager=manager,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Patch application failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if valid:
        print(f"✅ Patch successfully applied to {module_path}")
    else:
        print(f"[!] Patch validation failed. Flags: {list(flags)}")


if __name__ == "__main__":
    main()
