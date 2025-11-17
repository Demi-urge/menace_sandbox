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
from typing import Iterable, Tuple


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


def _validate_and_apply(
    module_path: Path, description: str, repo_root: Path
) -> Tuple[bool, Iterable[str]]:
    """Run validation and apply the patch when safe."""

    from menace_sandbox.context_builder import create_context_builder
    from menace_sandbox.quick_fix_engine import quick_fix

    builder = create_context_builder(repo_root=repo_root)
    provenance = getattr(builder, "provenance_token", None)
    if not provenance:
        raise RuntimeError("ContextBuilder did not expose a provenance token")

    valid, flags = quick_fix.validate_patch(
        module_path=str(module_path),
        description=description,
        repo_root=repo_root,
        provenance_token=provenance,
        manager=None,
        context_builder=builder,
    )

    if "missing_context" in flags:
        raise RuntimeError(
            "Patch validation requires both a SelfCodingManager and ContextBuilder. "
            "The helper was unable to locate a SelfCodingManager in this environment; "
            "start the sandbox via the self-coding entrypoints or provide a manager "
            "instance when invoking quick_fix.validate_patch."
        )

    if not valid or flags:
        raise RuntimeError(
            f"Quick-fix validation failed with flags: {sorted(flags)}"
        )

    context_meta = {
        "description": description,
        "target_module": str(module_path),
    }
    passed, _patch_id, apply_flags = quick_fix.apply_validated_patch(
        module_path=str(module_path),
        description=description,
        flags=flags,
        context_meta=context_meta,
        repo_root=repo_root,
        provenance_token=provenance,
        manager=None,
        context_builder=builder,
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

    try:
        valid, flags = _validate_and_apply(module_path, args.description, repo_root)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Patch application failed: {exc}")
        sys.exit(1)

    if valid:
        print(f"✅ Patch successfully applied to {module_path}")
    else:
        print(f"[!] Patch validation failed. Flags: {list(flags)}")


if __name__ == "__main__":
    main()
