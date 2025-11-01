"""CLI utility to audit cooperative ``__init__`` chains in a class hierarchy.

The tool inspects the method resolution order (MRO) for a target class and
reports each class' ``__init__`` owner, signature, and whether it accepts
``*args``/``**kwargs``.  Potential hazards are highlighted when a class accepts
keyword passthrough but the next initialiser in the chain does not.

Example usage::

    python -m tools.mro_diagnostic capital_management_bot:CapitalManagementBot
    python tools/mro_diagnostic.py module.submodule.ClassName
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_object(target: str) -> type:
    """Resolve ``target`` (``module:Class`` or ``module.Class``) to a class."""

    if ":" in target:
        module_name, _, attr_path = target.partition(":")
        attrs: Sequence[str] = attr_path.split(".")
    else:
        module_name, _, attr = target.rpartition(".")
        if not module_name:
            raise SystemExit("Fully qualified module path is required")
        attrs = [attr]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = _load_module_from_path(module_name)
        if module is None:  # pragma: no cover - CLI feedback
            raise SystemExit(
                f"Failed to import module '{module_name}': module not found"
            )

    obj: object = module
    try:
        for attr in attrs:
            obj = getattr(obj, attr)
    except AttributeError as exc:  # pragma: no cover - CLI feedback
        dotted = ".".join((module_name, *attrs))
        raise SystemExit(f"Failed to resolve '{dotted}': {exc}") from exc

    if not isinstance(obj, type):
        dotted = ".".join((module_name, *attrs))
        raise SystemExit(f"'{dotted}' is not a class")
    return obj


@dataclass
class InitInfo:
    cls: type
    init_owner: type | None
    signature: str
    accepts_kwargs: bool
    accepts_args: bool
    next_owner: type | None
    next_signature: str
    next_accepts_kwargs: bool | None
    hazard: bool


def _signature_for(obj: object | None) -> str:
    if obj is None:
        return "<no __init__>"
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return "<signature unavailable>"


def _unwrap_init(cls: type) -> tuple[type | None, object | None]:
    for base in cls.__mro__:
        if "__init__" in base.__dict__:
            return base, base.__dict__["__init__"]
    return None, None


def _accepts_kwargs(obj: object | None) -> bool | None:
    if obj is None:
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _accepts_args(obj: object | None) -> bool | None:
    if obj is None:
        return None
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    return any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in sig.parameters.values())


def _build_init_info(cls: type) -> List[InitInfo]:
    mro = list(cls.__mro__)
    details: List[InitInfo] = []

    for index, base in enumerate(mro):
        owner, init_func = _unwrap_init(base)
        signature = _signature_for(init_func)
        accepts_kwargs = bool(_accepts_kwargs(init_func))
        accepts_args = bool(_accepts_args(init_func))

        next_owner: type | None = None
        next_init: object | None = None
        for candidate in mro[index + 1 :]:
            next_owner, next_init = _unwrap_init(candidate)
            if next_init is not None:
                break
        else:
            next_owner = None
            next_init = None

        next_signature = _signature_for(next_init)
        next_accepts_kwargs = _accepts_kwargs(next_init)

        hazard = bool(
            accepts_kwargs
            and next_init is not None
            and next_accepts_kwargs is False
        )

        details.append(
            InitInfo(
                cls=base,
                init_owner=owner,
                signature=signature,
                accepts_kwargs=accepts_kwargs,
                accepts_args=accepts_args,
                next_owner=next_owner,
                next_signature=next_signature,
                next_accepts_kwargs=next_accepts_kwargs,
                hazard=hazard,
            )
        )

    return details


def _ensure_parent_packages(name: str, leaf: ModuleType, origin: Path) -> None:
    parts = name.split(".")
    for index in range(1, len(parts)):
        pkg_name = ".".join(parts[:index])
        pkg = sys.modules.get(pkg_name)
        if pkg is None:
            pkg = ModuleType(pkg_name)
            depth = len(parts) - index - 1
            parent_path = origin.parents[depth]
            pkg.__path__ = [str(parent_path)]  # type: ignore[attr-defined]
            sys.modules[pkg_name] = pkg
    sys.modules[name] = leaf


def _load_module_from_path(name: str) -> ModuleType | None:
    relative = Path(*name.split("."))
    candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        _ensure_parent_packages(name, module, candidate)
        spec.loader.exec_module(module)
        return module
    return None


def render_report(target_cls: type) -> str:
    rows = _build_init_info(target_cls)
    lines = [f"MRO diagnostic for {target_cls.__module__}.{target_cls.__qualname__}"]
    for idx, info in enumerate(rows):
        lines.append(f"[{idx}] {info.cls.__module__}.{info.cls.__qualname__}")
        if info.init_owner is None:
            lines.append("    __init__: <missing>")
        else:
            owner_name = f"{info.init_owner.__module__}.{info.init_owner.__qualname__}"
            lines.append(f"    __init__ owner: {owner_name}")
            lines.append(f"    signature: {info.signature}")
            accepts_parts = [part for part in ("*args" if info.accepts_args else None, "**kwargs" if info.accepts_kwargs else None) if part]
            if accepts_parts:
                lines.append(f"    accepts: {', '.join(accepts_parts)}")
            else:
                lines.append("    accepts: <standard parameters>")

        if info.next_owner is None:
            lines.append("    next __init__: <none>")
        else:
            next_owner_name = f"{info.next_owner.__module__}.{info.next_owner.__qualname__}"
            lines.append(f"    next __init__ owner: {next_owner_name}")
            lines.append(f"    next signature: {info.next_signature}")
            if info.next_accepts_kwargs is None:
                lines.append("    next accepts kwargs: <unknown>")
            else:
                lines.append(f"    next accepts kwargs: {'yes' if info.next_accepts_kwargs else 'no'}")

        if info.hazard:
            lines.append("    ⚠️  hazard: forwards **kwargs to init without support")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="Fully qualified class path (module:ClassName or module.ClassName)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    target_cls = _load_object(args.target)
    print(render_report(target_cls))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
