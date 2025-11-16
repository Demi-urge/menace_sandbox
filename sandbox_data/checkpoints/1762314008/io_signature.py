from __future__ import annotations

"""Lightweight module IO signature analysis.

This module exposes :func:`get_io_signature` which parses a Python module using
:mod:`ast` to extract high level structural information.  The analysis captures
function and class signatures, global variable assignments and basic file
operations such as :func:`open` and :class:`~pathlib.Path` methods.  Results are
cached in-memory based on the file modification time to avoid redundant parsing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Set
import ast


@dataclass
class ModuleSignature:
    """Structural information about a Python module."""

    name: str = ""
    functions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    globals: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_written: Set[str] = field(default_factory=set)


def _extract_path(node: ast.AST) -> str | None:
    """Return a string path from ``Path('foo')`` like nodes."""

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path":
            if node.args and isinstance(node.args[0], ast.Constant):
                val = node.args[0].value
                if isinstance(val, str):
                    return val
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _parse_module(path: Path) -> ModuleSignature:
    sig = ModuleSignature(name=path.stem)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return sig

    tree = ast.parse(source, filename=str(path))

    # ---- top level functions, classes and globals
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args: list[str] = []
            annotations: Dict[str, str] = {}

            def _handle_arg(a: ast.arg) -> None:
                args.append(a.arg)
                if getattr(a, "annotation", None) is not None:
                    try:
                        annotations[a.arg] = ast.unparse(a.annotation)
                    except Exception:  # pragma: no cover - ast.unparse fallback
                        pass

            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                _handle_arg(a)
            if node.args.vararg:
                _handle_arg(node.args.vararg)
            if node.args.kwarg:
                _handle_arg(node.args.kwarg)

            returns = None
            if getattr(node, "returns", None) is not None:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:  # pragma: no cover - ast.unparse fallback
                    returns = None

            sig.functions[node.name] = {
                "args": args,
                "annotations": annotations,
                "returns": returns,
            }
            sig.globals.update(args)

        elif isinstance(node, ast.ClassDef):
            init_info: Dict[str, Any] | None = None
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    args: list[str] = []
                    annotations: Dict[str, str] = {}

                    def _handle_init_arg(a: ast.arg) -> None:
                        args.append(a.arg)
                        if getattr(a, "annotation", None) is not None:
                            try:
                                annotations[a.arg] = ast.unparse(a.annotation)
                            except Exception:  # pragma: no cover
                                pass

                    for a in (
                        child.args.posonlyargs
                        + child.args.args
                        + child.args.kwonlyargs
                    ):
                        _handle_init_arg(a)
                    if child.args.vararg:
                        _handle_init_arg(child.args.vararg)
                    if child.args.kwarg:
                        _handle_init_arg(child.args.kwarg)

                    returns = None
                    if getattr(child, "returns", None) is not None:
                        try:
                            returns = ast.unparse(child.returns)
                        except Exception:  # pragma: no cover
                            returns = None

                    init_info = {
                        "args": args,
                        "annotations": annotations,
                        "returns": returns,
                    }
                    break
            sig.classes[node.name] = init_info or {
                "args": [],
                "annotations": {},
                "returns": None,
            }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    sig.globals.add(t.id)

    class Visitor(ast.NodeVisitor):
        def visit_Global(self, node: ast.Global) -> None:  # noqa: D401
            sig.globals.update(node.names)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                filename = None
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        filename = arg.value
                    else:
                        filename = _extract_path(arg)
                mode = None
                if len(node.args) > 1:
                    m = node.args[1]
                    if isinstance(m, ast.Constant) and isinstance(m.value, str):
                        mode = m.value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        val = kw.value.value
                        if isinstance(val, str):
                            mode = val
                if filename:
                    if mode and any(c in mode for c in "wa+"):
                        sig.files_written.add(filename)
                    else:
                        sig.files_read.add(filename)

            elif isinstance(func, ast.Name) and func.id == "Path":
                if node.args and isinstance(node.args[0], ast.Constant):
                    val = node.args[0].value
                    if isinstance(val, str):
                        sig.files_read.add(val)

            elif isinstance(func, ast.Attribute):
                attr = func.attr
                base = func.value
                filename = _extract_path(base)
                if filename:
                    if attr in {"read_text", "read_bytes"}:
                        sig.files_read.add(filename)
                    elif attr in {"write_text", "write_bytes"}:
                        sig.files_written.add(filename)
                    elif attr == "open":
                        mode = None
                        if node.args:
                            arg = node.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                mode = arg.value
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                val = kw.value.value
                                if isinstance(val, str):
                                    mode = val
                        if mode and any(c in mode for c in "wa+"):
                            sig.files_written.add(filename)
                        else:
                            sig.files_read.add(filename)
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:  # noqa: D401
            if isinstance(node.value, str):
                val = node.value
                if "/" in val or val.endswith(
                    (".txt", ".json", ".csv", ".yaml", ".yml", ".ini", ".cfg", ".db", ".py")
                ):
                    sig.files_read.add(val)

    Visitor().visit(tree)
    return sig


_CACHE: Dict[str, tuple[float, ModuleSignature]] = {}


def get_io_signature(module_path: str | Path) -> ModuleSignature:
    """Return :class:`ModuleSignature` for ``module_path`` with caching."""

    path = Path(module_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ModuleSignature(name=path.stem)

    key = str(path.resolve())
    cached = _CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    sig = _parse_module(path)
    _CACHE[key] = (mtime, sig)
    return sig


__all__ = ["ModuleSignature", "get_io_signature"]
