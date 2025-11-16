from __future__ import annotations

"""Utilities for resolving local Python dependencies within the sandbox."""

from pathlib import Path
import ast
import os
import logging
from typing import Callable, Iterable, List, Mapping, Set, Tuple

from dynamic_path_router import resolve_path, clear_cache


DependencyCallback = Callable[[str, str, List[str]], None]
"""Callback invoked for each discovered dependency.

Parameters
----------
dep_rel: str
    The relative path of the dependency within the repository.
parent_rel: str
    The relative path of the module that imported the dependency.
chain: list[str]
    The chain of parent modules leading to ``dep_rel``.
"""

ModuleCallback = Callable[[str, Path, List[str]], None]
"""Callback invoked when a module is visited during traversal."""


logger = logging.getLogger(__name__)


def collect_local_dependencies(
    paths: Iterable[str],
    *,
    initial_parents: Mapping[str, List[str]] | None = None,
    on_module: ModuleCallback | None = None,
    on_dependency: DependencyCallback | None = None,
    max_depth: int | None = None,
    strict: bool = False,
) -> Set[str]:
    """Return modules reachable from ``paths`` following local imports.

    Parameters
    ----------
    paths:
        Iterable of file paths to modules that should serve as traversal roots.
    initial_parents:
        Optional mapping providing existing parent chains for the root modules.
    on_module:
        Optional callback invoked for each visited module. The callback receives
        ``(rel_path, absolute_path, parents)``.
    on_dependency:
        Optional callback invoked when a dependency is discovered. Receives
        ``(dep_rel_path, parent_rel_path, chain)`` where ``chain`` represents the
        import lineage from the original roots to ``dep_rel_path``.
    max_depth:
        Optional maximum dependency chain length. When set, traversal stops once
        the chain of parents reaches this depth. ``None`` means unlimited.
    strict:
        When ``True``, propagate exceptions raised by callbacks instead of
        logging and continuing.
    """

    clear_cache()
    repo = Path(resolve_path(os.getenv("SANDBOX_REPO_PATH", ".")))
    queue: List[Tuple[Path, List[str]]] = []

    def _resolve_parts(parts: List[str]) -> Path | None:
        try:
            return Path(resolve_path(Path(*parts).with_suffix(".py")))
        except FileNotFoundError:
            try:
                return Path(resolve_path(Path(*parts) / "__init__.py"))
            except FileNotFoundError:
                return None

    def _call_on_module(rel: str, path: Path, parents: List[str]) -> None:
        if on_module is None:
            return
        try:
            on_module(rel, path, parents)
        except Exception:  # pragma: no cover - best effort
            if strict:
                raise
            logger.exception("on_module callback failed for %s", rel)

    def _call_on_dependency(dep_rel: str, rel: str, dep_parents: List[str]) -> None:
        if on_dependency is None:
            return
        try:
            on_dependency(dep_rel, rel, dep_parents)
        except Exception:  # pragma: no cover - best effort
            if strict:
                raise
            logger.exception("on_dependency callback failed for %s", dep_rel)

    for m in paths:
        p = Path(m)
        if not p.is_absolute():
            p = repo / p
        try:
            rel = p.resolve().relative_to(repo).as_posix()
        except Exception:
            rel = p.as_posix()
        parents = list(initial_parents.get(rel, []) if initial_parents else [])
        queue.append((p, parents))

    seen: Set[str] = set()
    while queue:
        path, parents = queue.pop()
        if not path.exists():
            continue
        try:
            rel = path.resolve().relative_to(repo).as_posix()
        except Exception:
            rel = path.as_posix()

        _call_on_module(rel, path, parents)

        if rel in seen:
            continue
        seen.add(rel)

        if max_depth is not None and len(parents) >= max_depth:
            continue

        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue

        pkg_parts = rel.split("/")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    dep = _resolve_parts(name.split("."))
                    if dep is not None:
                        dep_rel = dep.relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        _call_on_dependency(dep_rel, rel, dep_parents)
                        queue.append((dep, dep_parents))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level - 1 <= len(pkg_parts):
                        base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                    else:
                        base_prefix = []
                else:
                    base_prefix = pkg_parts
                if node.module:
                    parts = base_prefix + node.module.split(".")
                    dep = _resolve_parts(parts)
                    if dep is not None:
                        dep_rel = dep.relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        _call_on_dependency(dep_rel, rel, dep_parents)
                        queue.append((dep, dep_parents))
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        sub_parts = parts + alias.name.split(".")
                        dep = _resolve_parts(sub_parts)
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            _call_on_dependency(dep_rel, rel, dep_parents)
                            queue.append((dep, dep_parents))
                elif node.names:
                    for alias in node.names:
                        name = ".".join(base_prefix + alias.name.split("."))
                        dep = _resolve_parts(name.split("."))
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            _call_on_dependency(dep_rel, rel, dep_parents)
                            queue.append((dep, dep_parents))
            elif isinstance(node, ast.Call):
                mod_name: str | None = None
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                    and node.args
                ):
                    arg = node.args[0]
                    if isinstance(arg, ast.Str):
                        mod_name = arg.s
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        mod_name = arg.value
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"import_module", "__import__"}
                    and node.args
                ):
                    arg = node.args[0]
                    if isinstance(arg, ast.Str):
                        mod_name = arg.s
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        mod_name = arg.value
                if mod_name:
                    parts = mod_name.split(".")
                    dep = _resolve_parts(parts)
                    if dep is not None:
                        dep_rel = dep.relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        _call_on_dependency(dep_rel, rel, dep_parents)
                        queue.append((dep, dep_parents))

    return seen
