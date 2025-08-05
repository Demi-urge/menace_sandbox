from __future__ import annotations

"""Utilities for resolving local Python dependencies within the sandbox."""

from pathlib import Path
import ast
import os
from typing import Callable, Iterable, List, Mapping, Set, Tuple


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


def collect_local_dependencies(
    paths: Iterable[str],
    *,
    initial_parents: Mapping[str, List[str]] | None = None,
    on_module: ModuleCallback | None = None,
    on_dependency: DependencyCallback | None = None,
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
    """

    repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
    queue: List[Tuple[Path, List[str]]] = []

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

        if on_module is not None:
            try:
                on_module(rel, path, parents)
            except Exception:  # pragma: no cover - best effort
                pass

        if rel in seen:
            continue
        seen.add(rel)

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
                    mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                    pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                    dep = (
                        mod_path
                        if mod_path.exists()
                        else pkg_init if pkg_init.exists() else None
                    )
                    if dep is not None:
                        dep_rel = dep.relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        if on_dependency is not None:
                            try:
                                on_dependency(dep_rel, rel, dep_parents)
                            except Exception:  # pragma: no cover - best effort
                                pass
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
                    mod_path = repo / Path(*parts).with_suffix(".py")
                    pkg_init = repo / Path(*parts) / "__init__.py"
                    dep = (
                        mod_path
                        if mod_path.exists()
                        else pkg_init if pkg_init.exists() else None
                    )
                    if dep is not None:
                        dep_rel = dep.relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        if on_dependency is not None:
                            try:
                                on_dependency(dep_rel, rel, dep_parents)
                            except Exception:  # pragma: no cover - best effort
                                pass
                        queue.append((dep, dep_parents))
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        sub_parts = parts + alias.name.split(".")
                        mod_path = repo / Path(*sub_parts).with_suffix(".py")
                        pkg_init = repo / Path(*sub_parts) / "__init__.py"
                        dep = (
                            mod_path
                            if mod_path.exists()
                            else pkg_init if pkg_init.exists() else None
                        )
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            if on_dependency is not None:
                                try:
                                    on_dependency(dep_rel, rel, dep_parents)
                                except Exception:  # pragma: no cover - best effort
                                    pass
                            queue.append((dep, dep_parents))
                elif node.names:
                    for alias in node.names:
                        name = ".".join(base_prefix + alias.name.split("."))
                        mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                        pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                        dep = (
                            mod_path
                            if mod_path.exists()
                            else pkg_init if pkg_init.exists() else None
                        )
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            if on_dependency is not None:
                                try:
                                    on_dependency(dep_rel, rel, dep_parents)
                                except Exception:  # pragma: no cover - best effort
                                    pass
                            queue.append((dep, dep_parents))

    return seen

