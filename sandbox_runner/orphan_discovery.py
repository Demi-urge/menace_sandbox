from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import List

import orphan_analyzer



def discover_orphan_modules(repo_path: str, recursive: bool = True) -> List[str]:
    """Return module names that are never imported by other modules.

    By default this walks the import graph recursively and includes any modules
    whose importers are exclusively within the orphan set. Set ``recursive`` to
    ``False`` to return only top-level orphans.
    """

    repo_path = os.path.abspath(repo_path)

    modules: dict[str, str] = {}
    imported_by: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}

    for base, _, files in os.walk(repo_path):
        rel_base = os.path.relpath(base, repo_path)
        if rel_base.split(os.sep)[0] == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if name == "__init__.py":
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, repo_path)
            if rel.split(os.sep)[0] == "tests":
                continue
            module = os.path.splitext(rel)[0].replace(os.sep, ".")
            try:
                text = open(path, "r", encoding="utf-8").read()
            except Exception:
                continue
            if "if __name__ == '__main__'" in text or 'if __name__ == "__main__"' in text:
                continue

            modules[module] = path

            try:
                tree = ast.parse(text)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.setdefault(module, set()).add(alias.name)
                        imported_by.setdefault(alias.name, set()).add(module)
                elif isinstance(node, ast.ImportFrom):
                    pkg_parts = module.split(".")[:-1]
                    if node.level:
                        if node.level - 1 <= len(pkg_parts):
                            base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                        else:
                            base_prefix = []
                    else:
                        base_prefix = pkg_parts

                    if node.module:
                        name = ".".join(base_prefix + node.module.split("."))
                        imports.setdefault(module, set()).add(name)
                        imported_by.setdefault(name, set()).add(module)
                    elif node.names:
                        for alias in node.names:
                            name = ".".join(base_prefix + alias.name.split("."))
                            imports.setdefault(module, set()).add(name)
                            imported_by.setdefault(name, set()).add(module)

    orphans: set[str] = {m for m in modules if m not in imported_by}
    if not recursive:
        return sorted(orphans)

    queue = list(orphans)
    while queue:
        mod = queue.pop(0)
        for name in imports.get(mod, set()):
            if name not in modules:
                continue
            importers = imported_by.get(name, set())
            if importers and not importers.issubset(orphans):
                continue
            if name not in orphans:
                orphans.add(name)
                queue.append(name)

    return sorted(orphans)



def discover_recursive_orphans(
    repo_path: str, module_map: str | Path | None = None
) -> dict[str, list[str]]:
    """Return orphan modules and their local dependencies.

    The result maps each newly discovered module to the module(s) that imported
    it. Top level orphans will have an empty list of parents. Modules already
    present in ``module_map`` are ignored in the output. Modules flagged as
    redundant or legacy by ``orphan_analyzer.analyze_redundancy`` are skipped.
    """

    repo = Path(repo_path)
    if module_map is None:
        module_map = repo / "sandbox_data" / "module_map.json"

    known: set[str] = set()
    if module_map and Path(module_map).exists():
        try:
            data = json.loads(Path(module_map).read_text())
            if isinstance(data, dict):
                modules_dict = data.get("modules", data)
                if isinstance(modules_dict, dict):
                    for k in modules_dict.keys():
                        p = Path(str(k))
                        name = p.with_suffix("").as_posix().replace("/", ".")
                        known.add(name)
        except Exception:
            known = set()

    # seed traversal with top-level orphans to follow their local dependencies
    orphans = set(discover_orphan_modules(repo_path, recursive=False))
    found: set[str] = set()
    queue = list(orphans)
    seen: set[str] = set()
    parents: dict[str, set[str]] = {m: set() for m in orphans}

    while queue:
        mod = queue.pop(0)
        if mod in seen:
            continue
        seen.add(mod)
        found.add(mod)
        path = repo / Path(*mod.split(".")).with_suffix(".py")
        if not path.exists():
            continue
        if orphan_analyzer.analyze_redundancy(path):
            continue
        try:
            text = path.read_text()
            tree = ast.parse(text)
        except Exception:
            continue
        pkg_parts = mod.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                    pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                    target = mod_path if mod_path.exists() else pkg_init
                    if not target.exists() or orphan_analyzer.analyze_redundancy(target):
                        continue
                    parents.setdefault(name, set()).add(mod)
                    if name not in seen:
                        queue.append(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level - 1 <= len(pkg_parts):
                        base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                    else:
                        base_prefix = []
                else:
                    base_prefix = pkg_parts
                if node.module:
                    name = ".".join(base_prefix + node.module.split("."))
                    mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                    pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                    target = mod_path if mod_path.exists() else pkg_init
                    if target.exists() and not orphan_analyzer.analyze_redundancy(target):
                        parents.setdefault(name, set()).add(mod)
                        if name not in seen:
                            queue.append(name)
                elif node.names:
                    for alias in node.names:
                        name = ".".join(base_prefix + alias.name.split("."))
                        mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                        pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                        target = mod_path if mod_path.exists() else pkg_init
                        if target.exists() and not orphan_analyzer.analyze_redundancy(target):
                            parents.setdefault(name, set()).add(mod)
                            if name not in seen:
                                queue.append(name)

    return {
        m: sorted(parents.get(m, []))
        for m in sorted(found - known)
    }
