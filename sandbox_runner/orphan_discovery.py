from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, List

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
        result = sorted(orphans)
    else:
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
        result = sorted(orphans)

    try:
        repo = Path(repo_path)
        data_dir = repo / "sandbox_data"
        cache = data_dir / "orphan_modules.json"
        paths = [
            Path(*name.split(".")).with_suffix(".py").as_posix() for name in result
        ]
        existing: list[str] = []
        if cache.exists():
            try:
                existing = json.loads(cache.read_text()) or []
                if not isinstance(existing, list):
                    existing = []
            except Exception:  # pragma: no cover - best effort
                existing = []
        combined = sorted(set(existing).union(paths))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(combined, indent=2))
    except Exception:  # pragma: no cover - best effort
        pass

    return result



def discover_recursive_orphans(
    repo_path: str, module_map: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return orphan modules and their local dependencies.

    Modules reported by this function are orphaned within *repo_path* and are
    not known to the optional ``module_map``. Each result entry contains the
    list of orphan modules importing it under ``parents``. Any module marked as
    redundant by :func:`orphan_analyzer.analyze_redundancy` is skipped entirely.
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

    repo_path = os.path.abspath(repo_path)
    modules: dict[str, str] = {}
    imported_by: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}

    for base, _, files in os.walk(repo_path):
        rel_base = os.path.relpath(base, repo_path)
        if rel_base.split(os.sep)[0] == "tests":
            continue
        for name in files:
            if not name.endswith(".py") or name == "__init__.py":
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
    queue = list(orphans)
    seen: set[str] = set()
    parents: dict[str, set[str]] = {m: set() for m in orphans}
    redundant: dict[str, bool] = {}
    found: set[str] = set()

    while queue:
        mod = queue.pop(0)
        if mod in seen:
            continue
        seen.add(mod)

        path = repo / Path(*mod.split(".")).with_suffix(".py")
        if not path.exists():
            path = repo / Path(*mod.split(".")) / "__init__.py"
        if not path.exists():
            continue

        is_red = redundant.get(mod)
        if is_red is None:
            try:
                is_red = orphan_analyzer.analyze_redundancy(path)
            except Exception:
                is_red = False
            redundant[mod] = is_red
        found.add(mod)
        if is_red:
            continue

        for name in imports.get(mod, set()):
            if name not in modules:
                continue
            importers = imported_by.get(name, set())
            if importers and not importers.issubset(orphans):
                continue

            mod_path = repo / Path(*name.split(".")).with_suffix(".py")
            pkg_init = repo / Path(*name.split(".")) / "__init__.py"
            target = mod_path if mod_path.exists() else pkg_init
            if not target.exists():
                continue

            is_red = redundant.get(name)
            if is_red is None:
                try:
                    is_red = orphan_analyzer.analyze_redundancy(target)
                except Exception:
                    is_red = False
                redundant[name] = is_red

            parents.setdefault(name, set()).add(mod)
            found.add(name)
            if is_red:
                continue

            if name not in orphans:
                orphans.add(name)
                queue.append(name)

    return {
        m: {
            "parents": sorted(parents.get(m, [])),
            "redundant": redundant.get(m, False),
        }
        for m in sorted(found - known)
    }
