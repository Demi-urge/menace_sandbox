from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import List



def discover_orphan_modules(repo_path: str, recursive: bool = False) -> List[str]:
    """Return module names that are never imported by other modules.

    When ``recursive`` is ``True`` any modules imported exclusively by the
    discovered orphans are also included. The dependency walk continues until no
    new modules are found.
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



def discover_recursive_orphans(repo_path: str, module_map: str | Path | None = None) -> List[str]:
    """Return orphan modules along with local dependencies not tracked in the module map."""

    repo = Path(repo_path)
    if module_map is None:
        module_map = repo / "sandbox_data" / "module_map.json"

    known: set[str] = set()
    if module_map and Path(module_map).exists():
        try:
            data = json.loads(Path(module_map).read_text())
            if isinstance(data, dict):
                if "modules" in data:
                    keys = data.get("modules", {}).keys()
                else:
                    keys = data.keys()
                for k in keys:
                    p = Path(str(k))
                    name = p.with_suffix("").as_posix().replace("/", ".")
                    known.add(name)
        except Exception:
            known = set()

    # initial orphan set excluding modules already tracked in the module map
    found = {m for m in discover_orphan_modules(repo_path) if m not in known}
    queue = list(found)

    while queue:
        mod = queue.pop(0)
        if mod in known:
            # skip traversal for modules that are already known
            continue
        path = repo / Path(*mod.split(".")).with_suffix(".py")
        if not path.exists():
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
                    if not (mod_path.exists() or pkg_init.exists()):
                        continue
                    if name not in found and name not in known:
                        found.add(name)
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
                    if mod_path.exists() or pkg_init.exists():
                        if name not in found and name not in known:
                            found.add(name)
                            queue.append(name)
                elif node.names:
                    for alias in node.names:
                        name = ".".join(base_prefix + alias.name.split("."))
                        mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                        pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                        if mod_path.exists() or pkg_init.exists():
                            if name not in found and name not in known:
                                found.add(name)
                                queue.append(name)
    # ensure returned identifiers are relative to the repository root
    result = sorted(found - known)
    return result
