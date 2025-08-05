from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Dict

import orphan_analyzer


def _cache_path(repo: Path | str) -> Path:
    """Return path to the orphan module cache for ``repo``."""

    repo_path = Path(repo)
    data_dir = os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
    cache_dir = Path(data_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_path / cache_dir
    return cache_dir / "orphan_modules.json"


def load_orphan_cache(repo: Path | str) -> Dict[str, Dict[str, Any]]:
    """Load ``orphan_modules.json`` as a mapping.

    Older installations stored the data as a list of strings. This helper
    normalises the structure and always returns a dictionary mapping module
    paths to metadata dictionaries.
    """

    path = _cache_path(repo)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text()) or {}
    except Exception:  # pragma: no cover - best effort
        return {}
    if isinstance(data, list):
        return {str(p): {} for p in data}
    if isinstance(data, dict):
        norm: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                norm[str(k)] = {str(kk): vv for kk, vv in v.items()}
            else:
                norm[str(k)] = {}
        return norm
    return {}


def _save_orphan_cache(repo: Path | str, data: Dict[str, Dict[str, Any]]) -> None:
    path = _cache_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def append_orphan_cache(repo: Path | str, entries: Dict[str, Dict[str, Any]]) -> None:
    """Merge ``entries`` into ``orphan_modules.json`` for ``repo``."""

    if not entries:
        return
    data = load_orphan_cache(repo)
    changed = False
    for key, info in entries.items():
        cur = data.get(key, {})
        if isinstance(info, dict):
            cur.update(info)
        data[key] = cur
        changed = True
    if changed:
        _save_orphan_cache(repo, data)


def prune_orphan_cache(
    repo: Path | str,
    modules: Iterable[str],
    traces: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Remove ``modules`` from the orphan cache unless marked redundant."""

    data = load_orphan_cache(repo)
    changed = False
    for mod in modules:
        info = data.get(mod, {})
        redundant = info.get("redundant")
        if traces and mod in traces:
            redundant = traces[mod].get("redundant", redundant)
        if redundant:
            data[mod] = {"redundant": True}
        elif mod in data:
            del data[mod]
        else:
            continue
        changed = True
    if changed:
        _save_orphan_cache(repo, data)



def discover_orphan_modules(repo_path: str, recursive: bool = True) -> List[str]:
    """Return module names that are never imported by other modules.

    The function walks the repository rooted at ``repo_path`` and builds a
    simple import graph.  Modules listed in ``sandbox_data/module_map.json`` are
    ignored.  When ``recursive`` is ``True`` (the default) any modules imported
    solely by other orphan modules are also considered orphans.  Detected
    redundant modules, determined via :func:`orphan_analyzer.analyze_redundancy`,
    are filtered out.  Non-redundant orphan paths are cached in
    ``sandbox_data/orphan_modules.json`` for later inspection.
    """

    repo_path = os.path.abspath(repo_path)
    repo = Path(repo_path)

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

    # Exclude modules already known via module_map.json
    known: set[str] = set()
    try:
        repo = Path(repo_path)
        map_file = repo / "sandbox_data" / "module_map.json"
        if map_file.exists():
            data = json.loads(map_file.read_text())
            if isinstance(data, dict):
                modules_dict = data.get("modules", data)
                if isinstance(modules_dict, dict):
                    for k in modules_dict.keys():
                        p = Path(str(k))
                        name = p.with_suffix("").as_posix().replace("/", ".")
                        known.add(name)
    except Exception:  # pragma: no cover - best effort
        known = set()
    if known:
        result = [m for m in result if m not in known]

    # Filter redundant modules and cache non-redundant paths
    non_redundant: list[str] = []
    paths: list[str] = []
    for name in result:
        mod_path = repo / Path(*name.split(".")).with_suffix(".py")
        pkg_init = repo / Path(*name.split(".")) / "__init__.py"
        target = mod_path if mod_path.exists() else pkg_init
        if target.exists():
            try:
                if orphan_analyzer.analyze_redundancy(target):
                    continue
            except Exception:  # pragma: no cover - best effort
                pass
            paths.append(target.relative_to(repo).as_posix())
        non_redundant.append(name)
    result = non_redundant

    try:  # best effort cache update
        entries = {p: {} for p in paths}
        append_orphan_cache(repo, entries)
    except Exception:  # pragma: no cover - best effort
        try:
            cache = repo / "sandbox_data" / "orphan_modules.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if cache.exists():
                try:
                    existing = json.loads(cache.read_text()) or {}
                except Exception:
                    existing = {}
            if isinstance(existing, list):
                existing = {str(p): {} for p in existing}
            existing.update(entries)
            cache.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    return result



def discover_recursive_orphans(
    repo_path: str, module_map: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return orphan modules and their local dependencies.

    Modules reported by this function are orphaned within *repo_path* and are
    not known to the optional ``module_map``. Each result entry contains the
    list of orphan modules importing it under ``parents``.  Redundant modules
    identified by :func:`orphan_analyzer.analyze_redundancy` are included in the
    mapping with ``{"redundant": True}`` so callers can record and report them.
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

            is_red_child = redundant.get(name)
            if is_red_child is None:
                try:
                    is_red_child = orphan_analyzer.analyze_redundancy(target)
                except Exception:
                    is_red_child = False
                redundant[name] = is_red_child

            parents.setdefault(name, set()).add(mod)
            found.add(name)

            if name not in orphans:
                orphans.add(name)
                queue.append(name)

    result = {
        m: {
            "parents": sorted(parents.get(m, [])),
            "redundant": redundant.get(m, False),
        }
        for m in sorted(found - known)
    }

    try:  # best effort cache
        entries: Dict[str, Dict[str, Any]] = {}
        for name, info in result.items():
            mod_path = Path(*name.split(".")).with_suffix(".py")
            pkg_init = Path(*name.split(".")) / "__init__.py"
            target = mod_path if (repo / mod_path).exists() else pkg_init
            full_path = (repo / target)
            entries[full_path.relative_to(repo).as_posix()] = {
                "parents": info.get("parents", []),
                "redundant": bool(info.get("redundant")),
            }
        append_orphan_cache(repo, entries)
    except Exception:  # pragma: no cover - best effort
        pass

    return result
