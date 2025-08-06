from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Dict

import orphan_analyzer


logger = logging.getLogger(__name__)


def _cache_path(repo: Path | str) -> Path:
    """Return path to the orphan module cache for ``repo``."""

    repo_path = Path(repo)
    data_dir = os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
    cache_dir = Path(data_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_path / cache_dir
    return cache_dir / "orphan_modules.json"


def _classification_path(repo: Path | str) -> Path:
    """Return path to the orphan classification cache for ``repo``."""

    repo_path = Path(repo)
    data_dir = os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
    cache_dir = Path(data_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_path / cache_dir
    return cache_dir / "orphan_classifications.json"


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


def append_orphan_classifications(
    repo: Path | str, entries: Dict[str, Dict[str, Any]]
) -> None:
    """Merge ``entries`` into ``orphan_classifications.json`` for ``repo``.

    Each ``entries`` item may contain ``parents``, ``classification`` and
    ``redundant`` fields which are preserved in the resulting cache so the
    classification data mirrors the information stored in
    ``orphan_modules.json``.
    """

    if not entries:
        return
    path = _classification_path(repo)
    try:
        existing = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # pragma: no cover - best effort
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    changed = False
    for key, info in entries.items():
        if not isinstance(info, dict):
            continue
        current = existing.get(key, {}) if isinstance(existing.get(key), dict) else {}
        parents = info.get("parents")
        if parents is not None:
            current["parents"] = list(parents) if isinstance(parents, (set, list, tuple)) else parents
        cls = info.get("classification")
        if cls:
            current["classification"] = cls
        if "redundant" in info:
            current["redundant"] = bool(info["redundant"])
        if current:
            existing[key] = current
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True))


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
        cls = info.get("classification") if isinstance(info, dict) else None
        if traces and mod in traces:
            redundant = traces[mod].get("redundant", redundant)
            cls = traces[mod].get("classification", cls)
        if redundant:
            entry = {"redundant": True}
            if cls:
                entry["classification"] = cls
            data[mod] = entry
        elif mod in data:
            del data[mod]
        else:
            continue
        changed = True
    if changed:
        _save_orphan_cache(repo, data)



def discover_orphan_modules(repo_path: str, recursive: bool = True) -> List[str]:
    """Return module names that are never imported by other modules.

    This is a thin wrapper around :func:`discover_recursive_orphans`.  The
    helper performs the heavy lifting of walking the repository, building the
    import graph and caching results.  Here we simply adapt its output to the
    legacy return format of a list of module names.

    When ``recursive`` is ``False`` only modules that have no orphan parents are
    returned.  Otherwise all recursively discovered orphan modules are
    included.  Modules classified as ``redundant`` or ``legacy`` by
    :func:`orphan_analyzer.classify_module` are always excluded from the
    returned list.
    """

    data = discover_recursive_orphans(repo_path)
    if recursive:
        return sorted(m for m, info in data.items() if not info.get("redundant"))
    return sorted(
        m
        for m, info in data.items()
        if not info.get("parents") and not info.get("redundant")
    )



def discover_recursive_orphans(
    repo_path: str, module_map: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return orphan modules and their local dependencies.

    Modules reported by this function are orphaned within *repo_path* and are
    not known to the optional ``module_map``. Each result entry contains the
    list of orphan modules importing it under ``parents``.  Redundant modules
    identified by :func:`orphan_analyzer.analyze_redundancy` are included in the
    mapping with their ``classification`` so callers can record and report
    them. Entries labelled ``legacy`` or ``redundant`` should typically be
    excluded from further processing.
    """

    repo = Path(repo_path).resolve()
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

    repo_path = str(repo)
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
    classifications: dict[str, str] = {}
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

        cls = classifications.get(mod)
        if cls is None:
            try:
                cls = orphan_analyzer.classify_module(path)
            except Exception:
                cls = "candidate"
            classifications[mod] = cls
        found.add(mod)
        if cls in {"legacy", "redundant"}:
            logger.info("skipping %s module %s", cls, mod)
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

            child_cls = classifications.get(name)
            if child_cls is None:
                try:
                    child_cls = orphan_analyzer.classify_module(target)
                except Exception:
                    child_cls = "candidate"
                classifications[name] = child_cls

            parents.setdefault(name, set()).add(mod)
            found.add(name)

            if child_cls in {"legacy", "redundant"}:
                logger.info("skipping %s module %s", child_cls, name)
                continue

            if name not in orphans:
                orphans.add(name)
                queue.append(name)

    result = {
        m: {
            "parents": sorted(parents.get(m, [])),
            "classification": classifications.get(m, "candidate"),
            "redundant": classifications.get(m, "candidate") != "candidate",
        }
        for m in sorted(found - known)
    }

    try:  # best effort cache
        entries: Dict[str, Dict[str, Any]] = {}
        class_entries: Dict[str, Dict[str, Any]] = {}
        for name, info in result.items():
            mod_path = Path(*name.split(".")).with_suffix(".py")
            pkg_init = Path(*name.split(".")) / "__init__.py"
            target = mod_path if (repo / mod_path).exists() else pkg_init
            full_path = (repo / target)
            rel = full_path.relative_to(repo).as_posix()
            entry = {
                "parents": info.get("parents", []),
                "classification": info.get("classification", "candidate"),
                "redundant": bool(info.get("redundant")),
            }
            entries[rel] = entry
            class_entries[rel] = dict(entry)
        append_orphan_cache(repo, entries)
        append_orphan_classifications(repo, class_entries)
    except Exception:  # pragma: no cover - best effort
        pass

    return result
