from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Dict, Mapping, Sequence
import concurrent.futures

import orphan_analyzer


logger = logging.getLogger(__name__)


def _resolve_assignment(
    assignments: Mapping[str, Sequence[Tuple[int, ast.AST]]], name: str, lineno: int
) -> ast.AST | None:
    """Return the value node for ``name`` assigned prior to ``lineno``."""

    values = assignments.get(name)
    if not values:
        return None
    # ``values`` is sorted in visitation order; choose last assignment before lineno
    candidate: ast.AST | None = None
    best_line = -1
    for line, node in values:
        if line < lineno and line > best_line:
            candidate, best_line = node, line
    return candidate


SAFE_CALLS: dict[tuple[str, ...], Any] = {
    ("os", "getenv"): os.getenv,
    ("os", "environ", "get"): os.environ.get,
}


def _log_unresolved(node: ast.AST, lineno: int) -> None:
    logger.debug(
        "Unresolved expression at line %s: %s",
        lineno,
        ast.dump(node, include_attributes=False),
    )


def _eval_simple(
    node: ast.AST, assignments: Mapping[str, Sequence[Tuple[int, ast.AST]]], lineno: int
) -> str | List[str] | None:
    """Evaluate *node* to a string or list of strings if possible."""

    try:
        value = ast.literal_eval(node)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return list(value)
    except Exception:
        pass

    if isinstance(node, ast.Name):
        assigned = _resolve_assignment(assignments, node.id, lineno)
        if assigned is not None:
            return _eval_simple(assigned, assignments, lineno)
        _log_unresolved(node, lineno)
        return None

    if isinstance(node, (ast.List, ast.Tuple)):
        items: List[str] = []
        for elt in node.elts:
            val = _eval_simple(elt, assignments, lineno)
            if not isinstance(val, str):
                return _log_unresolved(node, lineno)
            items.append(val)
        return items

    if isinstance(node, ast.JoinedStr):  # f-string
        parts: list[str] = []
        for value in node.values:
            part = _eval_simple(
                value.value if isinstance(value, ast.FormattedValue) else value,
                assignments,
                lineno,
            )
            if part is None:
                return _log_unresolved(node, lineno)
            parts.append(part)
        return "".join(parts)

    if isinstance(node, ast.BinOp):
        left = _eval_simple(node.left, assignments, lineno)
        right = _eval_simple(node.right, assignments, lineno)
        if left is None or right is None:
            return _log_unresolved(node, lineno)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mod):
                if isinstance(right, list):
                    return left % tuple(right)
                return left % right
        except Exception:
            return _log_unresolved(node, lineno)
        return _log_unresolved(node, lineno)

    if isinstance(node, ast.Call):
        args: List[Any] = []
        for arg in node.args:
            val = _eval_simple(arg, assignments, lineno)
            if val is None:
                return _log_unresolved(node, lineno)
            args.append(val)
        kwargs: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                return _log_unresolved(node, lineno)
            val = _eval_simple(kw.value, assignments, lineno)
            if val is None:
                return _log_unresolved(node, lineno)
            kwargs[kw.arg] = val

        func = node.func
        if isinstance(func, ast.Attribute):
            # method on evaluated base value (e.g. str.lower, str.format)
            base_val = _eval_simple(func.value, assignments, lineno)
            if isinstance(base_val, str) and hasattr(base_val, func.attr):
                try:
                    return getattr(base_val, func.attr)(*args, **kwargs)
                except Exception:
                    return _log_unresolved(node, lineno)

            # walk attribute chain to lookup in SAFE_CALLS
            path: List[str] = [func.attr]
            cur = func.value
            while isinstance(cur, ast.Attribute):
                path.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                path.append(cur.id)
                key = tuple(reversed(path))
                target = SAFE_CALLS.get(key)
                if target is not None:
                    try:
                        return target(*args, **kwargs)
                    except Exception:
                        return _log_unresolved(node, lineno)
        elif isinstance(func, ast.Name):
            key = (func.id,)
            target = SAFE_CALLS.get(key)
            if target is not None:
                try:
                    return target(*args, **kwargs)
                except Exception:
                    return _log_unresolved(node, lineno)
        return _log_unresolved(node, lineno)

    return _log_unresolved(node, lineno)


def _extract_module_from_call(
    node: ast.Call,
    assignments: Mapping[str, Sequence[Tuple[int, ast.AST]]] | None = None,
    importlib_aliases: Iterable[str] | None = None,
    import_module_aliases: Iterable[str] | None = None,
) -> str | None:
    """Return module name if *node* represents a dynamic import call."""

    importlib_names = set(importlib_aliases or {"importlib"})
    import_module_names = set(import_module_aliases or {"import_module"})
    assigns = assignments or {}

    def _attr_parts(expr: ast.AST) -> List[str]:
        parts: List[str] = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            parts.append(expr.id)
            parts.reverse()
            return parts
        return []

    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_names
        and node.func.attr == "import_module"
        and node.args
    ) or (
        isinstance(node.func, ast.Name)
        and node.func.id in import_module_names.union({"__import__"})
        and node.args
    ):
        arg = node.args[0]
        # Attempt a generic evaluation first which now includes environment
        # variable lookups handled by ``_eval_simple``.
        resolved = _eval_simple(arg, assigns, node.lineno)
        if isinstance(resolved, str):
            return resolved
        if isinstance(arg, ast.BinOp):
            if isinstance(arg.op, ast.Add):
                left = _eval_simple(arg.left, assigns, node.lineno)
                right = _eval_simple(arg.right, assigns, node.lineno)
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
            if isinstance(arg.op, ast.Mod):
                left = _eval_simple(arg.left, assigns, node.lineno)
                right = _eval_simple(arg.right, assigns, node.lineno)
                if isinstance(left, str) and right is not None:
                    try:
                        if isinstance(right, list):
                            return left % tuple(right)
                        return left % right
                    except Exception:  # pragma: no cover - best effort
                        return None
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "join"
            and not arg.keywords
            and len(arg.args) == 1
        ):
            sep = _eval_simple(arg.func.value, assigns, node.lineno)
            parts = _eval_simple(arg.args[0], assigns, node.lineno)
            if isinstance(sep, str) and isinstance(parts, list):
                if all(isinstance(p, str) for p in parts):
                    return sep.join(parts)
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "format"
        ):
            fmt = _eval_simple(arg.func.value, assigns, node.lineno)
            if isinstance(fmt, str) and not any(
                isinstance(a, ast.Starred) for a in arg.args
            ):
                args: List[str] = []
                for a in arg.args:
                    val = _eval_simple(a, assigns, node.lineno)
                    if not isinstance(val, str):
                        return None
                    args.append(val)
                kwargs: Dict[str, str] = {}
                for kw in arg.keywords:
                    if kw.arg is None:
                        return None
                    val = _eval_simple(kw.value, assigns, node.lineno)
                    if not isinstance(val, str):
                        return None
                    kwargs[kw.arg] = val
                try:
                    return fmt.format(*args, **kwargs)
                except Exception:  # pragma: no cover - best effort
                    return None

    parts = _attr_parts(node.func)
    if parts:
        root = parts[0]
        if (
            parts[-1] in {"spec_from_file_location", "spec_from_loader"}
            and (
                (len(parts) >= 3 and parts[-2] == "util" and root in importlib_names)
                or len(parts) == 1
            )
            and node.args
        ):
            mod = _eval_simple(node.args[0], assigns, node.lineno)
            if isinstance(mod, str):
                return mod
        loader_names = {"SourceFileLoader", "SourcelessFileLoader", "ExtensionFileLoader"}
        if (
            parts[-1] in loader_names
            and (
                (len(parts) >= 3 and parts[-2] == "machinery" and root in importlib_names)
                or len(parts) == 1
            )
            and node.args
        ):
            mod = _eval_simple(node.args[0], assigns, node.lineno)
            if isinstance(mod, str):
                return mod
    return None


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


def _trace_path(repo: Path | str) -> Path:
    """Return path to the orphan trace history for ``repo``."""

    repo_path = Path(repo)
    data_dir = os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
    cache_dir = Path(data_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_path / cache_dir
    return cache_dir / "orphan_traces.json"


def load_orphan_traces(repo: Path | str) -> Dict[str, Dict[str, Any]]:
    """Return trace histories stored in ``orphan_traces.json``."""

    path = _trace_path(repo)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text()) or {}
    except Exception:  # pragma: no cover - best effort
        return {}
    if isinstance(data, dict):
        norm: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                norm[str(k)] = {
                    "classification_history": list(
                        v.get("classification_history", [])
                    ),
                    "roi_history": list(v.get("roi_history", [])),
                }
        return norm
    return {}


def append_orphan_traces(
    repo: Path | str, entries: Dict[str, Dict[str, Any]]
) -> None:
    """Merge ``entries`` into ``orphan_traces.json`` for ``repo``."""

    if not entries:
        return
    data = load_orphan_traces(repo)
    changed = False
    for key, info in entries.items():
        if not isinstance(info, dict):
            continue
        current = data.get(key, {}) if isinstance(data.get(key), dict) else {}
        cls_hist = info.get("classification_history")
        if cls_hist:
            existing = list(current.get("classification_history", []))
            existing.extend(str(c) for c in cls_hist)
            current["classification_history"] = existing
        roi_hist = info.get("roi_history")
        if roi_hist:
            existing_roi = list(current.get("roi_history", []))
            existing_roi.extend(float(r) for r in roi_hist)
            current["roi_history"] = existing_roi
        if current:
            data[key] = current
            changed = True
    if changed:
        path = _trace_path(repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))


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
    for mod, info in entries.items():
        if not isinstance(info, dict):
            continue
        current = data.get(mod, {}) if isinstance(data.get(mod), dict) else {}
        parents = info.get("parents")
        if parents is not None:
            existing = (
                set(current.get("parents", []))
                if isinstance(current.get("parents"), list)
                else set()
            )
            new_parents = (
                set(parents) if isinstance(parents, (set, list, tuple)) else {parents}
            )
            current["parents"] = sorted(existing | new_parents)
        redundant = info.get("redundant")
        if redundant is None:
            cls = info.get("classification")
            redundant = cls in {"legacy", "redundant"} if cls else False
        current["redundant"] = bool(redundant)
        data[mod] = current
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
    for mod, info in entries.items():
        if not isinstance(info, dict):
            continue
        current = existing.get(mod, {}) if isinstance(existing.get(mod), dict) else {}
        parents = info.get("parents")
        if parents is not None:
            existing_parents = (
                set(current.get("parents", []))
                if isinstance(current.get("parents"), list)
                else set()
            )
            new_parents = (
                set(parents) if isinstance(parents, (set, list, tuple)) else {parents}
            )
            current["parents"] = sorted(existing_parents | new_parents)
        cls = info.get("classification")
        if cls:
            current["classification"] = cls
        redundant = info.get("redundant")
        if redundant is None and cls is not None:
            redundant = cls in {"legacy", "redundant"}
        if redundant is not None:
            current["redundant"] = bool(redundant)
        if current:
            existing[mod] = current
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



def discover_orphan_modules(
    repo_path: str, recursive: bool = True, skip_dirs: Iterable[str] | None = None
) -> List[str]:
    """Return module names that are never imported by other modules.

    This is a thin wrapper around :func:`discover_recursive_orphans`.  The
    helper performs the heavy lifting of walking the repository, building the
    import graph and caching results.  Here we simply adapt its output to the
    legacy return format of a list of module names.

    When ``recursive`` is ``False`` only modules that have no orphan parents are
    returned.  Otherwise all recursively discovered orphan modules are
    included.  Modules classified as ``redundant`` or ``legacy`` by
    :func:`orphan_analyzer.classify_module` are always excluded from the
    returned list. Directories listed in ``skip_dirs`` or provided via the
    ``SANDBOX_SKIP_DIRS`` environment variable are ignored during discovery.
    """

    data = discover_recursive_orphans(repo_path, skip_dirs=skip_dirs)
    if recursive:
        return sorted(m for m, info in data.items() if not info.get("redundant"))
    return sorted(
        m
        for m, info in data.items()
        if not info.get("parents") and not info.get("redundant")
    )


def _parse_file(args: tuple[str, str]) -> tuple[str, set[str]] | None:
    """Parse *path* for imports and return mapping.

    Returns ``None`` if the file cannot be read, parsed or should be skipped
    (e.g. it contains a ``__main__`` guard).
    """

    module, path = args
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception:
        return None

    if "if __name__ == '__main__'" in text or 'if __name__ == "__main__"' in text:
        return None

    try:
        tree = ast.parse(text)
    except Exception:
        return None

    # Collect simple assignments for later resolution of import targets
    assignments: dict[str, list[tuple[int, ast.AST]]] = {}

    class _AssignVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if len(node.targets) != 1:
                return
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append((node.lineno, node.value))

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
            if isinstance(node.target, ast.Name) and node.value is not None:
                assignments.setdefault(node.target.id, []).append((node.lineno, node.value))

    _AssignVisitor().visit(tree)

    nodes = list(ast.walk(tree))
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    for n in nodes:
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(n, ast.ImportFrom) and n.module == "importlib":
            for alias in n.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)

    imports: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
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
                imports.add(name)
            elif node.names:
                for alias in node.names:
                    name = ".".join(base_prefix + alias.name.split("."))
                    imports.add(name)
        elif isinstance(node, ast.Call):
            mod_name = _extract_module_from_call(
                node, assignments, importlib_aliases, import_module_aliases
            )
            if mod_name:
                imports.add(mod_name)

    return module, imports



def discover_recursive_orphans(
    repo_path: str,
    module_map: str | Path | None = None,
    skip_dirs: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return orphan modules and their local dependencies.

    Modules reported by this function are orphaned within *repo_path* and are
    not known to the optional ``module_map``. Each result entry contains the
    chain of orphan modules importing it under ``parents`` so callers can trace
    the origin of a dependency.  Redundancy and legacy detection is performed
    via :func:`orphan_analyzer.classify_module` and the resulting
    classification information is written to ``sandbox_data/orphan_modules.json``
    and ``sandbox_data/orphan_classifications.json``.  Entries labelled
    ``legacy`` or ``redundant`` should typically be excluded from further
    processing. Directories listed in ``skip_dirs`` or provided via the
    ``SANDBOX_SKIP_DIRS`` environment variable are ignored during the scan.
    Parallel AST parsing is controlled via the ``SANDBOX_DISCOVERY_WORKERS``
    environment variable.
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
        except Exception as exc:
            logger.warning("failed to parse module map %s: %s", module_map, exc)
            known = set()

    repo_path = str(repo)
    modules: dict[str, str] = {}
    imported_by: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}

    skip: set[str] = {"tests", ".git", ".venv"}
    extra = os.getenv("SANDBOX_SKIP_DIRS")
    if extra:
        skip.update(p for p in extra.split(os.pathsep) if p)
    if skip_dirs:
        skip.update(skip_dirs)

    candidates: list[tuple[str, str]] = []
    for base, dirs, files in os.walk(repo_path):
        rel_base = os.path.relpath(base, repo_path)
        parts = [] if rel_base in {".", ""} else rel_base.split(os.sep)
        if parts and parts[0] in skip:
            continue
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if not name.endswith(".py") or name == "__init__.py":
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, repo_path)
            if rel.split(os.sep)[0] in skip:
                continue
            module = os.path.splitext(rel)[0].replace(os.sep, ".")
            candidates.append((module, path))

    workers_env = os.getenv("SANDBOX_DISCOVERY_WORKERS")
    try:
        workers = int(workers_env) if workers_env is not None else os.cpu_count() or 1
    except ValueError:
        workers = os.cpu_count() or 1

    module_paths = {mod: p for mod, p in candidates}
    parse_iter: Iterable[tuple[str, set[str]] | None]
    if workers <= 1:
        parse_iter = (_parse_file(c) for c in candidates)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            parse_iter = ex.map(_parse_file, candidates)
            results = list(parse_iter)
        parse_iter = results

    for result in parse_iter:
        if not result:
            continue
        module, imported = result
        path = module_paths[module]
        modules[module] = path
        if imported:
            for name in imported:
                imports.setdefault(module, set()).add(name)
                imported_by.setdefault(name, set()).add(module)

    orphans: set[str] = {m for m in modules if m not in imported_by}
    queue = list(orphans)
    seen: set[str] = set()
    parents: dict[str, set[str]] = {m: set() for m in orphans}
    classifications: dict[str, str] = {}
    meta: dict[str, Dict[str, Any]] = {}
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
                cls, info = orphan_analyzer.classify_module(path, include_meta=True)
            except Exception:
                cls, info = "candidate", {}
            classifications[mod] = cls
            meta[mod] = info
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
                    child_cls, info = orphan_analyzer.classify_module(target, include_meta=True)
                except Exception:
                    child_cls, info = "candidate", {}
                classifications[name] = child_cls
                meta[name] = info

            parents.setdefault(name, set()).add(mod)
            found.add(name)

            if child_cls in {"legacy", "redundant"}:
                logger.info("skipping %s module %s", child_cls, name)
                continue

            if name not in orphans:
                orphans.add(name)
                queue.append(name)

    result: Dict[str, Dict[str, Any]] = {}
    for m in sorted(found - known):
        cls = classifications.get(m, "candidate")
        mod_path = repo / Path(*m.split(".")).with_suffix(".py")
        pkg_init = repo / Path(*m.split(".")) / "__init__.py"
        target = mod_path if mod_path.exists() else pkg_init
        try:
            redundant_flag = orphan_analyzer.analyze_redundancy(target)
        except Exception:
            redundant_flag = cls in {"legacy", "redundant"}
        result[m] = {
            "parents": sorted(parents.get(m, [])),
            "classification": cls,
            "redundant": bool(redundant_flag),
            **meta.get(m, {}),
        }

    try:  # best effort cache
        entries: Dict[str, Dict[str, Any]] = {}
        class_entries: Dict[str, Dict[str, Any]] = {}
        for name, info in result.items():
            mod_path = Path(*name.split(".")).with_suffix(".py")
            pkg_init = Path(*name.split(".")) / "__init__.py"
            target = mod_path if (repo / mod_path).exists() else pkg_init
            full_path = repo / target
            rel = full_path.relative_to(repo).as_posix()
            cls = info.get("classification", "candidate")
            redundant_flag = info.get("redundant")
            if redundant_flag is None:
                redundant_flag = cls in {"legacy", "redundant"}
            entry = {
                "parents": info.get("parents", []),
                "classification": cls,
                "redundant": bool(redundant_flag),
            }
            entries[rel] = entry
            class_entries[rel] = dict(entry)
        append_orphan_cache(repo, entries)
        append_orphan_classifications(repo, class_entries)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to update orphan cache for %s", repo)

    return result
