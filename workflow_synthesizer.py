from __future__ import annotations

"""Lightweight workflow synthesizer utilities.

This module exposes :class:`WorkflowSynthesizer` which combines structural
signals from :class:`~module_synergy_grapher.ModuleSynergyGrapher` with
semantic intent matches provided by :class:`~intent_clusterer.IntentClusterer`.

The synthesizer is intentionally small and focuses on expanding an initial set
of modules either by following the synergy graph around a starting module or by
searching for modules related to a textual problem description.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# lightweight structural analysis helpers
try:  # Optional import; makes the synthesizer work in minimal environments
    from analysis import ModuleSignature, get_io_signature
except Exception:  # pragma: no cover - best effort fall back
    ModuleSignature = None  # type: ignore[misc]
    get_io_signature = None  # type: ignore[misc]

try:  # Optional imports; fall back to stubs in tests
    from module_synergy_grapher import (
        ModuleSynergyGrapher,
        get_synergy_cluster,
        load_graph,
    )
except Exception:  # pragma: no cover - graceful degradation
    ModuleSynergyGrapher = None  # type: ignore[misc]
    get_synergy_cluster = None  # type: ignore[misc]
    load_graph = None  # type: ignore[misc]

try:  # Optional dependency
    from intent_clusterer import IntentClusterer
except Exception:  # pragma: no cover - graceful degradation
    IntentClusterer = None  # type: ignore[misc]

try:  # Optional lightweight fallback
    from intent_db import IntentDB  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    IntentDB = None  # type: ignore[misc]


@dataclass
class ModuleIO:
    """Basic structural information about a Python module."""

    name: str = ""
    functions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    globals: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_written: Set[str] = field(default_factory=set)


@dataclass
class WorkflowStep:
    """Lightweight representation of an ordered workflow step."""

    module: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


def _parse_module_io(path: Path) -> ModuleIO:
    """Parse ``path`` and extract high level IO information."""

    module_io = ModuleIO(name=path.stem)

    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return module_io

    tree = ast.parse(source, filename=str(path))

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

    # ---- top level function definitions and globals
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args: List[str] = []
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

            module_io.functions[node.name] = {
                "args": args,
                "annotations": annotations,
                "returns": returns,
            }
            module_io.inputs.update(args)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    module_io.globals.add(t.id)

    class Visitor(ast.NodeVisitor):
        def visit_Global(self, node: ast.Global) -> None:  # noqa: D401
            module_io.globals.update(node.names)

        def visit_Return(self, node: ast.Return) -> None:  # noqa: D401
            targets: list[str] = []
            val = node.value
            if isinstance(val, ast.Name):
                targets.append(val.id)
            elif isinstance(val, ast.Tuple):
                for elt in val.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)
            module_io.outputs.update(targets)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                filename = None
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        filename = arg.value
                    elif isinstance(arg, ast.Name):
                        filename = arg.id
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
                        module_io.files_written.add(filename)
                    else:
                        module_io.files_read.add(filename)

            elif isinstance(func, ast.Name) and func.id == "Path":
                if node.args and isinstance(node.args[0], ast.Constant):
                    val = node.args[0].value
                    if isinstance(val, str):
                        module_io.files_read.add(val)

            elif isinstance(func, ast.Attribute):
                attr = func.attr
                base = func.value
                filename = _extract_path(base)
                if filename:
                    if attr in {"read_text", "read_bytes"}:
                        module_io.files_read.add(filename)
                    elif attr in {"write_text", "write_bytes"}:
                        module_io.files_written.add(filename)
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
                            module_io.files_written.add(filename)
                        else:
                            module_io.files_read.add(filename)

            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:  # noqa: D401
            if isinstance(node.value, str):
                val = node.value
                if "/" in val or val.endswith(
                    (".txt", ".json", ".csv", ".yaml", ".yml", ".ini", ".cfg", ".db", ".py")
                ):
                    module_io.files_read.add(val)

    Visitor().visit(tree)
    return module_io


_ANALYZE_CACHE: Dict[str, Tuple[float, ModuleIO]] = {}


def analyze_io(module_path: Path) -> ModuleIO:
    """Return :class:`ModuleIO` information for ``module_path`` with caching."""

    try:
        mtime = module_path.stat().st_mtime
    except OSError:
        return ModuleIO()

    key = str(module_path.resolve())
    cached = _ANALYZE_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    io = _parse_module_io(module_path)
    _ANALYZE_CACHE[key] = (mtime, io)
    return io


class ModuleIOAnalyzer:
    """Analyze modules to determine their IO signatures with caching."""

    def __init__(self, cache_path: Path = Path("sandbox_data/io_signatures.json")) -> None:
        self.cache_path = cache_path
        try:
            self._cache: Dict[str, Dict[str, Any]] = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
        except Exception:  # pragma: no cover - empty or corrupt cache
            self._cache = {}

    # ------------------------------------------------------------------
    def analyze(self, module_path: str | Path) -> Dict[str, List[str]]:
        """Return cached IO information for ``module_path``.

        The analysis inspects functions, globals, and basic file operations to
        produce high level input and output signatures.  Results are cached on
        disk so subsequent calls avoid re-parsing unchanged modules.
        """

        path = Path(module_path)
        try:
            data = path.read_bytes()
        except OSError:
            return {"inputs": [], "outputs": []}

        digest = hashlib.sha256(data).hexdigest()
        key = str(path)
        cached = self._cache.get(key)
        if cached and cached.get("hash") == digest:
            return {"inputs": cached.get("inputs", []), "outputs": cached.get("outputs", [])}

        io = analyze_io(path)
        record = {
            "hash": digest,
            "functions": io.functions,
            "inputs": sorted(io.inputs | io.files_read | io.globals),
            "outputs": sorted(io.outputs | io.files_written | io.globals),
            "globals": sorted(io.globals),
            "files_read": sorted(io.files_read),
            "files_written": sorted(io.files_written),
        }
        self._cache[key] = record
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        return {"inputs": record["inputs"], "outputs": record["outputs"]}


def inspect_module(module_name: str) -> ModuleIO:
    """Return :class:`ModuleIO` information for ``module_name``.

    Parameters
    ----------
    module_name:
        Dotted module name relative to the repository root.
    """

    path = Path(module_name.replace(".", "/")).with_suffix(".py")
    return analyze_io(path)


@dataclass(init=False)
class WorkflowSynthesizer:
    """Suggest modules for building a workflow."""

    module_synergy_grapher: ModuleSynergyGrapher | None
    intent_clusterer: IntentClusterer | None
    intent_db: "IntentDB" | None
    synergy_graph_path: Path
    intent_db_path: Path | None
    generated_workflows: List[List[Dict[str, Any]]]

    # ------------------------------------------------------------------
    def __init__(
        self,
        module_synergy_grapher: ModuleSynergyGrapher | None = None,
        *,
        synergy_graph_path: str | Path | None = None,
        intent_clusterer: IntentClusterer | None = None,
        intent_db_path: str | Path | None = None,
    ) -> None:
        """Initialise the synthesizer and load optional resources."""

        self.synergy_graph_path = (
            Path(synergy_graph_path)
            if synergy_graph_path is not None
            else Path("sandbox_data/module_synergy_graph.json")
        )

        # Load synergy graph if available
        self.module_synergy_grapher = module_synergy_grapher
        if self.module_synergy_grapher is None and ModuleSynergyGrapher is not None:
            try:
                self.module_synergy_grapher = ModuleSynergyGrapher()
            except Exception:  # pragma: no cover - best effort
                self.module_synergy_grapher = None
        if self.module_synergy_grapher is not None:
            try:
                self.module_synergy_grapher.load(self.synergy_graph_path)
            except Exception:  # pragma: no cover - ignore load failures
                pass

        self.intent_db_path = Path(intent_db_path) if intent_db_path else None
        self.intent_clusterer = intent_clusterer
        self.intent_db = None
        self.load_intent_clusters()

        self.generated_workflows = []

    # ------------------------------------------------------------------
    def load_intent_clusters(self) -> None:
        """Load intent search helpers if available."""

        if self.intent_clusterer is not None or self.intent_db is not None:
            return
        if IntentClusterer is not None:
            try:
                if self.intent_db_path is not None:
                    self.intent_clusterer = IntentClusterer(self.intent_db_path)
                else:
                    self.intent_clusterer = IntentClusterer()
                return
            except Exception:  # pragma: no cover - ignore init errors
                self.intent_clusterer = None
        if IntentDB is not None and self.intent_db is None:
            try:
                self.intent_db = IntentDB(self.intent_db_path or Path("intent.db"))
            except Exception:  # pragma: no cover - ignore
                self.intent_db = None

    # ------------------------------------------------------------------
    def resolve_dependencies(self, modules: List["ModuleSignature"]) -> List[WorkflowStep]:
        """Order ``modules`` based on their data dependencies.

        Parameters
        ----------
        modules:
            Sequence of :class:`ModuleSignature` objects. Each must provide a
            ``name`` identifying the module it represents.

        Returns
        -------
        list[WorkflowStep]
            Ordered steps where all required inputs for a step are produced by
            earlier steps.  If a dependency cycle is detected a ``ValueError``
            is raised.  Missing producers for required inputs are also reported
            via ``ValueError``.
        """

        from graphlib import CycleError, TopologicalSorter

        if not modules:
            return []

        if ModuleSignature is None:
            raise ValueError("ModuleSignature support is unavailable")

        # Helper to select the best producer for a given consumer based on
        # synergy weights.  Falls back to a deterministic alphabetical choice.
        def _select_best(consumer: str, candidates: Set[str]) -> str:
            best = None
            best_weight = float("-inf")
            graph = getattr(self.module_synergy_grapher, "graph", None)
            for cand in candidates:
                weight = 0.0
                if graph is not None:
                    if graph.has_edge(cand, consumer):
                        weight = float(graph[cand][consumer].get("weight", 0.0))
                    elif graph.has_edge(consumer, cand):
                        weight = float(graph[consumer][cand].get("weight", 0.0))
                if best is None or weight > best_weight:
                    best = cand
                    best_weight = weight
            if best is None:
                best = sorted(candidates)[0]
            return best

        # Map produced values/files/globals -> modules
        produced_by_name: Dict[str, Set[str]] = {}
        produced_by_type: Dict[str, Set[str]] = {}
        for mod in modules:
            name = mod.name or getattr(mod, "module", "")
            if not name:
                raise ValueError("ModuleSignature missing name attribute")

            outputs = (
                set(mod.files_written)
                | set(mod.globals)
                | set(getattr(mod, "outputs", []))
            )
            for out in outputs:
                produced_by_name.setdefault(out, set()).add(name)

            for fn in mod.functions.values():
                ret = fn.get("returns")
                if ret:
                    produced_by_type.setdefault(ret, set()).add(name)

        # Build dependency mapping for topological sort
        deps: Dict[str, Set[str]] = {}
        missing: Dict[str, Set[str]] = {}
        annotations_cache: Dict[str, Dict[str, str]] = {
            mod.name: {
                k: v
                for fn in mod.functions.values()
                for k, v in fn.get("annotations", {}).items()
            }
            for mod in modules
        }

        for mod in modules:
            name = mod.name
            required: Set[str] = set(mod.files_read) | set(mod.globals)
            for fn in mod.functions.values():
                required.update(fn.get("args", []))
            dependencies: Set[str] = set()

            for item in required:
                matched = False
                producers = produced_by_name.get(item)
                if producers:
                    producer = _select_best(name, producers)
                    if producer != name:
                        dependencies.add(producer)
                    matched = True
                else:
                    ann = annotations_cache[name].get(item)
                    if ann and ann in produced_by_type:
                        producer = _select_best(name, produced_by_type[ann])
                        if producer != name:
                            dependencies.add(producer)
                        matched = True
                if not matched:
                    missing.setdefault(name, set()).add(item)

            deps[name] = dependencies

        if missing:
            problems = ", ".join(f"{m}: {sorted(v)}" for m, v in sorted(missing.items()))
            raise ValueError(f"Unresolved dependencies: {problems}")

        sorter = TopologicalSorter(deps)
        try:
            order = list(sorter.static_order())
        except CycleError as exc:  # pragma: no cover - cycle detection
            raise ValueError(f"Cyclic dependency detected: {exc.args}") from exc

        steps: List[WorkflowStep] = []
        for name in order:
            mod = next(m for m in modules if m.name == name)
            inputs: Set[str] = set(mod.files_read) | set(mod.globals)
            for fn in mod.functions.values():
                inputs.update(fn.get("args", []))
            outputs = (
                set(mod.files_written)
                | set(mod.globals)
                | set(getattr(mod, "outputs", []))
            )
            step = WorkflowStep(
                module=name,
                inputs=sorted(inputs),
                outputs=sorted(outputs),
            )
            steps.append(step)

        return steps

    # ------------------------------------------------------------------
    def expand_cluster(
        self,
        start_module: str | None = None,
        *,
        problem: str | None = None,
        threshold: float = 0.0,
    ) -> Set[str]:
        """Expand from ``start_module`` and/or ``problem`` to related modules."""

        modules: Set[str] = set()

        if start_module:
            modules.add(start_module)
            try:
                if self.module_synergy_grapher is not None and hasattr(
                    self.module_synergy_grapher, "get_synergy_cluster"
                ):
                    try:
                        cluster = self.module_synergy_grapher.get_synergy_cluster(
                            start_module, threshold=threshold
                        )
                    except TypeError:  # pragma: no cover - threshold unsupported
                        cluster = self.module_synergy_grapher.get_synergy_cluster(
                            start_module
                        )
                elif get_synergy_cluster is not None:
                    cluster = get_synergy_cluster(
                        start_module, path=self.synergy_graph_path, threshold=threshold
                    )
                elif load_graph is not None:
                    graph = load_graph(self.synergy_graph_path)
                    cluster = list(graph.neighbors(start_module)) if start_module in graph else []
                else:
                    cluster = []
                modules.update(cluster)
            except Exception:  # pragma: no cover - best effort
                pass

        if problem:
            if self.intent_clusterer is None and self.intent_db is None:
                self.load_intent_clusters()
            if self.intent_clusterer is not None:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(
                        problem, top_k=20
                    )
                    for m in matches:
                        path = getattr(m, "path", None) or getattr(m, "module", None)
                        if path:
                            p = Path(str(path))
                            modules.add(p.stem)
                except Exception:  # pragma: no cover - best effort
                    pass
            elif self.intent_db is not None:
                try:
                    vec = self.intent_db.encode_text(problem)  # type: ignore[attr-defined]
                    results = self.intent_db.search_by_vector(vec, top_k=20)
                    for rid, _dist in results:
                        path = rid
                        if isinstance(rid, int):
                            row = self.intent_db.conn.execute(
                                "SELECT path FROM intent_modules WHERE id=?",
                                (rid,),
                            ).fetchone()
                            path = row["path"] if row else None
                        if path:
                            modules.add(Path(str(path)).stem)
                except Exception:  # pragma: no cover - best effort
                    pass

        return modules

    # ------------------------------------------------------------------
    def _synthesize_greedy(
        self,
        start_module: str | None = None,
        problem: str | None = None,
        limit: int = 10,
        overrides: Dict[str, Set[str]] | None = None,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return an ordered chain of modules based on dependency resolution.

        Modules are gathered using synergy graph expansion and optional intent
        search. They are then ordered using :meth:`resolve_dependencies` which
        analyses each module via :func:`analysis.get_io_signature`. ``overrides``
        allows callers to mark specific arguments as satisfied externally.
        ``threshold`` controls the minimum synergy weight when expanding the
        cluster around ``start_module``.
        """

        if ModuleSignature is None or get_io_signature is None:
            raise ValueError("Structural analysis helpers are unavailable")

        modules: Set[str] = self.expand_cluster(
            start_module=start_module, problem=problem, threshold=threshold
        )

        if start_module:
            modules.add(start_module)

        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = Path(mod.replace(".", "/")).with_suffix(".py")
            sig = get_io_signature(path)
            sig.name = mod
            if overrides and mod in overrides:
                for fn in sig.functions.values():
                    fn["args"] = [a for a in fn.get("args", []) if a not in overrides[mod]]
            signatures.append(sig)

        try:
            steps = self.resolve_dependencies(signatures)
        except ValueError as exc:  # pragma: no cover - surface errors
            raise ValueError(f"Dependency resolution failed: {exc}") from exc

        workflow: List[Dict[str, Any]] = []
        for step in steps[:limit]:
            workflow.append(
                {
                    "module": step.module,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                }
            )

        return workflow

    # ------------------------------------------------------------------
    def synthesize(
        self, start: str | Dict[str, Any], *, threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Expand ``start`` into a workflow description.

        ``start`` may be a module name or free text problem description. A
        mapping may also be supplied with ``module`` and ``problem`` keys. The
        optional ``threshold`` argument is forwarded to
        :meth:`expand_cluster`.
        """

        if isinstance(start, dict):
            start_module = start.get("module") or start.get("start")
            problem = start.get("problem")
        else:
            module_path = Path(start.replace(".", "/")).with_suffix(".py")
            if module_path.exists():
                start_module, problem = start, None
            else:
                start_module, problem = None, start
        try:
            steps = self._synthesize_greedy(
                start_module=start_module, problem=problem, threshold=threshold
            )
        except ValueError as exc:  # pragma: no cover - propagate helpful errors
            raise ValueError(f"Failed to synthesise workflow: {exc}") from exc
        return {"steps": steps}

    # ------------------------------------------------------------------
    def generate_workflows(
        self,
        start_module: str,
        *,
        problem: str | None = None,
        limit: int = 5,
        max_depth: int | None = None,
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidate workflows beginning at ``start_module``.

        The returned workflows are ordered using the same dependency resolution
        logic as :meth:`synthesize`.  Currently a single workflow is produced
        representing a best effort ordering of modules related to
        ``start_module`` and ``problem``.

        Parameters
        ----------
        start_module:
            Dotted name of the module that should start each workflow.
        problem:
            Optional textual description used for intent matching.  Only
            available when ``intent_clusterer`` is provided.
        limit:
            Maximum number of workflows to return. Only the top workflow is
            currently generated.
        max_depth:
            Unused but kept for API compatibility.
        """

        if ModuleSignature is None or get_io_signature is None:
            raise ValueError("Structural analysis helpers are unavailable")

        modules = self.expand_cluster(start_module=start_module, problem=problem)
        modules.add(start_module)

        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = Path(mod.replace(".", "/")).with_suffix(".py")
            sig = get_io_signature(path)
            sig.name = mod
            signatures.append(sig)

        try:
            ordered = self.resolve_dependencies(signatures)
        except ValueError as exc:  # pragma: no cover - surface errors
            raise ValueError(f"Failed to resolve dependencies: {exc}") from exc

        sig_map = {s.name: s for s in signatures}
        workflow: List[Dict[str, Any]] = []
        for step in ordered:
            sig = sig_map[step.module]
            fn = next(iter(sig.functions.keys()), None)
            workflow.append(
                {
                    "module": step.module,
                    "function": fn,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                }
            )

        self.generated_workflows = [workflow][:limit]

        out_dir = Path("sandbox_data/generated_workflows")
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, wf in enumerate(self.generated_workflows):
            name = wf[0]["module"].replace(".", "_") if wf else f"workflow_{idx}"
            path = out_dir / f"{name}_{idx}.workflow.json"
            path.write_text(to_json(wf), encoding="utf-8")

        return self.generated_workflows

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of generated workflows."""
        return {"workflows": [workflow_to_dict(wf) for wf in self.generated_workflows]}

    # ------------------------------------------------------------------
    def save(self, path: Path | str | None = None) -> Path:
        """Persist generated workflows to ``path`` as JSON and YAML.

        Parameters
        ----------
        path:
            Optional file or directory.  When omitted or when a directory is
            supplied, ``sandbox_data/generated_workflows`` is used and the file
            is named ``workflows.json``.

        Returns
        -------
        Path
            Location of the written JSON file.
        """

        data = self.to_dict()
        base = Path(path) if path is not None else Path("sandbox_data/generated_workflows")
        if base.suffix:
            out_dir = base.parent
            stem = base.stem
        else:
            out_dir = base
            stem = "workflows"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        try:  # pragma: no cover - YAML optional
            import yaml  # type: ignore

            yaml_path = out_dir / f"{stem}.yaml"
            with yaml_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False)  # type: ignore[arg-type]
        except Exception:
            pass

        return json_path


# ---------------------------------------------------------------------------
def workflow_to_dict(workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return ``workflow`` in the simplified serialisable format."""

    return {
        "steps": [
            {
                "module": step.get("module", ""),
                "inputs": step.get("inputs") or step.get("args", []),
                "outputs": step.get("outputs") or step.get("provides", []),
            }
            for step in workflow
        ]
    }


def to_json(workflow: List[Dict[str, Any]]) -> str:
    """Serialize ``workflow`` to a JSON string."""

    return json.dumps(workflow_to_dict(workflow), indent=2)


def to_yaml(workflow: List[Dict[str, Any]]) -> str:
    """Serialize ``workflow`` to a YAML string."""

    try:  # pragma: no cover - YAML optional
        import yaml  # type: ignore

        return yaml.safe_dump(workflow_to_dict(workflow), sort_keys=False)  # type: ignore[arg-type]
    except Exception:
        return to_json(workflow)


# ---------------------------------------------------------------------------
def to_workflow_spec(workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a lightweight workflow specification.

    The synthesizer internally represents workflows as a sequence of
    dictionaries containing at least the ``module`` name and optional ``inputs``
    and ``outputs``.  This helper adapts that structure to the format produced
    by :func:`workflow_spec.to_spec`.
    """

    from workflow_spec import to_spec as _to_spec

    steps = [
        {
            "module": step.get("module", ""),
            "inputs": step.get("inputs", []),
            "outputs": step.get("outputs", []),
            "files": step.get("files", [])
            or step.get("files_read", [])
            + step.get("files_written", []),
            "globals": step.get("globals", []),
        }
        for step in workflow
    ]
    return _to_spec(steps)


def save_workflow(workflow: List[Dict[str, Any]], path: Path | str | None = None) -> Path:
    """Persist ``workflow`` using :func:`workflow_spec.save_spec`."""

    from workflow_spec import save_spec as _save_spec

    spec = to_workflow_spec(workflow)
    out = Path(path) if path is not None else Path("workflow.workflow.json")
    return _save_spec(spec, out)


def synthesise_workflow(**kwargs: Any) -> Dict[str, Any]:
    """Generate a workflow using :class:`WorkflowSynthesizer`.

    Remote function signature:
    ``synthesise_workflow(start: str | dict[str, Any], threshold: float = 0.0) -> dict``

    Parameters
    ----------
    **kwargs:
        Forwarded to :meth:`WorkflowSynthesizer.synthesize`. Common keys are
        ``start`` or ``start_module`` describing the module or problem to
        expand and an optional ``threshold`` controlling synergy expansion.

    Returns
    -------
    dict[str, Any]
        Workflow description containing a ``steps`` list.
    """

    synthesizer = WorkflowSynthesizer()
    return synthesizer.synthesize(**kwargs)


def main(argv: List[str] | None = None) -> None:
    """Command line interface for :mod:`workflow_synthesizer`.

    When invoked with the ``synthesize`` command a single workflow is
    generated from the provided starting module or problem statement. Running
    the script without arguments starts an interactive loop prompting for a
    starting module or problem description.
    """

    parser = argparse.ArgumentParser(description="Workflow synthesizer CLI")
    sub = parser.add_subparsers(dest="command")

    synth_parser = sub.add_parser(
        "synthesize", help="Expand a starting module or problem into a workflow"
    )
    synth_parser.add_argument(
        "start",
        nargs="?",
        help="Starting module name or free text problem description",
    )
    synth_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Synergy threshold used when expanding the starting module",
    )

    args = parser.parse_args(argv)
    synthesizer = WorkflowSynthesizer()

    if args.command == "synthesize":
        start = args.start or input("Enter starting module or problem: ").strip()
        result = synthesizer.synthesize(start, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    else:
        while True:
            try:
                start = input("Start module or problem (blank to exit): ").strip()
            except EOFError:
                break
            if not start:
                break
            result = synthesizer.synthesize(start)
            print(json.dumps(result, indent=2))


__all__ = [
    "ModuleIO",
    "WorkflowStep",
    "ModuleIOAnalyzer",
    "WorkflowSynthesizer",
    "inspect_module",
    "to_json",
    "to_yaml",
    "to_workflow_spec",
    "save_workflow",
    "synthesise_workflow",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
