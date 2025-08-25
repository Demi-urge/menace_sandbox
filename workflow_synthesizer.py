from __future__ import annotations

"""Lightweight workflow synthesizer utilities.

This module exposes :class:`WorkflowSynthesizer` which combines structural
signals from :class:`~module_synergy_grapher.ModuleSynergyGrapher` with
semantic intent matches provided by :class:`~intent_clusterer.IntentClusterer`.

The synthesizer is intentionally small and focuses on expanding an initial set
of modules either by following the synergy graph around a starting module or by
searching for modules related to a textual problem description.
"""

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

try:  # Optional imports; fall back to stubs in tests
    from module_synergy_grapher import ModuleSynergyGrapher, get_synergy_cluster
except Exception:  # pragma: no cover - graceful degradation
    ModuleSynergyGrapher = None  # type: ignore[misc]
    get_synergy_cluster = None  # type: ignore[misc]

try:  # Optional dependency
    from intent_clusterer import IntentClusterer
except Exception:  # pragma: no cover - graceful degradation
    IntentClusterer = None  # type: ignore[misc]


@dataclass
class ModuleIO:
    """Basic structural information about a Python module."""

    functions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    globals: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_written: Set[str] = field(default_factory=set)


def _parse_module_io(path: Path) -> ModuleIO:
    """Parse ``path`` and extract high level IO information."""

    module_io = ModuleIO()

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

    # ---- top level function definitions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if node.args.vararg:
                args.append(node.args.vararg.arg)
            if node.args.kwarg:
                args.append(node.args.kwarg.arg)

            returns = None
            if getattr(node, "returns", None) is not None:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:  # pragma: no cover - ast.unparse fallback
                    returns = None

            module_io.functions[node.name] = {"args": args, "returns": returns}
            module_io.inputs.update(args)

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

    Visitor().visit(tree)
    return module_io


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

        io = _parse_module_io(path)
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
    return _parse_module_io(path)


@dataclass
class WorkflowSynthesizer:
    """Suggest modules for building a workflow.

    Parameters
    ----------
    module_synergy_grapher:
        Helper used to expand modules via structural relationships.
    intent_clusterer:
        Component used to search modules based on natural language problems.
    synergy_graph_path:
        Location of the persisted synergy graph JSON file.
    intent_db_path:
        Optional location of the intent vector database.  The synthesizer does
        not interact with the database directly but exposes this path so that
        callers can initialise :class:`IntentClusterer` with it if desired.
    """

    module_synergy_grapher: ModuleSynergyGrapher | None = None
    intent_clusterer: IntentClusterer | None = None
    synergy_graph_path: Path = Path("sandbox_data/module_synergy_graph.json")
    intent_db_path: Path | None = None
    generated_workflows: List[List[Dict[str, Any]]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def synthesize(
        self,
        start_module: str | None = None,
        problem: str | None = None,
        limit: int = 10,
        overrides: Dict[str, Set[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return a greedy chain of modules.

        Modules are gathered using synergy graph expansion and optional intent
        search.  They are then ordered greedily by matching outputs of previous
        modules to inputs of subsequent modules.  When no direct match exists the
        module with the smallest number of unresolved inputs is chosen as a
        fallback.  ``overrides`` allows callers to mark specific arguments as
        satisfied externally.
        """

        modules: Set[str] = set()

        # ----- expand via synergy graph
        if start_module:
            try:
                if self.module_synergy_grapher is not None:
                    if hasattr(self.module_synergy_grapher, "load"):
                        # Ensure the grapher has the latest graph loaded
                        self.module_synergy_grapher.load(self.synergy_graph_path)
                    cluster = self.module_synergy_grapher.get_synergy_cluster(start_module)
                elif get_synergy_cluster is not None:
                    # Fall back to module level helper which loads the graph on demand
                    cluster = get_synergy_cluster(start_module, path=self.synergy_graph_path)
                else:
                    cluster = {start_module}
                modules.update(cluster)
            except Exception:  # pragma: no cover - best effort
                modules.add(start_module)

        # ----- expand via intent search
        if problem and self.intent_clusterer is not None:
            try:
                matches = self.intent_clusterer.find_modules_related_to(
                    problem, top_k=limit
                )
                for match in matches:
                    path = getattr(match, "path", None)
                    if path:
                        modules.add(Path(path).stem)
            except Exception:  # pragma: no cover - ignore search failures
                pass

        if start_module:
            modules.add(start_module)

        # Greedily order modules based on IO overlap
        analyzer = ModuleIOAnalyzer()
        remaining = [m for m in sorted(modules) if m != start_module]
        provided: Set[str] = set()
        workflow: List[Dict[str, Any]] = []

        def _append(mod: str) -> None:
            io = analyzer.analyze(Path(mod.replace(".", "/")).with_suffix(".py"))
            inputs = set(io.get("inputs", []))
            outputs = set(io.get("outputs", []))
            unresolved = [i for i in inputs if i not in provided]
            if overrides and mod in overrides:
                unresolved = [i for i in unresolved if i not in overrides[mod]]
            workflow.append(
                {"module": mod, "args": unresolved, "provides": sorted(outputs)}
            )
            provided.update(outputs)

        if start_module:
            _append(start_module)

        while remaining and len(workflow) < limit:
            best = None
            best_overlap = -1
            best_missing = None
            for mod in remaining:
                io = analyzer.analyze(Path(mod.replace(".", "/")).with_suffix(".py"))
                ins = set(io.get("inputs", []))
                overlap = len(ins & provided)
                missing = len(ins - provided)
                if (
                    overlap > best_overlap
                    or (overlap == best_overlap and (best_missing is None or missing < best_missing))
                ):
                    best = mod
                    best_overlap = overlap
                    best_missing = missing

            if best is None:
                break
            _append(best)
            remaining.remove(best)

        return workflow

    # ------------------------------------------------------------------
    def generate_workflows(
        self,
        start_module: str,
        *,
        problem: str | None = None,
        limit: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Generate candidate workflows beginning at ``start_module``.

        The synthesizer builds a small dependency graph where edges connect
        modules whose outputs feed the inputs of another module or when one
        module writes a file another module reads.  The graph is seeded with
        ``start_module`` and expanded using :meth:`synthesize` which blends
        synergy relationships and intent matches.  A topological sort orders
        the modules when the graph is acyclic; otherwise a best effort ranking
        based on edge weights is used.  Simple paths starting from
        ``start_module`` are converted into workflow candidates and scored by
        combining edge weights with optional intent match scores.

        Parameters
        ----------
        start_module:
            Dotted name of the module that should start each workflow.
        problem:
            Optional textual description used for intent matching.  Only
            available when ``intent_clusterer`` is provided.
        limit:
            Maximum number of workflows to return.
        """

        import networkx as nx

        # gather candidate modules via synergy + intent expansion
        steps = self.synthesize(start_module=start_module, problem=problem, limit=50)
        modules = {s["module"] for s in steps}
        modules.add(start_module)

        # Inspect modules to gather IO information
        info: Dict[str, ModuleIO] = {m: inspect_module(m) for m in modules}

        # Build graph where outputs/files_written satisfy inputs/files_read
        G = nx.DiGraph()
        for m in modules:
            G.add_node(m)
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                io_a, io_b = info[a], info[b]
                if io_a.outputs & io_b.inputs or io_a.files_written & io_b.files_read:
                    weight = 1.0
                    if (
                        self.module_synergy_grapher
                        and getattr(self.module_synergy_grapher, "graph", None)
                        and self.module_synergy_grapher.graph is not None
                        and self.module_synergy_grapher.graph.has_edge(a, b)
                    ):
                        weight = float(
                            self.module_synergy_grapher.graph[a][b].get("weight", 1.0)
                        )
                    G.add_edge(a, b, weight=weight)

        # Determine node ordering
        try:
            order = list(nx.topological_sort(G))
        except Exception:  # pragma: no cover - cycle detected
            # Fallback heuristic: sort by total outgoing edge weight
            order = sorted(
                G.nodes,
                key=lambda n: sum(G[n][m].get("weight", 1.0) for m in G.successors(n)),
                reverse=True,
            )

        # Collect intent scores if problem provided
        intent_scores: Dict[str, float] = {}
        if problem and self.intent_clusterer is not None:
            try:
                matches = self.intent_clusterer.find_modules_related_to(problem, top_k=len(modules))
                for m in matches:
                    path = getattr(m, "path", None)
                    score = getattr(m, "score", None) or getattr(m, "similarity", None)
                    if path:
                        intent_scores[Path(path).stem] = float(score or 0.0)
            except Exception:  # pragma: no cover - best effort
                pass

        # Enumerate simple paths starting from start_module
        paths: List[List[str]] = []
        for target in order:
            if target == start_module or not nx.has_path(G, start_module, target):
                continue
            for p in nx.all_simple_paths(G, start_module, target):
                paths.append(p)
        if not paths:
            paths.append([start_module])

        workflows: List[List[Dict[str, Any]]] = []
        scores: List[float] = []
        for p in paths:
            wf: List[Dict[str, Any]] = []
            synergy_score = 0.0
            for i, mod in enumerate(p):
                io = info[mod]
                fn = next(iter(io.functions.keys()), None)
                wf.append(
                    {
                        "module": mod,
                        "function": fn,
                        "inputs": sorted(io.inputs),
                        "outputs": sorted(io.outputs),
                    }
                )
                if i < len(p) - 1 and G.has_edge(mod, p[i + 1]):
                    synergy_score += float(G[mod][p[i + 1]].get("weight", 1.0))

            intent_score = sum(intent_scores.get(step["module"], 0.0) for step in wf)
            scores.append(synergy_score + intent_score)
            workflows.append(wf)

        # Rank workflows by combined score
        ranked = sorted(zip(workflows, scores), key=lambda x: x[1], reverse=True)
        top = [wf for wf, _ in ranked[:limit]]
        self.generated_workflows = top
        return top

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of generated workflows."""

        return {"workflows": self.generated_workflows}

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
def to_workflow_spec(workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a workflow specification mapping for ``WorkflowDB``.

    This is a thin wrapper around :func:`workflow_spec.to_spec` that accepts the
    workflow format produced by :class:`WorkflowSynthesizer` where each step is
    represented by a ``module`` key.
    """

    from workflow_spec import to_spec as _to_spec

    steps = [{"name": step.get("name") or step.get("module", "")}
             for step in workflow]
    return _to_spec(steps)


def save_workflow(workflow: List[Dict[str, Any]], path: Path | str | None = None) -> Path:
    """Persist ``workflow`` as ``.workflow.json`` for ``WorkflowDB`` utilities."""

    from workflow_spec import save as _save

    spec = to_workflow_spec(workflow)
    out = Path(path) if path is not None else Path("sandbox_data/generated_workflows")
    return _save(spec, out)


__all__ = [
    "ModuleIOAnalyzer",
    "WorkflowSynthesizer",
    "inspect_module",
    "to_workflow_spec",
    "save_workflow",
]
