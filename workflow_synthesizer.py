from __future__ import annotations

"""Lightweight workflow synthesizer utilities.

This module exposes :class:`WorkflowSynthesizer` which combines structural
signals from :class:`~module_synergy_grapher.ModuleSynergyGrapher` with
semantic intent matches provided by :class:`~intent_clusterer.IntentClusterer`.

Workflows are scored by blending edge weights from the synergy graph with
intent match scores.  The :meth:`WorkflowSynthesizer.generate_workflows`
method exposes ``synergy_weight`` and ``intent_weight`` parameters to control
this blend.  Scores are normalised by the number of steps so longer workflows
are not automatically favoured.

The synthesizer is intentionally small and focuses on expanding an initial set
of modules either by following the synergy graph around a starting module or by
searching for modules related to a textual problem description.
"""

import argparse
import ast
import json
import logging
from collections import deque
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
    from module_synergy_grapher import ModuleSynergyGrapher, load_graph
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


logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Lightweight representation of an ordered workflow step."""

    module: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


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
    def analyze(self, module_path: str | Path) -> "ModuleSignature":
        """Return cached :class:`ModuleSignature` for ``module_path``."""

        if ModuleSignature is None or get_io_signature is None:
            raise ValueError("Structural analysis helpers are unavailable")

        path = Path(module_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return ModuleSignature(name=path.stem)

        key = str(path)
        cached = self._cache.get(key)
        if cached and cached.get("mtime") == mtime:
            data = cached.get("signature", {})
            sig = ModuleSignature(
                name=data.get("name", path.stem),
                functions=data.get("functions", {}),
                classes=data.get("classes", {}),
                globals=set(data.get("globals", [])),
                files_read=set(data.get("files_read", [])),
                files_written=set(data.get("files_written", [])),
            )
            sig.inputs = data.get("inputs", [])
            sig.outputs = data.get("outputs", [])
            return sig

        sig = get_io_signature(path)

        all_args: Set[str] = set()
        for fn in sig.functions.values():
            all_args.update(fn.get("args", []))
        module_globals = set(sig.globals) - all_args
        sig.globals = module_globals

        inputs: Set[str] = set(sig.files_read) | all_args | module_globals

        return_names: Set[str] = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    val = node.value
                    if isinstance(val, ast.Name):
                        return_names.add(val.id)
                    elif isinstance(val, ast.Tuple):
                        for elt in val.elts:
                            if isinstance(elt, ast.Name):
                                return_names.add(elt.id)
        except Exception:  # pragma: no cover - best effort
            pass

        outputs: Set[str] = set(sig.files_written) | module_globals | return_names
        sig.inputs = sorted(inputs)
        sig.outputs = sorted(outputs)

        record = {
            "mtime": mtime,
            "signature": {
                "name": sig.name,
                "functions": sig.functions,
                "classes": sig.classes,
                "globals": sorted(module_globals),
                "files_read": sorted(sig.files_read),
                "files_written": sorted(sig.files_written),
                "inputs": sig.inputs,
                "outputs": sig.outputs,
            },
        }
        self._cache[key] = record
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        return sig


def inspect_module(module_name: str) -> "ModuleSignature":
    """Return :class:`ModuleSignature` information for ``module_name``."""

    path = Path(module_name.replace(".", "/")).with_suffix(".py")
    if ModuleSignature is None or get_io_signature is None:
        return ModuleSignature(name=path.stem)
    analyzer = ModuleIOAnalyzer()
    sig = analyzer.analyze(path)
    sig.name = path.stem
    return sig


@dataclass(init=False)
class WorkflowSynthesizer:
    """Suggest modules for building a workflow."""

    module_synergy_grapher: ModuleSynergyGrapher | None
    intent_clusterer: IntentClusterer | None
    intent_db: "IntentDB" | None
    synergy_graph_path: Path
    intent_db_path: Path | None
    generated_workflows: List[List[WorkflowStep]]
    workflow_scores: List[float]

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
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to initialise ModuleSynergyGrapher: %s", exc)
                self.module_synergy_grapher = None
        if self.module_synergy_grapher is not None:
            if not self.synergy_graph_path.exists():
                logger.warning(
                    "Synergy graph path %s does not exist", self.synergy_graph_path
                )
            try:
                self.module_synergy_grapher.load(self.synergy_graph_path)
            except Exception as exc:  # pragma: no cover - ignore load failures
                logger.warning(
                    "Failed to load synergy graph from %s: %s",
                    self.synergy_graph_path,
                    exc,
                )

        self.intent_db_path = Path(intent_db_path) if intent_db_path else None
        self.intent_clusterer = intent_clusterer
        self.intent_db = None
        self.load_intent_clusters()

        self.generated_workflows = []
        self.workflow_scores = []

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
            except Exception as exc:  # pragma: no cover - ignore init errors
                logger.warning(
                    "Failed to initialise IntentClusterer with %s: %s",
                    self.intent_db_path or "<default>",
                    exc,
                )
                self.intent_clusterer = None
        if IntentDB is not None and self.intent_db is None:
            try:
                path = self.intent_db_path or Path("intent.db")
                self.intent_db = IntentDB(path)
            except Exception as exc:  # pragma: no cover - ignore
                logger.warning("Failed to load IntentDB from %s: %s", path, exc)
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
            is raised.  Missing producers for required inputs are accumulated
            per module and reported via the ``unresolved`` attribute of the
            returned :class:`WorkflowStep` objects.
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

        # Map produced values by category -> modules
        produced_globals: Dict[str, Set[str]] = {}
        produced_files: Dict[str, Set[str]] = {}
        produced_by_type: Dict[str, Set[str]] = {}
        func_args: Dict[str, Set[str]] = {}
        for mod in modules:
            name = mod.name or getattr(mod, "module", "")
            if not name:
                raise ValueError("ModuleSignature missing name attribute")

            args: Set[str] = set()
            for fn in mod.functions.values():
                args.update(fn.get("args", []))
                ret = fn.get("returns")
                if ret:
                    produced_by_type.setdefault(ret, set()).add(name)
            func_args[name] = args

            globals_out = set(mod.globals) | set(getattr(mod, "outputs", []))
            globals_out -= args
            for g in globals_out:
                produced_globals.setdefault(g, set()).add(name)

            for f in mod.files_written:
                produced_files.setdefault(f, set()).add(name)

        # Build dependency mapping for topological sort
        deps: Dict[str, Set[str]] = {}
        unresolved: Dict[str, Set[str]] = {}
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
            required_args: Set[str] = set()
            for fn in mod.functions.values():
                required_args.update(fn.get("args", []))
            required_files: Set[str] = set(mod.files_read)
            required_globals: Set[str] = set(mod.globals)

            dependencies: Set[str] = set()

            # function args
            for item in required_args:
                matched = False
                producers = produced_globals.get(item)
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
                    unresolved.setdefault(name, set()).add(item)

            # file dependencies
            for item in required_files:
                producers = produced_files.get(item)
                if producers:
                    producer = _select_best(name, producers)
                    if producer != name:
                        dependencies.add(producer)
                else:
                    unresolved.setdefault(name, set()).add(item)

            # globals
            for item in required_globals:
                producers = produced_globals.get(item)
                if producers:
                    producer = _select_best(name, producers)
                    if producer != name:
                        dependencies.add(producer)
                else:
                    unresolved.setdefault(name, set()).add(item)

            deps[name] = dependencies

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
            outputs = set(mod.files_written)
            globals_out = set(mod.globals) | set(getattr(mod, "outputs", []))
            globals_out -= func_args.get(name, set())
            outputs |= globals_out
            step = WorkflowStep(
                module=name,
                inputs=sorted(inputs),
                outputs=sorted(outputs),
                unresolved=sorted(unresolved.get(name, set())),
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
        max_depth: int = 1,
    ) -> Set[str]:
        """Expand from ``start_module`` and/or ``problem`` to related modules.

        Parameters
        ----------
        start_module:
            Module name to expand from.  If ``None`` only intent based expansion
            is performed.
        problem:
            Optional textual description used to search the intent database.
        threshold:
            Minimum edge weight when traversing the synergy graph.
        max_depth:
            Maximum graph depth to explore from ``start_module``.  ``1`` restricts
            expansion to direct neighbours.
        """

        modules: Set[str] = set()

        if start_module:
            modules.add(start_module)
            try:
                graph = None
                if self.module_synergy_grapher is not None:
                    graph = getattr(self.module_synergy_grapher, "graph", None)
                    if graph is None and hasattr(self.module_synergy_grapher, "load"):
                        graph = self.module_synergy_grapher.load(self.synergy_graph_path)
                elif load_graph is not None:
                    graph = load_graph(self.synergy_graph_path)
                if graph is not None and start_module in graph:
                    visited: Set[str] = {start_module}
                    queue: deque[tuple[str, int]] = deque([(start_module, 0)])
                    while queue:
                        node, depth = queue.popleft()
                        if depth >= max_depth:
                            continue
                        for neigh, data in graph[node].items():
                            weight = float(data.get("weight", 0.0))
                            if weight < threshold or neigh in visited:
                                continue
                            visited.add(neigh)
                            modules.add(neigh)
                            queue.append((neigh, depth + 1))
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
        analyses each module via :class:`ModuleIOAnalyzer`. ``overrides`` allows
        callers to mark specific arguments as satisfied externally.
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

        analyzer = ModuleIOAnalyzer()
        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = Path(mod.replace(".", "/")).with_suffix(".py")
            sig = analyzer.analyze(path)
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
        synergy_weight: float = 1.0,
        intent_weight: float = 1.0,
    ) -> List[List[WorkflowStep]]:
        """Generate candidate workflows beginning at ``start_module``.

        Candidate workflows are constructed by exploring different
        permutations of modules that satisfy dependency constraints.  Each
        candidate is scored and the best ``limit`` workflows are returned.

        Parameters
        ----------
        start_module:
            Dotted name of the module that should start each workflow.
        problem:
            Optional textual description used for intent matching.  Only
            available when ``intent_clusterer`` is provided.
        limit:
            Maximum number of workflows to return.
        max_depth:
            Unused but kept for API compatibility.
        synergy_weight:
            Multiplier applied to synergy graph edge weights when scoring
            workflows.
        intent_weight:
            Multiplier applied to intent match scores when scoring workflows.
        """

        if ModuleSignature is None or get_io_signature is None:
            raise ValueError("Structural analysis helpers are unavailable")

        modules = self.expand_cluster(start_module=start_module, problem=problem)
        modules.add(start_module)

        analyzer = ModuleIOAnalyzer()
        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = Path(mod.replace(".", "/")).with_suffix(".py")
            sig = analyzer.analyze(path)
            sig.name = mod
            signatures.append(sig)

        # Build dependency mapping for permutations
        produced_by_name: Dict[str, Set[str]] = {}
        produced_by_type: Dict[str, Set[str]] = {}
        for mod in signatures:
            outputs = (
                set(mod.files_written)
                | set(mod.globals)
                | set(getattr(mod, "outputs", []))
            )
            for out in outputs:
                produced_by_name.setdefault(out, set()).add(mod.name)
            for fn in mod.functions.values():
                ret = fn.get("returns")
                if ret:
                    produced_by_type.setdefault(ret, set()).add(mod.name)

        annotations_cache: Dict[str, Dict[str, str]] = {
            mod.name: {
                k: v
                for fn in mod.functions.values()
                for k, v in fn.get("annotations", {}).items()
            }
            for mod in signatures
        }

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

        deps: Dict[str, Set[str]] = {}
        missing: Dict[str, Set[str]] = {}
        step_map: Dict[str, WorkflowStep] = {}
        for mod in signatures:
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

            outputs = (
                set(mod.files_written)
                | set(mod.globals)
                | set(getattr(mod, "outputs", []))
            )
            step_map[name] = WorkflowStep(
                module=name,
                inputs=sorted(required),
                outputs=sorted(outputs),
            )

        if missing:
            problems = ", ".join(
                f"{m}: {sorted(v)}" for m, v in sorted(missing.items())
            )
            raise ValueError(f"Unresolved dependencies: {problems}")

        from graphlib import CycleError, TopologicalSorter

        try:  # Validate acyclic dependencies
            TopologicalSorter(deps).static_order()
        except CycleError as exc:  # pragma: no cover - cycle detection
            raise ValueError(f"Cyclic dependency detected: {exc.args}") from exc

        orders: List[List[str]] = []

        def _dfs(order: List[str], remaining: Set[str]) -> None:
            if not remaining:
                orders.append(order.copy())
                return
            available = [m for m in remaining if deps[m].issubset(order)]
            for mod in sorted(available):
                order.append(mod)
                remaining.remove(mod)
                _dfs(order, remaining)
                remaining.add(mod)
                order.pop()

        if start_module not in step_map:
            raise ValueError(f"Start module {start_module!r} not found")

        _dfs([start_module], set(step_map) - {start_module})

        graph = getattr(self.module_synergy_grapher, "graph", None)

        score_map: Dict[str, float] = {}
        if problem and self.intent_clusterer is not None:
            try:
                matches = self.intent_clusterer.find_modules_related_to(problem, top_k=50)
                for m in matches:
                    path = getattr(m, "path", None) or getattr(m, "module", None)
                    if path:
                        mod = Path(str(path)).stem
                        score_map[mod] = float(getattr(m, "score", 1.0))
            except Exception:  # pragma: no cover - best effort
                pass

        entries: List[Tuple[float, List[WorkflowStep]]] = []
        for order in orders:
            workflow = [step_map[m] for m in order]
            synergy_score = 0.0
            if graph is not None:
                for a, b in zip(order, order[1:]):
                    if graph.has_edge(a, b):
                        synergy_score += float(graph[a][b].get("weight", 0.0))
                    elif graph.has_edge(b, a):
                        synergy_score += float(graph[b][a].get("weight", 0.0))
            intent_score = sum(score_map.get(m, 0.0) for m in order)
            combined = synergy_weight * synergy_score + intent_weight * intent_score
            normalised = combined / max(len(order), 1)
            entries.append((normalised, workflow))

        entries.sort(key=lambda x: x[0], reverse=True)
        self.workflow_scores = [score for score, _wf in entries][:limit]
        self.generated_workflows = [wf for _score, wf in entries][:limit]

        out_dir = Path("sandbox_data/generated_workflows")
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, wf in enumerate(self.generated_workflows):
            name = wf[0].module.replace(".", "_") if wf else f"workflow_{idx}"
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
def workflow_to_dict(workflow: List[WorkflowStep] | List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return ``workflow`` in the simplified serialisable format."""

    steps: List[Dict[str, Any]] = []
    for step in workflow:
        if isinstance(step, WorkflowStep):
            steps.append(
                {
                    "module": step.module,
                    "inputs": list(step.inputs),
                    "outputs": list(step.outputs),
                }
            )
        else:
            steps.append(
                {
                    "module": step.get("module", ""),
                    "inputs": step.get("inputs") or step.get("args", []),
                    "outputs": step.get("outputs") or step.get("provides", []),
                }
            )
    return {"steps": steps}


def to_json(workflow: List[WorkflowStep] | List[Dict[str, Any]]) -> str:
    """Serialize ``workflow`` to a JSON string."""

    return json.dumps(workflow_to_dict(workflow), indent=2)


def to_yaml(workflow: List[WorkflowStep] | List[Dict[str, Any]]) -> str:
    """Serialize ``workflow`` to a YAML string."""

    try:  # pragma: no cover - YAML optional
        import yaml  # type: ignore

        return yaml.safe_dump(workflow_to_dict(workflow), sort_keys=False)  # type: ignore[arg-type]
    except Exception:
        return to_json(workflow)


# ---------------------------------------------------------------------------
def to_workflow_spec(workflow: List[WorkflowStep] | List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a lightweight workflow specification.

    The synthesizer internally represents workflows as a sequence of
    :class:`WorkflowStep` objects.  This helper adapts that structure (and the
    historical dictionary representation for backwards compatibility) to the
    format produced by :func:`workflow_spec.to_spec`.
    """

    from workflow_spec import to_spec as _to_spec

    steps: List[Dict[str, Any]] = []
    for step in workflow:
        if isinstance(step, WorkflowStep):
            steps.append(
                {
                    "module": step.module,
                    "inputs": list(step.inputs),
                    "outputs": list(step.outputs),
                    "files": [],
                    "globals": [],
                }
            )
        else:
            steps.append(
                {
                    "module": step.get("module", ""),
                    "inputs": step.get("inputs", []),
                    "outputs": step.get("outputs", []),
                    "files": step.get("files", [])
                    or step.get("files_read", [])
                    + step.get("files_written", []),
                    "globals": step.get("globals", []),
                }
            )
    return _to_spec(steps)


def save_workflow(
    workflow: List[WorkflowStep] | List[Dict[str, Any]],
    path: Path | str | None = None,
) -> Path:
    """Persist ``workflow`` using :func:`workflow_spec.save_spec`."""

    from workflow_spec import save_spec as _save_spec

    spec = to_workflow_spec(workflow)
    out = Path(path) if path is not None else Path("workflow.workflow.json")
    return _save_spec(spec, out)


def evaluate_workflow(workflow: Dict[str, Any]) -> bool:
    """Execute ``workflow`` using the sandbox runner.

    Parameters
    ----------
    workflow:
        Mapping describing a workflow in the format produced by
        :func:`to_workflow_spec` with a top-level ``steps`` list.

    Returns
    -------
    bool
        ``True`` when the sandbox reports success, ``False`` otherwise.
    """

    import logging

    try:  # pragma: no cover - sandbox_runner is optional in tests
        from sandbox_runner import run_generated_workflow
    except Exception as exc:  # pragma: no cover - provide clear error
        raise RuntimeError(
            "sandbox_runner.run_generated_workflow is unavailable"
        ) from exc

    logger = logging.getLogger(__name__)
    try:
        result = run_generated_workflow(workflow)
    except Exception:
        logger.exception("workflow evaluation failed")
        return False

    success = bool(result)
    logger.info("workflow evaluation %s", "succeeded" if success else "failed")
    return success


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
    "ModuleSignature",
    "WorkflowStep",
    "ModuleIOAnalyzer",
    "WorkflowSynthesizer",
    "inspect_module",
    "to_json",
    "to_yaml",
    "to_workflow_spec",
    "evaluate_workflow",
    "save_workflow",
    "synthesise_workflow",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
