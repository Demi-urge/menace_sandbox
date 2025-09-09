from __future__ import annotations

"""Lightweight workflow synthesizer utilities.

This module exposes :class:`WorkflowSynthesizer` which combines structural
signals from :class:`~module_synergy_grapher.ModuleSynergyGrapher` with
semantic intent matches provided by :class:`~intent_clusterer.IntentClusterer`.

Workflows are scored by blending edge weights from the synergy graph with
intent match scores.  The :meth:`WorkflowSynthesizer.generate_workflows`
method exposes ``synergy_weight`` and ``intent_weight`` parameters to control
this blend.  Scores are normalised by path length and a diversity ratio based
on the number of distinct modules.  Workflows accrue a penalty for unresolved
or duplicated dependencies.  The resulting score is::

    score = synergy_weight * (S / n * diversity)
          + intent_weight * (I / n * diversity)
          - penalty

where ``S`` is the sum of synergy edge weights, ``I`` is the sum of intent
match scores and ``penalty`` counts unresolved inputs and duplicate dependency
uses.  This normalisation prevents long or repetitive workflows from
dominating and penalises incomplete dependency chains.

The synthesizer is intentionally small and focuses on expanding an initial set
of modules either by following the synergy graph around a starting module or by
searching for modules related to a textual problem description.
"""

import argparse
import ast
import json
import logging
import importlib
import types
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Union,
    Sequence,
    Iterable,
    get_args,
    get_origin,
    get_type_hints,
)
from context_builder_util import create_context_builder

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - support running as a package
    from .fcntl_compat import flock, LOCK_EX, LOCK_SH, LOCK_UN
except Exception:  # pragma: no cover - allow running as script
    from fcntl_compat import flock, LOCK_EX, LOCK_SH, LOCK_UN

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


logger = logging.getLogger(__name__)

WINNING_SEQUENCES_PATH = resolve_path("sandbox_data") / "winning_sequences.json"


def record_winning_sequence(sequence: List[str]) -> None:
    """Append ``sequence`` to the shared winning sequence registry."""
    try:
        data: List[List[str]] = []
        if WINNING_SEQUENCES_PATH.exists():
            data = json.loads(WINNING_SEQUENCES_PATH.read_text())
        data.append(sequence)
        WINNING_SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        WINNING_SEQUENCES_PATH.write_text(json.dumps(data, indent=2))
    except Exception:  # pragma: no cover - best effort logging
        logger.exception("failed to record winning sequence")


def _extract_type_names(type_str: str | None) -> List[str]:
    """Return atomic type names from an annotation expression.

    The parser understands common generic containers like ``Tuple``/``list``/
    ``Dict`` and returns the constituent element types.  Unknown expressions fall
    back to the raw string representation.  The helper intentionally keeps
    dependencies light by operating directly on :mod:`ast` nodes.
    """

    if not type_str:
        return []
    try:
        node = ast.parse(type_str, mode="eval").body
    except SyntaxError:
        return [type_str]

    names: List[str] = []

    GENERIC_NAMES = {"tuple", "list", "set", "dict", "sequence", "union", "optional"}

    def _visit(n: ast.AST) -> None:
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            try:  # Python <3.9 compatibility
                names.append(ast.unparse(n))
            except Exception:  # pragma: no cover - best effort
                pass
        elif isinstance(n, ast.Subscript):
            base = getattr(n.value, "id", None)
            if base is None and isinstance(n.value, ast.Attribute):
                try:
                    base = ast.unparse(n.value)
                except Exception:  # pragma: no cover - best effort
                    base = None
            base_lower = base.lower() if isinstance(base, str) else ""
            if base and base_lower not in GENERIC_NAMES:
                names.append(base)
            slice_node = n.slice
            if isinstance(slice_node, (ast.Tuple, ast.List)):
                elts = slice_node.elts
            else:
                elts = [slice_node]
            if base_lower == "dict" and len(elts) == 2:
                _visit(elts[1])
            else:
                for elt in elts:
                    _visit(elt)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            _visit(n.left)
            _visit(n.right)
        elif isinstance(n, (ast.Tuple, ast.List)):
            for elt in n.elts:
                _visit(elt)
        elif isinstance(n, ast.Constant):
            # Skip ``None`` used in Optional types
            if n.value is None:
                return

    _visit(node)
    return names


@dataclass
class WorkflowStep:
    """Lightweight representation of an ordered workflow step."""

    module: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)


class ModuleIOAnalyzer:
    """Analyze modules to determine their IO signatures with caching."""

    def __init__(self, cache_path: Path = resolve_path("sandbox_data") / "io_signatures.json") -> None:
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                flock(fh.fileno(), LOCK_SH)
                try:
                    self._cache = json.load(fh)
                finally:
                    flock(fh.fileno(), LOCK_UN)
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

        # Extract return names and keys per function
        return_names: Dict[str, Set[str]] = {}

        class _ReturnVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: D401
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_Return(self, node: ast.Return) -> None:  # noqa: D401
                if not self.stack:
                    return
                fname = self.stack[-1]

                def _collect(v: ast.AST) -> Set[str]:
                    names: Set[str] = set()
                    if isinstance(v, ast.Name):
                        names.add(v.id)
                    elif isinstance(v, (ast.Tuple, ast.List)):
                        for elt in v.elts:
                            names.update(_collect(elt))
                    elif isinstance(v, ast.Dict):
                        for k, val in zip(v.keys, v.values):
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                names.add(str(k.value))
                            names.update(_collect(val))
                    return names

                return_names.setdefault(fname, set()).update(_collect(node.value))

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            _ReturnVisitor().visit(tree)
        except Exception:  # pragma: no cover - best effort
            pass

        # Augment function metadata with return types and names
        for fname, info in sig.functions.items():
            ret = info.get("returns")
            info["return_types"] = _extract_type_names(ret)
            info["return_names"] = sorted(return_names.get(fname, set()))

        flat_return_names: Set[str] = set()
        for names in return_names.values():
            flat_return_names.update(names)

        outputs: Set[str] = set(sig.files_written) | module_globals | flat_return_names
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

        lock_fh = self.cache_path.open("a+")
        try:
            flock(lock_fh.fileno(), LOCK_EX)
            tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.cache_path)
        finally:
            try:
                flock(lock_fh.fileno(), LOCK_UN)
            finally:
                lock_fh.close()
        return sig


def inspect_module(module_name: str) -> "ModuleSignature":
    """Return :class:`ModuleSignature` information for ``module_name``."""

    path = resolve_path(Path(module_name.replace(".", "/")).with_suffix(".py"))
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
    workflow_score_details: List[Dict[str, Any]]
    winning_sequences: List[List[str]]
    _winning_set: Set[Tuple[str, ...]]

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
            else resolve_path("sandbox_data") / "module_synergy_graph.json"
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
        self.workflow_score_details = []
        self._load_winning_sequences()

    def _load_winning_sequences(self) -> None:
        try:
            seqs = (
                json.loads(WINNING_SEQUENCES_PATH.read_text())
                if WINNING_SEQUENCES_PATH.exists()
                else []
            )
        except Exception:
            seqs = []
        self.winning_sequences = seqs
        self._winning_set = {tuple(s) for s in seqs}

    def reinforce_sequence(self, sequence: List[str]) -> None:
        record_winning_sequence(sequence)
        self.winning_sequences.append(sequence)
        self._winning_set.add(tuple(sequence))

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
            except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                logger.warning(
                    "IntentClusterer initialisation failed",
                    exc_info=exc,
                    extra={
                        "operation": "IntentClusterer.__init__",
                        "path": str(self.intent_db_path or "<default>"),
                    },
                )
                self.intent_clusterer = None
        if IntentDB is not None and self.intent_db is None:
            try:
                path = self.intent_db_path or Path("intent.db")
                self.intent_db = IntentDB(path)
            except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                logger.warning(
                    "IntentDB load failed",
                    exc_info=exc,
                    extra={"operation": "IntentDB", "path": str(path)},
                )
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
        annotations_cache: Dict[str, Dict[str, Set[str]]] = {}

        def _hint_to_names(hint: Any) -> Set[str]:
            names: Set[str] = set()
            origin = get_origin(hint)
            if origin in {list, set, tuple}:
                for arg in get_args(hint):
                    names.update(_hint_to_names(arg))
            elif origin is dict:
                args = get_args(hint)
                if len(args) == 2:
                    names.update(_hint_to_names(args[1]))
            elif origin in {types.UnionType, Union}:
                for arg in get_args(hint):
                    names.update(_hint_to_names(arg))
            elif origin is not None:
                names.update(_hint_to_names(origin))
                for arg in get_args(hint):
                    names.update(_hint_to_names(arg))
            else:
                if hint is type(None):
                    return names
                if hasattr(hint, "__module__") and hasattr(hint, "__qualname__"):
                    names.add(f"{hint.__module__}.{hint.__qualname__}")
            return names

        for mod in modules:
            name = mod.name or getattr(mod, "module", "")
            if not name:
                raise ValueError("ModuleSignature missing name attribute")

            args: Set[str] = set()
            annotations_cache[name] = {}

            module_obj = None
            try:
                module_obj = importlib.import_module(name)
            except Exception:  # pragma: no cover - best effort
                module_obj = None

            for fn_name, fn in mod.functions.items():
                args.update(fn.get("args", []))
                if module_obj is not None and hasattr(module_obj, fn_name):
                    try:
                        obj = getattr(module_obj, fn_name)
                        hints = get_type_hints(obj, globalns=vars(module_obj))
                    except Exception:  # pragma: no cover - best effort
                        hints = {}
                    for arg, hint in hints.items():
                        if arg == "return":
                            for t in _hint_to_names(hint):
                                produced_by_type.setdefault(t, set()).add(name)
                        else:
                            annotations_cache[name].setdefault(arg, set()).update(
                                _hint_to_names(hint)
                            )
                else:
                    for arg, ann in fn.get("annotations", {}).items():
                        for t in _extract_type_names(ann):
                            annotations_cache[name].setdefault(arg, set()).add(t)
                    for ret in fn.get("return_types", []):
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

        # Inspect function signatures to separate required and optional args
        sig_required: Dict[str, Set[str]] = {}
        sig_optional: Dict[str, Set[str]] = {}
        for mod in modules:
            req: Set[str] = set()
            opt: Set[str] = set()
            path = resolve_path(Path(mod.name.replace(".", "/")).with_suffix(".py"))
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name in mod.functions:
                        pos_args = node.args.posonlyargs + node.args.args
                        num_required = len(pos_args) - len(node.args.defaults)
                        for a in pos_args[:num_required]:
                            req.add(a.arg)
                        for a in pos_args[num_required:]:
                            opt.add(a.arg)
                        for a, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                            (req if default is None else opt).add(a.arg)
                        if node.args.vararg:
                            opt.add(node.args.vararg.arg)
                        if node.args.kwarg:
                            opt.add(node.args.kwarg.arg)
            except OSError:
                for fn in mod.functions.values():
                    req.update(fn.get("args", []))
            sig_required[mod.name] = req
            sig_optional[mod.name] = opt

        for mod in modules:
            name = mod.name
            required_args = sig_required.get(name, set())
            optional_args = sig_optional.get(name, set())
            all_args = required_args | optional_args
            required_files: Set[str] = set(mod.files_read)
            required_globals: Set[str] = set(mod.globals)

            dependencies: Set[str] = set()

            # function args
            for item in all_args:
                matched = False
                ann = annotations_cache.get(name, {}).get(item, set())
                for t in ann:
                    producers = produced_by_type.get(t)
                    if producers:
                        producer = _select_best(name, producers)
                        if producer != name:
                            dependencies.add(producer)
                        matched = True
                        break
                if not matched:
                    producers = produced_globals.get(item)
                    if producers:
                        producer = _select_best(name, producers)
                        if producer != name:
                            dependencies.add(producer)
                        matched = True
                if not matched and item in required_args:
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
        max_depth: int | None = None,
        intent_limit: int = 20,
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
            Maximum graph depth to explore from ``start_module``.  ``None``
            explores the entire reachable graph. ``1`` restricts expansion to
            direct neighbours.
        intent_limit:
            Maximum number of intent matches to consider for ``problem`` based
            expansion.  Controls the breadth of intent search; higher values may
            introduce less relevant modules while lower values focus on the most
            confident matches.
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
                    import heapq

                    visited: Set[str] = {start_module}
                    # priority queue ordered by negative weight so that the
                    # highest weight edges are expanded first
                    heap: list[tuple[float, int, str]] = []
                    for neigh, data in graph[start_module].items():
                        weight = float(data.get("weight", 0.0))
                        if weight < threshold:
                            continue
                        heapq.heappush(heap, (-weight, 1, neigh))

                    while heap:
                        neg_weight, depth, node = heapq.heappop(heap)
                        if node in visited:
                            continue
                        if max_depth is not None and depth > max_depth:
                            continue
                        visited.add(node)
                        modules.add(node)
                        if max_depth is not None and depth >= max_depth:
                            continue
                        for neigh, data in graph[node].items():
                            weight = float(data.get("weight", 0.0))
                            if weight < threshold or neigh in visited:
                                continue
                            heapq.heappush(heap, (-weight, depth + 1, neigh))
            except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                logger.warning(
                    "Synergy graph expansion failed",
                    exc_info=exc,
                    extra={
                        "operation": "expand_cluster_synergy",
                        "path": str(self.synergy_graph_path),
                    },
                )

        if problem:
            if self.intent_clusterer is None and self.intent_db is None:
                self.load_intent_clusters()
            if self.intent_clusterer is not None:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(
                        problem, top_k=intent_limit
                    )
                    for m in matches:
                        path = getattr(m, "path", None) or getattr(m, "module", None)
                        if path:
                            p = Path(str(path))
                            modules.add(p.stem)
                except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                    logger.warning(
                        "Intent search via clusterer failed",
                        exc_info=exc,
                        extra={
                            "operation": "IntentClusterer.find_modules_related_to",
                            "path": str(self.intent_db_path or "<default>"),
                        },
                    )
            elif self.intent_db is not None:
                try:
                    vec = self.intent_db.encode_text(problem)  # type: ignore[attr-defined]
                    results = self.intent_db.search_by_vector(vec, top_k=intent_limit)
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
                except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                    logger.warning(
                        "IntentDB search failed",
                        exc_info=exc,
                        extra={
                            "operation": "IntentDB.search",
                            "path": str(
                                getattr(
                                    self.intent_db,
                                    "path",
                                    self.intent_db_path or "intent.db",
                                )
                            ),
                        },
                    )

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
            start_module=start_module,
            problem=problem,
            threshold=threshold,
            max_depth=1,
        )

        if start_module:
            modules.add(start_module)

        analyzer = ModuleIOAnalyzer()
        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = resolve_path(Path(mod.replace(".", "/")).with_suffix(".py"))
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
            module_path = resolve_path(Path(start.replace(".", "/")).with_suffix(".py"))
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
        min_score: float = float("-inf"),
        auto_evaluate: bool = False,
        runner_config: Dict[str, Any] | None = None,
    ) -> List[List[WorkflowStep]]:
        """Generate candidate workflows beginning at ``start_module``.

        Candidate workflows are constructed by traversing the synergy graph
        from ``start_module`` to discover alternate module orderings that
        satisfy dependency constraints.  Each candidate is scored and the best
        ``limit`` workflows are returned.

        Parameters
        ----------
        start_module:
            Dotted name of the module that should start each workflow.
        problem:
            Optional textual description used for intent matching.  Requires
            either ``intent_clusterer`` or ``intent_db`` to be available.
        limit:
            Maximum number of workflows to return.
        max_depth:
            Maximum graph depth to explore from ``start_module``. ``None``
            explores the entire reachable graph.
        synergy_weight:
            Multiplier applied to synergy graph edge weights when scoring
            workflows.
        intent_weight:
            Multiplier applied to intent match scores when scoring workflows.
        min_score:
            Minimum partial score required to continue exploring a workflow
            extension.  Partial scores that fall below this threshold are
            pruned to reduce search space.
        auto_evaluate:
            When ``True``, each generated workflow is executed via
            :func:`evaluate_workflow` and the success flag is stored alongside
            score details.
        """

        if ModuleSignature is None or get_io_signature is None:
            raise ValueError("Structural analysis helpers are unavailable")

        modules = self.expand_cluster(
            start_module=start_module, problem=problem, max_depth=max_depth
        )
        modules.add(start_module)

        analyzer = ModuleIOAnalyzer()
        signatures: List[ModuleSignature] = []
        for mod in sorted(modules):
            path = resolve_path(Path(mod.replace(".", "/")).with_suffix(".py"))
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
                for ret in fn.get("return_types", []):
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
        step_map: Dict[str, WorkflowStep] = {}
        for mod in signatures:
            name = mod.name
            required: Set[str] = set(mod.files_read) | set(mod.globals)
            for fn in mod.functions.values():
                required.update(fn.get("args", []))
            dependencies: Set[str] = set()
            unresolved: List[str] = []
            for item in required:
                matched = False
                ann = annotations_cache[name].get(item)
                if ann:
                    for t in _extract_type_names(ann):
                        producers = produced_by_type.get(t)
                        if producers:
                            producer = _select_best(name, producers)
                            if producer != name:
                                dependencies.add(producer)
                            matched = True
                            break
                if not matched:
                    producers = produced_by_name.get(item)
                    if producers:
                        producer = _select_best(name, producers)
                        if producer != name:
                            dependencies.add(producer)
                        matched = True
                if not matched:
                    unresolved.append(item)
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
                unresolved=sorted(unresolved),
            )

        from graphlib import CycleError, TopologicalSorter

        try:  # Validate acyclic dependencies
            TopologicalSorter(deps).static_order()
        except CycleError as exc:  # pragma: no cover - cycle detection
            raise ValueError(f"Cyclic dependency detected: {exc.args}") from exc

        if start_module not in step_map:
            raise ValueError(f"Start module {start_module!r} not found")

        score_map: Dict[str, float] = {}
        if problem:
            if self.intent_clusterer is None and self.intent_db is None:
                self.load_intent_clusters()
            if self.intent_clusterer is not None:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(problem, top_k=50)
                    for m in matches:
                        path = getattr(m, "path", None) or getattr(m, "module", None)
                        if path:
                            mod = Path(str(path)).stem
                            score = float(
                                getattr(m, "score", getattr(m, "similarity", 1.0))
                            )
                            score_map[mod] = score
                except Exception:  # pragma: no cover - best effort
                    pass
            elif self.intent_db is not None:
                try:
                    vec = self.intent_db.encode_text(problem)  # type: ignore[attr-defined]
                    results = self.intent_db.search_by_vector(vec, top_k=50)
                    for rid, dist in results:
                        path = rid
                        if isinstance(rid, int) and hasattr(self.intent_db, "conn"):
                            row = self.intent_db.conn.execute(
                                "SELECT path FROM intent_modules WHERE id=?",
                                (rid,),
                            ).fetchone()
                            path = row["path"] if row else None
                        if path:
                            mod = Path(str(path)).stem
                            score_map[mod] = 1.0 / (1.0 + float(dist))
                except Exception:  # pragma: no cover - best effort
                    pass
            if score_map:
                max_score = max(score_map.values()) or 1.0
                for mod in list(score_map):
                    score_map[mod] /= max_score

        # Explore alternative chains by traversing the synergy graph rather
        # than following a single greedy dependency order.  A breadth first
        # search enumerates candidate paths starting from ``start_module`` and
        # only extends a path when the next module's dependencies are satisfied
        # by modules already in the path.  ``max_candidates`` bounds exploration
        # to avoid combinatorial explosion while still surfacing diverse
        # orderings.

        graph = getattr(self.module_synergy_grapher, "graph", None)
        orders: List[List[str]] = []
        max_candidates = max(limit * 10, 10)

        init_dep_counts = {dep: 1 for dep in deps[start_module]}
        init_unresolved = len(step_map[start_module].unresolved)
        init_intent = score_map.get(start_module, 0.0)
        init_synergy = 0.0
        init_dup = 0.0
        init_length = 1
        init_diversity = 1.0
        init_syn_norm = (init_synergy / init_length) * init_diversity
        init_int_norm = (init_intent / init_length) * init_diversity
        init_score = (
            synergy_weight * init_syn_norm
            + intent_weight * init_int_norm
            - (init_unresolved + init_dup)
        )

        if graph is None or start_module not in graph:
            queue: deque[
                Tuple[
                    List[str],
                    Set[str],
                    float,
                    float,
                    Dict[str, int],
                    float,
                    float,
                    float,
                ]
            ] = deque()
            queue.append(
                (
                    [start_module],
                    set(step_map) - {start_module},
                    init_synergy,
                    init_intent,
                    init_dep_counts,
                    init_unresolved,
                    init_dup,
                    init_score,
                )
            )
            seen: Set[Tuple[str, ...]] = set()
            while queue and len(orders) < max_candidates:
                (
                    order,
                    remaining,
                    syn_sum,
                    intent_sum,
                    dep_counts,
                    unresolved_pen,
                    dup_pen,
                    score,
                ) = queue.popleft()
                key = tuple(order)
                if key in seen:
                    continue
                seen.add(key)
                orders.append(order.copy())

                if (max_depth is not None and len(order) - 1 >= max_depth) or score < min_score:
                    continue

                available = [m for m in remaining if deps[m].issubset(order)]
                for mod in sorted(available):
                    prev = order[-1]
                    weight = 0.0
                    if graph is not None:
                        if graph.has_edge(prev, mod):
                            weight = float(graph[prev][mod].get("weight", 0.0))
                        elif graph.has_edge(mod, prev):
                            weight = float(graph[mod][prev].get("weight", 0.0))
                    new_syn = syn_sum + weight
                    new_int = intent_sum + score_map.get(mod, 0.0)
                    new_dep_counts = dep_counts.copy()
                    inc_dup = sum(1 for d in deps[mod] if new_dep_counts.get(d, 0) >= 1)
                    for d in deps[mod]:
                        new_dep_counts[d] = new_dep_counts.get(d, 0) + 1
                    new_unresolved = unresolved_pen + len(step_map[mod].unresolved)
                    new_dup = dup_pen + inc_dup
                    new_order = order + [mod]
                    length = len(new_order)
                    diversity = len(set(new_order)) / length
                    syn_norm = (new_syn / length) * diversity
                    int_norm = (new_int / length) * diversity
                    new_score = (
                        synergy_weight * syn_norm
                        + intent_weight * int_norm
                        - (new_unresolved + new_dup)
                    )
                    if new_score >= min_score:
                        queue.append(
                            (
                                new_order,
                                remaining - {mod},
                                new_syn,
                                new_int,
                                new_dep_counts,
                                new_unresolved,
                                new_dup,
                                new_score,
                            )
                        )
        else:
            queue: deque[
                Tuple[
                    List[str],
                    float,
                    float,
                    Dict[str, int],
                    float,
                    float,
                    float,
                ]
            ] = deque()
            queue.append(
                (
                    [start_module],
                    init_synergy,
                    init_intent,
                    init_dep_counts,
                    init_unresolved,
                    init_dup,
                    init_score,
                )
            )
            seen: Set[Tuple[str, ...]] = set()
            while queue and len(orders) < max_candidates:
                (
                    order,
                    syn_sum,
                    intent_sum,
                    dep_counts,
                    unresolved_pen,
                    dup_pen,
                    score,
                ) = queue.popleft()
                key = tuple(order)
                if key in seen:
                    continue
                seen.add(key)
                orders.append(order.copy())

                if (max_depth is not None and len(order) - 1 >= max_depth) or score < min_score:
                    continue

                last = order[-1]
                neighbours = graph[last]
                neigh_items = sorted(
                    neighbours.items(),
                    key=lambda item: float(item[1].get("weight", 0.0)),
                    reverse=True,
                )
                for neigh, data in neigh_items:
                    if neigh not in step_map:
                        continue
                    if neigh in order:
                        continue
                    if not deps[neigh].issubset(order):
                        continue
                    weight = float(data.get("weight", 0.0))
                    new_syn = syn_sum + weight
                    new_int = intent_sum + score_map.get(neigh, 0.0)
                    new_dep_counts = dep_counts.copy()
                    inc_dup = sum(1 for d in deps[neigh] if new_dep_counts.get(d, 0) >= 1)
                    for d in deps[neigh]:
                        new_dep_counts[d] = new_dep_counts.get(d, 0) + 1
                    new_unresolved = unresolved_pen + len(step_map[neigh].unresolved)
                    new_dup = dup_pen + inc_dup
                    new_order = order + [neigh]
                    length = len(new_order)
                    diversity = len(set(new_order)) / length
                    syn_norm = (new_syn / length) * diversity
                    int_norm = (new_int / length) * diversity
                    new_score = (
                        synergy_weight * syn_norm
                        + intent_weight * int_norm
                        - (new_unresolved + new_dup)
                    )
                    if new_score >= min_score:
                        queue.append(
                            (
                                new_order,
                                new_syn,
                                new_int,
                                new_dep_counts,
                                new_unresolved,
                                new_dup,
                                new_score,
                            )
                        )

        entries: List[Tuple[float, List[WorkflowStep], float, float, float]] = []
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

            length = max(len(order), 1)
            diversity = len(set(order)) / length
            synergy_norm = (synergy_score / length) * diversity
            intent_norm = (intent_score / length) * diversity

            unresolved_penalty = sum(len(step.unresolved) for step in workflow)
            dep_counts: Dict[str, int] = {}
            for m in order:
                for dep in deps[m]:
                    dep_counts[dep] = dep_counts.get(dep, 0) + 1
            duplicate_penalty = sum(v - 1 for v in dep_counts.values() if v > 1)

            penalty = unresolved_penalty + duplicate_penalty
            score = synergy_weight * synergy_norm + intent_weight * intent_norm - penalty
            if tuple(order) in self._winning_set:
                score += 0.5 * len(order)
            entries.append((score, workflow, synergy_norm, intent_norm, penalty))

        entries.sort(key=lambda x: x[0], reverse=True)
        self.workflow_scores = []
        self.generated_workflows = []
        self.workflow_score_details = []
        out_dir = resolve_path("sandbox_data") / "generated_workflows"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, (score, wf, syn, intent, penalty) in enumerate(entries[:limit]):
            details: Dict[str, Any] = {
                "score": score,
                "synergy": syn,
                "intent": intent,
                "penalty": penalty,
            }
            if auto_evaluate:
                try:
                    success = evaluate_workflow(
                        to_workflow_spec(wf), runner_config=runner_config
                    )
                except Exception:
                    logger.exception("auto evaluation failed")
                    success = False
                details["success"] = success
            self.workflow_scores.append(score)
            self.generated_workflows.append(wf)
            self.workflow_score_details.append(details)

            name = wf[0].module.replace(".", "_") if wf else f"workflow_{idx}"
            path = out_dir / f"{name}_{idx}.workflow.json"
            path.write_text(to_json(wf, metadata=details), encoding="utf-8")
        from db_router import GLOBAL_ROUTER

        os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
        try:
            import sandbox_runner

            builder = create_context_builder()
            added, syn_ok, intent_ok = sandbox_runner.post_round_orphan_scan(
                Path.cwd(), router=GLOBAL_ROUTER, context_builder=builder
            )
            logger.info(
                "post_round_orphan_scan added=%d synergy_ok=%s intent_ok=%s",
                len(added),
                syn_ok,
                intent_ok,
            )
        except Exception:  # pragma: no cover - best effort
            logger.warning("orphan integration failed", exc_info=True)

        return self.generated_workflows

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of generated workflows."""
        return {
            "workflows": [
                workflow_to_dict(wf, metadata=info)
                for wf, info in zip(self.generated_workflows, self.workflow_score_details)
            ]
        }

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
        base = Path(path) if path is not None else resolve_path("sandbox_data") / "generated_workflows"
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
def workflow_to_dict(
    workflow: List[WorkflowStep] | List[Dict[str, Any]],
    *,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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
    data: Dict[str, Any] = {"steps": steps}
    if metadata:
        data.update(metadata)
    return data


def to_json(
    workflow: List[WorkflowStep] | List[Dict[str, Any]],
    *,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Serialize ``workflow`` to a JSON string."""

    return json.dumps(workflow_to_dict(workflow, metadata=metadata), indent=2)


def to_yaml(
    workflow: List[WorkflowStep] | List[Dict[str, Any]],
    *,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Serialize ``workflow`` to a YAML string."""

    try:  # pragma: no cover - YAML optional
        import yaml  # type: ignore

        return yaml.safe_dump(
            workflow_to_dict(workflow, metadata=metadata), sort_keys=False
        )  # type: ignore[arg-type]
    except Exception:
        return to_json(workflow, metadata=metadata)


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
    *,
    parent_id: str | None = None,
    mutation_description: str = "",
) -> Tuple[Path, Dict[str, Any]]:
    """Persist ``workflow`` using :func:`workflow_spec.save_spec`.

    Parameters
    ----------
    workflow:
        Sequence of workflow steps.
    path:
        Optional destination for the workflow specification.
    parent_id:
        Identifier of the workflow this was derived from, if any.
    mutation_description:
        Human readable description of how the workflow was changed.

    Returns
    -------
    tuple
        ``(path, metadata)`` where ``path`` is the location of the saved
        specification and ``metadata`` contains details such as the generated
        ``workflow_id`` and ``created_at`` timestamp.
    """

    from workflow_spec import save_spec as _save_spec
    from workflow_run_summary import save_summary as _save_summary

    spec = to_workflow_spec(workflow)
    spec["metadata"] = {
        "parent_id": parent_id,
        "mutation_description": mutation_description,
    }
    out = Path(path) if path is not None else Path("workflow.workflow.json")
    out_path = _save_spec(spec, out)

    metadata = spec.get("metadata", {})
    try:
        saved = json.loads(out_path.read_text())
        metadata = saved.get("metadata", metadata)
    except Exception:  # pragma: no cover - best effort
        saved = spec
        metadata = dict(metadata)

    workflow_id = metadata.get("workflow_id")
    if workflow_id:
        try:
            summary_path = _save_summary(str(workflow_id), out_path.parent)
            metadata["summary_path"] = str(summary_path)
            _save_spec(saved, out_path, summary_path=summary_path)
        except Exception:
            pass

    return out_path, metadata


def evaluate_workflow(
    workflow: Dict[str, Any], runner_config: Dict[str, Any] | None = None
) -> bool:
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

    from sandbox_runner import WorkflowSandboxRunner

    logger = logging.getLogger(__name__)
    steps = workflow.get("steps", [])

    def _call() -> bool:
        ok = True
        for step in steps:
            module = step.get("module")
            func_name = step.get("func") or step.get("function") or "main"
            try:
                mod = importlib.import_module(module)
                fn = getattr(mod, func_name, getattr(mod, "main", None))
                if callable(fn):
                    fn()
            except Exception:
                ok = False
        return ok

    try:
        runner = WorkflowSandboxRunner()
        rc = dict(runner_config or {})
        rc.setdefault("safe_mode", True)
        metrics = runner.run(_call, **rc)
        success = all(m.success for m in getattr(metrics, "modules", []))
    except Exception:
        logger.exception("workflow evaluation failed")
        return False

    logger.info("workflow evaluation %s", "succeeded" if success else "failed")
    return success


def consume_planner_suggestions(chains: Iterable[Sequence[str]]) -> List[Path]:
    """Persist planner suggested chains as workflow specs."""
    paths: List[Path] = []
    for idx, chain in enumerate(chains, start=1):
        steps = [WorkflowStep(module=str(m)) for m in chain]
        path = resolve_path("sandbox_data") / f"planner_chain_{idx}.workflow.json"
        try:
            save_workflow(steps, path)
            paths.append(path)
        except Exception:
            logging.getLogger(__name__).exception(
                "failed to save planner suggestion", extra={"chain": chain}
            )
    return paths


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


def generate_workflow_variants(
    workflow_spec: Sequence[Any],
    *,
    limit: int = 5,
    validator: Callable[[List[str]], bool] | None = None,
    intent_clusterer: Any | None = None,
    synergy_threshold: float = 0.7,
) -> List[List[str]]:
    """Return up to ``limit`` workflow variations.

    Parameters
    ----------
    workflow_spec:
        Either a sequence of module names or a sequence of step mappings with
        ``module``, ``inputs`` and ``outputs`` fields as produced by
        :mod:`workflow_spec`.
    limit:
        Maximum number of variants to return.
    validator:
        Optional callback invoked with each candidate sequence.  Only sequences
        for which the callback returns ``True`` are yielded.  When omitted a
        lightweight structural check is performed using
        :meth:`WorkflowSynthesizer.resolve_dependencies`.
    intent_clusterer:
        Optional :class:`intent_clusterer.IntentClusterer` instance used to
        locate additional modules for injection.
    synergy_threshold:
        Threshold passed to :func:`module_synergy_grapher.get_synergy_cluster`
        when retrieving swap candidates.
    """

    base: List[str] = []
    steps: List[Dict[str, Set[str]]] = []
    for item in workflow_spec:
        if isinstance(item, str):
            base.append(item)
            steps.append({"inputs": set(), "outputs": set()})
        else:
            mod = getattr(item, "module", None) or item.get("module")
            ins = set(getattr(item, "inputs", None) or item.get("inputs", []) or [])
            outs = set(getattr(item, "outputs", None) or item.get("outputs", []) or [])
            base.append(str(mod))
            steps.append({"inputs": ins, "outputs": outs})

    variants: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = {tuple(base)}

    def _finalize() -> List[List[str]]:
        from db_router import GLOBAL_ROUTER

        os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
        try:  # pragma: no cover - optional dependency
            import sandbox_runner

            builder = create_context_builder()
            added, syn_ok, intent_ok = sandbox_runner.post_round_orphan_scan(
                Path.cwd(), router=GLOBAL_ROUTER, context_builder=builder
            )
            logger.info(
                "post_round_orphan_scan added=%d synergy_ok=%s intent_ok=%s",
                len(added),
                syn_ok,
                intent_ok,
            )
        except Exception:  # pragma: no cover - best effort
            logger.warning("orphan integration failed", exc_info=True)
        return variants

    if validator is None:
        analyzer = ModuleIOAnalyzer()
        checker = WorkflowSynthesizer()

        def _default_validator(order: List[str]) -> bool:
            try:
                modules = [analyzer.analyze(Path(m)) for m in order]
                resolved = checker.resolve_dependencies(modules)
                if any(step.unresolved for step in resolved):
                    return False
                return [s.module for s in resolved] == order
            except Exception:
                return False

        validator = _default_validator

    def _add(candidate: List[str]) -> None:
        if len(variants) >= limit:
            return
        key = tuple(candidate)
        if key in seen:
            return
        if validator(candidate):
            variants.append(candidate)
            seen.add(key)

    # 1. Swap modules using synergy clusters
    if get_synergy_cluster is not None:
        for idx, mod in enumerate(base):
            try:
                cluster = set(get_synergy_cluster(mod, threshold=synergy_threshold))
            except Exception:
                cluster = {mod}
            for alt in cluster - {mod}:
                variant = base.copy()
                variant[idx] = alt
                _add(variant)
                if len(variants) >= limit:
                    return _finalize()

    # 2. Reorder adjacent steps when allowed by dependency rules
    for i in range(len(base) - 1):
        a, b = steps[i], steps[i + 1]
        if a["outputs"] & b["inputs"]:
            continue
        if b["outputs"] & a["inputs"]:
            continue
        variant = base.copy()
        variant[i], variant[i + 1] = variant[i + 1], variant[i]
        _add(variant)
        if len(variants) >= limit:
            return _finalize()

    # 3. Inject additional modules suggested by intent clusterer
    if intent_clusterer is None and IntentClusterer is not None:
        try:
            intent_clusterer = IntentClusterer()
        except Exception:  # pragma: no cover - optional dependency
            intent_clusterer = None

    if intent_clusterer is not None:
        suggestions: List[str] = []
        try:
            query_text = " ".join(base)
            if hasattr(intent_clusterer, "query"):
                hits = intent_clusterer.query(query_text, top_k=limit)
            elif hasattr(intent_clusterer, "search"):
                hits = intent_clusterer.search(query_text, top_k=limit)
            elif hasattr(intent_clusterer, "_search_related"):
                hits = intent_clusterer._search_related(query_text, top_k=limit)
            else:
                hits = []
            for h in hits or []:
                path = getattr(h, "path", None)
                members = getattr(h, "members", None)
                if path:
                    suggestions.append(Path(path).stem)
                elif members:
                    suggestions.extend(Path(m).stem for m in members)
        except Exception:
            suggestions = []

        for mod in suggestions:
            if mod in base:
                continue
            for pos in range(len(base) + 1):
                variant = base.copy()
                variant.insert(pos, mod)
                _add(variant)
                if len(variants) >= limit:
                    return _finalize()

    return _finalize()


def generate_variants(
    workflow: Sequence[str],
    n: int,
    synergy_graph: Any | None,
    intent_clusterer: Any | None,
) -> List[List[str]]:
    """Generate up to ``n`` variant workflows.

    Variants are created by swapping modules using synergy suggestions,
    reordering steps when structurally valid and inserting modules suggested by
    ``intent_clusterer``.
    """

    base = list(workflow)
    variants: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = {tuple(base)}
    analyzer = ModuleIOAnalyzer()
    checker = WorkflowSynthesizer(
        module_synergy_grapher=synergy_graph
        if ModuleSynergyGrapher is not None
        and isinstance(synergy_graph, ModuleSynergyGrapher)
        else None
    )

    def _finalize() -> List[List[str]]:
        from db_router import GLOBAL_ROUTER

        os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
        try:  # pragma: no cover - optional dependency
            import sandbox_runner

            builder = create_context_builder()
            added, syn_ok, intent_ok = sandbox_runner.post_round_orphan_scan(
                Path.cwd(), router=GLOBAL_ROUTER, context_builder=builder
            )
            logger.info(
                "post_round_orphan_scan added=%d synergy_ok=%s intent_ok=%s",
                len(added),
                syn_ok,
                intent_ok,
            )
        except Exception:  # pragma: no cover - best effort
            logger.warning("orphan integration failed", exc_info=True)
        return variants[:n]

    def _is_valid(order: List[str]) -> bool:
        try:
            modules = [analyzer.analyze(Path(m)) for m in order]
            steps = checker.resolve_dependencies(modules)
            if any(step.unresolved for step in steps):
                return False
            return [s.module for s in steps] == order
        except Exception:
            return False

    def _add(order: List[str]) -> None:
        if len(variants) >= n:
            return
        key = tuple(order)
        if key in seen:
            return
        if _is_valid(order):
            variants.append(order)
            seen.add(key)

    # 1. Swap modules based on synergy suggestions
    if synergy_graph is not None:
        for idx, mod in enumerate(base):
            suggestions: Set[str] = set()
            try:
                if hasattr(synergy_graph, "get_synergy_cluster"):
                    suggestions = set(synergy_graph.get_synergy_cluster(mod))
                else:
                    graph = getattr(synergy_graph, "graph", None) or synergy_graph
                    neigh: Set[str] = set()
                    if hasattr(graph, "successors"):
                        try:
                            neigh.update(graph.successors(mod))
                        except Exception:
                            pass
                    if hasattr(graph, "predecessors"):
                        try:
                            neigh.update(graph.predecessors(mod))
                        except Exception:
                            pass
                    suggestions = neigh
            except Exception:
                suggestions = set()
            for s in suggestions:
                if s == mod:
                    continue
                variant = base.copy()
                variant[idx] = s
                _add(variant)
                if len(variants) >= n:
                    return _finalize()

    # 2. Reorder steps by swapping adjacent modules
    for i in range(len(base) - 1):
        variant = base.copy()
        variant[i], variant[i + 1] = variant[i + 1], variant[i]
        _add(variant)
        if len(variants) >= n:
            return _finalize()

    # 3. Insert modules suggested by intent_clusterer
    if intent_clusterer is not None:
        suggestions: List[str] = []
        try:
            query_text = " ".join(base)
            if hasattr(intent_clusterer, "query"):
                hits = intent_clusterer.query(query_text, top_k=n)
            elif hasattr(intent_clusterer, "search"):
                hits = intent_clusterer.search(query_text, top_k=n)
            elif hasattr(intent_clusterer, "_search_related"):
                hits = intent_clusterer._search_related(query_text, top_k=n)
            else:
                hits = []
            for h in hits or []:
                path = getattr(h, "path", None)
                members = getattr(h, "members", None)
                if path:
                    suggestions.append(Path(path).stem)
                elif members:
                    suggestions.extend(Path(m).stem for m in members)
        except Exception:
            suggestions = []

        for mod in suggestions:
            if mod in base:
                continue
            for pos in range(len(base) + 1):
                variant = base.copy()
                variant.insert(pos, mod)
                _add(variant)
                if len(variants) >= n:
                    return _finalize()

    return _finalize()


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
    "consume_planner_suggestions",
    "save_workflow",
    "synthesise_workflow",
    "generate_workflow_variants",
    "generate_variants",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
