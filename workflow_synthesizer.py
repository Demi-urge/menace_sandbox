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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

try:  # Optional imports; fall back to stubs in tests
    from module_synergy_grapher import ModuleSynergyGrapher
except Exception:  # pragma: no cover - graceful degradation
    ModuleSynergyGrapher = None  # type: ignore[misc]

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

    # ------------------------------------------------------------------
    def synthesize(
        self,
        start_module: str | None = None,
        problem: str | None = None,
        limit: int = 10,
    ) -> List[str]:
        """Return a list of module names relevant to ``start_module`` and ``problem``.

        The method first loads the synergy graph from ``synergy_graph_path`` and,
        if ``start_module`` is provided, expands the cluster around that module.
        When a textual ``problem`` is supplied, semantic matches from
        :class:`IntentClusterer` are merged with the synergy set.
        """

        modules: Set[str] = set()

        # ----- expand via synergy graph
        if start_module and self.module_synergy_grapher is not None:
            try:
                if hasattr(self.module_synergy_grapher, "load"):
                    # Ensure the grapher has the latest graph loaded
                    self.module_synergy_grapher.load(self.synergy_graph_path)
                cluster = self.module_synergy_grapher.get_synergy_cluster(start_module)
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

        return sorted(modules)[:limit]
