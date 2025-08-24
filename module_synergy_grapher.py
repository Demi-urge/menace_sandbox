from __future__ import annotations

"""Construct a composite module synergy graph."""

import ast
import json
import sqlite3
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from module_graph_analyzer import build_import_graph
from vector_service import SharedVectorService


@dataclass
class ModuleSynergyGrapher:
    """Build a synergy graph combining structural and historical signals."""

    coefficients: Dict[str, float] = field(
        default_factory=lambda: {
            "import": 1.0,
            "structure": 1.0,
            "workflow": 1.0,
            "semantic": 1.0,
        }
    )
    vector_service: SharedVectorService | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    @staticmethod
    def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
        import math

        la = list(a)
        lb = list(b)
        dot = sum(x * y for x, y in zip(la, lb))
        na = math.sqrt(sum(x * x for x in la))
        nb = math.sqrt(sum(y * y for y in lb))
        if not na or not nb:
            return 0.0
        return dot / (na * nb)

    # ------------------------------------------------------------------
    def _collect_ast_info(self, root: Path, modules: Iterable[str]) -> Tuple[
        Dict[str, set[str]],
        Dict[str, set[str]],
        Dict[str, set[str]],
        Dict[str, str],
    ]:
        vars_: Dict[str, set[str]] = {}
        funcs: Dict[str, set[str]] = {}
        classes: Dict[str, set[str]] = {}
        docs: Dict[str, str] = {}
        for mod in modules:
            file = root / f"{mod}.py"
            if not file.exists():
                continue
            try:
                tree = ast.parse(file.read_text())
            except Exception:
                continue
            vnames: set[str] = set()
            fnames: set[str] = set()
            cnames: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            vnames.add(tgt.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    vnames.add(node.target.id)
                elif isinstance(node, ast.FunctionDef):
                    params = [arg.arg for arg in node.args.args]
                    fnames.add(f"{node.name}({','.join(params)})")
                elif isinstance(node, ast.ClassDef):
                    cnames.add(node.name)
            vars_[mod] = vnames
            funcs[mod] = fnames
            classes[mod] = cnames
            docs[mod] = ast.get_docstring(tree) or ""
        return vars_, funcs, classes, docs

    def _workflow_pairs(self, root: Path, modules: set[str]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        db_path = root / "workflows.db"
        if not db_path.exists():
            return counts
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.execute("SELECT workflow, task_sequence FROM workflows")
            for workflow, sequence in cur.fetchall():
                mods: set[str] = set()
                for col in (workflow, sequence):
                    if col:
                        mods.update(m.strip() for m in col.split(",") if m.strip() in modules)
                for a, b in combinations(sorted(mods), 2):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
                    counts[(b, a)] = counts.get((b, a), 0) + 1
            conn.close()
        except Exception:
            pass
        return counts

    # ------------------------------------------------------------------
    def build_graph(self, root_path: str | Path) -> nx.DiGraph:
        """Return and persist a synergy graph for modules under ``root_path``."""

        root = Path(root_path)
        import_graph = build_import_graph(root)
        modules = list(import_graph.nodes)

        vars_, funcs, classes, docs = self._collect_ast_info(root, modules)

        # Structural similarity
        structure: Dict[Tuple[str, str], float] = {}
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                v = self._jaccard(vars_.get(a, set()), vars_.get(b, set()))
                f = self._jaccard(funcs.get(a, set()), funcs.get(b, set()))
                c = self._jaccard(classes.get(a, set()), classes.get(b, set()))
                if v or f or c:
                    structure[(a, b)] = (v + f + c) / 3

        # Workflow co-occurrences
        workflow_counts = self._workflow_pairs(root, set(modules))
        max_count = max(workflow_counts.values(), default=0)
        workflow = {
            k: (v / max_count if max_count else 0.0) for k, v in workflow_counts.items()
        }

        # Semantic similarity via embeddings
        svc = self.vector_service or SharedVectorService()
        embeds: Dict[str, Iterable[float]] = {}
        for mod, doc in docs.items():
            if not doc.strip():
                continue
            try:
                embeds[mod] = svc.vectorise("text", {"text": doc})
            except Exception:
                continue
        semantic: Dict[Tuple[str, str], float] = {}
        for a in embeds:
            for b in embeds:
                if a == b:
                    continue
                semantic[(a, b)] = self._cosine(embeds[a], embeds[b])

        # Combine metrics
        graph = nx.DiGraph()
        graph.add_nodes_from(modules)
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                imp = (
                    import_graph[a][b]["weight"]
                    if import_graph.has_edge(a, b)
                    else 0.0
                )
                total = (
                    self.coefficients.get("import", 1.0) * imp
                    + self.coefficients.get("structure", 1.0) * structure.get((a, b), 0.0)
                    + self.coefficients.get("workflow", 1.0) * workflow.get((a, b), 0.0)
                    + self.coefficients.get("semantic", 1.0) * semantic.get((a, b), 0.0)
                )
                if total > 0:
                    graph.add_edge(a, b, weight=total)

        # Persist graph
        out_dir = root / "sandbox_data"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "module_synergy_graph.json"
        data = json_graph.node_link_data(graph)
        out_file.write_text(json.dumps(data))
        return graph


__all__ = ["ModuleSynergyGrapher"]
