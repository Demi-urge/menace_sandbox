from __future__ import annotations

"""Construct a composite module synergy graph."""

import ast
import json
import pickle
import sqlite3
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from module_graph_analyzer import build_import_graph
from vector_service import SharedVectorService


def save_graph(graph: nx.Graph, path: str | Path) -> None:
    """Persist ``graph`` to ``path`` in JSON or pickle format."""

    path = Path(path)
    if path.suffix == ".json":
        data = json_graph.node_link_data(graph)
        path.write_text(json.dumps(data))
    elif path.suffix in {".pkl", ".pickle"}:
        with path.open("wb") as fh:
            pickle.dump(graph, fh)
    else:
        raise ValueError(f"Unsupported graph format: {path.suffix}")


def load_graph(path: str | Path) -> nx.Graph:
    """Load a graph previously persisted with :func:`save_graph`."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return json_graph.node_link_graph(data)
    elif path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            return pickle.load(fh)
    else:
        raise ValueError(f"Unsupported graph format: {path.suffix}")


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
        save_graph(graph, out_file)
        return graph


def get_synergy_cluster(
    module_name: str,
    threshold: float = 0.7,
    path: str | Path | None = None,
) -> set[str]:
    """Return modules reachable from ``module_name`` with edge weight ≥ ``threshold``."""

    path = Path(path) if path else Path("sandbox_data/module_synergy_graph.json")
    graph = load_graph(path)
    if module_name not in graph:
        return set()
    cluster: set[str] = {module_name}
    stack = [module_name]
    while stack:
        node = stack.pop()
        for neigh, data in graph[node].items():
            if data.get("weight", 0.0) >= threshold and neigh not in cluster:
                cluster.add(neigh)
                stack.append(neigh)
    return cluster


def _main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Module Synergy Grapher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="rebuild synergy graph")
    b.add_argument("root", nargs="?", default=".", help="root path for modules")
    b.add_argument("--out", default=None, help="output file (json or pickle)")

    c = sub.add_parser("cluster", help="query synergy cluster")
    c.add_argument("module", help="module name to query")
    c.add_argument("--threshold", type=float, default=0.7)
    c.add_argument("--path", default=None, help="path to saved graph")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        graph = ModuleSynergyGrapher().build_graph(args.root)
        if args.out:
            save_graph(graph, args.out)
    elif args.cmd == "cluster":
        cluster = get_synergy_cluster(args.module, args.threshold, args.path)
        for mod in sorted(cluster):
            print(mod)
    return 0


def main() -> int:  # pragma: no cover - thin wrapper
    return _main()


__all__ = [
    "ModuleSynergyGrapher",
    "save_graph",
    "load_graph",
    "get_synergy_cluster",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
