from __future__ import annotations

"""Construct a composite module synergy graph."""

import ast
import json
import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from embeddable_db_mixin import EmbeddableDBMixin
from governed_embeddings import governed_embed
from module_graph_analyzer import build_import_graph
from vector_utils import cosine_similarity

try:  # synergy history DB may need package import
    import synergy_history_db as shd  # type: ignore
except Exception:  # pragma: no cover - fallback
    try:
        import menace.synergy_history_db as shd  # type: ignore
    except Exception:  # pragma: no cover - final fallback
        shd = None  # type: ignore

try:  # task_handoff_bot may rely on package context
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - fallback to package import
    try:  # pragma: no cover - alternative package structure
        from menace.task_handoff_bot import WorkflowDB  # type: ignore
    except Exception:  # pragma: no cover - final fallback
        WorkflowDB = None  # type: ignore


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
            "cooccurrence": 1.0,
            "embedding": 1.0,
        }
    )
    graph: nx.DiGraph | None = None
    embedding_threshold: float = 0.8

    # ------------------------------------------------------------------
    @staticmethod
    def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    # ------------------------------------------------------------------
    def _collect_ast_info(
        self, root: Path, modules: Iterable[str]
    ) -> Tuple[
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

    def _workflow_pairs(
        self, root: Path, modules: set[str]
    ) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        db_path = root / "workflows.db"
        if not db_path.exists():
            return counts
        try:
            if WorkflowDB is None:
                return counts
            wfdb = WorkflowDB(db_path)  # type: ignore[call-arg]
            cur = wfdb.conn.execute("SELECT workflow, task_sequence FROM workflows")
            for workflow, sequence in cur.fetchall():
                mods: set[str] = set()
                for col in (workflow, sequence):
                    if col:
                        mods.update(
                            m.strip() for m in col.split(",") if m.strip() in modules
                        )
                for a, b in combinations(sorted(mods), 2):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
                    counts[(b, a)] = counts.get((b, a), 0) + 1
        except Exception:
            pass
        return counts

    def _history_pairs(
        self, root: Path, modules: set[str]
    ) -> Dict[Tuple[str, str], float]:
        counts: Dict[Tuple[str, str], float] = {}
        db_path = root / "synergy_history.db"
        if not db_path.exists():
            db_path = root / "sandbox_data" / "synergy_history.db"
        if not db_path.exists() or shd is None:  # type: ignore[operator]
            return counts
        try:
            history = shd.load_history(db_path)  # type: ignore[call-arg]
            for entry in history:
                keys = [k for k in entry if k in modules]
                for a, b in combinations(sorted(keys), 2):
                    val = min(float(entry.get(a, 0.0)), float(entry.get(b, 0.0)))
                    if val <= 0:
                        val = 1.0
                    counts[(a, b)] = counts.get((a, b), 0.0) + val
                    counts[(b, a)] = counts.get((b, a), 0.0) + val
        except Exception:
            pass
        return counts

    # ------------------------------------------------------------------
    def save(
        self,
        graph: nx.DiGraph | None = None,
        path: str | Path | None = None,
        *,
        format: str = "pickle",
    ) -> Path:
        """Persist ``graph`` to ``path`` in the requested ``format``."""

        graph = graph or self.graph
        if graph is None:
            raise ValueError("graph not built")

        fmt = format.lower()
        ext = ".json" if fmt == "json" else ".pkl"
        if path is None:
            path = Path("sandbox_data") / f"module_synergy_graph{ext}"
        else:
            path = Path(path)
            if not path.suffix:
                path = path.with_suffix(ext)

        path.parent.mkdir(parents=True, exist_ok=True)
        save_graph(graph, path)
        return path

    # ------------------------------------------------------------------
    def load(self, path: str | Path | None = None) -> nx.DiGraph:
        """Hydrate ``self.graph`` from a previously persisted graph file.

        Parameters
        ----------
        path:
            Optional location of the saved graph.  If omitted, the default
            ``sandbox_data/module_synergy_graph.json`` is used.

        Returns
        -------
        ``networkx.DiGraph``
            The loaded graph which is also stored in ``self.graph``.
        """

        if path is None:
            path = Path("sandbox_data/module_synergy_graph.json")
        self.graph = load_graph(path)
        return self.graph

    # ------------------------------------------------------------------
    def build_graph(self, root_path: str | Path) -> nx.DiGraph:
        """Return and persist a synergy graph for modules under ``root_path``."""

        root = Path(root_path)
        import_graph = build_import_graph(root)
        modules = list(import_graph.nodes)

        vars_, funcs, classes, docs = self._collect_ast_info(root, modules)

        # Module docstring embeddings
        embed_dir = root / "sandbox_data"
        embed_dir.mkdir(exist_ok=True)

        class _DocEmbedDB(EmbeddableDBMixin):
            def vector(self, record: str) -> list[float]:
                vec = governed_embed(record)
                return vec or []

        embed_db = _DocEmbedDB(
            index_path=embed_dir / "module_doc_embeddings.ann",
            metadata_path=embed_dir / "module_doc_embeddings.json",
        )
        embeddings: Dict[str, list[float]] = {}
        for mod, doc in docs.items():
            if not doc:
                continue
            vec = embed_db.get_vector(mod)
            if vec is None:
                embed_db.try_add_embedding(mod, doc, "module_doc")
                vec = embed_db.get_vector(mod)
            if vec:
                embeddings[mod] = vec

        # Direct import scores
        direct: Dict[Tuple[str, str], float] = {}
        for a, b, data in import_graph.edges(data=True):
            direct[(a, b)] = float(data.get("weight", 1.0))
        max_direct = max(direct.values(), default=0.0)
        direct_norm = {k: v / max_direct for k, v in direct.items()} if max_direct else {}

        # Shared dependencies
        deps = {m: set(import_graph.successors(m)) for m in modules}
        shared: Dict[Tuple[str, str], float] = {}
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                score = self._jaccard(deps.get(a, set()), deps.get(b, set()))
                if score:
                    shared[(a, b)] = score

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

        # Co-occurrence data
        workflow_counts = self._workflow_pairs(root, set(modules))
        history_counts = self._history_pairs(root, set(modules))
        max_wf = max(workflow_counts.values(), default=0)
        wf_norm = {k: v / max_wf for k, v in workflow_counts.items()} if max_wf else {}
        max_hist = max(history_counts.values(), default=0.0)
        hist_norm = {k: v / max_hist for k, v in history_counts.items()} if max_hist else {}
        co_occ: Dict[Tuple[str, str], float] = {}
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                score = wf_norm.get((a, b), 0.0) + hist_norm.get((a, b), 0.0)
                if score:
                    co_occ[(a, b)] = min(1.0, score)

        # Docstring embedding similarities
        embed_sim: Dict[Tuple[str, str], float] = {}
        thr = self.embedding_threshold
        for a in modules:
            va = embeddings.get(a)
            if not va:
                continue
            for b in modules:
                if a == b:
                    continue
                vb = embeddings.get(b)
                if not vb:
                    continue
                sim = cosine_similarity(va, vb)
                if sim >= thr:
                    embed_sim[(a, b)] = sim

        # Combine metrics
        graph = nx.DiGraph()
        graph.add_nodes_from(modules)
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                import_score = min(
                    1.0, direct_norm.get((a, b), 0.0) + shared.get((a, b), 0.0)
                )
                struct_score = structure.get((a, b), 0.0)
                co_score = co_occ.get((a, b), 0.0)
                emb_score = embed_sim.get((a, b), 0.0)
                total = (
                    self.coefficients.get("import", 1.0) * import_score
                    + self.coefficients.get("structure", 1.0) * struct_score
                    + self.coefficients.get("cooccurrence", 1.0) * co_score
                    + self.coefficients.get("embedding", 1.0) * emb_score
                )
                if total > 0:
                    graph.add_edge(a, b, weight=total)

        self.graph = graph
        out_dir = root / "sandbox_data"
        out_dir.mkdir(exist_ok=True)
        self.save(graph, out_dir / "module_synergy_graph.json", format="json")
        return graph

    # ------------------------------------------------------------------
    def get_synergy_cluster(
        self,
        module_name: str,
        threshold: float = 0.7,
        *,
        bfs: bool = False,
    ) -> set[str]:
        """Return modules whose cumulative synergy from ``module_name`` meets ``threshold``.

        Ensures the graph is loaded before traversal.  When ``bfs`` is ``True`` a
        breadth-first search is used, otherwise a depth-first search is
        performed.
        """

        graph = self.graph or self.load()
        if module_name not in graph:
            return set()

        from collections import deque

        container: deque[tuple[str, float]] | list[tuple[str, float]]
        if bfs:
            container = deque([(module_name, 0.0)])
            pop = container.popleft
            push = container.append
        else:
            container = [(module_name, 0.0)]
            pop = container.pop
            push = container.append

        best: Dict[str, float] = {module_name: 0.0}
        while container:
            node, score = pop()
            for neigh, data in graph[node].items():
                weight = float(data.get("weight", 0.0))
                new_score = score + weight
                if new_score > best.get(neigh, float("-inf")):
                    best[neigh] = new_score
                    push((neigh, new_score))

        cluster = {module_name}
        cluster.update(n for n, s in best.items() if n != module_name and s >= threshold)
        return cluster


def get_synergy_cluster(
    module_name: str,
    threshold: float = 0.7,
    path: str | Path | None = None,
    *,
    bfs: bool = False,
) -> set[str]:
    """Convenience wrapper around :class:`ModuleSynergyGrapher`.

    Parameters
    ----------
    module_name:
        Starting module for the cluster search.
    threshold:
        Minimum cumulative synergy score required for inclusion.
    path:
        Optional path to the persisted graph.  When omitted the default
        location is used.
    bfs:
        If ``True`` a breadth-first traversal is used, otherwise depth-first.
    """

    grapher = ModuleSynergyGrapher()
    if path is not None:
        grapher.load(path)
    return grapher.get_synergy_cluster(module_name, threshold, bfs=bfs)


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
    c.add_argument("--bfs", action="store_true", help="use BFS traversal")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        graph = ModuleSynergyGrapher().build_graph(args.root)
        if args.out:
            ModuleSynergyGrapher().save(graph, args.out)
    elif args.cmd == "cluster":
        cluster = get_synergy_cluster(
            args.module, args.threshold, args.path, bfs=args.bfs
        )
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
