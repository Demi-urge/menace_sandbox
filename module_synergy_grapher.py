from __future__ import annotations

"""Construct a composite module synergy graph."""

import ast
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older Pythons
    import toml as tomllib  # type: ignore
import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import networkx as nx
from networkx.readwrite import json_graph

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
    coefficients: Dict[str, float] = field(default_factory=dict)
    graph: nx.DiGraph | None = None
    embedding_threshold: float = 0.8
    root: Path | None = None
    weights_file: Path | None = None

    def __init__(
        self,
        coefficients: Dict[str, float] | None = None,
        *,
        config: Dict[str, float] | str | Path | None = None,
        graph: nx.DiGraph | None = None,
        embedding_threshold: float = 0.8,
        root: Path | None = None,
        weights_file: Path | None = None,
    ) -> None:
        self.coefficients = {
            "import": 1.0,
            "structure": 1.0,
            "cooccurrence": 1.0,
            "embedding": 1.0,
        }
        if coefficients:
            self.coefficients.update(coefficients)
        if config is not None:
            data: Dict[str, float] | Dict[str, Dict[str, float]]
            if isinstance(config, (str, Path)):
                path = Path(config)
                text = path.read_text()
                if path.suffix.lower() == ".json":
                    data = json.loads(text)
                elif path.suffix.lower() in {".toml", ".tml"}:
                    data = tomllib.loads(text)
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unsupported config format: {path.suffix}")
            else:
                data = config
            if isinstance(data, dict) and "coefficients" in data and isinstance(
                data["coefficients"], dict
            ):
                data = data["coefficients"]  # type: ignore[assignment]
            if isinstance(data, dict):
                self.coefficients.update({k: float(v) for k, v in data.items()})
        self.graph = graph
        self.embedding_threshold = embedding_threshold
        self.root = root
        self.weights_file = weights_file

    # ------------------------------------------------------------------
    def _load_weights(self, weights_file: Path | None = None) -> None:
        """Update ``self.coefficients`` from ``weights_file`` if it exists."""

        path = weights_file or self.weights_file
        if path is None:
            base = self.root if self.root is not None else Path.cwd()
            path = base / "sandbox_data" / "synergy_weights.json"
        else:
            path = Path(path)
        self.weights_file = path
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    self.coefficients.update({k: float(v) for k, v in data.items()})
        except Exception:  # pragma: no cover - ignore malformed files
            pass

    def reload_weights(self) -> None:
        """Reload coefficient weights from ``self.weights_file``."""

        self._load_weights()

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
        self,
        root: Path,
        modules: Iterable[str],
        cache: Dict[str, Dict[str, object]] | None = None,
        *,
        use_cache: bool = True,
    ) -> Tuple[
        Dict[str, set[str]],
        Dict[str, set[str]],
        Dict[str, set[str]],
        Dict[str, str],
        Dict[str, list[float]],
        Dict[str, Dict[str, object]],
        bool,
    ]:
        """Return AST details and doc embeddings for ``modules``.

        ``cache`` is a mapping of module names to previously computed details and
        is persisted to ``sandbox_data/synergy_cache.json`` by the caller. When
        ``use_cache`` is ``True`` the function reuses entries whose source file's
        modification time and hash are unchanged.
        """

        cache = {} if cache is None else dict(cache)

        vars_: Dict[str, set[str]] = {}
        funcs: Dict[str, set[str]] = {}
        classes: Dict[str, set[str]] = {}
        docs: Dict[str, str] = {}
        embeddings: Dict[str, list[float]] = {}

        updated = False
        for mod in modules:
            file = root / f"{mod}.py"
            if not file.exists():
                continue
            mtime = file.stat().st_mtime
            try:
                data = file.read_bytes()
            except Exception:
                continue
            hash_ = hashlib.sha256(data).hexdigest()
            cached = cache.get(mod) if use_cache else None
            if cached and cached.get("mtime") == mtime and cached.get("hash") == hash_:
                vars_[mod] = set(cached.get("vars", []))
                funcs[mod] = set(cached.get("funcs", []))
                classes[mod] = set(cached.get("classes", []))
                docs[mod] = str(cached.get("doc", ""))
                emb = cached.get("embedding")
                if emb:
                    embeddings[mod] = emb  # type: ignore[assignment]
                continue

            try:
                tree = ast.parse(data.decode())
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

            doc = ast.get_docstring(tree) or ""

            vars_[mod] = vnames
            funcs[mod] = fnames
            classes[mod] = cnames
            docs[mod] = doc

            cache[mod] = {
                "mtime": mtime,
                "hash": hash_,
                "vars": sorted(vnames),
                "funcs": sorted(fnames),
                "classes": sorted(cnames),
                "doc": doc,
            }
            updated = True

        if not use_cache:
            cache = {m: cache[m] for m in modules if m in cache}
            updated = True

        return vars_, funcs, classes, docs, embeddings, cache, updated

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
    def build_graph(
        self,
        root_path: str | Path,
        *,
        use_cache: bool = True,
        embed_workers: int = 4,
    ) -> nx.DiGraph:
        """Return and persist a synergy graph for modules under ``root_path``."""

        root = Path(root_path)
        self.root = root
        import_graph = build_import_graph(root)
        modules = list(import_graph.nodes)

        # Refresh coefficient weights from disk before scoring
        self._load_weights()

        cache_path = root / "sandbox_data" / "synergy_cache.json"
        cache: Dict[str, Dict[str, object]] = {}
        if use_cache and cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:  # pragma: no cover - corrupt cache
                cache = {}

        (
            vars_,
            funcs,
            classes,
            docs,
            embeddings,
            cache,
            updated,
        ) = self._collect_ast_info(root, modules, cache, use_cache=use_cache)

        # Fetch missing embeddings concurrently
        missing = [m for m in modules if m not in embeddings and docs.get(m)]
        if missing:
            class _EmbedDB:
                def try_add_embedding(self, record_id: str, text: str, kind: str = "doc"):
                    try:
                        return governed_embed(text) or []
                    except Exception:
                        return []

            embed_db = _EmbedDB()
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_to_mod = {
                    executor.submit(
                        embed_db.try_add_embedding, mod, docs.get(mod, ""), "module_doc"
                    ): mod
                    for mod in missing
                }
                for fut in as_completed(future_to_mod):
                    mod = future_to_mod[fut]
                    vec = fut.result()
                    if vec:
                        embeddings[mod] = vec
                        cache.setdefault(mod, {})["embedding"] = vec
            updated = True

        if updated:
            cache_path.parent.mkdir(exist_ok=True)
            try:
                cache_path.write_text(json.dumps(cache))
            except Exception:  # pragma: no cover - disk issues
                pass

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
        for mod in modules:
            graph.add_node(
                mod,
                vars=sorted(vars_.get(mod, set())),
                funcs=sorted(funcs.get(mod, set())),
                classes=sorted(classes.get(mod, set())),
                doc=docs.get(mod, ""),
                embedding=embeddings.get(mod),
            )
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
    def update_graph(
        self,
        changed_modules: Iterable[str],
        *,
        embed_workers: int = 4,
    ) -> nx.DiGraph:
        """Refresh graph data for ``changed_modules`` only.

        AST details, embeddings and edge weights touching the specified
        modules are recomputed and merged into ``self.graph`` which is then
        persisted.  ``changed_modules`` should contain module names relative to
        ``self.root``.
        """

        if self.graph is None:
            raise ValueError("graph not built")

        root = self.root or Path.cwd()
        changed: set[str] = {m for m in changed_modules}
        if not changed:
            return self.graph

        import_graph = build_import_graph(root)
        modules = set(import_graph.nodes)

        # Remove modules that disappeared from the codebase
        for mod in list(changed):
            if mod not in modules:
                if self.graph.has_node(mod):
                    self.graph.remove_node(mod)
                changed.remove(mod)

        if not changed:
            out_dir = root / "sandbox_data"
            out_dir.mkdir(exist_ok=True)
            self.save(self.graph, out_dir / "module_synergy_graph.json", format="json")
            return self.graph

        cache_path = root / "sandbox_data" / "synergy_cache.json"
        cache: Dict[str, Dict[str, object]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:  # pragma: no cover - corrupt cache
                cache = {}

        (
            vars_,
            funcs,
            classes,
            docs,
            new_embeddings,
            cache,
            updated_cache,
        ) = self._collect_ast_info(root, changed, cache, use_cache=False)

        missing = [m for m in changed if m not in new_embeddings and docs.get(m)]
        if missing:
            class _EmbedDB:
                def try_add_embedding(self, record_id: str, text: str, kind: str = "doc"):
                    try:
                        return governed_embed(text) or []
                    except Exception:
                        return []

            embed_db = _EmbedDB()
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_to_mod = {
                    executor.submit(
                        embed_db.try_add_embedding, mod, docs.get(mod, ""), "module_doc"
                    ): mod
                    for mod in missing
                }
                for fut in as_completed(future_to_mod):
                    mod = future_to_mod[fut]
                    vec = fut.result()
                    if vec:
                        new_embeddings[mod] = vec
                        cache.setdefault(mod, {})["embedding"] = vec
            updated_cache = True

        if updated_cache:
            cache_path.parent.mkdir(exist_ok=True)
            try:
                cache_path.write_text(json.dumps(cache))
            except Exception:  # pragma: no cover - disk issues
                pass

        embeddings: Dict[str, list[float]] = {}
        for mod in self.graph.nodes:
            vec = self.graph.nodes[mod].get("embedding")
            if vec:
                embeddings[mod] = vec  # existing vectors

        embeddings.update(new_embeddings)
        for mod in changed:
            if mod not in new_embeddings:
                embeddings.pop(mod, None)

        for mod in changed:
            self.graph.add_node(mod)
            self.graph.nodes[mod]["vars"] = sorted(vars_.get(mod, set()))
            self.graph.nodes[mod]["funcs"] = sorted(funcs.get(mod, set()))
            self.graph.nodes[mod]["classes"] = sorted(classes.get(mod, set()))
            self.graph.nodes[mod]["doc"] = docs.get(mod, "")
            self.graph.nodes[mod]["embedding"] = embeddings.get(mod)

        all_modules = set(self.graph.nodes)
        deps = {m: set(import_graph.successors(m)) for m in modules}

        direct: Dict[Tuple[str, str], float] = {}
        for a, b, data in import_graph.edges(data=True):
            if a in changed or b in changed:
                direct[(a, b)] = float(data.get("weight", 1.0))
        max_direct = max(direct.values(), default=0.0)
        direct_norm = {k: v / max_direct for k, v in direct.items()} if max_direct else {}

        shared: Dict[Tuple[str, str], float] = {}
        for a in changed:
            for b in modules:
                if a == b:
                    continue
                score = self._jaccard(deps.get(a, set()), deps.get(b, set()))
                if score:
                    shared[(a, b)] = score
                    shared[(b, a)] = score

        structure: Dict[Tuple[str, str], float] = {}
        for a in changed:
            va = set(vars_.get(a, set()))
            fa = set(funcs.get(a, set()))
            ca = set(classes.get(a, set()))
            for b in all_modules:
                if a == b:
                    continue
                vb = set(self.graph.nodes[b].get("vars", []))
                fb = set(self.graph.nodes[b].get("funcs", []))
                cb = set(self.graph.nodes[b].get("classes", []))
                v = self._jaccard(va, vb)
                f = self._jaccard(fa, fb)
                c = self._jaccard(ca, cb)
                if v or f or c:
                    s = (v + f + c) / 3
                    structure[(a, b)] = s
                    structure[(b, a)] = s

        workflow_counts = self._workflow_pairs(root, all_modules)
        history_counts = self._history_pairs(root, all_modules)
        max_wf = max(workflow_counts.values(), default=0)
        wf_norm = {k: v / max_wf for k, v in workflow_counts.items()} if max_wf else {}
        max_hist = max(history_counts.values(), default=0.0)
        hist_norm = {k: v / max_hist for k, v in history_counts.items()} if max_hist else {}
        co_occ: Dict[Tuple[str, str], float] = {}
        for a in changed:
            for b in all_modules:
                if a == b:
                    continue
                score = wf_norm.get((a, b), 0.0) + hist_norm.get((a, b), 0.0)
                if score:
                    co_occ[(a, b)] = min(1.0, score)
                    co_occ[(b, a)] = co_occ[(a, b)]

        embed_sim: Dict[Tuple[str, str], float] = {}
        thr = self.embedding_threshold
        for a in changed:
            va = embeddings.get(a)
            if not va:
                continue
            for b in all_modules:
                if a == b:
                    continue
                vb = embeddings.get(b)
                if not vb:
                    continue
                sim = cosine_similarity(va, vb)
                if sim >= thr:
                    embed_sim[(a, b)] = sim
                    embed_sim[(b, a)] = sim

        for a in all_modules:
            for b in all_modules:
                if a == b:
                    continue
                if a in changed or b in changed:
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
                        self.graph.add_edge(a, b, weight=total)
                    elif self.graph.has_edge(a, b):
                        self.graph.remove_edge(a, b)

        out_dir = root / "sandbox_data"
        out_dir.mkdir(exist_ok=True)
        self.save(self.graph, out_dir / "module_synergy_graph.json", format="json")
        return self.graph

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
    """Command line interface for :mod:`module_synergy_grapher`.

    ``--build`` rebuilds the synergy graph for the current repository while
    ``--cluster`` prints modules whose cumulative synergy with the supplied
    module meets the ``--threshold`` value.
    """

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Module Synergy Grapher CLI")
    parser.add_argument(
        "--build",
        action="store_true",
        help="regenerate the synergy graph for the current repository",
    )
    parser.add_argument(
        "--cluster",
        metavar="MODULE",
        help="module name whose synergistic neighbours should be printed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="minimum cumulative synergy required for inclusion",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="JSON/TOML file providing coefficient overrides",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute AST info and embeddings ignoring any caches",
    )
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=4,
        help="number of threads for embedding retrieval",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.build and not args.cluster:
        parser.print_help()
        return 1

    grapher = ModuleSynergyGrapher(config=args.config)
    if args.build:
        grapher.build_graph(
            Path.cwd(), use_cache=not args.no_cache, embed_workers=args.embed_workers
        )

    if args.cluster:
        cluster = grapher.get_synergy_cluster(args.cluster, threshold=args.threshold)
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
