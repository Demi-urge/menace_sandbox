from __future__ import annotations

"""Construct a composite module synergy graph."""

import ast
import json
import logging
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

try:  # pragma: no cover - optional dependency
    import networkx as nx
    from networkx.readwrite import json_graph
except Exception as exc:  # pragma: no cover - informative fallback
    raise ImportError(
        "module_synergy_grapher requires the 'networkx' package. "
        "Install it with 'pip install networkx'."
    ) from exc

from governed_embeddings import governed_embed
from module_graph_analyzer import build_import_graph
from vector_utils import cosine_similarity
from retry_utils import with_retry
from dynamic_path_router import resolve_path, get_project_root, resolve_module_path

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


logger = logging.getLogger(__name__)


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
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
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
        self.root = root or get_project_root()
        self.weights_file = (
            Path(weights_file)
            if weights_file
            else None
        )
        if self.weights_file and not self.weights_file.is_absolute():
            self.weights_file = (
                resolve_path(str(self.weights_file.parent)) / self.weights_file.name
            )
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    def _load_weights(self, weights_file: Path | None = None) -> None:
        """Update ``self.coefficients`` from ``weights_file`` if it exists."""

        path = weights_file or self.weights_file
        if path is None:
            path = resolve_path("sandbox_data") / "synergy_weights.json"
        else:
            p = Path(path)
            if not p.is_absolute():
                p = resolve_path(str(p.parent)) / p.name
            path = p
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

    def _embed_doc(self, mod: str, text: str) -> list[float]:
        """Return embedding for ``text`` with retry and logging."""

        def _call() -> list[float]:
            return governed_embed(text) or []

        try:
            return with_retry(
                _call,
                attempts=self.retry_attempts,
                delay=self.retry_delay,
                logger=logger,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding failed for %s: %s", mod, exc)
            return []

    def compute_optimal_weights(
        self, root_path: str | Path, *, weights_file: str | Path | None = None
    ) -> Dict[str, float]:
        """Learn coefficient weights from ``synergy_history.db``.

        This lightweight learner aggregates the historical contribution of each
        synergy signal and normalises the totals so that the resulting weights
        sum to one.  The weights are written to ``weights_file`` (defaulting to
        ``sandbox_data/synergy_weights.json``) and merged into
        ``self.coefficients``.
        """

        root = Path(root_path)
        self.root = root
        db_path = root / "synergy_history.db"
        if not db_path.exists():
            db_path = resolve_path("sandbox_data/synergy_history.db")
        if not db_path.exists() or shd is None:  # type: ignore[truthy-bool]
            return self.coefficients

        try:
            history = with_retry(
                lambda: shd.load_history(db_path),  # type: ignore[attr-defined]
                attempts=self.retry_attempts,
                delay=self.retry_delay,
                logger=logger,
            )
        except Exception as exc:  # pragma: no cover - DB failure
            logger.exception("failed to load synergy history from %s: %s", db_path, exc)
            return self.coefficients

        totals: Dict[str, float] = {k: 0.0 for k in self.coefficients}
        for entry in history:
            if not isinstance(entry, dict):
                continue
            for key in totals:
                try:
                    totals[key] += float(entry.get(key, 0.0))
                except Exception:
                    continue

        denom = sum(totals.values())
        if denom <= 0:
            return self.coefficients

        new_coeffs = {k: v / denom for k, v in totals.items()}
        self.coefficients.update(new_coeffs)

        path = Path(weights_file) if weights_file else None
        if path and not path.is_absolute():
            path = resolve_path(str(path.parent)) / path.name
        self._load_weights(path)
        out = self.weights_file or (resolve_path("sandbox_data") / "synergy_weights.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_text(json.dumps(self.coefficients))
        except Exception:  # pragma: no cover - disk issues
            pass
        return self.coefficients

    def learn_coefficients(
        self, root_path: str | Path, *, weights_file: str | Path | None = None
    ) -> Dict[str, float]:
        """Fit coefficient weights from historical synergy records.

        The function loads past synergy observations from ``synergy_history.db``
        and fits a simple linear regression model that predicts those known
        synergistic links using the available heuristic signals (imports,
        structure, workflow co-occurrence and docstring embeddings).  Learned
        weights are written to ``weights_file`` and applied to
        ``self.coefficients``.
        """

        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - numpy optional
            return self.coefficients

        root = Path(root_path)
        self.root = root

        import_graph = build_import_graph(root)
        modules = list(import_graph.nodes)

        # Reuse cached AST details and embeddings where possible
        cache_path = resolve_path("sandbox_data") / "synergy_cache.json"
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
            bases,
            docs,
            embeddings,
            cache,
            updated,
        ) = self._collect_ast_info(root, modules, cache, use_cache=True)

        # Fetch missing embeddings if any docstrings are present
        missing = [m for m in modules if m not in embeddings and docs.get(m)]
        if missing:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_mod = {
                    executor.submit(self._embed_doc, mod, docs.get(mod, "")): mod
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

        # Build feature matrices
        direct: Dict[Tuple[str, str], float] = {}
        for a, b, data in import_graph.edges(data=True):
            direct[(a, b)] = float(data.get("weight", 1.0))
        max_direct = max(direct.values(), default=0.0)
        direct_norm = {k: v / max_direct for k, v in direct.items()} if max_direct else {}

        deps = {m: set(import_graph.successors(m)) for m in modules}
        shared: Dict[Tuple[str, str], float] = {}
        for a in modules:
            for b in modules:
                if a == b:
                    continue
                score = self._jaccard(deps.get(a, set()), deps.get(b, set()))
                if score:
                    shared[(a, b)] = score

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

        wf_counts = self._workflow_pairs(root, set(modules))
        max_wf = max(wf_counts.values(), default=0)
        wf_norm = {k: v / max_wf for k, v in wf_counts.items()} if max_wf else {}

        hist_counts = self._history_pairs(root, set(modules))
        max_hist = max(hist_counts.values(), default=0.0)
        hist_norm = {k: v / max_hist for k, v in hist_counts.items()} if max_hist else {}

        # Embedding similarities
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

        if not hist_norm:
            return self.coefficients

        X: list[list[float]] = []
        y: list[float] = []
        for pair, target in hist_norm.items():
            import_score = min(1.0, direct_norm.get(pair, 0.0) + shared.get(pair, 0.0))
            struct_score = structure.get(pair, 0.0)
            wf_score = wf_norm.get(pair, 0.0)
            emb_score = embed_sim.get(pair, 0.0)
            X.append([import_score, struct_score, wf_score, emb_score])
            y.append(target)

        if not X:
            return self.coefficients

        A = np.array(X)
        b = np.array(y)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:  # pragma: no cover - regression failure
            return self.coefficients

        new_coeffs = {
            "import": max(0.0, float(coeffs[0])),
            "structure": max(0.0, float(coeffs[1])),
            "cooccurrence": max(0.0, float(coeffs[2])),
            "embedding": max(0.0, float(coeffs[3])),
        }

        self.coefficients.update(new_coeffs)

        # Persist learned weights
        path = Path(weights_file) if weights_file else None
        if path and not path.is_absolute():
            path = resolve_path(str(path.parent)) / path.name
        self._load_weights(path)
        out = self.weights_file or (resolve_path("sandbox_data") / "synergy_weights.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_text(json.dumps(self.coefficients))
        except Exception:  # pragma: no cover - disk issues
            pass
        return self.coefficients

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
        bases: Dict[str, set[str]] = {}
        docs: Dict[str, str] = {}
        embeddings: Dict[str, list[float]] = {}

        updated = False

        def _worker(mod: str) -> tuple[str, dict[str, object] | None, dict[str, object] | None, bool]:
            file = resolve_module_path(mod.replace("/", "."))
            if not file.exists():
                return mod, None, None, False
            mtime = file.stat().st_mtime
            try:
                data = file.read_bytes()
            except Exception:
                return mod, None, None, False
            hash_ = hashlib.sha256(data).hexdigest()
            cached = cache.get(mod) if use_cache else None
            if cached and cached.get("mtime") == mtime and cached.get("hash") == hash_:
                info = {
                    "vars": set(cached.get("vars", [])),
                    "funcs": set(cached.get("funcs", [])),
                    "classes": set(cached.get("classes", [])),
                    "bases": set(cached.get("bases", [])),
                    "doc": str(cached.get("doc", "")),
                    "embedding": cached.get("embedding"),
                }
                return mod, info, None, True

            try:
                tree = ast.parse(data.decode())
            except Exception:
                return mod, None, None, False

            vnames: set[str] = set()
            fnames: set[str] = set()
            cnames: set[str] = set()
            cbases: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            vnames.add(tgt.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    vnames.add(node.target.id)
                elif isinstance(node, ast.FunctionDef):
                    # Build parameter signatures including types and defaults
                    def _format_arg(arg: ast.arg, default: ast.expr | None) -> str:
                        ann = (
                            f":{ast.unparse(arg.annotation)}"
                            if getattr(arg, "annotation", None)
                            else ""
                        )
                        if default is not None:
                            return f"{arg.arg}{ann}={ast.unparse(default)}"
                        return f"{arg.arg}{ann}"

                    pos_args = list(node.args.posonlyargs) + list(node.args.args)
                    pos_defaults = [None] * (
                        len(pos_args) - len(node.args.defaults)
                    ) + list(node.args.defaults)
                    params = [
                        _format_arg(a, d) for a, d in zip(pos_args, pos_defaults)
                    ]
                    if node.args.vararg:
                        params.append("*" + _format_arg(node.args.vararg, None))
                    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
                        params.append(_format_arg(a, d))
                    if node.args.kwarg:
                        params.append("**" + _format_arg(node.args.kwarg, None))
                    ret = (
                        f" -> {ast.unparse(node.returns)}"
                        if getattr(node, "returns", None)
                        else ""
                    )
                    fnames.add(f"{node.name}({', '.join(params)}){ret}")
                elif isinstance(node, ast.ClassDef):
                    base_names: list[str] = []
                    for base in node.bases:
                        try:
                            base_names.append(ast.unparse(base))
                        except Exception:
                            if isinstance(base, ast.Name):
                                base_names.append(base.id)
                    if base_names:
                        cnames.add(f"{node.name}({','.join(base_names)})")
                        cbases.update(base_names)
                    else:
                        cnames.add(node.name)

            doc = ast.get_docstring(tree) or ""

            info = {
                "vars": vnames,
                "funcs": fnames,
                "classes": cnames,
                "bases": cbases,
                "doc": doc,
                "embedding": None,
            }

            cache_entry = {
                "mtime": mtime,
                "hash": hash_,
                "vars": sorted(vnames),
                "funcs": sorted(fnames),
                "classes": sorted(cnames),
                "bases": sorted(cbases),
                "doc": doc,
            }
            return mod, info, cache_entry, False

        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(_worker, mod) for mod in modules]
            for fut in as_completed(futures):
                mod, info, cache_entry, from_cache = fut.result()
                if info is None:
                    continue
                vars_[mod] = info["vars"]  # type: ignore[assignment]
                funcs[mod] = info["funcs"]  # type: ignore[assignment]
                classes[mod] = info["classes"]  # type: ignore[assignment]
                bases[mod] = info["bases"]  # type: ignore[assignment]
                docs[mod] = info["doc"]  # type: ignore[assignment]
                emb = info.get("embedding")
                if emb:
                    embeddings[mod] = emb  # type: ignore[assignment]
                if cache_entry is not None:
                    cache[mod] = cache_entry
                if not from_cache:
                    updated = True

        if not use_cache:
            cache = {m: cache[m] for m in modules if m in cache}
            updated = True

        return vars_, funcs, classes, bases, docs, embeddings, cache, updated

    def _workflow_pairs(
        self, root: Path, modules: set[str]
    ) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        db_path = root / "workflows.db"
        if not db_path.exists():
            return counts
        if WorkflowDB is None:
            return counts

        def _fetch() -> list[tuple[str, str]]:
            wfdb = WorkflowDB(db_path)  # type: ignore[call-arg]
            cur = wfdb.conn.execute("SELECT workflow, task_sequence FROM workflows")
            return cur.fetchall()

        try:
            rows = with_retry(
                _fetch,
                attempts=self.retry_attempts,
                delay=self.retry_delay,
                logger=logger,
            )
            for workflow, sequence in rows:
                mods: set[str] = set()
                for col in (workflow, sequence):
                    if col:
                        mods.update(
                            m.strip() for m in col.split(",") if m.strip() in modules
                        )
                for a, b in combinations(sorted(mods), 2):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
                    counts[(b, a)] = counts.get((b, a), 0) + 1
        except Exception as exc:  # pragma: no cover - DB failure
            logger.exception("failed to read WorkflowDB at %s: %s", db_path, exc)
        return counts

    def _history_pairs(
        self, root: Path, modules: set[str]
    ) -> Dict[Tuple[str, str], float]:
        counts: Dict[Tuple[str, str], float] = {}
        db_path = root / "synergy_history.db"
        if not db_path.exists():
            db_path = resolve_path("sandbox_data/synergy_history.db")
        if not db_path.exists() or shd is None:  # type: ignore[operator]
            return counts
        try:
            history = with_retry(
                lambda: shd.load_history(db_path),  # type: ignore[call-arg]
                attempts=self.retry_attempts,
                delay=self.retry_delay,
                logger=logger,
            )
            for entry in history:
                keys = [k for k in entry if k in modules]
                for a, b in combinations(sorted(keys), 2):
                    val = min(float(entry.get(a, 0.0)), float(entry.get(b, 0.0)))
                    if val <= 0:
                        val = 1.0
                    counts[(a, b)] = counts.get((a, b), 0.0) + val
                    counts[(b, a)] = counts.get((b, a), 0.0) + val
        except Exception as exc:  # pragma: no cover - DB failure
            logger.exception("failed to load synergy history from %s: %s", db_path, exc)
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
            path = resolve_path("sandbox_data") / f"module_synergy_graph{ext}"
        else:
            path = Path(path)
            if not path.is_absolute():
                path = resolve_path(str(path.parent)) / path.name
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
            path = resolve_path("sandbox_data/module_synergy_graph.json")
        else:
            path = Path(path)
            if not path.is_absolute():
                path = resolve_path(str(path))
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

        cache_path = resolve_path("sandbox_data") / "synergy_cache.json"
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
            bases,
            docs,
            embeddings,
            cache,
            updated,
        ) = self._collect_ast_info(root, modules, cache, use_cache=use_cache)

        # Fetch missing embeddings concurrently
        missing = [m for m in modules if m not in embeddings and docs.get(m)]
        if missing:
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_to_mod = {
                    executor.submit(self._embed_doc, mod, docs.get(mod, "")): mod
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
                bases=sorted(bases.get(mod, set())),
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
        out_dir = resolve_path("sandbox_data")
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

        root = self.root or get_project_root()
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
            out_dir = resolve_path("sandbox_data")
            out_dir.mkdir(exist_ok=True)
            self.save(self.graph, out_dir / "module_synergy_graph.json", format="json")
            return self.graph

        cache_path = resolve_path("sandbox_data") / "synergy_cache.json"
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
            bases,
            docs,
            new_embeddings,
            cache,
            updated_cache,
        ) = self._collect_ast_info(root, changed, cache, use_cache=False)

        missing = [m for m in changed if m not in new_embeddings and docs.get(m)]
        if missing:
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_to_mod = {
                    executor.submit(self._embed_doc, mod, docs.get(mod, "")): mod
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
            self.graph.nodes[mod]["bases"] = sorted(bases.get(mod, set()))
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

        out_dir = resolve_path("sandbox_data")
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

        A Dijkstra-style search is used that explores highest scoring paths first
        and finalises each node once its maximum cumulative score is known.  This
        prevents infinite exploration on graphs containing cycles.  The ``bfs``
        parameter is retained for backwards compatibility but has no effect on
        the search order.
        """

        graph = self.graph or self.load()
        if module_name not in graph:
            return set()

        from heapq import heappop, heappush

        # ``score`` values are negated so ``heapq`` acts as a max-heap.
        heap: list[tuple[float, str]] = [(-0.0, module_name)]
        best: Dict[str, float] = {module_name: 0.0}
        finalised: set[str] = set()

        while heap:
            neg_score, node = heappop(heap)
            score = -neg_score
            if node in finalised:
                continue
            finalised.add(node)
            for neigh, data in graph[node].items():
                weight = float(data.get("weight", 0.0))
                new_score = score + weight
                if neigh not in finalised and new_score > best.get(neigh, float("-inf")):
                    best[neigh] = new_score
                    heappush(heap, (-new_score, neigh))

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
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="recompute coefficient weights from synergy history before rebuilding",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.build and not args.cluster:
        parser.print_help()
        return 1

    grapher = ModuleSynergyGrapher(config=args.config)
    if args.build:
        if args.auto_tune:
            grapher.learn_coefficients(Path.cwd())
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
