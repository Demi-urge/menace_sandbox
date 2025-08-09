from __future__ import annotations

"""Simplified knowledge graph for cross-database relationships."""

from typing import Iterable, Optional, Callable, List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore


class _SimpleKMeans:
    """Fallback k-means clustering if scikit-learn is unavailable."""

    def __init__(self, n_clusters: int = 8, iters: int = 10) -> None:
        self.n_clusters = n_clusters
        self.iters = iters
        self.centers: List[List[float]] | None = None

    def fit(self, X: List[List[float]]) -> None:
        import random

        if not X:
            self.centers = []
            return
        self.centers = random.sample(X, min(self.n_clusters, len(X)))
        for _ in range(self.iters):
            clusters = [[] for _ in range(len(self.centers))]
            for vec in X:
                idx = self._closest(vec)[0]
                clusters[idx].append(vec)
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centers[i] = [sum(vals) / len(vals) for vals in zip(*cluster)]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._closest(vec)[0] for vec in X]

    def _closest(self, vec: List[float]) -> tuple[int, float]:
        import math

        best = 0
        best_dist = float("inf")
        for i, c in enumerate(self.centers or []):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, c)))
            if dist < best_dist:
                best = i
                best_dist = dist
        return best, best_dist


class KnowledgeGraph:
    """Lightweight wrapper around ``networkx`` to record entities."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph() if nx else None

    # ------------------------------------------------------------------
    def register_service_dependency(self, bot: str, service: str) -> None:
        """Record that ``bot`` depends on external ``service``."""
        if self.graph is None:
            return
        self.graph.add_node(f"bot:{bot}")
        self.graph.add_node(f"service:{service}")
        self.graph.add_edge(f"bot:{bot}", f"service:{service}", type="service")

    # ------------------------------------------------------------------
    def add_bot(
        self,
        bot_db: object,
        name: str,
        *,
        add_partial: bool = True,
    ) -> None:
        """Add ``name`` from ``bot_db`` with task and dependency edges."""

        if self.graph is None:
            return

        tasks: list[str] = []
        deps: list[str] = []
        try:
            row = bot_db.find_by_name(name)
            if row:
                tasks = row.get("tasks", "").split(",") if row.get("tasks") else []
                deps = (
                    row.get("dependencies", "").split(",")
                    if row.get("dependencies")
                    else []
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed fetching bot info for %s: %s", name, exc)
            if not add_partial:
                return

        node = f"bot:{name}"
        self.graph.add_node(node)
        self.graph.add_edge(node, f"tag:{name}", type="tag")
        for t in tasks:
            if not t:
                continue
            self.graph.add_node(f"task:{t}")
            self.graph.add_edge(node, f"task:{t}", type="task")
        for d in deps:
            if not d:
                continue
            self.graph.add_node(f"bot:{d}")
            self.graph.add_edge(node, f"bot:{d}", type="depends")

    def add_memory_entry(self, key: str, tags: Iterable[str] | None = None) -> None:
        if self.graph is None:
            return
        self.graph.add_node(f"memory:{key}")
        for t in tags or []:
            self.graph.add_node(f"tag:{t}")
            self.graph.add_edge(f"tag:{t}", f"memory:{key}", type="tag")

    def add_code_snippet(self, summary: str, bots: Iterable[str] | None = None) -> None:
        if self.graph is None:
            return
        node = f"code:{summary}"
        self.graph.add_node(node)
        for b in bots or []:
            self.graph.add_node(f"bot:{b}")
            self.graph.add_edge(f"bot:{b}", node, type="code")

    def add_pathway(self, actions: str) -> None:
        if self.graph is None:
            return
        pnode = f"pathway:{actions}"
        self.graph.add_node(pnode)
        steps = [s.strip() for s in actions.split("->") if s.strip()]
        for i in range(len(steps) - 1):
            self.graph.add_node(steps[i])
            self.graph.add_node(steps[i + 1])
            self.graph.add_edge(steps[i], steps[i + 1], type="next")
        if steps:
            self.graph.add_edge(pnode, steps[0], type="start")

    def link_code_to_bot(self, code_db: object, bot_db: object) -> None:
        """Create edges from bots to their code snippets."""

        if self.graph is None:
            return

        import sqlite3

        try:
            conn = sqlite3.connect(code_db.path)
        except Exception:
            return
        cur = conn.cursor()
        try:
            rows = cur.execute("SELECT id, summary FROM code").fetchall()
        except Exception:
            conn.close()
            return
        for cid, summary in rows:
            bots = cur.execute(
                "SELECT bot_id FROM code_bots WHERE code_id=?",
                (cid,),
            ).fetchall()
            for (bid,) in bots:
                try:
                    b_row = bot_db.conn.execute(
                        "SELECT name FROM bots WHERE id=?",
                        (int(bid),),
                    ).fetchone()
                    bname = b_row[0] if b_row else str(bid)
                except Exception:
                    bname = str(bid)
                self.add_code_snippet(summary, [bname])
        conn.close()

    def link_pathway_to_memory(self, pathway_db: object, memory_mgr: object) -> None:
        """Link pathways to memory entries with matching keys."""

        if self.graph is None:
            return

        try:
            p_rows = pathway_db.conn.execute("SELECT actions FROM pathways").fetchall()
            m_rows = memory_mgr.conn.execute("SELECT key FROM memory").fetchall()
        except Exception:
            return

        mem_keys = {k for (k,) in m_rows}
        for (actions,) in p_rows:
            if actions in mem_keys:
                self.graph.add_node(f"pathway:{actions}")
                self.graph.add_node(f"memory:{actions}")
                self.graph.add_edge(
                    f"pathway:{actions}", f"memory:{actions}", type="memory"
                )

    # ------------------------------------------------------------------
    def ingest_bots(self, bot_db: object) -> None:
        """Load all bots from ``bot_db``."""

        try:
            rows = bot_db.fetch_all()
        except Exception:
            rows = []
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else None
            if name:
                self.add_bot(bot_db, name)

    def ingest_memory(self, memory_mgr: object) -> None:
        """Load memory entries from ``memory_mgr``."""

        try:
            rows = memory_mgr.conn.execute("SELECT key, tags FROM memory").fetchall()
        except Exception:
            rows = []
        for key, tags in rows:
            tag_list = str(tags).split() if tags else []
            self.add_memory_entry(key, tag_list)

    def ingest_pathways(self, pathway_db: object) -> None:
        """Load pathways from ``pathway_db``."""

        try:
            rows = pathway_db.conn.execute("SELECT actions FROM pathways").fetchall()
        except Exception:
            rows = []
        for (actions,) in rows:
            self.add_pathway(actions)

    def ingest_code(self, code_db: object, bot_db: object) -> None:
        """Load code snippets and create edges to bots."""

        self.link_code_to_bot(code_db, bot_db)

    def related(self, key: str, depth: int = 1) -> list[str]:
        if self.graph is None:
            return []
        if key not in self.graph:
            return []
        nodes = nx.single_source_shortest_path_length(self.graph, key, depth).keys()
        return [n for n in nodes if n != key]

    # ------------------------------------------------------------------
    # Error and telemetry ingestion
    # ------------------------------------------------------------------

    def add_error(
        self,
        error_id: int,
        message: str,
        *,
        bots: Iterable[str] | None = None,
        models: Iterable[int] | None = None,
        codes: Iterable[int] | None = None,
        summary_lookup: Callable[[int], str] | None = None,
    ) -> None:
        """Add an error node linking to bots, models and code."""

        if self.graph is None:
            return

        enode = f"error:{error_id}"
        self.graph.add_node(enode, message=message)
        for b in bots or []:
            self.graph.add_node(f"bot:{b}")
            self.graph.add_edge(enode, f"bot:{b}", type="bot")
        for m in models or []:
            self.graph.add_node(f"model:{m}")
            self.graph.add_edge(enode, f"model:{m}", type="model")
        for c in codes or []:
            label = str(c)
            if summary_lookup:
                try:
                    label = summary_lookup(c)
                except Exception:
                    label = str(c)
            self.graph.add_node(f"code:{label}")
            self.graph.add_edge(enode, f"code:{label}", type="code")

    def add_telemetry_event(
        self,
        bot_id: str,
        error_type: str | None = None,
        root_module: str | None = None,
        module_counts: dict[str, int] | None = None,
        *,
        patch_id: int | None = None,
        deploy_id: int | None = None,
    ) -> None:
        """Add telemetry relationship from error type to bot.

        Error nodes track their occurrence frequency via the ``weight``
        attribute.  Edges from an error ``cause`` (``error_type``) to the
        affected ``module`` also store a ``weight`` representing how often the
        pair has been observed.
        """

        if self.graph is None:
            return

        bnode = f"bot:{bot_id}"
        self.graph.add_node(bnode)
        if error_type:
            enode = f"error_type:{error_type}"
            self.graph.add_node(enode)
            # increment node weight for frequency of this error type
            self.graph.nodes[enode]["weight"] = self.graph.nodes[enode].get("weight", 0) + 1
            self.graph.add_edge(enode, bnode, type="telemetry")
            mods = module_counts or ({root_module: 1} if root_module else {})
            for mod, cnt in mods.items():
                mnode = f"module:{mod}"
                self.graph.add_node(mnode)
                prev = self.graph.get_edge_data(enode, mnode, {}).get("weight", 0)
                self.graph.add_edge(enode, mnode, type="module", weight=prev + cnt)
            if patch_id is not None:
                pnode = f"patch:{patch_id}"
                self.graph.add_node(pnode)
                self.graph.add_edge(enode, pnode, type="patch")
            if deploy_id is not None:
                dnode = f"deploy:{deploy_id}"
                self.graph.add_node(dnode)
                self.graph.add_edge(enode, dnode, type="deploy")

    def update_error_stats(self, err_db: object) -> None:
        """Synchronise error statistics from ``err_db``.

        The :class:`ErrorDB` aggregates telemetry into an ``error_stats`` table
        with counts for ``(error_type, module)`` pairs.  This method updates the
        corresponding edges and node weights in the graph.
        """

        if self.graph is None:
            return

        try:
            stats = err_db.get_error_stats()  # type: ignore[call-arg]
        except Exception:
            return

        totals: Dict[str, int] = {}
        for row in stats:
            try:
                etype = row["error_type"]
                module = row["module"]
                count = int(row["count"])
            except Exception:
                continue

            enode = f"error_type:{etype}"
            mnode = f"module:{module}"
            self.graph.add_node(enode)
            self.graph.add_node(mnode)
            self.graph.add_edge(enode, mnode, type="module", weight=count)
            totals[enode] = totals.get(enode, 0) + count

        for enode, weight in totals.items():
            self.graph.nodes[enode]["weight"] = weight

    def ingest_error_db(self, err_db: object, code_db: object | None = None) -> None:
        """Load errors and telemetry from ``err_db``."""

        if self.graph is None:
            return

        try:
            cur = err_db.conn
        except Exception:
            return

        def _summary(cid: int) -> str:
            if not code_db:
                return str(cid)
            try:
                row = code_db.conn.execute("SELECT summary FROM code WHERE id=?", (cid,)).fetchone()
                return row[0] if row else str(cid)
            except Exception:
                return str(cid)

        try:
            errs = cur.execute("SELECT id, message FROM errors").fetchall()
        except Exception:
            errs = []
        for eid, msg in errs:
            bots = [r[0] for r in cur.execute("SELECT bot_id FROM error_bot WHERE error_id=?", (eid,)).fetchall()]
            models = [r[0] for r in cur.execute("SELECT model_id FROM error_model WHERE error_id=?", (eid,)).fetchall()]
            codes = [r[0] for r in cur.execute("SELECT code_id FROM error_code WHERE error_id=?", (eid,)).fetchall()]
            self.add_error(eid, msg, bots=bots, models=models, codes=codes, summary_lookup=_summary)

        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(telemetry)").fetchall()]
            sel = ["bot_id", "error_type", "root_module"]
            has_patch = "patch_id" in cols
            has_deploy = "deploy_id" in cols
            if has_patch:
                sel.append("patch_id")
            if has_deploy:
                sel.append("deploy_id")
            query = f"SELECT {', '.join(sel)} FROM telemetry"
            telemetry = cur.execute(query).fetchall()
        except Exception:
            telemetry = []
        for row in telemetry:
            bot, etype, mod = row[0], row[1], row[2]
            idx = 3
            patch = row[idx] if has_patch else None
            if has_patch:
                idx += 1
            deploy = row[idx] if has_deploy else None
            if not bot:
                continue
            for b in str(bot).split(";"):
                if not b:
                    continue
                self.add_telemetry_event(
                    b,
                    etype,
                    mod,
                    patch_id=patch,
                    deploy_id=deploy,
                )

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def cascading_effects(self, key: str, order: int = 5) -> list[str]:
        """Return nodes reachable from ``key`` within ``order`` hops."""

        if self.graph is None:
            return []
        if key not in self.graph:
            return []
        nodes = nx.single_source_shortest_path_length(self.graph, key, cutoff=order).keys()
        return [n for n in nodes if n != key]

    def root_causes(self, bot_name: str, *, hops: Optional[int] = 5) -> list[str]:
        """Return potential root cause nodes for ``bot_name``.

        ``hops`` can be ``None`` for unlimited traversal depth.
        """

        if self.graph is None:
            return []
        bnode = f"bot:{bot_name}"
        if bnode not in self.graph:
            return []
        rev = self.graph.reverse(copy=False)
        nodes = nx.single_source_shortest_path_length(rev, bnode, cutoff=hops).keys()
        return [n for n in nodes if n != bnode and (n.startswith("error:") or n.startswith("error_type:") or n.startswith("code:") or n.startswith("model:"))]

    def suggest_root_cause(
        self,
        bot_name: str,
        *,
        hops: Optional[int] = 5,
        min_cluster_size: int = 2,
    ) -> List[List[str]]:
        """Return clusters of potential root causes for ``bot_name``."""

        causes = self.root_causes(bot_name, hops=hops)
        if not causes:
            return []
        if len(causes) <= min_cluster_size:
            return [causes]

        vectors: List[List[float]] = []
        for n in causes:
            degree = float(self.graph.degree(n)) if self.graph else 0.0
            msg = str(self.graph.nodes[n].get("message", "")) if self.graph else ""
            vectors.append([degree, float(hash(msg) % 1000)])

        labels: List[int]
        if hdbscan:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = list(clusterer.fit_predict(vectors))
            except Exception:
                labels = []
        else:
            n_clusters = max(1, len(causes) // min_cluster_size)
            if KMeans:
                km = KMeans(n_clusters=n_clusters, n_init="auto")  # type: ignore[arg-type]
            else:
                km = _SimpleKMeans(n_clusters=n_clusters)
            km.fit(vectors)
            labels = km.predict(vectors)
            labels = list(labels)

        clusters: Dict[int, List[str]] = {}
        if labels:
            for node, lbl in zip(causes, labels):
                clusters.setdefault(int(lbl), []).append(node)
        else:
            clusters[0] = causes

        return [nodes for _, nodes in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]

    # ------------------------------------------------------------------
    def add_trending_item(self, name: str) -> None:
        """Add a trending item node."""
        if self.graph is None:
            return
        self.graph.add_node(f"trend:{name}")

    # ------------------------------------------------------------------
    def add_crash_trace(self, bot: str, trace: str) -> None:
        """Record a crash trace for *bot*."""
        if self.graph is None:
            return
        node = f"crash:{abs(hash(trace))}"
        self.graph.add_node(node, trace=trace)
        self.graph.add_node(f"bot:{bot}")
        self.graph.add_edge(f"bot:{bot}", node, type="crash")


__all__ = ["KnowledgeGraph"]
