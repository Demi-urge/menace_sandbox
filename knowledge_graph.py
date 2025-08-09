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

        Error nodes track their occurrence frequency via both ``weight`` and a
        ``frequency`` attribute for backward compatibility.  Edges from an
        error ``cause`` (``error_type``) to the affected ``module`` store a
        ``weight`` representing how often the pair has been observed.  The
        calling ``root_module`` is also linked to the error type via a ``cause``
        edge so upstream modules can be traced.
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
            self.graph.nodes[enode]["frequency"] = (
                self.graph.nodes[enode].get("frequency", 0) + 1
            )
            self.graph.add_edge(enode, bnode, type="telemetry")
            mods = module_counts or ({root_module: 1} if root_module else {})
            for mod, cnt in mods.items():
                mnode = f"module:{mod}"
                self.graph.add_node(mnode)
                prev = self.graph.get_edge_data(enode, mnode, {}).get("weight", 0)
                self.graph.add_edge(enode, mnode, type="module", weight=prev + cnt)
            if root_module:
                mnode = f"module:{root_module}"
                self.graph.add_node(mnode)
                prev = self.graph.get_edge_data(mnode, enode, {}).get("weight", 0)
                self.graph.add_edge(mnode, enode, type="cause", weight=prev + 1)
            if patch_id is not None:
                pnode = f"patch:{patch_id}"
                self.graph.add_node(pnode)
                self.graph.add_edge(enode, pnode, type="patch")
            if deploy_id is not None:
                dnode = f"deploy:{deploy_id}"
                self.graph.add_node(dnode)
                self.graph.add_edge(enode, dnode, type="deploy")

    def add_error_instance(self, category: str, module: str, cause: str | None = None) -> None:
        """Record an observed ``category``/``module``/``cause`` chain.

        This creates nodes ``error_category:<category>``, ``module:<module>`` and
        ``cause:<cause>`` (if provided) with edges ``category -> module`` and
        ``module -> cause``.  Occurrence counts are stored on the edges and also
        aggregated on the ``error_category`` node so frequency queries can be
        performed later.
        """

        if self.graph is None:
            return

        cnode = f"error_category:{category}"
        mnode = f"module:{module}"
        self.graph.add_node(cnode)
        self.graph.add_node(mnode)

        # increment weights for category node and category->module edge
        self.graph.nodes[cnode]["weight"] = self.graph.nodes[cnode].get("weight", 0) + 1
        prev = self.graph.get_edge_data(cnode, mnode, {}).get("weight", 0)
        self.graph.add_edge(cnode, mnode, type="module", weight=prev + 1)

        if cause:
            conode = f"cause:{cause}"
            self.graph.add_node(conode)
            self.graph.nodes[conode]["weight"] = self.graph.nodes[conode].get("weight", 0) + 1
            prev = self.graph.get_edge_data(mnode, conode, {}).get("weight", 0)
            self.graph.add_edge(mnode, conode, type="cause", weight=prev + 1)

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

        has_patch = has_deploy = has_category = False
        has_module = has_cause = has_freq = False
        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(telemetry)").fetchall()]
            sel = ["rowid", "bot_id", "error_type", "root_module"]
            has_patch = "patch_id" in cols
            has_deploy = "deploy_id" in cols
            has_category = "category" in cols
            has_module = "module" in cols
            has_cause = "cause" in cols
            has_freq = "frequency" in cols
            if has_patch:
                sel.append("patch_id")
            if has_deploy:
                sel.append("deploy_id")
            if has_category:
                sel.append("category")
            if has_module:
                sel.append("module")
            if has_cause:
                sel.append("cause")
            if has_freq:
                sel.append("frequency")
            query = f"SELECT {', '.join(sel)} FROM telemetry"
            telemetry = cur.execute(query).fetchall()
        except Exception:
            telemetry = []
        for row in telemetry:
            rowid, bot, etype, root_mod = row[0], row[1], row[2], row[3]
            idx = 4
            patch = row[idx] if has_patch else None
            if has_patch:
                idx += 1
            deploy = row[idx] if has_deploy else None
            if has_deploy:
                idx += 1
            category = row[idx] if has_category else None
            if has_category:
                idx += 1
            module = row[idx] if has_module else root_mod
            if has_module:
                idx += 1
            cause = row[idx] if has_cause else None
            if has_cause:
                idx += 1
            freq = row[idx] if has_freq else 1
            try:
                cnt = int(freq)
            except Exception:
                cnt = 1
            if not bot:
                continue
            for b in str(bot).split(";"):
                if not b:
                    continue
                for _ in range(max(cnt, 1)):
                    self.add_telemetry_event(
                        b,
                        etype,
                        root_mod,
                        patch_id=patch,
                        deploy_id=deploy,
                    )
            if category and module:
                for _ in range(max(cnt, 1)):
                    self.add_error_instance(category, module, cause)
            try:
                cur.execute(
                    "UPDATE telemetry SET frequency=COALESCE(frequency,0)+? WHERE rowid=?",
                    (cnt, rowid),
                )
            except Exception:
                pass
        try:
            cur.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Error clustering and failure chain analysis
    # ------------------------------------------------------------------

    def error_clusters(
        self, min_weight: int = 1, min_cluster_size: int = 2
    ) -> Dict[str, int]:
        """Group ``error_type`` nodes by frequency and shared modules.

        Returns a mapping of error node ids to their cluster label. Cluster
        assignments are also stored on each ``error_type`` node under the
        ``cluster`` attribute for later lookups.
        """

        if self.graph is None:
            return {}

        errors: List[str] = [
            n
            for n, data in self.graph.nodes(data=True)
            if n.startswith("error_type:") and data.get("weight", 0) >= min_weight
        ]
        if not errors:
            return {}

        modules = sorted(
            {
                m
                for e in errors
                for _, m, _ in self.graph.out_edges(e, data=True)
                if m.startswith("module:")
            }
        )
        if not modules:
            return {}
        mod_index = {m: i for i, m in enumerate(modules)}
        vectors: List[List[float]] = []
        for e in errors:
            vec = [0.0] * (len(modules) + 1)
            for _, m, d in self.graph.out_edges(e, data=True):
                if m.startswith("module:"):
                    idx = mod_index[m]
                    vec[idx] = float(d.get("weight", 0.0))
            vec[-1] = float(self.graph.nodes[e].get("weight", 0.0))
            vectors.append(vec)

        labels: List[int]
        if hdbscan and len(errors) >= min_cluster_size:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = list(clusterer.fit_predict(vectors))
            except Exception:
                labels = [0] * len(errors)
        else:
            n_clusters = max(1, len(errors) // max(1, min_cluster_size))
            if KMeans:
                km = KMeans(n_clusters=n_clusters, n_init="auto")  # type: ignore[arg-type]
            else:
                km = _SimpleKMeans(n_clusters=n_clusters)
            km.fit(vectors)
            labels = list(km.predict(vectors))

        mapping: Dict[str, int] = {}
        for node, lbl in zip(errors, labels):
            mapping[node] = int(lbl)
            self.graph.nodes[node]["cluster"] = int(lbl)
        return mapping

    def cluster_failure_chain(self, error_type: str, top: int = 5) -> List[str]:
        """Return modules most associated with the cluster of ``error_type``."""

        if self.graph is None:
            return []
        enode = (
            error_type if error_type.startswith("error_type:") else f"error_type:{error_type}"
        )
        if enode not in self.graph:
            return []
        cluster = self.graph.nodes[enode].get("cluster")
        if cluster is None:
            self.error_clusters()
            cluster = self.graph.nodes[enode].get("cluster")
            if cluster is None:
                return []
        modules: Dict[str, int] = {}
        for node, data in self.graph.nodes(data=True):
            if node.startswith("error_type:") and data.get("cluster") == cluster:
                for _, m, d in self.graph.out_edges(node, data=True):
                    if m.startswith("module:"):
                        modules[m] = modules.get(m, 0) + int(d.get("weight", 1))
        return [m for m, _ in sorted(modules.items(), key=lambda x: x[1], reverse=True)[:top]]

    def bot_failure_chain(self, bot: str, top: int = 5) -> List[str]:
        """Return likely module failure chain for ``bot`` based on cluster history."""

        if self.graph is None:
            return []
        bnode = f"bot:{bot}"
        if bnode not in self.graph:
            return []
        modules: Dict[str, int] = {}
        for enode, _, _ in self.graph.in_edges(bnode, data=True):
            if enode.startswith("error_type:"):
                chain = self.cluster_failure_chain(enode, top=top)
                for m in chain:
                    modules[m] = modules.get(m, 0) + 1
        return [m for m, _ in sorted(modules.items(), key=lambda x: x[1], reverse=True)[:top]]

    def bot_patch_candidates(self, bot: str, top: int = 3) -> List[str]:
        """Return patch nodes linked to error clusters affecting ``bot``."""

        if self.graph is None:
            return []
        bnode = f"bot:{bot}"
        if bnode not in self.graph:
            return []
        patches: Dict[str, int] = {}
        for enode, _, _ in self.graph.in_edges(bnode, data=True):
            if not enode.startswith("error_type:"):
                continue
            cluster = self.graph.nodes[enode].get("cluster")
            if cluster is None:
                self.error_clusters()
                cluster = self.graph.nodes[enode].get("cluster")
            for node, data in self.graph.nodes(data=True):
                if node.startswith("error_type:") and data.get("cluster") == cluster:
                    for _, pnode, d in self.graph.out_edges(node, data=True):
                        if pnode.startswith("patch:"):
                            patches[pnode] = patches.get(pnode, 0) + int(d.get("weight", 1))
        return [p for p, _ in sorted(patches.items(), key=lambda x: x[1], reverse=True)[:top]]

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
