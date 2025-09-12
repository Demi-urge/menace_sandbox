from __future__ import annotations

"""Automatically propose fixes for recurring errors.

The public :func:`generate_patch` helper expects callers to provide a
pre-configured :class:`vector_service.ContextBuilder`. The builder should
have its database weights refreshed before use to ensure accurate context
retrieval.
"""

__version__ = "1.0.0"

from pathlib import Path
import logging
import subprocess
import shutil
import tempfile
import os
import uuid
import time
from typing import Tuple, Iterable, Dict, Any, List, TYPE_CHECKING

from .snippet_compressor import compress_snippets
from .codebase_diff_checker import generate_code_diff, flag_risky_changes

try:  # pragma: no cover - optional dependency
    from context_builder_util import ensure_fresh_weights
except Exception:  # pragma: no cover - fallback when utility missing
    def ensure_fresh_weights(builder) -> None:  # type: ignore
        builder.refresh_db_weights()
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
from collections import Counter
import numpy as np


class ErrorClusterPredictor:
    """Cluster stack traces for a module using a lightweight k-means algorithm."""

    def __init__(self, db: "ErrorDB") -> None:
        self.db = db

    @staticmethod
    def _vectorize(traces: list[str]) -> np.ndarray:
        """Convert traces to simple bag-of-words vectors."""
        vocab: dict[str, int] = {}
        rows: list[Counter[str]] = []
        for trace in traces:
            counts: Counter[str] = Counter(trace.split())
            rows.append(counts)
            for token in counts:
                if token not in vocab:
                    vocab[token] = len(vocab)
        vecs = np.zeros((len(traces), len(vocab)), dtype=float)
        for i, counts in enumerate(rows):
            for token, count in counts.items():
                vecs[i, vocab[token]] = float(count)
        return vecs

    @staticmethod
    def _kmeans(vecs: np.ndarray, n_clusters: int, max_iter: int = 100) -> list[int]:
        """Minimal k-means implementation using Euclidean distance."""
        n_samples = vecs.shape[0]
        n_clusters = min(n_clusters, n_samples) or 1
        rng = np.random.default_rng(0)
        indices: list[int] = []
        for idx in rng.permutation(n_samples):
            if len(indices) == n_clusters:
                break
            if not any(np.array_equal(vecs[idx], vecs[j]) for j in indices):
                indices.append(idx)
        while len(indices) < n_clusters:
            indices.append(rng.integers(0, n_samples))
        centroids = vecs[indices].copy()
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            distances = ((vecs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for j in range(n_clusters):
                members = vecs[labels == j]
                if len(members):
                    centroids[j] = members.mean(axis=0)
        return labels.tolist()

    def best_cluster(self, module: str, n_clusters: int = 3) -> tuple[int | None, list[str]]:
        """Return ``(cluster_id, traces)`` for ``module``.

        The ``cluster_id`` corresponds to the cluster with the highest number of
        stack traces. ``traces`` contains stack traces belonging to that
        cluster.
        """

        cur = self.db.conn.execute(
            "SELECT stack_trace FROM telemetry WHERE module=? AND stack_trace!=''",
            (module,),
        )
        traces = [row[0] for row in cur.fetchall()]
        if not traces:
            return None, []
        vecs = self._vectorize(traces)
        labels = self._kmeans(vecs, n_clusters)
        cluster_id, _ = Counter(labels).most_common(1)[0]
        cluster_traces = [t for t, lbl in zip(traces, labels) if lbl == cluster_id]
        return int(cluster_id), cluster_traces

from .error_bot import ErrorDB
try:  # pragma: no cover - optional dependency
    from .self_coding_manager import SelfCodingManager
except Exception:  # pragma: no cover - fallback
    class SelfCodingManager:  # type: ignore
        pass
try:  # pragma: no cover - optional dependency
    from .knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - fallback
    class KnowledgeGraph:  # type: ignore
        pass
try:  # pragma: no cover - optional dependency
    from .coding_bot_interface import self_coding_managed, manager_generate_helper
except Exception:  # pragma: no cover - fallback when coding engine unavailable
    def self_coding_managed(cls):  # type: ignore
        return cls

    def manager_generate_helper(manager, description: str, **kwargs):  # type: ignore
        engine = getattr(manager, "engine", None)
        if engine is None:
            raise ImportError("Self-coding engine is required for operation")
        return engine.generate_helper(description, **kwargs)
try:  # pragma: no cover - optional dependency
    from .data_bot import DataBot
except Exception:  # pragma: no cover - fallback when unavailable
    DataBot = object  # type: ignore
try:  # pragma: no cover - fail fast if vector service missing
    from vector_service.context_builder import (
        ContextBuilder,
        Retriever,
        FallbackResult,
        EmbeddingBackfill,
    )
except Exception as exc:  # pragma: no cover - provide actionable error
    raise RuntimeError(
        "vector_service is required for quick_fix_engine. "
        "Install it via `pip install vector_service`."
    ) from exc
try:  # pragma: no cover - optional dependency
    from patch_provenance import PatchLogger
except Exception:  # pragma: no cover - fallback when unavailable
    PatchLogger = object  # type: ignore
try:  # pragma: no cover - optional dependency
    from chunking import get_chunk_summaries
except Exception:  # pragma: no cover - chunking unavailable
    get_chunk_summaries = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .target_region import extract_target_region
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from target_region import extract_target_region  # type: ignore
    except Exception:  # pragma: no cover - extractor unavailable
        extract_target_region = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from self_improvement.prompt_strategies import PromptStrategy, render_prompt
except Exception:  # pragma: no cover - fallback for tests
    class PromptStrategy(str):  # type: ignore
        """Minimal stand-in when prompt strategies are unavailable."""

        def __new__(cls, value: str = ""):
            logging.getLogger("QuickFixEngine").warning(
                "self_improvement.prompt_strategies missing; using dummy PromptStrategy"
            )
            return str.__new__(cls, value)

    def render_prompt(*a: object, **k: object) -> str:  # type: ignore
        logging.getLogger("QuickFixEngine").warning(
            "self_improvement.prompt_strategies missing; render_prompt returning empty string"
        )
        return ""

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .self_coding_engine import SelfCodingEngine
    from .self_improvement.target_region import TargetRegion
try:  # pragma: no cover - required dependency
    from vector_service import ErrorResult  # type: ignore
except Exception as exc:  # pragma: no cover - fail fast when unavailable
    raise RuntimeError(
        "vector_service.ErrorResult is required for quick_fix_engine. "
        "Install or update `vector_service` to include ErrorResult."
    ) from exc
try:  # pragma: no cover - optional dependency
    from .human_alignment_flagger import _collect_diff_data
except Exception:  # pragma: no cover - fallback for tests
    def _collect_diff_data(*a, **k):
        logging.getLogger("QuickFixEngine").warning(
            "human_alignment_flagger._collect_diff_data missing; returning empty diff data"
        )
        return {}
try:  # pragma: no cover - optional dependency
    from .human_alignment_agent import HumanAlignmentAgent
except Exception:  # pragma: no cover - fallback for tests
    class HumanAlignmentAgent:  # type: ignore
        """Fallback agent when human alignment module is unavailable."""

        def __init__(self, *a: object, **k: object) -> None:
            logging.getLogger("QuickFixEngine").warning(
                "HumanAlignmentAgent missing; alignment checks skipped"
            )

        def evaluate_changes(self, *a: object, **k: object) -> dict:
            logging.getLogger("QuickFixEngine").warning(
                "Fallback HumanAlignmentAgent.evaluate_changes called"
            )
            return {}
try:  # pragma: no cover - optional dependency
    from .violation_logger import log_violation
except Exception:  # pragma: no cover - fallback for tests
    def log_violation(*a, **k):
        logging.getLogger("QuickFixEngine").warning(
            "violation_logger.log_violation missing; nothing will be recorded"
        )
        return None


_VEC_METRICS = None


def generate_patch(
    module: str,
    manager: SelfCodingManager,
    engine: "SelfCodingEngine" | None = None,
    *,
    context_builder: ContextBuilder,
    description: str | None = None,
    strategy: PromptStrategy | None = None,
    patch_logger: PatchLogger | None = None,
    context: Dict[str, Any] | None = None,
    effort_estimate: float | None = None,
    target_region: "TargetRegion" | None = None,
    return_flags: bool = False,
) -> int | tuple[int | None, list[str]] | None:
    """Attempt a quick patch for *module* and return the patch id.

    A provided :class:`vector_service.ContextBuilder` is used to gather context
    for the patch.

    Parameters
    ----------
    module:
        Target module path or module name without ``.py``.
    manager:
        :class:`~self_coding_manager.SelfCodingManager` providing helper
        generation context and telemetry hooks.
    engine:
        :class:`~self_coding_engine.SelfCodingEngine` instance. If ``None``,
        the value from ``manager.engine`` is used. A ``RuntimeError`` is raised
        when no engine is available.
    context_builder:
        :class:`vector_service.ContextBuilder` instance used to retrieve
        contextual information from local databases. The builder must be able
        to query the databases associated with the current repository.
    description:
        Optional patch description.  When omitted, a generic description is
        used.
    strategy:
        Optional :class:`self_improvement.prompt_strategies.PromptStrategy` to
        tailor the prompt. When provided the corresponding template is appended
        to the description.
    patch_logger:
        Optional :class:`patch_provenance.PatchLogger` for recording vector
        provenance.
    context:
        Optional dictionary merged into the patch's ``context_meta`` prior to
        patch application.
    target_region:
        Optional region within the file to patch. When provided only this
        slice is modified.
    """

    logger = logging.getLogger("QuickFixEngine")
    risk_flags: list[str] = []
    if context_builder is None:
        raise TypeError("context_builder is required")
    if manager is None:
        raise RuntimeError("manager is required")
    builder = context_builder
    try:
        builder.refresh_db_weights()
    except Exception as exc:  # pragma: no cover - validation
        raise RuntimeError(
            "provided ContextBuilder cannot query local databases"
        ) from exc
    mod_str = module if module.endswith(".py") else f"{module}.py"
    try:
        path = resolve_path(mod_str)
    except FileNotFoundError:
        logger.error("module not found: %s", module)
        return (None, risk_flags) if return_flags else None

    prompt_path = path_for_prompt(path.as_posix())
    description = description or f"preemptive fix for {prompt_path}"
    context_meta: Dict[str, Any] = {"module": prompt_path, "reason": "preemptive_fix"}
    if context:
        context_meta.update(context)
    cluster_id: int | None = None
    cluster_traces: list[str] = []
    error_db = getattr(manager, "error_db", None)
    if error_db is not None:
        try:
            predictor = ErrorClusterPredictor(error_db)
            cluster_id, cluster_traces = predictor.best_cluster(prompt_path)
            if cluster_id is not None:
                context_meta["error_cluster_id"] = cluster_id
        except Exception:
            cluster_id = None
            cluster_traces = []
    if cluster_traces:
        try:
            description += "\n\n" + cluster_traces[0]
            if extract_target_region is not None and target_region is None:
                region = extract_target_region(cluster_traces[0])
                if region and region.filename.endswith(prompt_path):
                    target_region = region
        except Exception:
            pass
    context_block = ""
    cb_session = uuid.uuid4().hex
    context_meta["context_session_id"] = cb_session
    vectors: List[Tuple[str, str, float]] = []
    try:
        ctx_res = builder.build(
            description, session_id=cb_session, include_vectors=True
        )
        if isinstance(ctx_res, tuple):
            context_block, _, vectors = ctx_res
        else:
            context_block = ctx_res
        if isinstance(context_block, (FallbackResult, ErrorResult)):
            context_block = ""
    except Exception:
        context_block = ""
        vectors = []
    if context_block:
        compressed = compress_snippets({"snippet": context_block}).get("snippet", "")
        if compressed:
            description += "\n\n" + compressed
    if strategy is not None:
        try:
            template = render_prompt(strategy, {"module": prompt_path})
        except Exception:
            template = ""
        if template:
            description += "\n\n" + template
        context_meta["prompt_strategy"] = str(strategy)
    base_description = description

    try:
        meta = dict(context_meta)
        meta.setdefault("trigger", "quick_fix_engine")
        manager.register_patch_cycle(description, meta)
    except Exception:
        logger.exception("failed to register patch cycle")

    if patch_logger is None:
        try:
            patch_logger = PatchLogger()
        except Exception:
            patch_logger = None

    if engine is None:
        engine = getattr(manager, "engine", None)
    if engine is None:
        raise RuntimeError(
            "generate_patch requires a SelfCodingEngine instance. Pass"
            " manager.engine so improvements originate from SelfCodingManager."
        )

    try:
        patch_id: int | None
        with (
            tempfile.TemporaryDirectory() as before_dir,
            tempfile.TemporaryDirectory() as after_dir,
        ):
            rel = path.name if path.is_absolute() else path
            before_target = Path(before_dir) / rel
            before_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, before_target)
            chunks: List[Any] = []
            if get_chunk_summaries is not None:
                token_limit = getattr(engine, "prompt_chunk_token_threshold", 1000)
                try:
                    chunks = get_chunk_summaries(
                        path, token_limit, context_builder=builder
                    )
                except Exception:
                    chunks = []
            patch_ids: List[int | None] = []

            def _gen(desc: str) -> str:
                """Generate helper code for *desc* using the current context."""
                owner = getattr(manager, "engine", manager)
                attempts = int(getattr(owner, "helper_retry_attempts", 3))
                delay = float(getattr(owner, "helper_retry_delay", 1.0))
                event_bus = getattr(owner, "event_bus", None)
                data_bot: DataBot | None = getattr(owner, "data_bot", None)
                bot_name = getattr(owner, "bot_name", "quick_fix_engine")
                module_name = Path(context_meta.get("module", path.name)).name
                for i in range(attempts):
                    try:
                        return manager_generate_helper(
                            manager,
                            desc,
                            path=path,
                            metadata=context_meta,
                            target_region=target_region,
                        )
                    except TypeError:
                        try:
                            return manager_generate_helper(manager, desc)
                        except Exception as exc2:  # fall through to logging
                            err: Exception = exc2
                    except Exception as exc:
                        err = exc
                    logger.exception("helper generation failed", exc_info=err)
                    if event_bus:
                        payload = {
                            "module": module_name,
                            "description": desc,
                            "error": str(err),
                            "attempt": i + 1,
                        }
                        try:
                            event_bus.publish("bot:helper_failed", payload)
                        except Exception:
                            logger.exception("event bus publish failed")
                    if i < attempts - 1:
                        time.sleep(delay)
                        delay *= 2
                if data_bot:
                    try:
                        data_bot.record_validation(
                            bot_name,
                            module_name,
                            False,
                            ["helper_generation_failed"],
                        )
                    except Exception:
                        logger.exception(
                            "failed to record validation in DataBot"
                        )
                return ""
            if chunks and target_region is None:
                for chunk in chunks:
                    summary = (
                        chunk.get("summary", "")
                        if isinstance(chunk, dict)
                        else str(chunk)
                    )
                    chunk_desc = base_description
                    if summary:
                        chunk_desc = f"{base_description}\n\n{summary}"
                    try:
                        apply = getattr(engine, "apply_patch_with_retry")
                    except AttributeError:
                        apply = getattr(engine, "apply_patch")
                    helper = _gen(chunk_desc)
                    try:
                        pid, _, _ = apply(
                            path,
                            helper,
                            description=chunk_desc,
                            reason="preemptive_fix",
                            trigger="quick_fix_engine",
                            context_meta=context_meta,
                            target_region=target_region,
                        )
                    except TypeError:
                        try:
                            pid, _, _ = apply(
                                path,
                                chunk_desc,
                                reason="preemptive_fix",
                                trigger="quick_fix_engine",
                                context_meta=context_meta,
                                target_region=target_region,
                            )
                        except AttributeError:
                            engine.patch_file(
                                path,
                                "preemptive_fix",
                                context_meta=context_meta,
                                target_region=target_region,
                            )
                            pid = None
                    except AttributeError:
                        engine.patch_file(
                            path,
                            "preemptive_fix",
                            context_meta=context_meta,
                            target_region=target_region,
                        )
                        pid = None
                    patch_ids.append(pid)
                patch_id = patch_ids[-1] if patch_ids else None
            else:
                try:
                    apply = getattr(engine, "apply_patch_with_retry")
                except AttributeError:
                    apply = getattr(engine, "apply_patch")
                helper = _gen(base_description)
                try:
                    patch_id, _, _ = apply(
                        path,
                        helper,
                        description=base_description,
                        reason="preemptive_fix",
                        trigger="quick_fix_engine",
                        context_meta=context_meta,
                        target_region=target_region,
                    )
                except TypeError:
                    try:
                        patch_id, _, _ = apply(
                            path,
                            base_description,
                            reason="preemptive_fix",
                            trigger="quick_fix_engine",
                            context_meta=context_meta,
                            target_region=target_region,
                        )
                    except AttributeError:
                        engine.patch_file(
                            path,
                            "preemptive_fix",
                            context_meta=context_meta,
                            target_region=target_region,
                        )
                        patch_id = None
                except AttributeError:
                    engine.patch_file(
                        path,
                        "preemptive_fix",
                        context_meta=context_meta,
                        target_region=target_region,
                    )
                    patch_id = None
            after_target = Path(after_dir) / rel
            after_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, after_target)
            diff_struct = generate_code_diff(before_dir, after_dir)
            risk_flags = flag_risky_changes(diff_struct)
            if risk_flags:
                logger.warning("risky changes detected: %s", risk_flags)
            diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
            workflow_changes = [
                {"file": path_for_prompt(f), "code": "\n".join(d["added"])}
                for f, d in diff_data.items()
                if d["added"]
            ]
            if workflow_changes:
                agent = HumanAlignmentAgent()
                warnings = agent.evaluate_changes(workflow_changes, None, [])
                if any(warnings.values()):
                    log_violation(
                        str(patch_id) if patch_id is not None else str(path),
                        "alignment_warning",
                        1,
                        {"warnings": warnings},
                        alignment_warning=True,
                    )
            if patch_logger is not None:
                for attempt in range(2):
                    try:
                        patch_logger.track_contributors(
                            [(f"{o}:{vid}", score) for o, vid, score in vectors],
                            patch_id is not None,
                            patch_id=str(patch_id) if patch_id is not None else "",
                            session_id=cb_session,
                            effort_estimate=effort_estimate,
                        )
                    except Exception:
                        logger.warning(
                            "patch_logger.track_contributors failed",
                            exc_info=True,
                            extra={"attempt": attempt + 1},
                        )
                        if attempt == 0:
                            time.sleep(0.5)
                        else:
                            break
                    else:  # run embedding backfill on success
                        try:
                            EmbeddingBackfill().run(
                                db="code",
                                backend=os.getenv("VECTOR_BACKEND", "annoy"),
                            )
                        except BaseException:  # pragma: no cover - best effort
                            logger.debug(
                                "embedding backfill failed", exc_info=True
                            )
                        break
            if patch_id is not None:
                registry = getattr(manager, "bot_registry", None)
                if registry is not None:
                    commit_hash: str | None = None
                    try:
                        commit_hash = (
                            subprocess.check_output(["git", "rev-parse", "HEAD"])
                            .decode()
                            .strip()
                        )
                    except Exception:
                        logger.debug("failed to capture commit hash", exc_info=True)
                    try:
                        registry.update_bot(
                            getattr(manager, "bot_name", ""),
                            path.as_posix(),
                            patch_id=patch_id,
                            commit=commit_hash,
                        )
                    except Exception:
                        logger.exception("failed to update bot registry")
            try:
                from sandbox_runner import post_round_orphan_scan

                post_round_orphan_scan(Path.cwd(), context_builder=builder)
            except Exception:
                logger.exception(
                    "post_round_orphan_scan after preemptive patch failed"
                )
            return (patch_id, risk_flags) if return_flags else patch_id
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.error("quick fix generation failed for %s: %s", prompt_path, exc)
        return (None, [str(exc)]) if return_flags else None


@self_coding_managed
class QuickFixEngine:
    """Analyse frequent errors and trigger small patches."""

    def __init__(
        self,
        error_db: ErrorDB,
        manager: SelfCodingManager,
        *,
        threshold: int = 3,
        graph: KnowledgeGraph | None = None,
        risk_threshold: float = 0.5,
        predictor: ErrorClusterPredictor | None = None,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder,
        patch_logger: PatchLogger | None = None,
        min_reliability: float | None = None,
        redundancy_limit: int | None = None,
    ) -> None:
        self.db = error_db
        self.manager = manager
        self.threshold = threshold
        self.graph = graph or KnowledgeGraph()
        self.risk_threshold = risk_threshold
        self.predictor = predictor
        self.retriever = retriever
        logger = logging.getLogger(self.__class__.__name__)
        try:
            ensure_fresh_weights(context_builder)
        except Exception as exc:  # pragma: no cover - validation
            raise RuntimeError(
                "provided ContextBuilder cannot query local databases"
            ) from exc
        self.context_builder = context_builder
        if patch_logger is None:
            try:
                eng = getattr(manager, "engine", None)
                pdb = getattr(eng, "patch_db", None)
                patch_logger = PatchLogger(patch_db=pdb)
            except Exception:
                patch_logger = None
        self.patch_logger = patch_logger
        self.logger = logger
        env_min_rel = float(os.getenv("DB_MIN_RELIABILITY", "0.0"))
        env_red = int(os.getenv("DB_REDUNDANCY_LIMIT", "1"))
        self.min_reliability = (
            float(min_reliability)
            if min_reliability is not None
            else env_min_rel
        )
        self.redundancy_limit = (
            int(redundancy_limit)
            if redundancy_limit is not None
            else env_red
        )

    # ------------------------------------------------------------------
    def _top_error(
        self, bot: str
    ) -> Tuple[str, str, dict[str, int], int] | None:
        try:
            info = self.db.top_error_module(bot, scope="local")
        except Exception:
            return None
        if not info:
            return None
        etype, module, mods, count, _ = info
        return etype, module, mods, count

    def _redundant_retrieve(
        self, query: str, top_k: int
    ) -> Tuple[List[Any], str, List[Tuple[str, str, float]]]:
        if self.retriever is None:
            return [], "", []
        session_id = uuid.uuid4().hex
        try:
            hits = self.retriever.search(query, top_k=top_k, session_id=session_id)
            if isinstance(hits, (FallbackResult, ErrorResult)):
                if isinstance(hits, FallbackResult):
                    self.logger.debug(
                        "retriever returned fallback for %s: %s",
                        query,
                        getattr(hits, "reason", ""),
                    )
                return [], "", []
        except Exception:
            self.logger.debug("retriever lookup failed", exc_info=True)
            return [], "", []
        vectors = [
            (
                h.get("origin_db", ""),
                str(h.get("record_id", "")),
                float(h.get("score") or 0.0),
            )
            for h in hits
        ]
        return hits, session_id, vectors

    def run(self, bot: str) -> None:
        """Attempt a quick patch for the most frequent error of ``bot``."""
        if self.predictor is not None:
            try:
                modules = self.predictor.predict_high_risk_modules()
            except Exception as exc:
                self.logger.error("high risk prediction failed: %s", exc)
                modules = []
            if modules:
                if isinstance(modules[0], str):
                    modules_with_scores = [(m, 1.0) for m in modules]
                else:
                    modules_with_scores = modules  # type: ignore[assignment]
                self.preemptive_patch_modules(modules_with_scores)

        info = self._top_error(bot)
        if not info:
            return
        etype, module, mods, count = info
        if count < self.threshold:
            return
        try:
            path = resolve_path(f"{module}.py")
        except FileNotFoundError:
            return
        prompt_path = path_for_prompt(path)
        context_meta = {"error_type": etype, "module": prompt_path, "bot": bot}
        builder = ContextBuilder()
        try:
            ensure_fresh_weights(builder)
        except Exception:
            self.logger.debug("context builder refresh failed", exc_info=True)
        self.context_builder = builder
        ctx_block = ""
        cb_vectors: list[tuple[str, str, float]] = []
        cb_session = uuid.uuid4().hex
        context_meta["context_session_id"] = cb_session
        try:
            query = f"{etype} in {prompt_path}"
            ctx_res = builder.build(
                query, session_id=cb_session, include_vectors=True
            )
            if isinstance(ctx_res, tuple):
                ctx_block, _sid, cb_vectors = ctx_res
            else:
                ctx_block = ctx_res
            if isinstance(ctx_block, (FallbackResult, ErrorResult)):
                ctx_block = ""
        except Exception:
            ctx_block = ""
            cb_vectors = []
        if cb_vectors:
            context_meta["retrieval_vectors"] = cb_vectors
        desc = f"quick fix {etype}"
        if ctx_block:
            try:
                compressed = compress_snippets({"snippet": ctx_block}).get(
                    "snippet", ""
                )
            except Exception:
                compressed = ""
            if compressed:
                desc += "\n\n" + compressed
        session_id = ""
        vectors: list[tuple[str, str, float]] = []
        retrieval_metadata: dict[str, dict[str, Any]] = {}
        if self.retriever is not None:
            _hits, session_id, vectors = self._redundant_retrieve(prompt_path, top_k=1)
            retrieval_metadata = {
                f"{h.get('origin_db', '')}:{h.get('record_id', '')}": {
                    "license": h.get("license"),
                    "license_fingerprint": h.get("license_fingerprint"),
                    "semantic_alerts": h.get("semantic_alerts"),
                    "alignment_severity": h.get("alignment_severity"),
                }
                for h in _hits
            }
        if session_id:
            context_meta["retrieval_session_id"] = session_id
            context_meta.setdefault("retrieval_vectors", vectors)
        patch_id = None
        event_bus = getattr(self.manager, "event_bus", None)
        if event_bus:
            try:
                event_bus.publish(
                    "quick_fix:patch_start",
                    {"module": prompt_path, "bot": bot, "description": desc},
                )
            except Exception:
                self.logger.exception("failed to publish patch_start event")
        try:
            try:
                result = self.manager.run_patch(
                    path,
                    desc,
                    context_meta=context_meta,
                    context_builder=builder,
                )
            except TypeError:
                result = self.manager.run_patch(
                    path,
                    desc,
                    context_meta=context_meta,
                )
            patch_id = getattr(result, "patch_id", None)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.error("quick fix failed for %s: %s", bot, exc)
            if event_bus:
                try:
                    event_bus.publish(
                        "quick_fix:patch_failed",
                        {
                            "module": prompt_path,
                            "bot": bot,
                            "description": desc,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    self.logger.exception(
                        "failed to publish patch_failed event",
                    )
        tests_ok = True
        try:
            subprocess.run(["pytest", "-q"], check=True)
        except Exception as exc:
            tests_ok = False
            self.logger.error("quick fix validation failed: %s", exc)
        try:
            self.graph.add_telemetry_event(
                bot, etype, prompt_path, mods, patch_id=patch_id, resolved=tests_ok
            )
            self.graph.update_error_stats(self.db)
        except Exception as exc:
            self.logger.exception("telemetry update failed: %s", exc)
            raise
        if self.patch_logger is not None:
            ids = {f"{o}:{v}": s for o, v, s in vectors}
            try:
                result = bool(patch_id) and tests_ok
                self.patch_logger.track_contributors(
                    ids,
                    result,
                    patch_id=str(patch_id or ""),
                    session_id=session_id,
                    contribution=1.0 if result else 0.0,
                    retrieval_metadata=retrieval_metadata,
                )
            except Exception:
                self.logger.debug("patch logging failed", exc_info=True)

    # ------------------------------------------------------------------
    def preemptive_patch_modules(
        self,
        modules: Iterable[tuple[str, float]] | Iterable[str],
        *,
        risk_threshold: float | None = None,
    ) -> None:
        """Proactively patch ``modules`` that exceed ``risk_threshold``.

        Parameters
        ----------
        modules:
            Iterable of ``(module, risk_score)`` pairs as produced by
            :meth:`~error_cluster_predictor.ErrorClusterPredictor.predict_high_risk_modules`.
        risk_threshold:
            Minimum risk score required to trigger a patch.  Defaults to the
            instance's ``risk_threshold`` attribute.
        """

        thresh = self.risk_threshold if risk_threshold is None else risk_threshold
        for item in modules:
            if isinstance(item, tuple):
                module, risk = item
            else:
                module, risk = item, 1.0
            if risk < thresh:
                continue
            try:
                path = resolve_path(f"{module}.py")
            except FileNotFoundError:
                continue
            prompt_path = path_for_prompt(path)
            meta = {"module": prompt_path, "reason": "preemptive_patch"}
            builder = ContextBuilder()
            try:
                ensure_fresh_weights(builder)
            except Exception:
                self.logger.debug("context builder refresh failed", exc_info=True)
            ctx = ""
            cb_vectors: list[tuple[str, str, float]] = []
            cb_session = uuid.uuid4().hex
            try:
                query = f"preemptive patch {prompt_path}"
                ctx_res = builder.build(
                    query, session_id=cb_session, include_vectors=True
                )
                if isinstance(ctx_res, tuple):
                    ctx, cb_session, cb_vectors = ctx_res
                else:
                    ctx, cb_session = ctx_res, cb_session
                if isinstance(ctx, (FallbackResult, ErrorResult)):
                    ctx = ""
                    cb_vectors = []
            except Exception:
                ctx = ""
                cb_vectors = []
            meta["context_session_id"] = cb_session
            if cb_vectors:
                meta["retrieval_vectors"] = cb_vectors
            desc = "preemptive_patch"
            if ctx:
                try:
                    compressed = compress_snippets({"snippet": ctx}).get("snippet", "")
                except Exception:
                    compressed = ""
                if compressed:
                    desc += "\n\n" + compressed
                    meta["context"] = compressed
            session_id = ""
            vectors: list[tuple[str, str, float]] = []
            retrieval_metadata: dict[str, dict[str, Any]] = {}
            if self.retriever is not None:
                _hits, session_id, vectors = self._redundant_retrieve(prompt_path, top_k=1)
                retrieval_metadata = {
                    f"{h.get('origin_db', '')}:{h.get('record_id', '')}": {
                        "license": h.get("license"),
                        "license_fingerprint": h.get("license_fingerprint"),
                        "semantic_alerts": h.get("semantic_alerts"),
                        "alignment_severity": h.get("alignment_severity"),
                    }
                    for h in _hits
                }
            if session_id:
                meta["retrieval_session_id"] = session_id
                meta.setdefault("retrieval_vectors", vectors)
            patch_id = None
            event_bus = getattr(self.manager, "event_bus", None)
            if event_bus:
                try:
                    event_bus.publish(
                        "quick_fix:patch_start",
                        {"module": prompt_path, "bot": "predictor", "description": desc},
                    )
                except Exception:
                    self.logger.exception("failed to publish patch_start event")
            try:
                try:
                    result = self.manager.run_patch(
                        path,
                        desc,
                        context_meta=meta,
                        context_builder=builder,
                    )
                except TypeError:
                    result = self.manager.run_patch(
                        path,
                        desc,
                        context_meta=meta,
                    )
                patch_id = getattr(result, "patch_id", None)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.error("preemptive patch failed for %s: %s", prompt_path, exc)
                if event_bus:
                    try:
                        event_bus.publish(
                            "quick_fix:patch_failed",
                            {
                                "module": prompt_path,
                                "bot": "predictor",
                                "description": desc,
                                "error": str(exc),
                            },
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish patch_failed event",
                        )
                try:
                    patch_id = generate_patch(
                        prompt_path,
                        getattr(self.manager, "engine", None),
                        self.manager,
                        context_builder=self.context_builder,
                    )
                except Exception:
                    patch_id = None
            try:
                self.graph.add_telemetry_event(
                    "predictor",
                    "predicted_high_risk",
                    prompt_path,
                    None,
                    patch_id=patch_id,
                )
            except Exception as exc:  # pragma: no cover - graph issues
                self.logger.error(
                    "failed to record telemetry for %s: %s", prompt_path, exc
                )
            try:
                self.db.log_preemptive_patch(prompt_path, risk, patch_id)
            except Exception as exc:  # pragma: no cover - db issues
                self.logger.error(
                    "failed to record preemptive patch for %s: %s", prompt_path, exc
                )
            if self.patch_logger is not None:
                ids = {
                    f"{o}:{v}": s for o, v, s in meta.get("retrieval_vectors", [])
                }
                try:
                    result = bool(patch_id)
                    self.patch_logger.track_contributors(
                        ids,
                        result,
                        patch_id=str(patch_id or ""),
                        session_id=meta.get("context_session_id", ""),
                        contribution=1.0 if result else 0.0,
                        retrieval_metadata=retrieval_metadata,
                    )
                except Exception:
                    self.logger.debug("patch logging failed", exc_info=True)

    # ------------------------------------------------------------------
    def apply_validated_patch(
        self,
        module_path: str | Path,
        description: str = "",
        context_meta: Dict[str, Any] | None = None,
    ) -> Tuple[bool, int | None, List[str]]:
        """Generate and apply a patch returning its success status, id and flags.

        The patch is first generated via :func:`generate_patch` which performs
        internal risk checks.  When any validation flags are raised the change
        is reverted and ``False`` is returned along with ``None`` for the
        ``patch_id`` and the list of ``flags``.
        """

        ctx = context_meta or {}
        try:
            patch_id, flags = generate_patch(
                str(module_path),
                getattr(self.manager, "engine", None),
                self.manager,
                context_builder=self.context_builder,
                description=description,
                context=ctx,
                return_flags=True,
            )
        except Exception:
            self.logger.exception("quick fix patch failed")
            return False, None, ["generation_error"]
        flags_list = list(flags)
        if flags_list:
            try:
                subprocess.run([
                    "git",
                    "checkout",
                    "--",
                    str(module_path),
                ], check=True)
            except Exception:
                self.logger.exception("failed to revert invalid patch")
            event_bus = getattr(self.manager, "event_bus", None)
            if event_bus:
                payload = {
                    "bot": getattr(self.manager, "bot_name", "quick_fix_engine"),
                    "module": str(module_path),
                    "flags": flags_list,
                }
                try:
                    event_bus.publish("self_coding:patch_rejected", payload)
                except Exception:
                    self.logger.exception(
                        "failed to publish patch_rejected event",
                    )
            data_bot = getattr(self.manager, "data_bot", None)
            if data_bot:
                try:
                    data_bot.record_validation(
                        getattr(self.manager, "bot_name", "quick_fix_engine"),
                        str(module_path),
                        False,
                        flags_list,
                    )
                except Exception:
                    self.logger.exception(
                        "failed to record validation in DataBot",
                    )
            return False, None, flags_list
        return True, patch_id, []

    # ------------------------------------------------------------------
    def validate_patch(
        self,
        module_name: str,
        description: str = "",
        *,
        target_region: "TargetRegion | None" = None,
        repo_root: Path | str | None = None,
    ) -> Tuple[bool, List[str]]:
        """Run quick-fix validation on ``module_name`` without applying it."""
        flags: List[str]
        try:
            _pid, flags = generate_patch(
                module_name,
                getattr(self.manager, "engine", None),
                self.manager,
                context_builder=self.context_builder,
                description=description,
                target_region=target_region,
                return_flags=True,
            )
        except Exception:
            self.logger.exception("quick fix validation failed")
            flags = ["validation_error"]
        finally:
            try:
                if repo_root is not None:
                    rel = Path(module_name).resolve().relative_to(Path(repo_root).resolve())
                    subprocess.run(
                        ["git", "checkout", "--", str(rel)],
                        check=True,
                        cwd=str(repo_root),
                    )
                else:
                    path = resolve_path(module_name)
                    subprocess.run(["git", "checkout", "--", str(path)], check=True)
            except Exception:
                self.logger.exception("failed to revert validation patch")
        return (not bool(flags), flags)

    # ------------------------------------------------------------------
    def run_and_validate(self, bot: str) -> None:
        """Run :meth:`run` then execute the test suite."""
        self.run(bot)


__all__ = ["QuickFixEngine", "generate_patch"]
