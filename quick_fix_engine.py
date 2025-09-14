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
import py_compile
import shlex
from typing import Tuple, Iterable, Dict, Any, List, TYPE_CHECKING, Callable

from .snippet_compressor import compress_snippets
from .codebase_diff_checker import generate_code_diff, flag_risky_changes

try:  # pragma: no cover - optional dependency
    from context_builder_util import ensure_fresh_weights
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "context_builder_util.ensure_fresh_weights is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class QuickFixEngineError(RuntimeError):
    """Exception raised when quick fix generation fails.

    Parameters
    ----------
    code:
        Machine-readable error code describing the failure.
    message:
        Human-readable explanation of the failure.
    """

    def __init__(self, code: str, message: str) -> None:  # pragma: no cover - simple
        super().__init__(message)
        self.code = code


class ErrorClusterPredictor:
    """Cluster stack traces for a module using TF-IDF and scikit-learn k-means."""

    def __init__(
        self,
        db: "ErrorDB",
        *,
        max_clusters: int = 8,
        min_cluster_size: int = 2,
        tfidf_kwargs: dict | None = None,
        kmeans_kwargs: dict | None = None,
    ) -> None:
        self.db = db
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.vectorizer = TfidfVectorizer(**(tfidf_kwargs or {"stop_words": "english"}))
        self.kmeans_kwargs = {"n_init": "auto"}
        if kmeans_kwargs:
            self.kmeans_kwargs.update(kmeans_kwargs)

    def _cluster(self, traces: list[str], n_clusters: int) -> list[int]:
        """Cluster ``traces`` into ``n_clusters`` groups using TF-IDF + KMeans."""
        matrix = self.vectorizer.fit_transform(traces)
        if n_clusters <= 1:
            return [0] * len(traces)
        km = KMeans(n_clusters=n_clusters, **self.kmeans_kwargs)
        return km.fit_predict(matrix).tolist()

    def best_cluster(
        self, module: str, n_clusters: int | None = None
    ) -> tuple[int | None, list[str], int]:
        """Return ``(cluster_id, traces, size)`` for ``module``.

        The ``cluster_id`` corresponds to the cluster with the highest number of
        stack traces. ``traces`` contains stack traces belonging to that
        cluster and ``size`` is the number of traces in that cluster.
        """

        cur = self.db.conn.execute(
            "SELECT stack_trace FROM telemetry WHERE module=? AND stack_trace!=''",
            (module,),
        )
        traces = [row[0] for row in cur.fetchall()]
        if not traces:
            return None, [], 0
        n_clusters = n_clusters or max(
            1, round(len(traces) / self.min_cluster_size)
        )
        n_clusters = min(self.max_clusters, n_clusters)
        labels = self._cluster(traces, n_clusters)
        cluster_id, count = Counter(labels).most_common(1)[0]
        cluster_traces = [t for t, lbl in zip(traces, labels) if lbl == cluster_id]
        return int(cluster_id), cluster_traces, int(count)


from .error_bot import ErrorDB  # noqa: E402
from .self_coding_manager import SelfCodingManager  # noqa: E402
try:  # pragma: no cover - optional helper
    from .self_coding_manager import _manager_generate_helper_with_builder  # noqa: E402
except Exception:  # pragma: no cover - helper unavailable
    _manager_generate_helper_with_builder = None  # type: ignore
from .knowledge_graph import KnowledgeGraph  # noqa: E402
try:  # pragma: no cover - optional dependency
    from .coding_bot_interface import (  # noqa: E402
        manager_generate_helper as _base_manager_generate_helper,
    )
except Exception:  # pragma: no cover - fallback when coding engine unavailable
    def _base_manager_generate_helper(manager, description: str, **kwargs):  # type: ignore
        raise ImportError("Self-coding engine is required for operation")

manager_generate_helper = (
    _manager_generate_helper_with_builder or _base_manager_generate_helper
)
try:  # pragma: no cover - optional dependency
    from .data_bot import DataBot  # noqa: E402
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "data_bot.DataBot is required for quick_fix_engine"
    ) from exc
from .resilience import retry_with_backoff  # noqa: E402
try:  # pragma: no cover - fail fast if vector service missing
    from vector_service.context_builder import (  # noqa: E402
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
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "patch_provenance.PatchLogger is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from chunking import get_chunk_summaries
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "chunking.get_chunk_summaries is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .target_region import extract_target_region
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from target_region import extract_target_region  # type: ignore
    except Exception:  # pragma: no cover - extractor unavailable
        extract_target_region = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from self_improvement.prompt_strategies import PromptStrategy, render_prompt
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "self_improvement.prompt_strategies is required for quick_fix_engine"
    ) from exc

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
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "human_alignment_flagger._collect_diff_data is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .human_alignment_agent import HumanAlignmentAgent
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "human_alignment_agent is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .violation_logger import log_violation
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "violation_logger.log_violation is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .advanced_error_management import AutomatedRollbackManager
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "advanced_error_management.AutomatedRollbackManager is required for quick_fix_engine"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .code_database import PatchHistoryDB
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "code_database.PatchHistoryDB is required for quick_fix_engine"
    ) from exc

_VEC_METRICS = None


def generate_patch(
    module: str,
    manager: SelfCodingManager,
    engine: "SelfCodingEngine" | None = None,
    *,
    context_builder: ContextBuilder,
    provenance_token: str,
    description: str | None = None,
    strategy: PromptStrategy | None = None,
    patch_logger: PatchLogger | None = None,
    context: Dict[str, Any] | None = None,
    effort_estimate: float | None = None,
    target_region: "TargetRegion" | None = None,
    return_flags: bool = False,
    helper_fn: Callable[..., str] | None = None,
    graph: KnowledgeGraph | None = None,
    test_command: List[str] | None = None,
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
        generation context and telemetry hooks. A ``RuntimeError`` is raised if
        no :class:`SelfCodingManager` is supplied.
    engine:
        :class:`~self_coding_engine.SelfCodingEngine` instance. If ``None``,
        the value from ``manager.engine`` is used. A ``RuntimeError`` is raised
        when no engine is available.
    context_builder:
        :class:`vector_service.ContextBuilder` instance used to retrieve
        contextual information from local databases. The builder must be able
        to query the databases associated with the current repository.
    provenance_token:
        Token from :class:`SelfCodingManager.validate_provenance` confirming the
        call originates from the active ``EvolutionOrchestrator``.
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
    helper = helper_fn or manager_generate_helper
    risk_flags: list[str] = []
    if test_command is None:
        env_cmd = os.getenv("SELF_CODING_TEST_COMMAND")
        if env_cmd:
            try:
                test_command = shlex.split(env_cmd)
            except Exception:
                test_command = [env_cmd]
    if context_builder is None:
        raise QuickFixEngineError(
            "quick_fix_validation_error", "context_builder is required"
        )
    if manager is None:
        raise QuickFixEngineError(
            "quick_fix_init_error", "generate_patch requires a manager"
        )
    if not isinstance(manager, SelfCodingManager):
        raise QuickFixEngineError(
            "quick_fix_init_error",
            "generate_patch requires a SelfCodingManager instance",
        )
    if not provenance_token:
        raise QuickFixEngineError(
            "quick_fix_validation_error", "provenance_token is required"
        )
    try:
        manager.validate_provenance(provenance_token)
    except Exception as exc:  # pragma: no cover - validation
        raise QuickFixEngineError(
            "quick_fix_validation_error", "invalid provenance token"
        ) from exc
    builder = context_builder
    try:
        builder.refresh_db_weights()
    except Exception as exc:  # pragma: no cover - validation
        raise QuickFixEngineError(
            "quick_fix_validation_error",
            "provided ContextBuilder cannot query local databases",
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
    graph_lines: list[str] = []
    if graph is not None:
        def _fetch() -> list[str]:
            return graph.related(f"code:{prompt_path}")
        try:
            related_nodes = retry_with_backoff(_fetch, attempts=2, delay=0.1, logger=logger)
        except Exception:
            related_nodes = []
        modules = [n.split(":", 1)[1] for n in related_nodes if n.startswith("code:")]
        errors = [n.split(":", 1)[1] for n in related_nodes if n.startswith("error:")]
        if modules:
            graph_lines.append("Related modules: " + ", ".join(modules[:5]))
            context_meta["related_modules"] = modules[:5]
        if errors:
            graph_lines.append("Related errors: " + ", ".join(errors[:5]))
            context_meta["related_errors"] = errors[:5]
    if graph_lines:
        description += "\n\n" + "\n".join(graph_lines)
    cluster_id: int | None = None
    cluster_traces: list[str] = []
    cluster_size = 0
    error_db = getattr(manager, "error_db", None)
    if error_db is not None:
        try:
            predictor = ErrorClusterPredictor(error_db)
            cluster_id, cluster_traces, cluster_size = predictor.best_cluster(
                prompt_path
            )
            if cluster_id is not None:
                context_meta["error_cluster_id"] = cluster_id
                context_meta["error_cluster_size"] = cluster_size
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
            logger.exception("failed to incorporate cluster trace")
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
        manager.register_patch_cycle(
            description, meta, provenance_token=provenance_token
        )
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
        raise QuickFixEngineError(
            "quick_fix_init_error",
            "generate_patch requires a SelfCodingEngine instance. Pass"
            " manager.engine so improvements originate from SelfCodingManager.",
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
                        return helper(
                            manager,
                            desc,
                            path=path,
                            metadata=context_meta,
                            target_region=target_region,
                        )
                    except TypeError:
                        try:
                            return helper(manager, desc)
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
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:
                shutil.copy2(before_target, path)
                logger.error("patch compilation failed: %s", exc)
                raise RuntimeError("patch compilation failed") from exc
            if test_command:
                try:
                    subprocess.run(test_command, check=True)
                except Exception as exc:
                    shutil.copy2(before_target, path)
                    logger.error("patch tests failed: %s", exc)
                    raise RuntimeError("patch tests failed") from exc
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
                    approved = True
                    ap = getattr(manager, "approval_policy", None)
                    if ap is not None:
                        try:
                            approved = bool(ap.approve(path))
                        except Exception:
                            logger.exception("approval policy execution failed")
                            approved = False
                    if not approved:
                        rbm = getattr(ap, "rollback_mgr", None)
                        if rbm is None and AutomatedRollbackManager is not None:
                            try:
                                rbm = AutomatedRollbackManager()
                            except Exception:
                                rbm = None
                        if rbm is not None:
                            try:
                                rbm.rollback(str(patch_id))
                            except Exception:
                                logger.exception("failed to rollback invalid patch")
                        if PatchHistoryDB is not None:
                            try:
                                PatchHistoryDB().record_vector_metrics(
                                    "",
                                    [],
                                    patch_id=patch_id,
                                    contribution=0.0,
                                    win=False,
                                    regret=True,
                                )
                            except Exception:
                                logger.exception("failed to log patch outcome")
                        event_bus = getattr(manager, "event_bus", None)
                        if event_bus:
                            try:
                                event_bus.publish(
                                    "quick_fix:approval_failed",
                                    {
                                        "bot": getattr(manager, "bot_name", ""),
                                        "module": path.as_posix(),
                                        "patch_id": patch_id,
                                    },
                                )
                            except Exception:
                                logger.exception("failed to publish approval_failed event")
                        return (None, risk_flags) if return_flags else None
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
        helper_fn: Callable[..., str] = manager_generate_helper,
    ) -> None:
        if context_builder is None:
            raise RuntimeError("context_builder is required")
        try:
            ensure_fresh_weights(context_builder)
        except Exception as exc:  # pragma: no cover - validation
            raise RuntimeError(
                "provided ContextBuilder cannot query local databases"
            ) from exc
        if getattr(manager, "bot_registry", None) is None or getattr(
            manager, "data_bot", None
        ) is None:
            raise RuntimeError(
                "manager must provide bot_registry and data_bot"
            )
        self.db = error_db
        self.manager = manager
        logger = logging.getLogger(self.__class__.__name__)
        try:
            self.manager.register_bot(self.__class__.__name__)
        except Exception:
            logger.exception("bot registration failed")
        self.threshold = threshold
        self.graph = graph or KnowledgeGraph()
        self.risk_threshold = risk_threshold
        self.predictor = predictor
        self.retriever = retriever
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
        self.helper_fn = helper_fn

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
        cluster_id: int | None = None
        cluster_traces: list[str] = []
        cluster_size = count
        try:
            predictor = ErrorClusterPredictor(self.db)
            cluster_id, cluster_traces, cluster_size = predictor.best_cluster(module)
        except Exception:
            cluster_id = None
            cluster_traces = []
            cluster_size = count
        if cluster_size < self.threshold:
            return
        try:
            path = resolve_path(f"{module}.py")
        except FileNotFoundError:
            return
        prompt_path = path_for_prompt(path)
        context_meta = {"error_type": etype, "module": prompt_path, "bot": bot}
        if cluster_id is not None:
            context_meta["error_cluster_id"] = cluster_id
            context_meta["error_cluster_size"] = cluster_size
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
        if cluster_traces:
            try:
                desc += "\n\n" + cluster_traces[0]
            except Exception:
                self.logger.exception("failed to append cluster trace to description")
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
        token = getattr(
            getattr(self.manager, "evolution_orchestrator", None),
            "provenance_token",
            None,
        )
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
                    provenance_token=token,
                    context_meta=context_meta,
                    context_builder=builder,
                )
            except TypeError:
                result = self.manager.run_patch(
                    path,
                    desc,
                    provenance_token=token,
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
        predictor = ErrorClusterPredictor(self.db)
        ranked: list[tuple[str, float, int, int | None, list[str]]] = []
        for item in modules:
            if isinstance(item, tuple):
                module, risk = item
            else:
                module, risk = item, 1.0
            cid, traces, size = predictor.best_cluster(module)
            impact = size * risk
            ranked.append((module, risk, size, cid, traces, impact))
        ranked.sort(key=lambda x: x[5], reverse=True)
        for module, risk, size, cid, traces, _impact in ranked:
            if risk < thresh:
                continue
            try:
                path = resolve_path(f"{module}.py")
            except FileNotFoundError:
                continue
            prompt_path = path_for_prompt(path)
            meta = {"module": prompt_path, "reason": "preemptive_patch"}
            if cid is not None:
                meta["error_cluster_id"] = cid
                meta["error_cluster_size"] = size
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
            if traces:
                try:
                    desc += "\n\n" + traces[0]
                except Exception:
                    self.logger.exception("failed to append trace to description")
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
            token = getattr(
                getattr(self.manager, "evolution_orchestrator", None),
                "provenance_token",
                None,
            )
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
                        provenance_token=token,
                        context_meta=meta,
                        context_builder=builder,
                    )
                except TypeError:
                    result = self.manager.run_patch(
                        path,
                        desc,
                        provenance_token=token,
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
                        self.manager,
                        engine=getattr(self.manager, "engine", None),
                        context_builder=self.context_builder,
                        provenance_token=token,
                        helper_fn=self.helper_fn,
                        graph=self.graph,
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
        *,
        provenance_token: str,
    ) -> Tuple[bool, int | None, List[str]]:
        """Generate and apply a patch returning its success status, id and flags.

        The patch is first generated via :func:`generate_patch` which performs
        internal risk checks.  When any validation flags are raised the change
        is reverted and ``False`` is returned along with ``None`` for the
        ``patch_id`` and the list of ``flags``.

        Parameters
        ----------
        provenance_token:
            Verified token confirming the call originates from the active
            ``EvolutionOrchestrator``.
        """

        ctx = context_meta or {}
        try:
            patch_id, flags = generate_patch(
                str(module_path),
                self.manager,
                engine=getattr(self.manager, "engine", None),
                context_builder=self.context_builder,
                provenance_token=provenance_token,
                description=description,
                context=ctx,
                return_flags=True,
                helper_fn=self.helper_fn,
                patch_logger=self.patch_logger,
                graph=self.graph,
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
        provenance_token: str,
    ) -> Tuple[bool, List[str]]:
        """Run quick-fix validation on ``module_name`` without applying it.

        Parameters
        ----------
        provenance_token:
            Verified token ensuring the request originates from the active
            ``EvolutionOrchestrator``.
        """
        flags: List[str]
        try:
            _pid, flags = generate_patch(
                module_name,
                self.manager,
                engine=getattr(self.manager, "engine", None),
                context_builder=self.context_builder,
                provenance_token=provenance_token,
                description=description,
                target_region=target_region,
                return_flags=True,
                helper_fn=self.helper_fn,
                patch_logger=self.patch_logger,
                graph=self.graph,
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


__all__ = ["QuickFixEngine", "generate_patch", "QuickFixEngineError"]
