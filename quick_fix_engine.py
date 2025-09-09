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
from typing import Tuple, Iterable, Dict, Any, List, TYPE_CHECKING

from .snippet_compressor import compress_snippets

from .codebase_diff_checker import generate_code_diff, flag_risky_changes
from context_builder_util import ensure_fresh_weights
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore
try:  # pragma: no cover - optional dependency
    from .error_cluster_predictor import ErrorClusterPredictor
except Exception:  # pragma: no cover - optional dependency
    ErrorClusterPredictor = object  # type: ignore

from .error_bot import ErrorDB
from .self_coding_manager import SelfCodingManager
from .knowledge_graph import KnowledgeGraph
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
    from self_improvement.prompt_strategies import PromptStrategy, render_prompt
except Exception:  # pragma: no cover - fallback for tests
    class PromptStrategy(str):  # type: ignore
        pass

    def render_prompt(*a: object, **k: object) -> str:  # type: ignore
        return ""

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .self_coding_engine import SelfCodingEngine
    from .self_improvement.target_region import TargetRegion
try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when unavailable
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service lacks explicit class."""

        pass
try:  # pragma: no cover - optional dependency
    from .human_alignment_flagger import _collect_diff_data
except Exception:  # pragma: no cover - fallback for tests
    def _collect_diff_data(*a, **k):
        return {}
try:  # pragma: no cover - optional dependency
    from .human_alignment_agent import HumanAlignmentAgent
except Exception:  # pragma: no cover - fallback for tests
    HumanAlignmentAgent = object  # type: ignore
try:  # pragma: no cover - optional dependency
    from .violation_logger import log_violation
except Exception:  # pragma: no cover - fallback for tests
    def log_violation(*a, **k):
        return None


_VEC_METRICS = None


def generate_patch(
    module: str,
    engine: "SelfCodingEngine" | None = None,
    *,
    context_builder: ContextBuilder,
    description: str | None = None,
    strategy: PromptStrategy | None = None,
    patch_logger: PatchLogger | None = None,
    context: Dict[str, Any] | None = None,
    effort_estimate: float | None = None,
    target_region: "TargetRegion" | None = None,
) -> int | None:
    """Attempt a quick patch for *module* and return the patch id.

    A provided :class:`vector_service.ContextBuilder` is used to gather context
    for the patch.

    Parameters
    ----------
    module:
        Target module path or module name without ``.py``.
    engine:
        Optional :class:`~self_coding_engine.SelfCodingEngine` instance.  If not
        provided, a minimal engine is instantiated on demand.  The function
        tolerates missing dependencies and simply returns ``None`` on failure.
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
        return None

    prompt_path = path_for_prompt(path.as_posix())
    description = description or f"preemptive fix for {prompt_path}"
    context_meta: Dict[str, Any] = {"module": prompt_path, "reason": "preemptive_fix"}
    if context:
        context_meta.update(context)
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

    if patch_logger is None:
        try:
            patch_logger = PatchLogger()
        except Exception:
            patch_logger = None

    if engine is None:
        try:  # pragma: no cover - heavy dependencies
            from .self_coding_engine import SelfCodingEngine
            from .code_database import CodeDB
            from .menace_memory_manager import MenaceMemoryManager

            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=builder
            )
        except Exception as exc:  # pragma: no cover - optional deps
            logger.error("self coding engine unavailable: %s", exc)
            return None

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
                    chunks = get_chunk_summaries(path, token_limit)
                except Exception:
                    chunks = []
            patch_ids: List[int | None] = []
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
                    patch_ids.append(pid)
                patch_id = patch_ids[-1] if patch_ids else None
            else:
                try:
                    apply = getattr(engine, "apply_patch_with_retry")
                except AttributeError:
                    apply = getattr(engine, "apply_patch")
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
            after_target = Path(after_dir) / rel
            after_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, after_target)
            diff_struct = generate_code_diff(before_dir, after_dir)
            risky = flag_risky_changes(diff_struct)
            if risky:
                logger.warning("risky changes detected: %s", risky)
                shutil.copy2(before_target, path)
                return None
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
                try:
                    patch_logger.track_contributors(
                        [(f"{o}:{vid}", score) for o, vid, score in vectors],
                        patch_id is not None,
                        patch_id=str(patch_id) if patch_id is not None else "",
                        session_id=cb_session,
                        effort_estimate=effort_estimate,
                    )
                except Exception:
                    pass
                else:  # run embedding backfill on success
                    try:
                        EmbeddingBackfill().run(
                            db="code",
                            backend=os.getenv("VECTOR_BACKEND", "annoy"),
                        )
                    except BaseException:  # pragma: no cover - best effort
                        logger.debug("embedding backfill failed", exc_info=True)
            try:
                from sandbox_runner import post_round_orphan_scan

                post_round_orphan_scan(Path.cwd())
            except Exception:
                logger.exception(
                    "post_round_orphan_scan after preemptive patch failed"
                )
            return patch_id
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.error("quick fix generation failed for %s: %s", prompt_path, exc)
        return None


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
        builder = self.context_builder
        ctx_block = ""
        cb_session = uuid.uuid4().hex
        context_meta["context_session_id"] = cb_session
        try:
            query = f"{etype} in {prompt_path}"
            ctx_block = builder.build(query, session_id=cb_session)
            if isinstance(ctx_block, (FallbackResult, ErrorResult)):
                ctx_block = ""
        except Exception:
            ctx_block = ""
        desc = f"quick fix {etype}"
        if ctx_block:
            try:
                ctx_block = compress_snippets({"snippet": ctx_block}).get("snippet", "")
            except Exception:
                ctx_block = ""
            if ctx_block:
                desc += "\n\n" + ctx_block
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
            context_meta["retrieval_vectors"] = vectors
        patch_id = None
        try:
            try:
                result = self.manager.run_patch(path, desc, context_meta=context_meta)
            except TypeError:
                result = self.manager.run_patch(path, desc)
            patch_id = getattr(result, "patch_id", None)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.error("quick fix failed for %s: %s", bot, exc)
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
            builder = self.context_builder
            ctx = ""
            cb_session = uuid.uuid4().hex
            meta["context_session_id"] = cb_session
            try:
                query = f"preemptive patch {prompt_path}"
                ctx = builder.build(query, session_id=cb_session)
                if isinstance(ctx, (FallbackResult, ErrorResult)):
                    ctx = ""
            except Exception:
                ctx = ""
            desc = "preemptive_patch"
            if ctx:
                desc += "\n\n" + ctx
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
                meta["retrieval_vectors"] = vectors
            patch_id = None
            try:
                try:
                    result = self.manager.run_patch(path, desc, context_meta=meta)
                except TypeError:
                    result = self.manager.run_patch(path, desc)
                patch_id = getattr(result, "patch_id", None)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.error("preemptive patch failed for %s: %s", prompt_path, exc)
                try:
                    patch_id = generate_patch(
                        prompt_path,
                        getattr(self.manager, "engine", None),
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
                ids = {f"{o}:{v}": s for o, v, s in vectors}
                try:
                    result = bool(patch_id)
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
    def run_and_validate(self, bot: str) -> None:
        """Run :meth:`run` then execute the test suite."""
        self.run(bot)


__all__ = ["QuickFixEngine", "generate_patch"]
