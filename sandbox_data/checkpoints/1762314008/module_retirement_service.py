from __future__ import annotations

"""Service for archiving, compressing, or replacing modules flagged for retirement."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import logging
import shutil

from dynamic_path_router import resolve_path

from module_graph_analyzer import build_import_graph
from metrics_exporter import (
    update_module_retirement_metrics,
    retired_modules_total,
    compressed_modules_total,
    replaced_modules_total,
)


ContextBuilder: Optional[type[Any]]
_context_builder_import_error: Optional[Exception]
try:  # pragma: no cover - optional heavy dependency
    from vector_service.context_builder import ContextBuilder as _RealContextBuilder
except Exception as exc:  # pragma: no cover - allow module import without optional deps
    _context_builder_import_error = exc
    ContextBuilder = None
else:
    _context_builder_import_error = None
    ContextBuilder = _RealContextBuilder

SelfCodingManager: Optional[type[Any]]
_self_coding_manager_import_error: Optional[Exception]
try:  # pragma: no cover - optional heavy dependency
    from self_coding_manager import SelfCodingManager as _RealSelfCodingManager
except Exception as exc:  # pragma: no cover - allow module import without optional deps
    _self_coding_manager_import_error = exc
    SelfCodingManager = None
else:
    _self_coding_manager_import_error = None
    SelfCodingManager = _RealSelfCodingManager


_LOGGER = logging.getLogger(__name__)


def _make_stub_context_builder() -> Any:
    class _StubContextBuilder:
        def refresh_db_weights(self) -> None:
            _LOGGER.debug("stub context builder refresh requested; skipping")

    return _StubContextBuilder()


def _make_stub_manager() -> Any:
    class _StubRegistry:
        def update_bot(self, *args: Any, **kwargs: Any) -> None:
            _LOGGER.debug(
                "stub registry update ignored",
                extra={"args": args, "kwargs": kwargs},
            )

    class _StubOrchestrator:
        provenance_token: Any = None

        def register_patch_cycle(self, *_args: Any, **_kwargs: Any) -> None:
            _LOGGER.debug("stub orchestrator register_patch_cycle ignored")

    class _StubManager:
        bot_registry = _StubRegistry()
        evolution_orchestrator = _StubOrchestrator()
        quick_fix: Any = None
        error_db: Any = None
        _last_commit_hash: Any = None
        _retirement_stub = True

        def generate_patch(self, *_args: Any, **_kwargs: Any) -> Any:
            _LOGGER.debug("stub manager generate_patch invoked; returning None")
            return None

    _LOGGER.warning(
        "ModuleRetirementService operating in stub mode; self-coding actions will be skipped"
    )
    return _StubManager()


def _is_valid_context_builder(candidate: Any) -> bool:
    if candidate is None:
        return False
    if ContextBuilder is not None and isinstance(candidate, ContextBuilder):
        return True
    return callable(getattr(candidate, "refresh_db_weights", None))


def _normalise_manager(manager: Any) -> Any:
    if manager is None:
        return None
    required = ("generate_patch", "bot_registry", "evolution_orchestrator")
    if all(hasattr(manager, attr) for attr in required):
        return manager
    raise TypeError(
        "manager must provide generate_patch, bot_registry, and evolution_orchestrator"
    )


def _build_default_context_builder() -> Any:
    if ContextBuilder is None:
        _LOGGER.warning(
            "vector_service.context_builder.ContextBuilder unavailable; using stub"
        )
        return _make_stub_context_builder()
    try:  # pragma: no cover - heavy dependency path
        from context_builder_util import create_context_builder

        builder = create_context_builder()
    except Exception as exc:  # pragma: no cover - fallback path exercised in CI
        _LOGGER.warning(
            "Failed to create context builder; using stub fallback: %s", exc
        )
        return _make_stub_context_builder()
    if not _is_valid_context_builder(builder):
        _LOGGER.warning(
            "Context builder factory returned incompatible instance; using stub"
        )
        return _make_stub_context_builder()
    return builder


def _build_default_manager() -> Any:
    if SelfCodingManager is None:
        return _make_stub_manager()
    try:  # pragma: no cover - heavy dependency path
        from coding_bot_interface import get_shared_manager  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fall back to stub when helper missing
        return _make_stub_manager()
    try:
        manager = get_shared_manager()  # type: ignore[call-arg]
    except Exception:
        return _make_stub_manager()
    if not isinstance(manager, SelfCodingManager):
        return _make_stub_manager()
    return manager


class ModuleRetirementService:
    """Handle archival, compression, or replacement of modules based on relevancy flags."""

    def __init__(
        self,
        repo_root: Path | str = ".",
        *,
        context_builder: Any | None = None,
        manager: Any | None = None,
    ) -> None:
        context_builder = (
            context_builder if _is_valid_context_builder(context_builder) else None
        ) or _build_default_context_builder()
        manager = _normalise_manager(manager) or _build_default_manager()
        self.root = Path(resolve_path(repo_root))
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            graph = build_import_graph(self.root)
        except Exception:  # pragma: no cover - dependency graph failures
            graph = None
            self.logger.exception("failed to build import graph")
        self._graph = graph
        self._context_builder = context_builder
        self.manager = manager
        self._using_stub_manager = bool(getattr(manager, "_retirement_stub", False))
        try:
            self._context_builder.refresh_db_weights()
        except Exception:
            self.logger.exception("failed to initialise ContextBuilder")
            raise

    # ------------------------------------------------------------------
    def _normalise(self, module: str) -> str:
        mod = module
        if mod.endswith(".py"):
            mod = mod[:-3]
        return mod.replace("\\", "/")

    def _dependents(self, module: str) -> Iterable[str]:
        if self._graph is None:
            return []
        mod = self._normalise(module)
        if mod in self._graph:
            return list(self._graph.predecessors(mod))
        return []

    # ------------------------------------------------------------------
    def retire_module(self, module: str) -> bool:
        """Archive ``module`` if no other modules depend on it."""

        path = Path(
            resolve_path(
                self.root
                / (module if module.endswith(".py") else f"{module}.py")
            )
        )
        dependents = list(self._dependents(module))
        if dependents:
            self.logger.warning("cannot retire %s; dependents exist: %s", module, dependents)
            return False
        archive_dir = self.root / "sandbox_data" / "retired_modules"
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_dir = Path(resolve_path(archive_dir))
            if path.exists():
                shutil.move(str(path), archive_dir / path.name)
                retired_modules_total.inc()
                return True
            log_level = logging.DEBUG if self._using_stub_manager else logging.ERROR
            self.logger.log(log_level, "module not found: %s", module)
        except Exception:  # pragma: no cover - filesystem issues
            self.logger.exception("failed to retire module %s", module)
        return False

    def compress_module(self, module: str) -> bool:
        """Invoke quick fix tooling to minimise ``module``."""

        path = Path(
            resolve_path(
                self.root
                / (module if module.endswith(".py") else f"{module}.py")
            )
        )
        if not path.exists():
            log_level = logging.DEBUG if self._using_stub_manager else logging.ERROR
            self.logger.log(log_level, "module not found: %s", module)
            return False
        orchestrator = getattr(self.manager, "evolution_orchestrator", None)
        token = getattr(orchestrator, "provenance_token", None)
        if not token:
            self.logger.error("missing EvolutionOrchestrator provenance token")
            return False
        try:
            patch_id = self.manager.generate_patch(
                str(path),
                context_builder=self._context_builder,
                provenance_token=token,
                description=f"compress:{module}",
            )
            if patch_id is not None:
                compressed_modules_total.inc()
                registry = getattr(self.manager, "bot_registry", None)
                if registry:
                    try:
                        commit = getattr(self.manager, "_last_commit_hash", None)
                        registry.update_bot(
                            getattr(self.manager, "bot_name", module),
                            str(path),
                            patch_id=patch_id,
                            commit=commit,
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to update bot registry for %s", module)
                if orchestrator:
                    try:
                        orchestrator.register_patch_cycle(
                            {"bot": getattr(self.manager, "bot_name", "")}
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to notify EvolutionOrchestrator")
                try:
                    from sandbox_runner import integrate_new_orphans

                    integrate_new_orphans(
                        self.root, context_builder=self._context_builder
                    )
                except Exception:
                    self.logger.exception(
                        "integrate_new_orphans after compression failed",
                    )
                return True
        except Exception:  # pragma: no cover - patching issues
            self.logger.exception("compression failed for %s", module)
        return False

    def replace_module(self, module: str) -> bool:
        """Invoke quick fix tooling to propose replacement for ``module``."""

        path = Path(
            resolve_path(
                self.root
                / (module if module.endswith(".py") else f"{module}.py")
            )
        )
        if not path.exists():
            log_level = logging.DEBUG if self._using_stub_manager else logging.ERROR
            self.logger.log(log_level, "module not found: %s", module)
            return False
        orchestrator = getattr(self.manager, "evolution_orchestrator", None)
        token = getattr(orchestrator, "provenance_token", None)
        if not token:
            self.logger.error("missing EvolutionOrchestrator provenance token")
            return False
        try:
            patch_id = self.manager.generate_patch(
                str(path),
                context_builder=self._context_builder,
                provenance_token=token,
                description=f"replace:{module}",
            )
            if patch_id is not None:
                replaced_modules_total.inc()
                self.logger.info(
                    "generated replacement patch %s for %s", patch_id, module
                )
                registry = getattr(self.manager, "bot_registry", None)
                if registry:
                    try:
                        commit = getattr(self.manager, "_last_commit_hash", None)
                        registry.update_bot(
                            getattr(self.manager, "bot_name", module),
                            str(path),
                            patch_id=patch_id,
                            commit=commit,
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to update bot registry for %s", module)
                if orchestrator:
                    try:
                        orchestrator.register_patch_cycle(
                            {"bot": getattr(self.manager, "bot_name", "")}
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to notify EvolutionOrchestrator")
                try:
                    from sandbox_runner import integrate_new_orphans

                    integrate_new_orphans(
                        self.root, context_builder=self._context_builder
                    )
                except Exception:
                    self.logger.exception(
                        "integrate_new_orphans after replacement failed",
                    )
                return True
            self.logger.info("no replacement patch generated for %s", module)
        except Exception:  # pragma: no cover - patching issues
            self.logger.exception("replacement failed for %s", module)
        return False

    # ------------------------------------------------------------------
    def process_flags(self, flags: Dict[str, str]) -> Dict[str, str]:
        """Process relevancy ``flags`` mapping modules to actions."""

        results: Dict[str, str] = {}
        for mod, flag in flags.items():
            success = False
            if flag == "retire":
                success = self.retire_module(mod)
                if success:
                    results[mod] = "retired"
            elif flag == "compress":
                success = self.compress_module(mod)
                if success:
                    results[mod] = "compressed"
            elif flag == "replace":
                success = self.replace_module(mod)
                if success:
                    results[mod] = "replaced"
                    self.logger.info("replaced %s", mod)
                else:
                    results[mod] = "skipped"
                    self.logger.info("skipped %s", mod)
                continue
            if not success:
                results.setdefault(mod, "skipped")
        update_module_retirement_metrics(results)
        return results


__all__ = ["ModuleRetirementService"]
