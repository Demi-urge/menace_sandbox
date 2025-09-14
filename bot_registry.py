from __future__ import annotations

"""Graph based registry capturing bot interactions.

The registry persists bot connections to a database and allows bots to be
hot swapped at runtime. Updating a bot's backing module via ``update_bot``
broadcasts a ``bot:updated`` event so other components can react to the
change.
"""

from typing import List, Tuple, Union, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import time
import importlib
import importlib.util
import sys
import subprocess
import json
import os
import threading
from dataclasses import asdict, is_dataclass

try:
    from .databases import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore
try:
    from .neuroplasticity import PathwayDB
except Exception:  # pragma: no cover - optional dependency
    PathwayDB = None  # type: ignore

import networkx as nx
import logging

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover
    class UnifiedEventBus:  # type: ignore[override]
        pass
try:  # pragma: no cover - allow flat imports
    from .shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore
import db_router
from db_router import DBRouter, init_db_router

from .threshold_service import threshold_service
from .retry_utils import with_retry

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .self_coding_manager import SelfCodingManager
    from .data_bot import DataBot
else:  # pragma: no cover - runtime placeholders
    SelfCodingManager = Any  # type: ignore
    DataBot = Any  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .data_bot import persist_sc_thresholds
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from data_bot import persist_sc_thresholds  # type: ignore
    except Exception:  # pragma: no cover - last resort stub
        def persist_sc_thresholds(*_a, **_k):  # type: ignore
            return None

try:  # pragma: no cover - optional dependency
    from .rollback_manager import RollbackManager
except Exception:  # pragma: no cover - optional dependency
    RollbackManager = None  # type: ignore

logger = logging.getLogger(__name__)


class BotRegistry:
    """Store connections between bots using a directed graph."""

    def __init__(
        self,
        *,
        persist: Optional[Path | str] = None,
        event_bus: Optional["UnifiedEventBus"] = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.modules: Dict[str, str] = {}
        self.persist_path = Path(persist) if persist else None
        # Default to the shared event bus so all registries participate in the
        # same publish/subscribe channel unless explicitly overridden.
        self.event_bus = event_bus or _SHARED_EVENT_BUS
        self.heartbeats: Dict[str, float] = {}
        self.interactions_meta: List[Dict[str, object]] = []
        self._lock = threading.RLock()
        if self.persist_path and self.persist_path.exists():
            try:
                self.load(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to load bot registry from %s: %s", self.persist_path, exc
                )
        try:
            self.schedule_unmanaged_scan()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to schedule unmanaged bot scan")

    def register_bot(
        self,
        name: str,
        *,
        roi_threshold: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        manager: "SelfCodingManager" | None = None,
        data_bot: "DataBot" | None = None,
        is_coding_bot: bool = False,
    ) -> None:
        """Ensure *name* exists in the graph and persist metadata."""
        with self._lock:
            self.graph.add_node(name)
            node = self.graph.nodes[name]
            existing_mgr = node.get("selfcoding_manager") or node.get("manager")
            existing_data = node.get("data_bot")
            is_coding_bot = bool(is_coding_bot)
            if is_coding_bot:
                missing: list[str] = []
                mgr = manager or existing_mgr
                db = data_bot or existing_data
                if mgr is None:
                    missing.append("manager")
                if db is None:
                    missing.append("data_bot")
                if missing:
                    try:
                        from .self_coding_manager import internalize_coding_bot
                        from .self_coding_engine import SelfCodingEngine
                        from .model_automation_pipeline import ModelAutomationPipeline
                        from .data_bot import DataBot
                        from .code_database import CodeDB
                        from .gpt_memory import GPTMemoryManager
                        from vector_service.context_builder import ContextBuilder
                        from .self_coding_thresholds import get_thresholds

                        ctx = ContextBuilder()
                        engine = SelfCodingEngine(
                            CodeDB(), GPTMemoryManager(), context_builder=ctx
                        )
                        pipeline = ModelAutomationPipeline(
                            context_builder=ctx, bot_registry=self
                        )
                        db = db or DataBot(start_server=False)
                        th = get_thresholds(name)
                        internalize_coding_bot(
                            name,
                            engine,
                            pipeline,
                            data_bot=db,
                            bot_registry=self,
                            roi_threshold=getattr(th, "roi_drop", None),
                            error_threshold=getattr(th, "error_increase", None),
                            test_failure_threshold=getattr(
                                th, "test_failure_increase", None
                            ),
                        )
                        if self.event_bus:
                            try:
                                self.event_bus.publish(
                                    "bot:internalized", {"bot": name}
                                )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(
                                    "Failed to publish bot:internalized event: %s",
                                    exc,
                                )
                        return
                    except Exception as exc:
                        if self.event_bus:
                            try:
                                self.event_bus.publish(
                                    "bot:internalization_failed",
                                    {"bot": name, "error": str(exc)},
                                )
                            except Exception as exc2:  # pragma: no cover - best effort
                                logger.error(
                                    "Failed to publish bot:internalization_failed event: %s",
                                    exc2,
                                )
                        raise RuntimeError(
                            "coding bot could not be internalized"
                        ) from exc
            if roi_threshold is not None:
                node["roi_threshold"] = float(roi_threshold)
            if error_threshold is not None:
                node["error_threshold"] = float(error_threshold)
            if test_failure_threshold is not None:
                node["test_failure_threshold"] = float(test_failure_threshold)
            node.setdefault("patch_history", [])
            if manager is not None:
                node["selfcoding_manager"] = manager
            if data_bot is not None:
                node["data_bot"] = data_bot
                try:
                    data_bot.check_degradation(
                        name, roi=0.0, errors=0.0, test_failures=0.0
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "failed to initialise baseline for %s: %s", name, exc
                    )
                if manager is not None:
                    orchestrator = getattr(manager, "evolution_orchestrator", None)
                    if orchestrator is not None and hasattr(
                        orchestrator, "register_patch_cycle"
                    ):
                        bus = getattr(data_bot, "event_bus", None)
                        if bus:
                            handler = (
                                lambda _t, e: orchestrator.register_patch_cycle(e)
                            )

                            def _subscribe() -> None:
                                try:
                                    bus.subscribe("degradation:detected", handler)
                                except Exception as exc:  # pragma: no cover - best effort
                                    logger.error(
                                        "failed to subscribe degradation callback for %s: %s",
                                        name,
                                        exc,
                                    )
                                    try:
                                        bus.subscribe("bus:restarted", lambda *_: _subscribe())
                                    except Exception as sub_exc:  # pragma: no cover - best effort
                                        logger.error(
                                            "failed to schedule resubscription for %s: %s",
                                            name,
                                            sub_exc,
                                        )

                            _subscribe()
                        else:
                            try:
                                data_bot.subscribe_degradation(
                                    orchestrator.register_patch_cycle
                                )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(
                                    "failed to subscribe degradation callback for %s: %s",
                                    name,
                                    exc,
                                )
                    else:
                        def _on_degraded(event: dict, _bot=name, _mgr=manager):
                            if str(event.get("bot")) != _bot:
                                return
                            try:
                                desc = f"auto_patch_due_to_degradation:{_bot}"
                                token = getattr(
                                    getattr(_mgr, "evolution_orchestrator", None),
                                    "provenance_token",
                                    None,
                                )
                                result_vals = _mgr.register_patch_cycle(
                                    desc, event, provenance_token=token
                                )
                                if isinstance(result_vals, tuple):
                                    patch_id, commit = result_vals
                                else:
                                    patch_id, commit = (None, None)
                                module = self.graph.nodes[_bot].get("module")
                                result = None
                                if module and hasattr(_mgr, "generate_and_patch"):
                                    result, new_commit = _mgr.generate_and_patch(
                                        Path(module),
                                        desc,
                                        context_meta=event,
                                        provenance_token=token,
                                    )
                                    commit = commit or new_commit
                                try:
                                    ph = self.graph.nodes[_bot].setdefault(
                                        "patch_history", []
                                    )
                                    ph.append(
                                        {
                                            "patch_id": patch_id,
                                            "commit": commit,
                                            "ts": time.time(),
                                        }
                                    )
                                except Exception:
                                    logger.exception(
                                        "failed to record patch history for %s",
                                        _bot,
                                    )
                                if result is not None and self.event_bus:
                                    try:
                                        payload: Dict[str, Any] = {
                                            "bot": _bot,
                                            "patch_id": patch_id,
                                            "commit": commit,
                                            "result": (
                                                asdict(result)
                                                if is_dataclass(result)
                                                else getattr(result, "__dict__", result)
                                            ),
                                        }
                                        self.event_bus.publish(
                                            "bot:patch_applied", payload
                                        )
                                    except Exception as exc:
                                        logger.error(
                                            "Failed to publish bot:patch_applied event: %s",
                                            exc,
                                        )
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(
                                    "degradation callback failed for %s: %s",
                                    _bot,
                                    exc,
                                )

                        bus = getattr(data_bot, "event_bus", None)

                        def _subscribe() -> None:
                            if bus:
                                bus.subscribe(
                                    "degradation:detected",
                                    lambda _t, e: _on_degraded(e),
                                )
                            else:
                                data_bot.subscribe_degradation(_on_degraded)

                        try:
                            with_retry(_subscribe, attempts=3, delay=1.0, logger=logger)
                        except Exception as exc:
                            logger.error(
                                "failed to subscribe degradation callback for %s: %s",
                                name,
                                exc,
                            )
                            if self.event_bus:
                                try:
                                    self.event_bus.publish(
                                        "bot:subscription_failed",
                                        {"bot": name, "error": str(exc)},
                                    )
                                except Exception as exc2:  # pragma: no cover - best effort
                                    logger.error(
                                        "Failed to publish bot:subscription_failed event: %s",
                                        exc2,
                                    )
                            raise
            if (
                roi_threshold is not None
                or error_threshold is not None
                or test_failure_threshold is not None
            ):
                try:
                    threshold_service.update(
                        name,
                        roi_drop=roi_threshold,
                        error_threshold=error_threshold,
                        test_failure_threshold=test_failure_threshold,
                    )
                    persist_sc_thresholds(
                        name,
                        roi_drop=roi_threshold,
                        error_increase=error_threshold,
                        test_failure_increase=test_failure_threshold,
                        event_bus=self.event_bus,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "failed to persist thresholds for %s: %s", name, exc
                    )
            if self.event_bus:
                try:
                    self.event_bus.publish("bot:new", {"name": name})
                except Exception as exc:
                    logger.error("Failed to publish bot:new event: %s", exc)
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as exc:
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, exc
                    )

    def schedule_unmanaged_scan(self, interval: float = 3600.0) -> None:
        """Periodically scan for unmanaged coding bots and register them."""

        root = Path(__file__).resolve().parent
        script = root / "tools" / "find_unmanaged_bots.py"
        if not script.exists():
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    result = subprocess.run(
                        [sys.executable, str(script), str(root)],
                        capture_output=True,
                        text=True,
                    )
                    for line in result.stdout.splitlines():
                        if "unmanaged bot classes" in line:
                            cls_part = line.split("unmanaged bot classes:", 1)[1]
                            for bot in [c.strip() for c in cls_part.split(",") if c.strip()]:
                                try:
                                    self.register_bot(bot, is_coding_bot=True)
                                except Exception:  # pragma: no cover - best effort
                                    logger.exception(
                                        "auto-registration failed for %s", bot
                                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("scheduled unmanaged bot scan failed")

        threading.Thread(target=_loop, daemon=True).start()

    def _verify_signed_provenance(self, patch_id: int, commit: str) -> bool:
        """Return ``True`` if a signed provenance file confirms the update."""

        prov_file = os.environ.get("PATCH_PROVENANCE_FILE")
        pubkey = os.environ.get("PATCH_PROVENANCE_PUBKEY") or os.environ.get(
            "PATCH_PROVENANCE_PUBLIC_KEY"
        )
        if not prov_file or not pubkey:
            raise RuntimeError("signed provenance required")
        try:
            with open(prov_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            data = payload.get("data") or {}
            signature = payload.get("signature")
            if not signature:
                raise RuntimeError("missing signature")
            if str(data.get("patch_id")) != str(patch_id) or str(
                data.get("commit")
            ) != str(commit):
                raise RuntimeError("provenance mismatch")
            from .override_validator import verify_signature

            if not verify_signature(data, signature, pubkey):
                raise RuntimeError("invalid signature")
            logger.info(
                "verified signed provenance for patch_id=%s commit=%s",
                patch_id,
                commit,
            )
            return True
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - best effort
            raise RuntimeError(f"provenance verification failed: {exc}") from exc

    def update_bot(
        self,
        name: str,
        module_path: str,
        *,
        patch_id: int | None = None,
        commit: str | None = None,
    ) -> None:
        """Update stored module path for ``name`` and emit ``bot:updated``.

        ``patch_id`` and ``commit`` are expected from the ``SelfCodingManager``
        so changes can be traced back to their origin.  If either piece of
        metadata is missing this method attempts to retrieve it from
        :mod:`patch_provenance` and retries once.  If the metadata still cannot
        be determined a :class:`RuntimeError` is raised.
        """

        if patch_id is None or commit is None:
            logger.warning(
                "update_bot called without provenance for %s (patch_id=%s commit=%s)",
                name,
                patch_id,
                commit,
            )
            if patch_id is not None:
                try:
                    from .patch_provenance import PatchProvenanceService

                    service = PatchProvenanceService()
                    rec = service.db.get(patch_id)
                    if rec and getattr(rec, "summary", None):
                        try:
                            commit = json.loads(rec.summary).get("commit")
                        except Exception:  # pragma: no cover - best effort
                            commit = None
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to fetch patch provenance for %s: %s", patch_id, exc
                    )
            if patch_id is None or commit is None:
                raise RuntimeError("patch provenance required")

        with self._lock:
            self.register_bot(name, is_coding_bot=False)
            node = self.graph.nodes[name]

            try:
                self._verify_signed_provenance(patch_id, commit)
            except RuntimeError as exc:
                logger.error("Signed provenance verification failed: %s", exc)
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:update_blocked",
                            {
                                "name": name,
                                "module": module_path,
                                "patch_id": patch_id,
                                "commit": commit,
                                "reason": "unverified_provenance",
                                "error": str(exc),
                            },
                        )
                    except Exception as exc2:
                        logger.error(
                            "Failed to publish bot:update_blocked event: %s", exc2
                        )
                node["update_blocked"] = True
                if self.persist_path:
                    try:
                        self.save(self.persist_path)
                    except Exception as exc2:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to save bot registry to %s: %s",
                            self.persist_path,
                            exc2,
                        )
                raise RuntimeError(
                    "update blocked: provenance verification failed"
                ) from exc

            prev_state = dict(node)
            prev_module_entry = self.modules.get(name)
            node["module"] = module_path
            node["version"] = int(node.get("version", 0)) + 1
            node["patch_id"] = patch_id
            node["commit"] = commit
            try:
                ph = node.setdefault("patch_history", [])
                ph.append({"patch_id": patch_id, "commit": commit, "ts": time.time()})
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to record patch history for %s", name)
            self.modules[name] = module_path

            if self.event_bus:
                try:
                    payload = {
                        "name": name,
                        "module": module_path,
                        "version": node["version"],
                        "patch_id": patch_id,
                        "commit": commit,
                    }
                    self.event_bus.publish("bot:updated", payload)
                except Exception as exc:
                    logger.error("Failed to publish bot:updated event: %s", exc)
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as exc:
                    logger.error(
                        "Failed to save bot registry to %s: %s",
                        self.persist_path,
                        exc,
                    )
            update_ok = False
            try:
                self.hot_swap_bot(name)
                self.health_check_bot(name, prev_state)
                update_ok = True
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "thresholds:refresh", {"bot": name}
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to publish thresholds:refresh event for %s: %s",
                            name,
                            exc,
                        )
            except Exception as exc:
                if prev_module_entry is None:
                    self.modules.pop(name, None)
                else:
                    self.modules[name] = prev_module_entry
                node.clear()
                node.update(prev_state)
                if self.persist_path:
                    try:
                        self.save(self.persist_path)
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error(
                            "Failed to save bot registry to %s: %s",
                            self.persist_path,
                            exc,
                        )
                if self.event_bus:
                    try:
                        self.event_bus.publish(
                            "bot:update_rolled_back",
                            {
                                "name": name,
                                "module": module_path,
                                "patch_id": patch_id,
                                "commit": commit,
                                "error": str(exc),
                            },
                        )
                    except Exception as pub_exc:
                        logger.error(
                            "Failed to publish bot:update_rolled_back event: %s",
                            pub_exc,
                        )
                if RollbackManager is not None:
                    try:
                        RollbackManager().rollback(
                            str(patch_id), requesting_bot=name
                        )
                    except Exception as rb_exc:  # pragma: no cover - best effort
                        logger.error(
                            "RollbackManager rollback failed for %s: %s",
                            name,
                            rb_exc,
                        )
                raise
        return update_ok

    def hot_swap(self, name: str, module_path: str) -> None:
        """Update ``module_path`` for ``name`` and reload the bot.

        This helper ensures the registry records the new module before
        delegating to :meth:`hot_swap_bot` which performs the actual import
        and validation.  The bot entry must already contain provenance
        metadata (commit hash and patch id) which is typically provided by
        :meth:`update_bot`.
        """

        with self._lock:
            self.register_bot(name, is_coding_bot=False)
            node = self.graph.nodes[name]
            node["module"] = module_path
            self.modules[name] = module_path
        self.hot_swap_bot(name)

    def hot_swap_bot(self, name: str) -> None:
        """Import or reload the module backing ``name`` and refresh references."""

        node = self.graph.nodes.get(name)
        if not node or "module" not in node:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node["module"]
        commit = node.get("commit")
        patch_id = node.get("patch_id")
        prev_module = node.get("last_good_module")
        prev_version = node.get("last_good_version")
        prev_commit = node.get("last_good_commit")
        prev_patch = node.get("last_good_patch_id")
        if not commit or patch_id is None:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "missing_provenance"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "missing_provenance",
                        },
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:update_blocked event: %s", exc)
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: missing provenance metadata")

        stored_commit: str | None = None
        try:
            from .patch_provenance import PatchProvenanceService

            service = PatchProvenanceService()
            rec = service.db.get(patch_id)
            if rec and getattr(rec, "summary", None):
                try:
                    stored_commit = json.loads(rec.summary).get("commit")
                except Exception:  # pragma: no cover - best effort
                    stored_commit = None
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to fetch patch provenance for %s: %s", patch_id, exc)
        if stored_commit != commit:
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "provenance_mismatch"},
                    )
                    self.event_bus.publish(
                        "bot:update_blocked",
                        {
                            "name": name,
                            "module": module_path,
                            "reason": "provenance_mismatch",
                            "expected": stored_commit,
                            "actual": commit,
                        },
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to publish bot:update_blocked event: %s", exc
                    )
            manager = node.get("selfcoding_manager") or node.get("manager")
            if manager and hasattr(manager, "register_patch_cycle"):
                try:
                    token = getattr(
                        getattr(manager, "evolution_orchestrator", None),
                        "provenance_token",
                        None,
                    )
                    manager.register_patch_cycle(
                        f"manual change detected for {name}",
                        {
                            "reason": "provenance_mismatch",
                            "module": module_path,
                        },
                        provenance_token=token,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to notify SelfCodingManager for %s: %s", name, exc
                    )
            node["update_blocked"] = True
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise RuntimeError("update blocked: provenance mismatch")

        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain", module_path]
            ).decode()
            if status.strip() and self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:manual_change",
                        {"name": name, "module": module_path, "reason": "uncommitted_changes"},
                    )
                except Exception as exc:
                    logger.error("Failed to publish bot:manual_change event: %s", exc)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to check manual changes for %s: %s", module_path, exc)
        try:
            path_obj = Path(module_path)
            if path_obj.suffix == ".py" or "/" in module_path:
                module_name = path_obj.stem
                spec = importlib.util.spec_from_file_location(
                    module_name, module_path
                )
                if not spec or not spec.loader:
                    raise ImportError(f"cannot load module from {module_path}")
                importlib.invalidate_caches()
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    module.__file__ = module_path
                    module.__spec__ = spec
                    spec.loader.exec_module(module)
                else:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            else:
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)
            node["last_good_module"] = module_path
            node["last_good_version"] = node.get("version")
            node["last_good_commit"] = commit
            node["last_good_patch_id"] = patch_id
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(
                "Failed to hot swap bot %s from %s: %s", name, module_path, exc
            )
            if prev_module is not None:
                node["module"] = prev_module
            if prev_version is not None:
                node["version"] = prev_version
            if prev_commit is not None:
                node["commit"] = prev_commit
            if prev_patch is not None:
                node["patch_id"] = prev_patch
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s", self.persist_path, save_exc
                    )
            raise

    def health_check_bot(self, name: str, prev_state: Optional[Dict[str, object]] = None) -> None:
        """Import the bot module and record a heartbeat to verify health."""

        node = self.graph.nodes.get(name)
        if not node or "module" not in node:
            raise KeyError(f"bot {name!r} has no module path")
        module_path = node["module"]
        try:
            path_obj = Path(module_path)
            if path_obj.suffix == ".py" or "/" in module_path:
                module_name = path_obj.stem
            else:
                module_name = module_path
            importlib.invalidate_caches()
            importlib.import_module(module_name)
            self.record_heartbeat(name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Health check failed for bot %s: %s", name, exc)
            if prev_state is not None:
                node.clear()
                node.update(prev_state)
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "bot:hot_swap_failed",
                        {"name": name, "module": module_path, "error": str(exc)},
                    )
                except Exception as pub_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to publish bot:hot_swap_failed event: %s", pub_exc
                    )
            if self.persist_path:
                try:
                    self.save(self.persist_path)
                except Exception as save_exc:  # pragma: no cover - best effort
                    logger.error(
                        "Failed to save bot registry to %s: %s",
                        self.persist_path,
                        save_exc,
                    )
            raise

    def register_interaction(self, from_bot: str, to_bot: str, weight: float = 1.0) -> None:
        """Record that *from_bot* interacted with *to_bot*."""
        self.register_bot(from_bot, is_coding_bot=False)
        self.register_bot(to_bot, is_coding_bot=False)
        if self.graph.has_edge(from_bot, to_bot):
            self.graph[from_bot][to_bot]["weight"] += weight
        else:
            self.graph.add_edge(from_bot, to_bot, weight=weight)
        self.interactions_meta.append(
            {"from": from_bot, "to": to_bot, "weight": weight, "ts": time.time()}
        )
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:interaction", {"from": from_bot, "to": to_bot, "weight": weight}
                )
            except Exception as exc:
                logger.error("Failed to publish bot:interaction event: %s", exc)
        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception as exc:
                logger.error(
                    "Failed to save bot registry to %s: %s", self.persist_path, exc
                )

    def connections(self, bot: str, depth: int = 1) -> List[Tuple[str, float]]:
        """Return outgoing connections up to *depth* hops."""
        results: List[Tuple[str, float]] = []
        if bot not in self.graph:
            return results
        for nbr in self.graph.successors(bot):
            w = float(self.graph[bot][nbr].get("weight", 1.0))
            results.append((nbr, w))
            if depth > 1:
                results.extend(self.connections(nbr, depth - 1))
        return results

    # ------------------------------------------------------------------
    def record_heartbeat(self, name: str) -> None:
        """Update last seen timestamp for *name*."""
        self.heartbeats[name] = time.time()
        if self.event_bus:
            try:
                self.event_bus.publish("bot:heartbeat", {"name": name})
            except Exception as exc:
                logger.error("Failed to publish bot:heartbeat event: %s", exc)
                try:
                    self.event_bus.publish(
                        "bot:heartbeat_error", {"name": name, "error": str(exc)}
                    )
                except Exception:
                    logger.exception("Failed publishing heartbeat error")

    def record_validation(self, bot: str, module: str, passed: bool) -> None:
        """Record patch validation outcome for ``bot``."""
        self.interactions_meta.append(
            {"bot": bot, "module": module, "passed": bool(passed), "ts": time.time()}
        )
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "bot:patch_validation",
                    {"bot": bot, "module": module, "passed": bool(passed)},
                )
            except Exception as exc:
                logger.error("Failed to publish bot:patch_validation event: %s", exc)

    def active_bots(self, timeout: float = 60.0) -> Dict[str, float]:
        """Return bots seen within ``timeout`` seconds."""
        now = time.time()
        return {n: ts for n, ts in self.heartbeats.items() if now - ts <= timeout}

    def record_interaction_metadata(
        self,
        from_bot: str,
        to_bot: str,
        *,
        duration: float,
        success: bool,
        resources: str = "",
    ) -> None:
        """Store detailed metadata for an interaction."""
        self.interactions_meta.append(
            {
                "from": from_bot,
                "to": to_bot,
                "duration": duration,
                "success": success,
                "resources": resources,
                "ts": time.time(),
            }
        )

    def aggregate_statistics(self) -> Dict[str, float]:
        """Return simple aggregate metrics about interactions."""
        if not self.interactions_meta:
            return {"count": 0, "success_rate": 0.0, "avg_duration": 0.0}
        count = len(self.interactions_meta)
        successes = sum(1 for rec in self.interactions_meta if rec.get("success"))
        total_dur = sum(float(rec.get("duration", 0.0)) for rec in self.interactions_meta)
        return {
            "count": count,
            "success_rate": successes / count,
            "avg_duration": total_dur / count,
        }

    # ------------------------------------------------------------------
    def save(
        self, dest: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Persist the current graph to a SQLite-backed database."""
        if isinstance(dest, (str, Path)):
            path = Path(dest)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(dest, DBRouter):
            conn = dest.get_connection("bots")
            close_conn = False
        elif MenaceDB is not None and isinstance(dest, MenaceDB):
            conn = dest.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(dest, PathwayDB):
            conn = dest.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported destination for save")

        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS bot_nodes(" "name TEXT PRIMARY KEY, "
            "module TEXT, "
            "version INTEGER, "
            "last_good_module TEXT, "
            "last_good_version INTEGER)"
        )
        # Ensure columns exist for databases created before they were introduced.
        try:  # pragma: no cover - only executed on legacy schemas
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
            if "module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN module TEXT")
            if "version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN version INTEGER")
            if "last_good_module" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_module TEXT")
            if "last_good_version" not in cols:
                cur.execute("ALTER TABLE bot_nodes ADD COLUMN last_good_version INTEGER")
        except Exception:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_edges(
                from_bot TEXT,
                to_bot TEXT,
                weight REAL,
                PRIMARY KEY(from_bot, to_bot)
            )
            """
        )
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            module = data.get("module")
            version = data.get("version")
            last_mod = data.get("last_good_module")
            last_ver = data.get("last_good_version")
            cur.execute(
                """
                INSERT OR REPLACE INTO bot_nodes(
                    name, module, version, last_good_module, last_good_version
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (node, module, version, last_mod, last_ver),
            )
        for u, v, data in self.graph.edges(data=True):
            cur.execute(
                "REPLACE INTO bot_edges(from_bot,to_bot,weight) VALUES(?,?,?)",
                (u, v, float(data.get("weight", 1.0))),
            )
        conn.commit()
        if close_conn:
            conn.close()

    # ------------------------------------------------------------------
    def load(
        self, src: Union[Path, str, "MenaceDB", "PathwayDB", DBRouter]
    ) -> None:
        """Populate ``self.graph`` from ``src`` tables."""
        if isinstance(src, (str, Path)):
            path = Path(src)
            router = db_router.GLOBAL_ROUTER or init_db_router(
                "bot_registry", str(path), str(path)
            )
            conn = router.get_connection("bots")
            close_conn = False
        elif isinstance(src, DBRouter):
            conn = src.get_connection("bots")
            close_conn = False
        elif MenaceDB is not None and isinstance(src, MenaceDB):
            conn = src.engine.raw_connection()
            close_conn = False
        elif PathwayDB is not None and isinstance(src, PathwayDB):  # pragma: no cover - rarely used
            conn = src.conn
            close_conn = False
        else:  # pragma: no cover - invalid type
            raise TypeError("Unsupported source for load")

        self.graph.clear()
        cur = conn.cursor()
        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(bot_nodes)").fetchall()]
        except Exception:
            cols = []

        module_col = "module" in cols
        version_col = "version" in cols
        last_mod_col = "last_good_module" in cols
        last_ver_col = "last_good_version" in cols
        select_cols = [
            c for c in ["module", "version", "last_good_module", "last_good_version"] if c in cols
        ]
        col_sql = ", ".join(select_cols)
        try:
            if col_sql:
                node_rows = cur.execute(
                    f"SELECT name, {col_sql} FROM bot_nodes"
                ).fetchall()
            else:
                node_rows = cur.execute("SELECT name FROM bot_nodes").fetchall()
        except Exception:  # pragma: no cover - corrupted table
            node_rows = []

        for row in node_rows:
            name = row[0]
            self.graph.add_node(name)
            idx = 1
            if module_col:
                module = row[idx]
                idx += 1
                if module is not None:
                    self.graph.nodes[name]["module"] = module
            if version_col:
                version = row[idx]
                idx += 1
                if version is not None:
                    self.graph.nodes[name]["version"] = int(version)
            if last_mod_col:
                last_mod = row[idx]
                idx += 1
                if last_mod is not None:
                    self.graph.nodes[name]["last_good_module"] = last_mod
            if last_ver_col:
                last_ver = row[idx]
                idx += 1
                if last_ver is not None:
                    self.graph.nodes[name]["last_good_version"] = int(last_ver)

        try:
            edge_rows = cur.execute(
                "SELECT from_bot, to_bot, weight FROM bot_edges"
            ).fetchall()
        except Exception:
            edge_rows = []
        for u, v, w in edge_rows:
            self.graph.add_edge(u, v, weight=float(w))

        if close_conn:
            conn.close()


__all__ = ["BotRegistry"]
