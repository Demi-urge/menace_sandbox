from __future__ import annotations

"""Periodic self-improvement engine for the Menace system."""

import logging
from .logging_utils import log_record, get_logger, setup_logging, set_correlation_id
import time
import threading
import asyncio
import os

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    setup_logging()
from sandbox_settings import SandboxSettings
from .metrics_exporter import (
    synergy_weight_updates_total,
    synergy_weight_update_failures_total,
    synergy_weight_update_alerts_total,
)
from alert_dispatcher import dispatch_alert
import json
import sqlite3
import pickle
import io
import tempfile
import math
from pathlib import Path
import numpy as np
import socket
import contextlib

logger = get_logger(__name__)

BACKUP_COUNT = 3


# Default synergy weight values used when no valid file is available
DEFAULT_SYNERGY_WEIGHTS: dict[str, float] = {
    "roi": 1.0,
    "efficiency": 1.0,
    "resilience": 1.0,
    "antifragility": 1.0,
    "reliability": 1.0,
    "maintainability": 1.0,
    "throughput": 1.0,
}


def _rotate_backups(path: Path) -> None:
    """Rotate backup files for ``path``."""
    backups = [path.with_suffix(path.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)]
    for i in range(BACKUP_COUNT - 1, 0, -1):
        if backups[i - 1].exists():
            if backups[i].exists():
                backups[i].unlink()
            os.replace(backups[i - 1], backups[i])
    if path.exists():
        os.replace(path, backups[0])


def _atomic_write(path: Path, data: bytes | str, *, binary: bool = False) -> None:
    """Atomically write data to ``path`` with backup rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    with tempfile.NamedTemporaryFile(mode, encoding=encoding, dir=path.parent, delete=False) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    _rotate_backups(path)
    os.replace(tmp, path)

from .self_model_bootstrap import bootstrap
from .research_aggregator_bot import (
    ResearchAggregatorBot,
    ResearchItem,
    InfoDB,
)
from .model_automation_pipeline import (
    ModelAutomationPipeline,
    AutomationResult,
)
from .diagnostic_manager import DiagnosticManager
from .error_bot import ErrorBot, ErrorDB
from .data_bot import MetricsDB, DataBot
from .code_database import PatchHistoryDB
from .patch_score_backend import PatchScoreBackend, backend_from_url
from .capital_management_bot import CapitalManagementBot
from .learning_engine import LearningEngine
from .unified_event_bus import UnifiedEventBus
from .neuroplasticity import PathwayRecord, Outcome
from .self_coding_engine import SelfCodingEngine
from .action_planner import ActionPlanner
from .evolution_history_db import EvolutionHistoryDB
from . import synergy_weight_cli
from . import synergy_history_db as shd
from .self_improvement_policy import (
    SelfImprovementPolicy,
    ConfigurableSelfImprovementPolicy,
    DQNStrategy,
    DoubleDQNStrategy,
    ActorCriticStrategy,
    torch as sip_torch,
)
from .pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
from .env_config import PRE_ROI_SCALE, PRE_ROI_BIAS, PRE_ROI_CAP

POLICY_STATE_LEN = 21


class SynergyWeightLearner:
    """Learner adjusting synergy weights using a simple RL policy."""

    def __init__(self, path: Path | None = None, lr: float = 0.1) -> None:
        if sip_torch is None:
            logger.warning(
                "PyTorch not installed - using ActorCritic strategy"
            )
        self.path = Path(path) if path else None
        self.lr = lr
        self.weights = DEFAULT_SYNERGY_WEIGHTS.copy()
        self.strategy = ActorCriticStrategy()
        self._state: tuple[float, ...] = (0.0,) * 7
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("failed to load synergy weights %s: %s", self.path, exc)
            self.weights = DEFAULT_SYNERGY_WEIGHTS.copy()
            return

        valid = isinstance(data, dict) and all(
            k in data and isinstance(data[k], (int, float)) for k in self.weights
        )
        if not valid:
            logger.warning(
                "invalid synergy weight data in %s - using defaults", self.path
            )
            self.weights = DEFAULT_SYNERGY_WEIGHTS.copy()
        else:
            for k in self.weights:
                self.weights[k] = float(data[k])
        try:
            base = os.path.splitext(self.path)[0]
            pkl = base + ".policy.pkl"
            if os.path.exists(pkl):
                with open(pkl, "rb") as fh:
                    self.strategy = pickle.load(fh)
        except Exception as exc:
            logger.exception("failed to load strategy pickle: %s", exc)

        logger.info("loaded synergy weights", extra=log_record(weights=self.weights))

    # ------------------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        try:
            _atomic_write(self.path, json.dumps(self.weights))
        except Exception as exc:
            logger.exception("failed to save synergy weights: %s", exc)
        try:
            base = os.path.splitext(self.path)[0]
            pkl = Path(base + ".policy.pkl")
            _atomic_write(pkl, pickle.dumps(self.strategy), binary=True)
        except Exception as exc:
            logger.exception("failed to save strategy pickle: %s", exc)
        try:
            synergy_weight_updates_total.inc()
        except Exception:
            pass

        logger.info("saved synergy weights", extra=log_record(weights=self.weights))

    # ------------------------------------------------------------------
    def update(self, roi_delta: float, deltas: dict[str, float]) -> None:
        names = [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]
        self._state = tuple(float(deltas.get(n, 0.0)) for n in names)
        q_vals_list: list[float] = []
        for idx, _ in enumerate(names):
            reward = roi_delta * self._state[idx]
            q = self.strategy.update({}, self._state, idx, reward, self._state, 1.0, 0.9)
            q_vals_list.append(float(q))
        if hasattr(self.strategy, "predict"):
            q_vals = self.strategy.predict(self._state)
            q_vals_list = [float(v.item() if hasattr(v, "item") else v) for v in q_vals]
        mapping = {
            "roi": 0,
            "efficiency": 1,
            "resilience": 2,
            "antifragility": 3,
            "reliability": 4,
            "maintainability": 5,
            "throughput": 6,
        }
        for key, idx in mapping.items():
            val = q_vals_list[idx]
            self.weights[key] = max(0.0, min(val, 10.0))
        logger.info(
            "updated synergy weights",
            extra=log_record(
                weights=self.weights, state=self._state, roi_delta=roi_delta
            ),
        )
        self.save()


class DQNSynergyLearner(SynergyWeightLearner):
    """Learner using a simple DQN to adapt synergy weights."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float = 1e-3,
        *,
        strategy: str | None = None,
        target_sync: int = 10,
    ) -> None:
        super().__init__(path, lr)
        if sip_torch is None:
            logger.warning(
                "PyTorch not available - DQNSynergyLearner falling back to DQN strategy"
            )
        name = (strategy or "dqn").lower()
        self.target_sync = max(1, int(target_sync))
        if name in {"double", "double_dqn", "double-dqn", "ddqn", "td3"} and sip_torch is not None:
            self.strategy = DoubleDQNStrategy(action_dim=7, lr=lr, target_sync=target_sync)
            self.strategy_name = "td3" if name == "td3" else "double_dqn"
        elif name in {"policy", "policy_gradient", "actor_critic", "actor-critic", "sac"}:
            self.strategy = ActorCriticStrategy()
            self.strategy_name = "sac" if name == "sac" else "policy_gradient"
        else:
            # prefer Double DQN when PyTorch is available
            if sip_torch is not None:
                self.strategy = DoubleDQNStrategy(action_dim=7, lr=lr, target_sync=target_sync)
                self.strategy_name = "double_dqn"
            else:
                self.strategy = DQNStrategy(action_dim=7, lr=lr)
                self.strategy_name = "dqn"
        self._state: tuple[float, ...] = (0.0,) * 7
        self._steps = 0

    # ------------------------------------------------------------------
    def load(self) -> None:
        super().load()
        if not self.path:
            return
        base = os.path.splitext(self.path)[0]
        if self.path and sip_torch is not None:
            try:
                if hasattr(self.strategy, "_ensure_model"):
                    self.strategy._ensure_model(7)
                pt = base + ".pt"
                if os.path.exists(pt) and hasattr(self.strategy, "model"):
                    state_dict = sip_torch.load(pt)
                    assert self.strategy.model is not None
                    self.strategy.model.load_state_dict(state_dict)
                tgt = base + ".target.pt"
                if os.path.exists(tgt) and hasattr(self.strategy, "target_model"):
                    assert self.strategy.target_model is not None
                    self.strategy.target_model.load_state_dict(sip_torch.load(tgt))
            except Exception as exc:
                logger.exception("failed to load DQN models: %s", exc)
        try:
            pkl = base + ".policy.pkl"
            if os.path.exists(pkl):
                with open(pkl, "rb") as fh:
                    self.strategy = pickle.load(fh)
        except Exception as exc:
            logger.exception("failed to load strategy pickle: %s", exc)

    # ------------------------------------------------------------------
    def save(self) -> None:
        super().save()
        if not self.path:
            return
        base = os.path.splitext(self.path)[0]
        if self.path and sip_torch is not None:
            try:
                if hasattr(self.strategy, "model") and self.strategy.model is not None:
                    buf = io.BytesIO()
                    sip_torch.save(self.strategy.model.state_dict(), buf)
                    _atomic_write(Path(base + ".pt"), buf.getvalue(), binary=True)
                if hasattr(self.strategy, "target_model") and self.strategy.target_model is not None:
                    buf = io.BytesIO()
                    sip_torch.save(self.strategy.target_model.state_dict(), buf)
                    _atomic_write(Path(base + ".target.pt"), buf.getvalue(), binary=True)
            except Exception as exc:
                logger.exception("failed to save DQN models: %s", exc)
        try:
            pkl = Path(base + ".policy.pkl")
            _atomic_write(pkl, pickle.dumps(self.strategy), binary=True)
        except Exception as exc:
            logger.exception("failed to save strategy pickle: %s", exc)

    # ------------------------------------------------------------------
    def update(self, roi_delta: float, deltas: dict[str, float]) -> None:
        names = [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]
        self._state = tuple(float(deltas.get(n, 0.0)) for n in names)
        q_vals_list: list[float] = []
        for idx, name in enumerate(names):
            reward = roi_delta * self._state[idx]
            q = self.strategy.update({}, self._state, idx, reward, self._state, 1.0, 0.9)
            q_vals_list.append(float(q))
        if hasattr(self.strategy, "predict"):
            q_vals = self.strategy.predict(self._state)
            q_vals_list = [
                float(v.item() if hasattr(v, "item") else v) for v in q_vals
            ]
        mapping = {
            "roi": 0,
            "efficiency": 1,
            "resilience": 2,
            "antifragility": 3,
            "reliability": 4,
            "maintainability": 5,
            "throughput": 6,
        }
        for key, idx in mapping.items():
            val = q_vals_list[idx]
            self.weights[key] = max(0.0, min(val, 10.0))
        self._steps += 1
        if (
            self._steps % self.target_sync == 0
            and hasattr(self.strategy, "target_model")
            and hasattr(self.strategy, "model")
            and getattr(self.strategy, "target_model") is not None
            and getattr(self.strategy, "model") is not None
        ):
            try:
                self.strategy.target_model.load_state_dict(
                    self.strategy.model.state_dict()
                )
            except Exception as exc:
                logger.exception("target model sync failed: %s", exc)
        logger.info(
            "updated synergy weights",
            extra=log_record(
                weights=self.weights, state=self._state, roi_delta=roi_delta
            ),
        )
        self.save()


class DoubleDQNSynergyLearner(DQNSynergyLearner):
    """Synergy learner using a Double DQN strategy."""

    def __init__(
        self, path: Path | None = None, lr: float = 1e-3, *, target_sync: int = 10
    ) -> None:
        super().__init__(path, lr, strategy="double_dqn", target_sync=target_sync)


class SACSynergyLearner(DQNSynergyLearner):
    """Synergy learner using a simplified SAC strategy."""

    def __init__(
        self, path: Path | None = None, lr: float = 1e-3, *, target_sync: int = 10
    ) -> None:
        super().__init__(path, lr, strategy="sac", target_sync=target_sync)


class TD3SynergyLearner(DQNSynergyLearner):
    """Synergy learner using a simplified TD3 strategy."""

    def __init__(self, path: Path | None = None, lr: float = 1e-3, *, target_sync: int = 10) -> None:
        super().__init__(path, lr, strategy="td3", target_sync=target_sync)


class SelfImprovementEngine:
    """Run the automation pipeline on a configurable bot."""

    def __init__(
        self,
        *,
        interval: int = 3600,
        pipeline: ModelAutomationPipeline | None = None,
        bot_name: str = "menace",
        diagnostics: DiagnosticManager | None = None,
        info_db: InfoDB | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        energy_threshold: float = 0.5,
        learning_engine: LearningEngine | None = None,
        self_coding_engine: SelfCodingEngine | None = None,
        action_planner: "ActionPlanner" | None = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_history: EvolutionHistoryDB | None = None,
        data_bot: DataBot | None = None,
        patch_db: PatchHistoryDB | None = None,
        policy: SelfImprovementPolicy | None = None,
        policy_strategy: str | None = None,
        optimize_self: bool = False,
        meta_logger: object | None = None,
        module_index: "ModuleIndexDB" | None = None,
        module_clusters: dict[str, int] | None = None,
        module_groups: dict[str, str] | None = None,
        pre_roi_bot: PreExecutionROIBot | None = None,
        pre_roi_scale: float | None = None,
        pre_roi_bias: float | None = None,
        pre_roi_cap: float | None = None,
        synergy_weight_roi: float | None = None,
        synergy_weight_efficiency: float | None = None,
        synergy_weight_resilience: float | None = None,
        synergy_weight_antifragility: float | None = None,
        synergy_weight_reliability: float | None = None,
        synergy_weight_maintainability: float | None = None,
        synergy_weight_throughput: float | None = None,
        state_path: Path | str | None = None,
        roi_ema_alpha: float | None = None,
        synergy_weights_path: Path | str | None = None,
        synergy_weights_lr: float | None = None,
        synergy_learner_cls: Type[SynergyWeightLearner] = SynergyWeightLearner,
        score_backend: PatchScoreBackend | None = None,
    ) -> None:
        self.interval = interval
        self.bot_name = bot_name
        self.info_db = info_db or InfoDB()
        self.aggregator = ResearchAggregatorBot(
            [bot_name], info_db=self.info_db
        )
        self.pipeline = pipeline or ModelAutomationPipeline(
            aggregator=self.aggregator, action_planner=action_planner
        )
        err_bot = ErrorBot(ErrorDB(), MetricsDB())
        self.error_bot = err_bot
        self.diagnostics = diagnostics or DiagnosticManager(
            MetricsDB(), err_bot
        )
        self.last_run = 0.0
        self.capital_bot = capital_bot
        self.energy_threshold = energy_threshold
        self.learning_engine = learning_engine
        self.self_coding_engine = self_coding_engine
        self.event_bus = event_bus
        self.evolution_history = evolution_history
        self.data_bot = data_bot
        self.patch_db = patch_db or (data_bot.patch_db if data_bot else None)
        if policy is None:
            policy = ConfigurableSelfImprovementPolicy(
                strategy=policy_strategy
            )
        self.policy = policy
        if self.policy and getattr(self.policy, "path", None):
            try:
                self.policy.load(self.policy.path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("policy load failed: %s", exc)
        self.optimize_self_flag = optimize_self
        self.meta_logger = meta_logger
        self.pre_roi_bot = pre_roi_bot
        self.pre_roi_scale = (
            pre_roi_scale if pre_roi_scale is not None else PRE_ROI_SCALE
        )
        self.pre_roi_bias = (
            pre_roi_bias if pre_roi_bias is not None else PRE_ROI_BIAS
        )
        self.pre_roi_cap = (
            pre_roi_cap if pre_roi_cap is not None else PRE_ROI_CAP
        )
        settings = SandboxSettings()
        self.synergy_weight_roi = (
            synergy_weight_roi
            if synergy_weight_roi is not None
            else settings.synergy_weight_roi
        )
        self.synergy_weight_efficiency = (
            synergy_weight_efficiency
            if synergy_weight_efficiency is not None
            else settings.synergy_weight_efficiency
        )
        self.synergy_weight_resilience = (
            synergy_weight_resilience
            if synergy_weight_resilience is not None
            else settings.synergy_weight_resilience
        )
        self.synergy_weight_antifragility = (
            synergy_weight_antifragility
            if synergy_weight_antifragility is not None
            else settings.synergy_weight_antifragility
        )
        self.synergy_weight_reliability = (
            synergy_weight_reliability
            if synergy_weight_reliability is not None
            else getattr(settings, "synergy_weight_reliability", 1.0)
        )
        self.synergy_weight_maintainability = (
            synergy_weight_maintainability
            if synergy_weight_maintainability is not None
            else getattr(settings, "synergy_weight_maintainability", 1.0)
        )
        self.synergy_weight_throughput = (
            synergy_weight_throughput
            if synergy_weight_throughput is not None
            else getattr(settings, "synergy_weight_throughput", 1.0)
        )
        self.roi_ema_alpha = (
            roi_ema_alpha
            if roi_ema_alpha is not None
            else settings.roi_ema_alpha
        )
        default_path = Path(settings.sandbox_data_dir) / "synergy_weights.json"
        self.synergy_weights_path = (
            Path(synergy_weights_path)
            if synergy_weights_path is not None
            else default_path
        )
        self.synergy_weights_lr = (
            synergy_weights_lr
            if synergy_weights_lr is not None
            else settings.synergy_weights_lr
        )

        if synergy_learner_cls is SynergyWeightLearner:
            env_name = os.getenv("SYNERGY_LEARNER", "").lower()
            mapping = {
                "dqn": DQNSynergyLearner,
                "double": DoubleDQNSynergyLearner,
                "double_dqn": DoubleDQNSynergyLearner,
                "ddqn": DoubleDQNSynergyLearner,
                "sac": SACSynergyLearner,
                "td3": TD3SynergyLearner,
            }
            synergy_learner_cls = mapping.get(env_name, synergy_learner_cls)

        self.synergy_learner = synergy_learner_cls(
            self.synergy_weights_path, lr=self.synergy_weights_lr
        )
        if synergy_weight_roi is None:
            self.synergy_weight_roi = self.synergy_learner.weights["roi"]
        else:
            self.synergy_learner.weights["roi"] = self.synergy_weight_roi
        if synergy_weight_efficiency is None:
            self.synergy_weight_efficiency = self.synergy_learner.weights[
                "efficiency"
            ]
        else:
            self.synergy_learner.weights["efficiency"] = self.synergy_weight_efficiency
        if synergy_weight_resilience is None:
            self.synergy_weight_resilience = self.synergy_learner.weights[
                "resilience"
            ]
        else:
            self.synergy_learner.weights["resilience"] = self.synergy_weight_resilience
        if synergy_weight_antifragility is None:
            self.synergy_weight_antifragility = self.synergy_learner.weights[
                "antifragility"
            ]
        else:
            self.synergy_learner.weights[
                "antifragility"
            ] = self.synergy_weight_antifragility
        if synergy_weight_reliability is None:
            self.synergy_weight_reliability = self.synergy_learner.weights[
                "reliability"
            ]
        else:
            self.synergy_learner.weights[
                "reliability"
            ] = self.synergy_weight_reliability
        if synergy_weight_maintainability is None:
            self.synergy_weight_maintainability = self.synergy_learner.weights[
                "maintainability"
            ]
        else:
            self.synergy_learner.weights[
                "maintainability"
            ] = self.synergy_weight_maintainability
        if synergy_weight_throughput is None:
            self.synergy_weight_throughput = self.synergy_learner.weights[
                "throughput"
            ]
        else:
            self.synergy_learner.weights[
                "throughput"
            ] = self.synergy_weight_throughput
        self.state_path = Path(state_path) if state_path else None
        self.roi_history: list[float] = []
        self.roi_group_history: dict[int, list[float]] = {}
        self.roi_delta_ema: float = 0.0
        self._synergy_cache: dict | None = None
        self.logger = get_logger("SelfImprovementEngine")
        self._load_state()
        self._load_synergy_weights()
        from .module_index_db import ModuleIndexDB

        auto_map = os.getenv("SANDBOX_AUTO_MAP") == "1"
        self.module_index = module_index or ModuleIndexDB(auto_map=auto_map)
        if module_clusters is None and self.module_index is not None:
            try:
                module_clusters = dict(getattr(self.module_index, "_map", {}))
            except Exception:
                module_clusters = None
        self.module_clusters: dict[str, int] = module_clusters or {}
        if module_groups:
            grp_map: dict[str, int] = {}
            for mod, grp in module_groups.items():
                try:
                    idx = self.module_index.group_id(str(grp)) if self.module_index else abs(hash(grp)) % 1000
                except Exception:
                    idx = abs(hash(grp)) % 1000
                grp_map[mod] = idx
            if self.module_index:
                try:
                    self.module_index.merge_groups(grp_map)
                except Exception:
                    pass
            self.module_clusters.update(grp_map)
        logging.basicConfig(level=logging.INFO)
        self._score_backend: PatchScoreBackend | None = None
        if score_backend is not None:
            self._score_backend = score_backend
        else:
            backend_url = os.getenv("PATCH_SCORE_BACKEND_URL")
            if backend_url:
                try:
                    self._score_backend = backend_from_url(backend_url)
                except Exception:
                    self.logger.exception("patch score backend init failed")
        self._cycle_running = False
        self._schedule_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._trainer_stop: threading.Event | None = None
        self._trainer_thread: threading.Thread | None = None
        self._cycle_count = 0
        if self.event_bus:
            if self.learning_engine:
                try:
                    self.event_bus.subscribe(
                        "pathway:new", self._on_new_pathway
                    )
                except Exception as exc:
                    self.logger.exception(
                        "failed to subscribe to pathway events: %s", exc
                    )
            try:
                self.event_bus.subscribe(
                    "evolve:self_improve", lambda *_: self.run_cycle()
                )
            except Exception as exc:
                self.logger.exception(
                    "failed to subscribe to self_improve events: %s", exc
                )

        if os.getenv("AUTO_TRAIN_SYNERGY") == "1":
            try:
                interval = float(os.getenv("AUTO_TRAIN_INTERVAL", "600"))
            except Exception:
                interval = 600.0
            hist_file = Path(settings.sandbox_data_dir) / "synergy_history.db"
            self._start_synergy_trainer(hist_file, interval)

    # ------------------------------------------------------------------
    def recent_scores(self, limit: int = 20) -> list[tuple]:
        """Return recently stored patch scores."""
        if self._score_backend:
            try:
                rows = self._score_backend.fetch_recent(limit)
                if rows:
                    return rows
            except Exception:
                self.logger.exception("patch score backend fetch failed")
        return []

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.roi_history = [float(x) for x in data.get("roi_history", [])]
            self.roi_group_history = {
                int(k): [float(vv) for vv in v]
                for k, v in data.get("roi_group_history", {}).items()
            }
            self.last_run = float(data.get("last_run", self.last_run))
            self.roi_delta_ema = float(
                data.get("roi_delta_ema", self.roi_delta_ema)
            )
        except Exception as exc:
            self.logger.exception("failed to load state: %s", exc)

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=self.state_path.parent, encoding="utf-8"
            ) as fh:
                json.dump(
                    {
                        "roi_history": self.roi_history,
                        "roi_group_history": self.roi_group_history,
                        "last_run": self.last_run,
                        "roi_delta_ema": self.roi_delta_ema,
                    },
                    fh,
                )
                tmp = Path(fh.name)
            os.replace(tmp, self.state_path)
        except Exception as exc:
            self.logger.exception("failed to save state: %s", exc)

    # ------------------------------------------------------------------
    def _load_synergy_weights(self) -> None:
        """Load persisted synergy weights from JSON file."""
        self.synergy_learner.load()
        self.synergy_weight_roi = self.synergy_learner.weights["roi"]
        self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
        self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
        self.synergy_weight_antifragility = self.synergy_learner.weights[
            "antifragility"
        ]
        self.synergy_weight_reliability = self.synergy_learner.weights["reliability"]
        self.synergy_weight_maintainability = self.synergy_learner.weights["maintainability"]
        self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
        self.logger.info(
            "synergy weights loaded",
            extra=log_record(weights=self.synergy_learner.weights),
        )

    # ------------------------------------------------------------------
    def _save_synergy_weights(self) -> None:
        """Persist synergy weights to JSON file."""
        self.synergy_learner.weights["roi"] = self.synergy_weight_roi
        self.synergy_learner.weights["efficiency"] = self.synergy_weight_efficiency
        self.synergy_learner.weights["resilience"] = self.synergy_weight_resilience
        self.synergy_learner.weights["antifragility"] = self.synergy_weight_antifragility
        self.synergy_learner.weights["reliability"] = self.synergy_weight_reliability
        self.synergy_learner.weights["maintainability"] = self.synergy_weight_maintainability
        self.synergy_learner.weights["throughput"] = self.synergy_weight_throughput
        self.synergy_learner.save()
        self.logger.info(
            "synergy weights saved",
            extra=log_record(weights=self.synergy_learner.weights),
        )

    # ------------------------------------------------------------------
    def _train_synergy_weights_once(self, history_file: Path) -> None:
        if not history_file.exists():
            return
        try:
            with sqlite3.connect(history_file) as conn:
                hist = shd.fetch_all(conn)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to load history: %s", exc)
            return
        if not hist:
            return
        try:
            synergy_weight_cli.train_from_history(hist, self.synergy_weights_path)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("training failed: %s", exc)

    def _synergy_trainer_loop(self, history_file: Path, interval: float) -> None:
        assert self._trainer_stop is not None
        while not self._trainer_stop.is_set():
            self._train_synergy_weights_once(history_file)
            self._trainer_stop.wait(interval)

    def _start_synergy_trainer(self, history_file: Path, interval: float) -> None:
        if self._trainer_thread:
            return
        self.logger.info(
            "starting synergy trainer thread",
            extra=log_record(
                history_file=str(history_file), interval=float(interval)
            ),
        )
        self._trainer_stop = threading.Event()
        self._trainer_thread = threading.Thread(
            target=self._synergy_trainer_loop,
            args=(history_file, interval),
            daemon=True,
        )
        self._trainer_thread.start()

    def stop_synergy_trainer(self) -> None:
        if self._trainer_thread and self._trainer_stop:
            self.logger.info("stopping synergy trainer thread")
            self._trainer_stop.set()
            self._trainer_thread.join(timeout=1.0)
            self._trainer_thread = None
            self._trainer_stop = None

    # ------------------------------------------------------------------
    def _metric_delta(self, name: str, window: int = 3) -> float:
        """Return rolling average delta for *name* metric."""
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return 0.0
        try:
            vals = tracker.metrics_history.get(name, [])
        except Exception:
            return 0.0
        if not vals:
            return 0.0
        w = min(window, len(vals))
        current_avg = sum(vals[-w:]) / w
        if len(vals) > w:
            prev_w = min(w, len(vals) - w)
            prev_avg = sum(vals[-w - prev_w : -w]) / prev_w
        elif len(vals) >= 2:
            prev_avg = vals[-2]
        else:
            return float(vals[-1])
        return float(current_avg - prev_avg)

    # ------------------------------------------------------------------
    def _weighted_synergy_adjustment(self, window: int = 3) -> float:
        """Compute weighted synergy adjustment factor.

        The weights for each synergy metric are derived from recent patch
        history when available.  Moving averages and variance normalise the
        metric deltas so that the adjustment adapts to historical trends.
        """

        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)

        learner_weights = getattr(self, "synergy_learner", None)
        if learner_weights is not None:
            lw = learner_weights.weights
            base_weights = {
                "synergy_roi": lw.get("roi", self.synergy_weight_roi),
                "synergy_efficiency": lw.get("efficiency", self.synergy_weight_efficiency),
                "synergy_resilience": lw.get("resilience", self.synergy_weight_resilience),
                "synergy_antifragility": lw.get("antifragility", self.synergy_weight_antifragility),
                "synergy_reliability": lw.get("reliability", self.synergy_weight_reliability),
                "synergy_maintainability": lw.get("maintainability", self.synergy_weight_maintainability),
                "synergy_throughput": lw.get("throughput", self.synergy_weight_throughput),
            }
        else:
            base_weights = {
                "synergy_roi": self.synergy_weight_roi,
                "synergy_efficiency": self.synergy_weight_efficiency,
                "synergy_resilience": self.synergy_weight_resilience,
                "synergy_antifragility": self.synergy_weight_antifragility,
                "synergy_reliability": self.synergy_weight_reliability,
                "synergy_maintainability": self.synergy_weight_maintainability,
                "synergy_throughput": self.synergy_weight_throughput,
            }

        weights: dict[str, float]
        stats: dict[str, tuple[float, float]]
        weights = dict(base_weights)
        stats = {}

        cache = getattr(self, "_synergy_cache", None)

        if pdb:
            try:
                records = pdb.filter()
                records.sort(key=lambda r: getattr(r, "ts", ""))
                patch_count = len(records)
                if not cache or cache.get("count") != patch_count:
                    recent = records[-20:]
                    roi_vals: list[float] = []
                    data: list[list[float]] = []
                    for rec in recent:
                        roi_vals.append(float(getattr(rec, "roi_delta", 0.0)))
                        data.append(
                            [float(getattr(rec, name, 0.0)) for name in base_weights]
                        )

                    if len(data) >= 2 and any(any(row) for row in data):
                        import numpy as np

                        X = np.array(data, dtype=float)
                        y = np.array(roi_vals, dtype=float)
                        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
                        coef_abs = np.abs(coefs)
                        total = float(coef_abs.sum())
                        if total > 0:
                            for i, name in enumerate(base_weights):
                                w = coef_abs[i] / total
                                weights[name] = w * base_weights[name]
                        else:
                            weights = {k: base_weights[k] / len(base_weights) for k in base_weights}
                        for i, name in enumerate(base_weights):
                            col = X[:, i]
                            stats[name] = (float(col.mean()), float(col.std() or 1.0))
                    cache = {"count": patch_count, "weights": weights, "stats": stats}
                    self._synergy_cache = cache
                else:
                    weights = cache.get("weights", weights)
                    stats = cache.get("stats", stats)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "synergy weight history processing failed: %s", exc
                )
                raise shd.HistoryParseError(str(exc)) from exc
        if cache is None:
            self._synergy_cache = {"count": 0, "weights": weights, "stats": stats}

        def norm_delta(name: str) -> float:
            val = self._metric_delta(name, window)
            mean, std = stats.get(name, (0.0, 1.0))
            return (val - mean) / (std + 1e-6)

        try:
            syn_adj = sum(
                norm_delta(name) * weights.get(name, 0.0) for name in weights
            )
        except Exception:
            syn_adj = 0.0
        self.logger.info(
            "weighted synergy adjustment",
            extra=log_record(factor=float(syn_adj), weights=weights),
        )
        self.logger.debug(
            "synergy weights used for adjustment",
            extra=log_record(weights=weights),
        )
        return float(syn_adj)

    # ------------------------------------------------------------------
    def _update_synergy_weights(self, roi_delta: float) -> None:
        """Adjust synergy weights using reinforcement learning."""
        names = [
            "synergy_roi",
            "synergy_efficiency",
            "synergy_resilience",
            "synergy_antifragility",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ]
        deltas = {n: self._metric_delta(n) for n in names}
        try:
            self.synergy_learner.update(roi_delta, deltas)
            self.logger.info(
                "synergy weights updated",
                extra=log_record(
                    weights=self.synergy_learner.weights,
                    roi_delta=roi_delta,
                    state=self.synergy_learner._state,
                ),
            )
        except Exception as exc:  # pragma: no cover - runtime issues
            try:
                synergy_weight_update_failures_total.inc()
            except Exception:
                pass
            try:
                dispatch_alert(
                    "synergy_weight_update_failure",
                    2,
                    "Weight update failed",
                    {"roi_delta": roi_delta},
                )
                synergy_weight_update_alerts_total.inc()
            except Exception:
                pass
            self.logger.exception("synergy weight update failed: %s", exc)
            return
        self.synergy_weight_roi = self.synergy_learner.weights["roi"]
        self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
        self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
        self.synergy_weight_antifragility = self.synergy_learner.weights[
            "antifragility"
        ]
        self.synergy_weight_reliability = self.synergy_learner.weights["reliability"]
        self.synergy_weight_maintainability = self.synergy_learner.weights["maintainability"]
        self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
        self.logger.info(
            "synergy weights after update",
            extra=log_record(
                weights=self.synergy_learner.weights, roi_delta=roi_delta
            ),
        )

    # ------------------------------------------------------------------
    def _policy_state(self) -> tuple[int, ...]:
        """Return the state tuple used by :class:`SelfImprovementPolicy`."""
        energy = 0.0
        if self.capital_bot:
            try:
                energy = self.capital_bot.energy_score(
                    load=0.0,
                    success_rate=1.0,
                    deploy_eff=1.0,
                    failure_rate=0.0,
                )
            except Exception as exc:
                self.logger.exception("energy_score failed: %s", exc)
                energy = 0.0
        profit = 0.0
        if self.capital_bot:
            try:
                profit = self.capital_bot.profit()
            except Exception as exc:
                self.logger.exception("profit check failed: %s", exc)
                profit = 0.0
        trend = anomaly = patch_rate = 0.0
        if self.data_bot:
            try:
                trend = self.data_bot.long_term_roi_trend(limit=200)
            except Exception as exc:
                self.logger.exception("ROI trend fetch failed: %s", exc)
                trend = 0.0
            try:
                df = self.data_bot.db.fetch(100)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        df["roi"] = df["revenue"] - df["expense"]
                        anomaly = float(
                            len(DataBot.detect_anomalies(df, "roi"))
                        ) / len(df)
                elif isinstance(df, list) and df:
                    rois = [
                        float(r.get("revenue", 0.0) - r.get("expense", 0.0))
                        for r in df
                    ]
                    df_list = [{"roi": r} for r in rois]
                    anomaly = float(
                        len(DataBot.detect_anomalies(df_list, "roi"))
                    ) / len(rois)
            except Exception as exc:
                self.logger.exception("anomaly detection failed: %s", exc)
                anomaly = 0.0
            if getattr(self.data_bot, "patch_db", None):
                try:
                    patch_rate = self.data_bot.patch_db.success_rate()
                except Exception as exc:
                    self.logger.exception(
                        "patch success rate lookup failed: %s", exc
                    )
                    patch_rate = 0.0
        avg_roi = avg_complex = revert_rate = 0.0
        module_idx = 0
        module_trend = 0.0
        dim_flag = 0
        tracker = getattr(self, "tracker", None)
        syn_roi = syn_eff = syn_res = syn_af = 0.0
        if tracker is not None:
            try:
                syn_roi = self._metric_delta("synergy_roi")
                syn_eff = self._metric_delta("synergy_efficiency")
                syn_res = self._metric_delta("synergy_resilience")
                syn_af = self._metric_delta("synergy_antifragility")
            except Exception:
                syn_roi = syn_eff = syn_res = syn_af = 0.0
        syn_roi *= self.synergy_weight_roi
        syn_eff *= self.synergy_weight_efficiency
        syn_res *= self.synergy_weight_resilience
        syn_af *= self.synergy_weight_antifragility
        profit += syn_roi
        energy = max(0.0, energy + syn_eff)
        pdb = self.patch_db or (
            self.data_bot.patch_db if self.data_bot else None
        )
        if pdb:
            try:
                with pdb._connect() as conn:
                    rows = conn.execute(
                        "SELECT roi_delta, complexity_delta, reverted, filename "
                        "FROM patch_history ORDER BY id DESC LIMIT ?",
                        (10,),
                    ).fetchall()
                if rows:
                    avg_roi = float(sum(r[0] for r in rows) / len(rows))
                    avg_complex = float(sum(r[1] for r in rows) / len(rows))
                    revert_rate = float(
                        sum(1 for r in rows if r[2]) / len(rows)
                    )
                    mod_name = Path(rows[0][3]).name
                    module_idx = self.module_index.get(mod_name)
                    mods = [m for m, idx in self.module_clusters.items() if idx == module_idx]
                    try:
                        if mods:
                            placeholders = ",".join("?" * len(mods))
                            total = conn.execute(
                                f"SELECT SUM(roi_delta) FROM patch_history WHERE filename IN ({placeholders})",
                                mods,
                            ).fetchone()
                            module_trend = float(total[0] or 0.0)
                        else:
                            module_trend = 0.0
                    except Exception:
                        module_trend = 0.0
                    if self.meta_logger:
                        try:
                            dim_flag = (
                                1
                                if mod_name in self.meta_logger.diminishing()
                                else 0
                            )
                            if module_trend == 0.0:
                                module_trend = dict(
                                    self.meta_logger.rankings()
                                ).get(mod_name, 0.0)
                        except (
                            Exception
                        ) as exc:  # pragma: no cover - best effort
                            self.logger.exception(
                                "meta logger stats failed: %s", exc
                            )
            except Exception as exc:
                self.logger.exception("patch metrics failed: %s", exc)
                avg_roi = avg_complex = revert_rate = 0.0
                module_idx = 0
            try:
                kw_count, kw_recent = pdb.keyword_features()
            except Exception as exc:
                self.logger.exception("keyword feature fetch failed: %s", exc)
                kw_count = kw_recent = 0
        else:
            kw_count = kw_recent = 0
        avg_roi_delta = avg_eff = 0.0
        if self.evolution_history:
            try:
                stats = self.evolution_history.averages(limit=5)
                avg_roi_delta = float(stats.get("avg_roi_delta", 0.0))
                avg_eff = float(stats.get("avg_efficiency", 0.0))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "evolution history stats failed: %s", exc
                )
                avg_roi_delta = avg_eff = 0.0
        short_avg = 0.0
        if self.roi_history:
            n = min(len(self.roi_history), 5)
            short_avg = float(sum(self.roi_history[-n:]) / n)
        return (
            int(round(profit)),
            int(round(energy * 10)),
            int(round(trend * 10)),
            int(round(anomaly * 10)),
            int(round(patch_rate * 10)),
            int(round(avg_roi * 10)),
            int(round(avg_complex * 10)),
            int(round(revert_rate * 10)),
            int(module_idx),
            int(round(module_trend * 10)),
            int(dim_flag),
            int(kw_count),
            int(kw_recent),
            int(round(avg_roi_delta * 10)),
            int(round(avg_eff)),
            int(round(short_avg * 10)),
            int(round(self.roi_delta_ema * 10)),
            int(round(syn_roi * 10)),
            int(round(syn_eff * 10)),
            int(round(syn_res * 10)),
            int(round(syn_af * 10)),
        )

    # ------------------------------------------------------------------
    def _should_trigger(self) -> bool:
        if time.time() - self.last_run >= self.interval:
            return True
        if self.policy:
            try:
                if self.policy.score(self._policy_state()) > 0:
                    return True
            except Exception as exc:
                self.logger.exception("policy scoring failed: %s", exc)
        if self.pre_roi_bot:
            try:
                forecast = self.pre_roi_bot.predict_model_roi(
                    self.bot_name, []
                )
                if forecast.roi > self.pre_roi_bias:
                    return True
            except Exception as exc:
                self.logger.exception("pre ROI forecast failed: %s", exc)
        issues = self.diagnostics.diagnose()
        return bool(issues)

    def _record_state(self) -> None:
        """Store metrics and discrepancies as research items."""
        mdb = self.diagnostics.metrics
        edb = self.diagnostics.error_bot.db
        df = mdb.fetch(20)
        for row in df.itertuples(index=False):
            item = ResearchItem(
                topic="metrics",
                content=str(row._asdict()),
                timestamp=time.time(),
            )
            try:
                self.info_db.add(item)
            except Exception as exc:
                self.logger.exception("failed to record metric item: %s", exc)
        disc = edb.discrepancies()
        if "message" in disc:
            for msg in disc["message"]:
                item = ResearchItem(
                    topic="error",
                    content=str(msg),
                    timestamp=time.time(),
                    tags=["error"],
                )
                try:
                    self.info_db.add(item)
                except Exception as exc:
                    self.logger.exception(
                        "failed to record error item: %s", exc
                    )

    def _evaluate_learning(self) -> None:
        """Benchmark the learning engine via cross-validation."""
        if not self.learning_engine:
            return
        try:
            if hasattr(self.learning_engine, "evaluate"):
                result = self.learning_engine.evaluate()
                mean_score = float(result.get("cv_score", 0.0))
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(result)
                    except Exception as exc:
                        self.logger.exception(
                            "persist_evaluation failed: %s", exc
                        )
            else:
                X, y = self.learning_engine._dataset()  # type: ignore[attr-defined]
                if not X or len(set(y)) < 2:
                    return
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(
                    self.learning_engine.model, X, y, cv=3
                )
                mean_score = float(scores.mean())
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(
                            {
                                "cv_score": mean_score,
                                "holdout_score": mean_score,
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as exc:
                        self.logger.exception(
                            "persist_evaluation failed: %s", exc
                        )
        except Exception as exc:
            self.logger.exception("learning evaluation failed: %s", exc)
            mean_score = 0.0
        item = ResearchItem(
            topic="learning_eval",
            content=str({"cv_score": mean_score}),
            timestamp=time.time(),
        )
        try:
            self.info_db.add(item)
        except Exception as exc:
            self.logger.exception("failed to record learning eval: %s", exc)

    def _optimize_self(self) -> tuple[int | None, bool, float]:
        """Apply a patch to this engine via :class:`SelfCodingEngine`."""
        if not self.self_coding_engine:
            return None, False, 0.0
        try:
            patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                Path(__file__), "self_improvement"
            )
            return patch_id, reverted, delta
        except Exception as exc:
            self.logger.exception("self optimization failed: %s", exc)
            return None, False, 0.0

    def _on_new_pathway(self, topic: str, payload: object) -> None:
        """Incrementally train when a new pathway is logged."""
        if not self.learning_engine:
            return
        if isinstance(payload, dict):
            try:
                rec = PathwayRecord(
                    actions=payload.get("actions", ""),
                    inputs=payload.get("inputs", ""),
                    outputs=payload.get("outputs", ""),
                    exec_time=float(payload.get("exec_time", 0.0)),
                    resources=payload.get("resources", ""),
                    outcome=Outcome(payload.get("outcome", "FAILURE")),
                    roi=float(payload.get("roi", 0.0)),
                    ts=payload.get("ts", ""),
                )
                self.learning_engine.partial_train(rec)
            except Exception as exc:
                self.logger.exception(
                    "failed to process pathway record: %s", exc
                )

    # ------------------------------------------------------------------
    def run_cycle(self, energy: int = 1) -> AutomationResult:
        """Execute a self-improvement cycle."""
        self._cycle_running = True
        self._cycle_count += 1
        cid = f"cycle-{self._cycle_count}"
        set_correlation_id(cid)
        try:
            state = (
                self._policy_state()
                if self.policy
                else (0,) * POLICY_STATE_LEN
            )
            predicted = self.policy.score(state) if self.policy else 0.0
            roi_pred: float | None = None
            self.logger.info(
                "cycle start",
                extra=log_record(
                    energy=energy, predicted_roi=predicted, state=state
                ),
            )
            if self.policy:
                self.logger.info(
                    "policy predicted roi",
                    extra=log_record(
                        predicted_roi=predicted,
                        state=state,
                        weights=self.synergy_learner.weights,
                    ),
                )
            before_roi = 0.0
            if self.capital_bot:
                try:
                    before_roi = self.capital_bot.profit()
                    self.logger.info(
                        "initial ROI", extra=log_record(roi=before_roi)
                    )
                except Exception as exc:
                    self.logger.exception("profit lookup failed: %s", exc)
                    before_roi = 0.0
            if self.capital_bot:
                try:
                    energy = int(
                        round(
                            self.capital_bot.energy_score(
                                load=0.0,
                                success_rate=1.0,
                                deploy_eff=1.0,
                                failure_rate=0.0,
                                reward=None,
                            )
                            * 5
                        )
                    )
                    self.logger.info(
                        "available energy", extra=log_record(value=energy)
                    )
                except Exception as exc:
                    self.logger.exception("energy calculation failed: %s", exc)
                    energy = 1
            if self.policy:
                try:
                    energy = max(
                        1, int(round(energy * (1 + max(0.0, predicted))))
                    )
                    self.logger.info(
                        "policy adjusted energy",
                        extra=log_record(
                            value=energy,
                            predicted_roi=predicted,
                            state=state,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                except Exception as exc:
                    self.logger.exception(
                        "policy energy adjustment failed: %s", exc
                    )
            if self.pre_roi_bot:
                try:
                    forecast = self.pre_roi_bot.predict_model_roi(
                        self.bot_name, []
                    )
                    roi_pred = float(getattr(forecast, "roi", 0.0))
                    scale = (
                        1
                        + max(0.0, roi_pred + self.pre_roi_bias)
                        * self.pre_roi_scale
                    )
                    if self.pre_roi_cap:
                        scale = min(scale, self.pre_roi_cap)
                    energy = max(1, int(round(energy * scale)))
                    self.logger.info(
                        "pre_roi adjusted energy",
                        extra=log_record(
                            value=energy, roi_prediction=roi_pred
                        ),
                    )
                except Exception as exc:
                    self.logger.exception(
                        "pre ROI energy adjustment failed: %s", exc
                    )
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                try:
                    syn_adj = self._weighted_synergy_adjustment()
                    self.logger.info(
                        "synergy adjustment",
                        extra=log_record(
                            factor=syn_adj,
                            energy_before=energy,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                    if syn_adj:
                        energy = int(round(energy * (1.0 + syn_adj)))
                        self.logger.info(
                            "synergy adjusted energy",
                            extra=log_record(
                                value=energy, weights=self.synergy_learner.weights
                            ),
                        )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "synergy energy adjustment failed: %s", exc
                    )
            try:
                roi_scale = 1.0 + max(0.0, self.roi_delta_ema)
                self.logger.info(
                    "roi ema adjustment",
                    extra=log_record(factor=roi_scale, energy_before=energy),
                )
                energy = int(round(energy * roi_scale))
                self.logger.info(
                    "roi ema adjusted energy", extra=log_record(value=energy)
                )
            except Exception as exc:
                self.logger.exception("roi energy adjustment failed: %s", exc)
            energy = max(1, min(int(energy), 100))
            model_id = bootstrap()
            self.logger.info(
                "model bootstrapped", extra=log_record(model_id=model_id)
            )
            self.info_db.set_current_model(model_id)
            self._record_state()
            if self.learning_engine:
                try:
                    self.logger.info("training learning engine")
                    self.learning_engine.train()
                    self._evaluate_learning()
                except Exception as exc:
                    self.logger.exception(
                        "learning engine run failed: %s", exc
                    )
            self.logger.info(
                "pipeline pre-run metrics",
                extra=log_record(
                    predicted_roi=roi_pred if roi_pred is not None else predicted,
                    policy_score=predicted,
                    energy=energy,
                    synergy_weights=self.synergy_learner.weights,
                ),
            )
            self.logger.info(
                "running automation pipeline", extra=log_record(energy=energy)
            )
            result = self.pipeline.run(self.bot_name, energy=energy)
            self.logger.info(
                "pipeline complete",
                extra=log_record(roi=getattr(result.roi, "roi", 0.0)),
            )
            trending_topic = getattr(result, "trending_topic", None)
            patch_id = None
            reverted = False
            if self.self_coding_engine and result.package:
                try:
                    self.logger.info(
                        "applying helper patch",
                        extra=log_record(trending_topic=trending_topic),
                    )
                    patch_id, reverted, delta = (
                        self.self_coding_engine.apply_patch(
                            Path("auto_helpers.py"),
                            "helper",
                            trending_topic=trending_topic,
                        )
                    )
                    if self.policy:
                        try:
                            self.logger.info(
                                "patch applied",
                                extra=log_record(
                                    patch_id=patch_id,
                                    reverted=reverted,
                                    delta=delta,
                                ),
                            )
                            st = self._policy_state()
                            syn_reward = st[-2] / 10.0 + st[-1] / 10.0
                            self.policy.update(
                                st,
                                delta + syn_reward,
                                synergy_roi_delta=st[-4] / 10.0,
                                synergy_efficiency_delta=st[-3] / 10.0,
                            )
                            if getattr(self.policy, "path", None):
                                try:
                                    self.policy.save()
                                except (
                                    Exception
                                ) as exc:  # pragma: no cover - best effort
                                    self.logger.exception(
                                        "policy save failed: %s", exc
                                    )
                        except Exception as exc:
                            self.logger.exception(
                                "policy patch update failed: %s", exc
                            )
                    if self.optimize_self_flag:
                        self._optimize_self()
                except Exception as exc:
                    self.logger.exception("helper patch failed: %s", exc)
                    patch_id = None
                    reverted = False
            if self.error_bot:
                try:
                    self.error_bot.auto_patch_recurrent_errors()
                    self.logger.info("error auto-patching complete")
                except Exception as exc:
                    self.logger.exception(
                        "auto patch recurrent errors failed: %s", exc
                    )
            after_roi = before_roi
            if self.capital_bot:
                try:
                    after_roi = self.capital_bot.profit()
                    self.logger.info(
                        "post-cycle ROI",
                        extra=log_record(before=before_roi, after=after_roi),
                    )
                except Exception as exc:
                    self.logger.exception(
                        "post-cycle profit lookup failed: %s", exc
                    )
                    after_roi = before_roi
            roi_value = result.roi.roi if result.roi else 0.0
            if self.evolution_history:
                try:
                    from .evolution_history_db import EvolutionEvent

                    self.evolution_history.add(
                        EvolutionEvent(
                            action="self_improvement",
                            before_metric=before_roi,
                            after_metric=after_roi,
                            roi=roi_value,
                            predicted_roi=predicted,
                            trending_topic=trending_topic,
                        )
                    )
                except Exception as exc:
                    self.logger.exception(
                        "evolution history logging failed: %s", exc
                    )
            eff = bottleneck = patch_rate = trend = anomaly = 0.0
            if self.data_bot:
                try:
                    df = self.data_bot.db.fetch(20)
                    if hasattr(df, "empty"):
                        if not getattr(df, "empty", True):
                            eff = float(max(0.0, 100.0 - df["cpu"].mean()))
                            if "errors" in df.columns:
                                bottleneck = float(df["errors"].mean())
                    elif isinstance(df, list) and df:
                        avg_cpu = sum(r.get("cpu", 0.0) for r in df) / len(df)
                        eff = float(max(0.0, 100.0 - avg_cpu))
                        bottleneck = float(
                            sum(r.get("errors", 0.0) for r in df) / len(df)
                        )
                except Exception as exc:
                    self.logger.exception("data fetch failed: %s", exc)
                    eff = bottleneck = 0.0
                if self.self_coding_engine and getattr(
                    self.self_coding_engine, "patch_db", None
                ):
                    try:
                        patch_rate = (
                            self.self_coding_engine.patch_db.success_rate()
                        )
                    except Exception as exc:
                        self.logger.exception(
                            "self_coding patch rate lookup failed: %s", exc
                        )
                        patch_rate = 0.0
                if getattr(self.data_bot, "patch_db", None) and not patch_rate:
                    try:
                        patch_rate = self.data_bot.patch_db.success_rate()
                    except Exception as exc:
                        self.logger.exception(
                            "data_bot patch rate lookup failed: %s", exc
                        )
                        patch_rate = 0.0
                try:
                    trend = self.data_bot.long_term_roi_trend(limit=200)
                except Exception as exc:
                    self.logger.exception("trend retrieval failed: %s", exc)
                    trend = 0.0
                try:
                    df_anom = self.data_bot.db.fetch(100)
                    if hasattr(df_anom, "empty"):
                        if not getattr(df_anom, "empty", True):
                            df_anom["roi"] = (
                                df_anom["revenue"] - df_anom["expense"]
                            )
                            anomaly = float(
                                len(DataBot.detect_anomalies(df_anom, "roi"))
                            ) / len(df_anom)
                    elif isinstance(df_anom, list) and df_anom:
                        rois = [
                            float(
                                r.get("revenue", 0.0) - r.get("expense", 0.0)
                            )
                            for r in df_anom
                        ]
                        df_list = [{"roi": r} for r in rois]
                        anomaly = float(
                            len(DataBot.detect_anomalies(df_list, "roi"))
                        ) / len(rois)
                except Exception as exc:
                    self.logger.exception(
                        "anomaly calculation failed: %s", exc
                    )
                    anomaly = 0.0
                try:
                    self.data_bot.log_evolution_cycle(
                        "self_improvement",
                        before_roi,
                        after_roi,
                        roi_value,
                        0.0,
                        patch_success=patch_rate,
                        roi_delta=after_roi - before_roi,
                        roi_trend=trend,
                        anomaly_count=anomaly,
                        efficiency=eff,
                        bottleneck=bottleneck,
                        patch_id=patch_id,
                        trending_topic=trending_topic,
                        reverted=reverted,
                    )
                    self.logger.info(
                        "cycle metrics",
                        extra=log_record(
                            patch_success=patch_rate,
                            roi_delta=after_roi - before_roi,
                            roi_trend=trend,
                            anomaly=anomaly,
                        ),
                    )
                    if self.capital_bot:
                        try:
                            self.capital_bot.log_evolution_event(
                                "self_improvement",
                                before_roi,
                                after_roi,
                            )
                        except Exception as exc:
                            self.logger.exception(
                                "capital bot evolution log failed: %s", exc
                            )
                except Exception as exc:
                    self.logger.exception(
                        "data_bot evolution logging failed: %s", exc
                    )
            self.last_run = time.time()
            delta = after_roi - before_roi
            self.roi_delta_ema = (
                1 - self.roi_ema_alpha
            ) * self.roi_delta_ema + self.roi_ema_alpha * delta
            group_idx = None
            if self.patch_db:
                try:
                    with self.patch_db._connect() as conn:
                        row = conn.execute(
                            "SELECT filename FROM patch_history ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        mod_name = Path(row[0]).name
                        group_idx = self.module_clusters.get(mod_name)
                        if group_idx is None and self.module_index:
                            group_idx = self.module_index.get(mod_name)
                            self.module_clusters[mod_name] = group_idx
                except Exception as exc:
                    self.logger.exception("group index lookup failed: %s", exc)
            if group_idx is not None:
                self.roi_group_history.setdefault(int(group_idx), []).append(delta)
            self.roi_history.append(delta)
            self._save_state()
            self._update_synergy_weights(delta)
            self.logger.info(
                "cycle summary",
                extra=log_record(
                    roi_delta=delta,
                    patch_success=patch_rate,
                    roi_trend=trend,
                    anomaly=anomaly,
                    synergy_weights=self.synergy_learner.weights,
                ),
            )
            if self._score_backend:
                try:
                    self._score_backend.store(
                        {
                            "description": self.bot_name,
                            "result": "success" if delta >= 0 else "decline",
                            "roi_delta": float(delta),
                            "score": float(delta),
                        }
                    )
                except Exception:
                    self.logger.exception("patch score backend store failed")
            if self.policy:
                try:
                    next_state = self._policy_state()
                    syn_reward = next_state[-2] / 10.0 + next_state[-1] / 10.0
                    self.policy.update(
                        state,
                        after_roi - before_roi + syn_reward,
                        next_state,
                        synergy_roi_delta=next_state[-4] / 10.0,
                        synergy_efficiency_delta=next_state[-3] / 10.0,
                    )
                    if getattr(self.policy, "path", None):
                        try:
                            self.policy.save()
                        except (
                            Exception
                        ) as exc:  # pragma: no cover - best effort
                            self.logger.exception(
                                "policy save failed: %s", exc
                            )
                    self.logger.info(
                        "policy updated",
                        extra=log_record(
                            reward=after_roi - before_roi,
                            state=state,
                            next_state=next_state,
                            weights=self.synergy_learner.weights,
                        ),
                    )
                except Exception as exc:
                    self.logger.exception("policy update failed: %s", exc)
            if self.policy and getattr(self.policy, "path", None):
                try:
                    self.policy.save()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("policy save failed: %s", exc)
            self.logger.info("cycle complete", extra=log_record(roi=roi_value))
            return result
        finally:
            self._cycle_running = False
            set_correlation_id(None)

    async def _schedule_loop(self, energy: int = 1) -> None:
        while not self._stop_event.is_set():
            current_energy = energy
            if self.capital_bot:
                try:
                    current_energy = self.capital_bot.energy_score(
                        load=0.0,
                        success_rate=1.0,
                        deploy_eff=1.0,
                        failure_rate=0.0,
                    )
                except Exception as exc:
                    self.logger.exception("energy check failed: %s", exc)
                    current_energy = energy
            if (
                current_energy >= self.energy_threshold
                and not self._cycle_running
            ):
                try:
                    await asyncio.to_thread(
                        self.run_cycle, energy=int(round(current_energy * 5))
                    )
                except Exception as exc:
                    self.logger.exception(
                        "self improvement run_cycle failed with energy %s: %s",
                        int(round(current_energy * 5)),
                        exc,
                    )
            else:
                if current_energy < self.energy_threshold:
                    self.logger.info(
                        "energy below threshold - skipping cycle",
                        extra=log_record(
                            energy=current_energy,
                            threshold=self.energy_threshold,
                        ),
                    )
            await asyncio.sleep(self.interval)

    def schedule(
        self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.Task:
        """Start the scheduling loop in the background."""
        if self._schedule_task and not self._schedule_task.done():
            return self._schedule_task
        self.logger.info(
            "scheduling started",
            extra=log_record(energy=energy),
        )
        self._stop_event = asyncio.Event()
        loop = loop or asyncio.get_event_loop()
        self._schedule_task = loop.create_task(self._schedule_loop(energy))
        return self._schedule_task

    async def shutdown_schedule(self) -> None:
        """Stop the scheduler and wait for the task to finish."""
        if self._schedule_task:
            assert self._stop_event is not None
            self.logger.info("schedule shutdown initiated")
            self._stop_event.set()
            try:
                await self._schedule_task
                self.logger.info("schedule task finished")
            finally:
                self._schedule_task = None


from typing import Callable, Optional, Type


class ImprovementEngineRegistry:
    """Register and run multiple :class:`SelfImprovementEngine` instances."""

    def __init__(self) -> None:
        self.engines: dict[str, SelfImprovementEngine] = {}

    def register_engine(
        self, name: str, engine: SelfImprovementEngine
    ) -> None:
        """Add *engine* under *name*."""
        self.engines[name] = engine

    def unregister_engine(self, name: str) -> None:
        """Remove the engine referenced by *name* if present."""
        self.engines.pop(name, None)

    def run_all_cycles(self, energy: int = 1) -> dict[str, AutomationResult]:
        """Execute ``run_cycle`` on all registered engines."""
        results: dict[str, AutomationResult] = {}
        for name, eng in self.engines.items():
            if eng._should_trigger():
                results[name] = eng.run_cycle(energy=energy)
        return results

    async def run_all_cycles_async(
        self, energy: int = 1
    ) -> dict[str, AutomationResult]:
        """Asynchronously execute ``run_cycle`` on all registered engines."""

        async def _run(name: str, eng: SelfImprovementEngine):
            if eng._should_trigger():
                res = await asyncio.to_thread(eng.run_cycle, energy=energy)
                return name, res
            return None

        tasks = [
            asyncio.create_task(_run(n, e)) for n, e in self.engines.items()
        ]
        results: dict[str, AutomationResult] = {}
        for t in tasks:
            out = await t
            if out:
                results[out[0]] = out[1]
        return results

    def schedule_all(
        self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> list[asyncio.Task]:
        """Start schedules for all engines and return the created tasks."""
        tasks: list[asyncio.Task] = []
        for eng in self.engines.values():
            tasks.append(eng.schedule(energy=energy, loop=loop))
        return tasks

    async def shutdown_all(self) -> None:
        """Gracefully stop all running schedules."""
        for eng in self.engines.values():
            await eng.shutdown_schedule()

    def autoscale(
        self,
        *,
        capital_bot: CapitalManagementBot,
        data_bot: DataBot,
        factory: Callable[[str], SelfImprovementEngine],
        max_engines: int = 5,
        min_engines: int = 1,
        create_energy: float = 0.8,
        remove_energy: float = 0.3,
        roi_threshold: float = 0.0,
        cost_per_engine: float = 0.0,
        approval_callback: Optional[Callable[[], bool]] = None,
        max_instances: Optional[int] = None,
    ) -> None:
        """Dynamically create or remove engines based on ROI and resources."""
        try:
            energy = capital_bot.energy_score(
                load=0.0,
                success_rate=1.0,
                deploy_eff=1.0,
                failure_rate=0.0,
            )
        except Exception as exc:
            self.logger.exception("autoscale energy check failed: %s", exc)
            energy = 0.0
        try:
            trend = data_bot.long_term_roi_trend(limit=200)
        except Exception as exc:
            self.logger.exception("autoscale trend fetch failed: %s", exc)
            trend = 0.0
        if not capital_bot.check_budget():
            return
        if max_instances is not None and len(self.engines) >= max_instances:
            return
        projected_roi = trend - cost_per_engine
        if (
            energy >= create_energy
            and trend > roi_threshold
            and projected_roi > 0.0
            and len(self.engines) < max_engines
        ):
            if approval_callback and not approval_callback():
                return
            name = f"engine{len(self.engines)}"
            self.register_engine(name, factory(name))
            return
        if (
            energy <= remove_energy
            or trend <= roi_threshold
            or projected_roi <= 0.0
        ) and len(self.engines) > min_engines:
            name = next(iter(self.engines))
            self.unregister_engine(name)


__all__ = [
    "SelfImprovementEngine",
    "ImprovementEngineRegistry",
    "auto_x",
    "SynergyDashboard",
    "load_synergy_history",
    "synergy_stats",
    "synergy_ma",
]


def auto_x(
    engines: list[SelfImprovementEngine] | None = None,
    *,
    energy: int = 1,
) -> dict[str, AutomationResult]:
    """Convenience helper to run a self‑improvement cycle."""
    registry = ImprovementEngineRegistry()
    if engines:
        for idx, eng in enumerate(engines):
            registry.register_engine(f"engine{idx}", eng)
    else:
        registry.register_engine("default", SelfImprovementEngine())
    return registry.run_all_cycles(energy=energy)


def load_synergy_history(path: str | Path) -> list[dict[str, float]]:
    """Return synergy history entries from ``path`` SQLite database."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with sqlite3.connect(p) as conn:
            rows = conn.execute(
                "SELECT entry FROM synergy_history ORDER BY id"
            ).fetchall()
        hist: list[dict[str, float]] = []
        for (text,) in rows:
            data = json.loads(text)
            if isinstance(data, dict):
                hist.append({str(k): float(v) for k, v in data.items()})
        return hist
    except Exception:
        return []


def synergy_stats(history: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return average and variance for each synergy metric."""
    metrics: dict[str, list[float]] = {}
    for entry in history:
        for k, v in entry.items():
            metrics.setdefault(str(k), []).append(float(v))
    stats: dict[str, dict[str, float]] = {}
    for name, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        stats[name] = {
            "average": float(arr.mean()) if arr.size else 0.0,
            "variance": float(arr.var()) if arr.size else 0.0,
        }
    return stats


def synergy_ma(
    history: list[dict[str, float]], window: int = 5
) -> list[dict[str, float]]:
    """Return rolling averages over ``window`` samples for each metric."""
    if window < 1:
        raise ValueError("window must be positive")
    metrics = sorted({k for d in history for k in d})
    ma_history: list[dict[str, float]] = []
    for idx in range(len(history)):
        ma_entry: dict[str, float] = {}
        start = max(0, idx + 1 - window)
        for name in metrics:
            vals = [history[j].get(name, 0.0) for j in range(start, idx + 1)]
            arr = np.array(vals, dtype=float)
            ma_entry[name] = float(arr.mean()) if arr.size else 0.0
        ma_history.append(ma_entry)
    return ma_history


class SynergyDashboard:
    """Expose synergy metrics via a small Flask app.

    Parameters
    ----------
    history_file:
        Path to the SQLite history database.
    ma_window:
        Window size for moving averages.
    exporter_host/exporter_port:
        When set, fetch metrics from a Prometheus exporter instead of the
        history file.
    refresh_interval:
        How often to poll the exporter.
    max_history:
        Maximum number of history entries to keep in memory when using an
        exporter. ``None`` disables trimming.
    """

    def __init__(
        self,
        history_file: str | Path = "synergy_history.db",
        *,
        ma_window: int = 5,
        exporter_host: str | None = None,
        exporter_port: int = 8003,
        refresh_interval: float = 5.0,
        max_history: int | None = None,
    ) -> None:
        from flask import Flask, jsonify  # type: ignore

        self.logger = get_logger(self.__class__.__name__)
        self.history_file = Path(history_file)
        self.ma_window = ma_window
        self.exporter_host = exporter_host
        self.exporter_port = exporter_port
        self.refresh_interval = float(refresh_interval)
        self.max_history = max_history
        self._history: list[dict[str, float]] = []
        self._last_metrics: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.jsonify = jsonify
        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/stats", "stats", self.stats)
        self.app.add_url_rule("/plot.png", "plot", self.plot)
        self.app.add_url_rule("/history", "history", self.history)

        if self.exporter_host:
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def _load(self) -> list[dict[str, float]]:
        if self.exporter_host:
            return list(self._history)
        return load_synergy_history(self.history_file)

    # --------------------------------------------------------------
    def _parse_metrics(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, value = parts
            if name.startswith("synergy_"):
                try:
                    metrics[name] = float(value)
                except ValueError:
                    continue
        return metrics

    def _fetch_exporter_metrics(self) -> dict[str, float]:
        import requests  # type: ignore

        url = f"http://{self.exporter_host}:{self.exporter_port}/metrics"
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.status_code != 200:
                raise RuntimeError(f"status {resp.status_code}")
            metrics = self._parse_metrics(resp.text)
            if metrics:
                self._last_metrics = metrics
            return metrics
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.warning("failed to fetch metrics from %s: %s", url, exc)
            return dict(self._last_metrics)

    def _update_loop(self) -> None:
        while not self._stop.is_set():
            vals = self._fetch_exporter_metrics()
            if vals:
                self._history.append(vals)
                if self.max_history and len(self._history) > self.max_history:
                    self._history = self._history[-self.max_history :]
            self._stop.wait(self.refresh_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def index(self) -> tuple[str, int]:
        return "Synergy dashboard running. Access /stats", 200

    def stats(self) -> tuple[str, int]:
        hist = self._load()
        ma_hist = synergy_ma(hist, self.ma_window)
        data = {
            "stats": synergy_stats(hist),
            "latest": hist[-1] if hist else {},
            "rolling_average": ma_hist[-1] if ma_hist else {},
        }
        return self.jsonify(data), 200

    def history(self) -> tuple[list[dict[str, float]], int]:
        return self.jsonify(self._load()), 200

    def plot(self) -> tuple[bytes, int, dict[str, str]]:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        history = self._load()
        ma_history = synergy_ma(history, self.ma_window)
        metrics = sorted({k for d in history for k in d})
        labels = list(range(len(history)))
        fig, ax = plt.subplots()
        for name in metrics:
            vals = [d.get(name, 0.0) for d in history]
            ax.plot(labels, vals, label=name)
            ma_vals = [d.get(name, 0.0) for d in ma_history]
            ax.plot(labels, ma_vals, label=f"{name}_ma", linestyle="--")
        if metrics:
            ax.legend()
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        fig.tight_layout()
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        *,
        wsgi: str = "flask",
    ) -> None:
        """Run the dashboard using the selected WSGI/ASGI server."""
        with contextlib.closing(socket.socket()) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                self.logger.error("port %d in use", port)
                raise

        server = wsgi.lower()
        if server == "flask":
            self.app.run(host=host, port=port)
            return

        if server == "gunicorn":
            try:
                from gunicorn.app.base import BaseApplication  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("gunicorn is required for --wsgi gunicorn") from exc

            class _GunicornApp(BaseApplication):
                def __init__(self, app):
                    self.application = app
                    super().__init__()

                def load_config(self):  # pragma: no cover - runtime setup
                    self.cfg.set("bind", f"{host}:{port}")
                    self.cfg.set("workers", 1)

                def load(self):  # pragma: no cover - runtime setup
                    return self.application

            _GunicornApp(self.app).run()
            return

        if server == "uvicorn":
            try:
                import uvicorn  # type: ignore
                from starlette.middleware.wsgi import WSGIMiddleware
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("uvicorn is required for --wsgi uvicorn") from exc

            uvicorn.run(WSGIMiddleware(self.app), host=host, port=port, workers=1)
            return

        raise ValueError(f"unknown wsgi server: {wsgi}")


def cli(argv: list[str] | None = None) -> None:
    """Command line interface for synergy utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Self-improvement utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser(
        "synergy-dashboard", help="start synergy metrics dashboard"
    )
    p_dash.add_argument("--file", default="synergy_history.db")
    p_dash.add_argument("--port", type=int, default=5001)
    p_dash.add_argument("--exporter-host")
    p_dash.add_argument("--exporter-port", type=int, default=8003)
    p_dash.add_argument("--refresh-interval", type=float, default=5.0)
    p_dash.add_argument(
        "--max-history",
        type=int,
        default=1000,
        help="max entries to keep in memory when polling an exporter",
    )
    p_dash.add_argument(
        "--wsgi",
        choices=["flask", "gunicorn", "uvicorn"],
        default="flask",
        help="WSGI/ASGI server to use",
    )

    p_plot = sub.add_parser("plot-synergy", help="plot synergy metrics")
    p_plot.add_argument("history", help="synergy_history.db file")
    p_plot.add_argument("output", help="output PNG file")

    args = parser.parse_args(argv)

    if args.cmd == "synergy-dashboard":
        dash = SynergyDashboard(
            args.file,
            exporter_host=args.exporter_host,
            exporter_port=args.exporter_port,
            refresh_interval=args.refresh_interval,
            max_history=args.max_history,
        )
        dash.run(port=args.port, wsgi=args.wsgi)
        return

    if args.cmd == "plot-synergy":
        hist = load_synergy_history(args.history)
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            parser.error("matplotlib is required for plotting")
        labels = list(range(len(hist)))
        metrics = sorted({k for d in hist for k in d})
        for name in metrics:
            vals = [d.get(name, 0.0) for d in hist]
            plt.plot(labels, vals, label=name)
        if metrics:
            plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig(args.output)
        plt.close()
        return

    parser.error("unknown command")


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
