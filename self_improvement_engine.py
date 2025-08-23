from __future__ import annotations

"""Periodic self-improvement engine for the Menace system."""

import logging

try:
    from .logging_utils import log_record, get_logger, setup_logging, set_correlation_id
except Exception:  # pragma: no cover - simplified environments
    try:
        from logging_utils import log_record  # type: ignore
    except Exception:  # pragma: no cover - last resort

        def log_record(**fields: object) -> dict[str, object]:  # type: ignore
            return fields

    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

    def setup_logging() -> None:  # type: ignore
        return

    def set_correlation_id(_: str | None) -> None:  # type: ignore
        return


import time
import threading
import asyncio
import os

from db_router import init_db_router, GLOBAL_ROUTER

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    setup_logging()
from sandbox_settings import SandboxSettings
from .metrics_exporter import (
    synergy_weight_updates_total,
    synergy_weight_update_failures_total,
    synergy_weight_update_alerts_total,
    orphan_modules_reintroduced_total,
    orphan_modules_passed_total,
    orphan_modules_tested_total,
    orphan_modules_failed_total,
    orphan_modules_reclassified_total,
    orphan_modules_redundant_total,
    orphan_modules_legacy_total,
    prediction_mae,
    prediction_reliability,
)

init_db_router("self_improvement_engine")
from alert_dispatcher import dispatch_alert
import json
import inspect
import sqlite3
import pickle
import io
import tempfile
import math
import shutil
import ast
import yaml
from pathlib import Path
from typing import Mapping
from datetime import datetime
from dynamic_module_mapper import build_module_map, discover_module_groups
try:
    from . import security_auditor
except Exception:  # pragma: no cover - fallback for flat layout
    import security_auditor  # type: ignore
import sandbox_runner.environment as environment
from .self_test_service import SelfTestService
try:
    from . import self_test_service as sts
except Exception:  # pragma: no cover - fallback for flat layout
    import self_test_service as sts  # type: ignore
from orphan_analyzer import classify_module, analyze_redundancy

import numpy as np
import socket
import contextlib
import subprocess
from .error_cluster_predictor import ErrorClusterPredictor
from .quick_fix_engine import generate_patch
from .error_logger import TelemetryEvent
from . import mutation_logger as MutationLogger
from .gpt_memory import GPTMemoryManager
from .local_knowledge_module import init_local_knowledge
from gpt_memory_interface import GPTMemoryInterface
try:
    from .gpt_knowledge_service import GPTKnowledgeService
except Exception:  # pragma: no cover - fallback for flat layout
    from gpt_knowledge_service import GPTKnowledgeService  # type: ignore
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # helper for standardised GPT memory logging
    from .memory_logging import log_with_tags
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_logging import log_with_tags  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .local_knowledge_module import LocalKnowledgeModule
except Exception:  # pragma: no cover - fallback for flat layout
    from local_knowledge_module import LocalKnowledgeModule  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .knowledge_retriever import (
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_improvement_path,
        recent_error_fix,
)
except Exception:  # pragma: no cover - fallback for flat layout
    from knowledge_retriever import (  # type: ignore
        get_feedback,
        get_error_fixes,
        recent_feedback,
        recent_improvement_path,
        recent_error_fix,
    )
try:  # pragma: no cover - allow flat imports
    from .relevancy_radar import RelevancyRadar, scan as radar_scan, radar
except Exception:  # pragma: no cover - fallback for flat layout
    from relevancy_radar import RelevancyRadar, scan as radar_scan, radar  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .module_retirement_service import ModuleRetirementService
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from module_retirement_service import ModuleRetirementService  # type: ignore
    except Exception:  # pragma: no cover - last resort
        ModuleRetirementService = object  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .relevancy_metrics_db import RelevancyMetricsDB
except Exception:  # pragma: no cover - fallback for flat layout
    from relevancy_metrics_db import RelevancyMetricsDB  # type: ignore
try:  # pragma: no cover - optional dependency
    from sandbox_runner.orphan_discovery import append_orphan_classifications
except Exception:  # pragma: no cover - best effort fallback
    append_orphan_classifications = None  # type: ignore
from .human_alignment_flagger import (
    HumanAlignmentFlagger,
    flag_improvement,
    flag_alignment_issues,
    _collect_diff_data,
)
from .human_alignment_agent import HumanAlignmentAgent
from .audit_logger import log_event as audit_log_event, get_recent_events
from .violation_logger import log_violation
from .alignment_review_agent import AlignmentReviewAgent
from .governance import check_veto, load_rules
from .evaluation_dashboard import append_governance_result
try:  # pragma: no cover - allow flat imports
    from .deployment_governance import evaluate as deployment_evaluate
except Exception:  # pragma: no cover - fallback for flat layout
    from deployment_governance import evaluate as deployment_evaluate  # type: ignore
try:
    from .borderline_bucket import BorderlineBucket
except Exception:  # pragma: no cover - fallback for flat layout
    from borderline_bucket import BorderlineBucket  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .foresight_gate import ForesightDecision, is_foresight_safe_to_promote
except Exception:  # pragma: no cover - fallback for flat layout
    from foresight_gate import ForesightDecision, is_foresight_safe_to_promote  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .upgrade_forecaster import UpgradeForecaster
except Exception:  # pragma: no cover - fallback for flat layout
    from upgrade_forecaster import UpgradeForecaster  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .workflow_graph import WorkflowGraph
except Exception:  # pragma: no cover - fallback for flat layout
    from workflow_graph import WorkflowGraph  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .forecast_logger import ForecastLogger, log_forecast_record
except Exception:  # pragma: no cover - fallback for flat layout
    from forecast_logger import ForecastLogger, log_forecast_record  # type: ignore

logger = get_logger(__name__)

BACKUP_COUNT = 3
ADAPTIVE_ROI_TRAIN_INTERVAL = 3600  # seconds between scheduled retraining


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
    backups = [
        path.with_suffix(path.suffix + f".bak{i}") for i in range(1, BACKUP_COUNT + 1)
    ]
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
    with tempfile.NamedTemporaryFile(
        mode, encoding=encoding, dir=path.parent, delete=False
    ) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    _rotate_backups(path)
    os.replace(tmp, path)


def _update_alignment_baseline(settings: SandboxSettings | None = None) -> None:
    """Write current test counts and complexity scores to baseline metrics file."""
    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return
        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        test_count = 0
        total_complexity = 0
        for file in repo.rglob("*.py"):
            rel = file.relative_to(repo)
            name = rel.name
            rel_posix = rel.as_posix()
            if (
                rel_posix.startswith("tests")
                or name.startswith("test_")
                or name.endswith("_test.py")
            ):
                test_count += 1
            try:
                code = file.read_text(encoding="utf-8")
                tree = ast.parse(code)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    score = 1
                    for sub in ast.walk(node):
                        if isinstance(
                            sub,
                            (
                                ast.If,
                                ast.For,
                                ast.AsyncFor,
                                ast.While,
                                ast.With,
                                ast.AsyncWith,
                                ast.IfExp,
                                ast.ListComp,
                                ast.DictComp,
                                ast.SetComp,
                                ast.GeneratorExp,
                            ),
                        ):
                            score += 1
                        elif isinstance(sub, ast.BoolOp):
                            score += len(getattr(sub, "values", [])) - 1
                        elif isinstance(sub, ast.Try):
                            score += len(sub.handlers)
                            if sub.orelse:
                                score += 1
                            if sub.finalbody:
                                score += 1
                    total_complexity += score
        data = {"tests": int(test_count), "complexity": int(total_complexity)}
        _atomic_write(Path(path_str), yaml.safe_dump(data))
    except Exception:
        logger.exception("alignment baseline update failed")

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
try:  # pragma: no cover - optional dependency
    from .adaptive_roi_predictor import AdaptiveROIPredictor, load_training_data
except Exception:  # pragma: no cover - fallback for tests
    AdaptiveROIPredictor = object  # type: ignore
    def load_training_data(*a, **k):  # type: ignore
        return []
from .adaptive_roi_dataset import build_dataset
from .roi_tracker import ROITracker
from .foresight_tracker import ForesightTracker
from .truth_adapter import TruthAdapter
from .evaluation_history_db import EvaluationHistoryDB
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
            logger.warning("PyTorch not installed - using ActorCritic strategy")
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
    def update(
        self,
        roi_delta: float,
        deltas: dict[str, float],
        extra: dict[str, float] | None = None,
    ) -> None:
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
            if extra:
                reward *= 1.0 + float(extra.get("avg_roi", 0.0))
                reward *= 1.0 + float(extra.get("pass_rate", 0.0))
            q = self.strategy.update(
                {}, self._state, idx, reward, self._state, 1.0, 0.9
            )
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
        if (
            name in {"double", "double_dqn", "double-dqn", "ddqn", "td3"}
            and sip_torch is not None
        ):
            self.strategy = DoubleDQNStrategy(
                action_dim=7, lr=lr, target_sync=target_sync
            )
            self.strategy_name = "td3" if name == "td3" else "double_dqn"
        elif name in {
            "policy",
            "policy_gradient",
            "actor_critic",
            "actor-critic",
            "sac",
        }:
            self.strategy = ActorCriticStrategy()
            self.strategy_name = "sac" if name == "sac" else "policy_gradient"
        else:
            # prefer Double DQN when PyTorch is available
            if sip_torch is not None:
                self.strategy = DoubleDQNStrategy(
                    action_dim=7, lr=lr, target_sync=target_sync
                )
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
                if (
                    hasattr(self.strategy, "target_model")
                    and self.strategy.target_model is not None
                ):
                    buf = io.BytesIO()
                    sip_torch.save(self.strategy.target_model.state_dict(), buf)
                    _atomic_write(
                        Path(base + ".target.pt"), buf.getvalue(), binary=True
                    )
            except Exception as exc:
                logger.exception("failed to save DQN models: %s", exc)
        try:
            pkl = Path(base + ".policy.pkl")
            _atomic_write(pkl, pickle.dumps(self.strategy), binary=True)
        except Exception as exc:
            logger.exception("failed to save strategy pickle: %s", exc)

    # ------------------------------------------------------------------
    def update(
        self,
        roi_delta: float,
        deltas: dict[str, float],
        extra: dict[str, float] | None = None,
    ) -> None:
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
            if extra:
                reward *= 1.0 + float(extra.get("avg_roi", 0.0))
                reward *= 1.0 + float(extra.get("pass_rate", 0.0))
            q = self.strategy.update(
                {}, self._state, idx, reward, self._state, 1.0, 0.9
            )
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

    def __init__(
        self, path: Path | None = None, lr: float = 1e-3, *, target_sync: int = 10
    ) -> None:
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
        auto_refresh_map: bool = False,
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
        roi_compounding_weight: float | None = None,
        synergy_weights_path: Path | str | None = None,
        synergy_weights_lr: float | None = None,
        synergy_learner_cls: Type[SynergyWeightLearner] = SynergyWeightLearner,
        score_backend: PatchScoreBackend | None = None,
        error_predictor: ErrorClusterPredictor | None = None,
        roi_predictor: AdaptiveROIPredictor | None = None,
        roi_tracker: ROITracker | None = None,
        foresight_tracker: ForesightTracker | None = None,
        gpt_memory: GPTMemoryInterface | None = None,
        knowledge_service: GPTKnowledgeService | None = None,
        relevancy_radar: RelevancyRadar | None = None,
        tau: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if gpt_memory is None:
            gpt_memory = kwargs.get("gpt_memory_manager")
        self.interval = interval
        self.bot_name = bot_name
        self.info_db = info_db or InfoDB()
        self.aggregator = ResearchAggregatorBot([bot_name], info_db=self.info_db)
        self.pipeline = pipeline or ModelAutomationPipeline(
            aggregator=self.aggregator, action_planner=action_planner
        )
        self.action_planner = action_planner
        err_bot = ErrorBot(ErrorDB(), MetricsDB())
        self.error_bot = err_bot
        self.diagnostics = diagnostics or DiagnosticManager(MetricsDB(), err_bot)
        self._alignment_agent: AlignmentReviewAgent | None = None
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
            policy = ConfigurableSelfImprovementPolicy(strategy=policy_strategy)
        self.policy = policy
        if self.policy and getattr(self.policy, "path", None):
            try:
                self.policy.load(self.policy.path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("policy load failed: %s", exc)
        self.optimize_self_flag = optimize_self
        self.meta_logger = meta_logger
        self.metrics_db = getattr(meta_logger, "metrics_db", None) if meta_logger else None
        if self.metrics_db is None:
            try:
                data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
                self.metrics_db = RelevancyMetricsDB(data_dir / "relevancy_metrics.db")
            except Exception:
                self.metrics_db = None
        self.auto_refresh_map = bool(auto_refresh_map)
        self.pre_roi_bot = pre_roi_bot
        self.pre_roi_scale = (
            pre_roi_scale if pre_roi_scale is not None else PRE_ROI_SCALE
        )
        self.pre_roi_bias = pre_roi_bias if pre_roi_bias is not None else PRE_ROI_BIAS
        self.pre_roi_cap = pre_roi_cap if pre_roi_cap is not None else PRE_ROI_CAP
        settings = SandboxSettings()
        self.borderline_bucket = BorderlineBucket()
        self.workflow_ready = False
        self.workflow_high_risk = False
        self.workflow_risk: dict[str, object] | None = None
        self.borderline_raroi_threshold = getattr(
            settings, "borderline_raroi_threshold", 0.0
        )
        self.tau = tau if tau is not None else getattr(
            settings, "borderline_confidence_threshold", 0.0
        )
        self.use_adaptive_roi = getattr(settings, "adaptive_roi_prioritization", True)
        if self.use_adaptive_roi:
            self.roi_predictor = roi_predictor or AdaptiveROIPredictor()
            self.roi_tracker = roi_tracker or ROITracker(
                confidence_threshold=self.tau,
                raroi_borderline_threshold=self.borderline_raroi_threshold,
                borderline_bucket=self.borderline_bucket,
            )
            self._adaptive_roi_last_train = time.time()
            self.adaptive_roi_train_interval = getattr(
                settings, "adaptive_roi_train_interval", ADAPTIVE_ROI_TRAIN_INTERVAL
            )
        else:
            self.roi_predictor = None
            self.roi_tracker = None
            self._adaptive_roi_last_train = 0.0
            self.adaptive_roi_train_interval = ADAPTIVE_ROI_TRAIN_INTERVAL
        self.foresight_tracker = foresight_tracker or ForesightTracker()
        self.truth_adapter = TruthAdapter()
        self._truth_adapter_needs_retrain = False
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
            roi_ema_alpha if roi_ema_alpha is not None else settings.roi_ema_alpha
        )
        self.roi_compounding_weight = (
            roi_compounding_weight
            if roi_compounding_weight is not None
            else getattr(settings, "roi_compounding_weight", 0.0)
        )
        self.growth_weighting = getattr(settings, "roi_growth_weighting", True)
        self.growth_multipliers = {
            "exponential": getattr(settings, "growth_multiplier_exponential", 3.0),
            "linear": getattr(settings, "growth_multiplier_linear", 2.0),
            "marginal": getattr(settings, "growth_multiplier_marginal", 1.0),
        }
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
        self.auto_patch_high_risk = getattr(settings, "auto_patch_high_risk", True)
        if getattr(settings, "menace_mode", "").lower() == "autonomous":
            self.auto_patch_high_risk = True

        self.gpt_memory = (
            gpt_memory
            or getattr(self_coding_engine, "gpt_memory", None)
            or getattr(self_coding_engine, "gpt_memory_manager", None)
            or init_local_knowledge(
                os.getenv("GPT_MEMORY_DB", "gpt_memory.db")
            ).memory
        )
        self.gpt_memory_manager = self.gpt_memory  # backward compatibility
        self.local_knowledge = LocalKnowledgeModule(
            manager=self.gpt_memory, service=knowledge_service
        )
        self.knowledge_service = self.local_knowledge.knowledge
        self.relevancy_radar = relevancy_radar or RelevancyRadar()
        self.relevancy_flags: dict[str, str] = {}
        self.entropy_ceiling_modules: set[str] = set()
        # Track when the relevancy radar last ran so we can evaluate at intervals
        self._last_relevancy_eval = 0.0
        try:
            # Allow settings to override the default cadence
            self.relevancy_eval_interval = getattr(
                settings, "relevancy_eval_interval", 3600.0
            )
        except Exception:  # pragma: no cover - fallback for minimal settings
            self.relevancy_eval_interval = 3600.0

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
            self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
        else:
            self.synergy_learner.weights["efficiency"] = self.synergy_weight_efficiency
        if synergy_weight_resilience is None:
            self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
        else:
            self.synergy_learner.weights["resilience"] = self.synergy_weight_resilience
        if synergy_weight_antifragility is None:
            self.synergy_weight_antifragility = self.synergy_learner.weights[
                "antifragility"
            ]
        else:
            self.synergy_learner.weights["antifragility"] = (
                self.synergy_weight_antifragility
            )
        if synergy_weight_reliability is None:
            self.synergy_weight_reliability = self.synergy_learner.weights[
                "reliability"
            ]
        else:
            self.synergy_learner.weights["reliability"] = (
                self.synergy_weight_reliability
            )
        if synergy_weight_maintainability is None:
            self.synergy_weight_maintainability = self.synergy_learner.weights[
                "maintainability"
            ]
        else:
            self.synergy_learner.weights["maintainability"] = (
                self.synergy_weight_maintainability
            )
        if synergy_weight_throughput is None:
            self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
        else:
            self.synergy_learner.weights["throughput"] = self.synergy_weight_throughput
        self.state_path = Path(state_path) if state_path else None
        if error_predictor is None:
            graph = getattr(getattr(self.error_bot, "error_logger", None), "graph", None)
            if graph is None:
                graph = getattr(self.error_bot, "graph", None)
            if graph is not None and hasattr(self.error_bot, "db"):
                try:
                    self.error_predictor = ErrorClusterPredictor(graph, self.error_bot.db)
                except Exception:
                    logger.exception(
                        "error predictor init failed",
                        extra=log_record(component="ErrorClusterPredictor"),
                    )
                    self.error_predictor = None
            else:
                self.error_predictor = None
        else:
            self.error_predictor = error_predictor
        self.roi_history: list[float] = []
        self.raroi_history: list[float] = []
        self.roi_group_history: dict[int, list[float]] = {}
        self.roi_delta_ema: float = 0.0
        self._last_growth_type: str | None = None
        self._synergy_cache: dict | None = None
        self.alignment_flagger = HumanAlignmentFlagger()
        self.cycle_logs: list[dict[str, Any]] = []
        self.warning_summary: list[dict[str, Any]] = []
        self.logger = get_logger("SelfImprovementEngine")
        self._load_state()
        self._load_synergy_weights()
        from .module_index_db import ModuleIndexDB

        auto_map = os.getenv("SANDBOX_AUTO_MAP") == "1"
        if not auto_map and os.getenv("SANDBOX_AUTODISCOVER_MODULES") == "1":
            self.logger.warning(
                "SANDBOX_AUTODISCOVER_MODULES is deprecated; use SANDBOX_AUTO_MAP"
            )
            auto_map = True
        self.module_index = module_index or ModuleIndexDB(auto_map=auto_map)
        map_path = getattr(
            self.module_index,
            "path",
            Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")) / "module_map.json",
        )
        try:
            self._last_map_refresh = map_path.stat().st_mtime
        except Exception:
            self._last_map_refresh = 0.0
        if self.module_index and self.patch_db:
            try:
                repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                with self.patch_db._connect() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT filename FROM patch_history"
                    ).fetchall()
                mods: list[str] = []
                for r in rows:
                    p = Path(r[0])
                    if not p.is_absolute():
                        p = repo / p
                    try:
                        rel = p.relative_to(repo).as_posix()
                    except Exception:
                        rel = p.as_posix()
                    mods.append(rel)
                self.module_index.refresh(mods)
            except Exception:
                self.logger.exception("module map refresh failed during init")
        if module_clusters is None and self.module_index is not None:
            try:
                module_clusters = dict(getattr(self.module_index, "_map", {}))
            except Exception:
                module_clusters = None
        self.module_clusters: dict[str, int] = module_clusters or {}
        # Filled by ``_update_orphan_modules`` when recursive orphan discovery
        # finds new modules. Maps module paths to metadata such as parents and
        # redundancy classification.
        self.orphan_traces: dict[str, dict[str, Any]] = {}
        # queue of modules needing preventative fixes
        self._preventative_queue: list[str] = []

        if module_groups is None:
            try:
                repo_path = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                discovered = discover_module_groups(repo_path)
                module_groups = {
                    (m if m.endswith(".py") else f"{m}.py"): grp
                    for grp, mods in discovered.items()
                    for m in mods
                }
            except Exception:
                module_groups = None

        if module_groups:
            grp_map: dict[str, int] = {}
            for mod, grp in module_groups.items():
                try:
                    idx = (
                        self.module_index.group_id(str(grp))
                        if self.module_index
                        else abs(hash(grp)) % 1000
                    )
                except Exception:
                    idx = abs(hash(grp)) % 1000
                grp_map[mod] = idx
            if self.module_index:
                try:
                    self.module_index.merge_groups(grp_map)
                    grp_map = {m: self.module_index.get(m) for m in grp_map}
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
        self._last_mutation_id: int | None = None
        self._last_patch_id: int | None = None
        self._last_scenario_metrics: dict[str, float] = {}
        self._last_scenario_trend: dict[str, float] = {}
        self._scenario_pass_rate: float = 0.0
        self._force_rerun = False
        if self.event_bus:
            if self.learning_engine:
                try:
                    self.event_bus.subscribe("pathway:new", self._on_new_pathway)
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

    def _memory_summaries(self, key: str) -> str:
        """Return a summary of similar past actions from memory."""
        summaries: list[str] = []
        try:
            entries = get_feedback(self.gpt_memory, key, limit=5)
        except Exception:
            entries = []
        for ent in entries:
            resp = (getattr(ent, "response", "") or "").strip()
            tag = "success" if "success" in resp.lower() else "failure"
            snippet = resp.splitlines()[0]
            summaries.append(f"{tag}: {snippet}")
        try:
            fixes = get_error_fixes(self.gpt_memory, key, limit=3)
        except Exception:
            fixes = []
        for fix in fixes:
            resp = (getattr(fix, "response", "") or "").strip()
            if resp:
                summaries.append(f"fix: {resp.splitlines()[0]}")
        if getattr(self, "knowledge_service", None):
            try:
                insight = recent_feedback(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{FEEDBACK} insight: {insight}")
            except Exception:
                pass
            try:
                insight = recent_improvement_path(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{IMPROVEMENT_PATH} insight: {insight}")
            except Exception:
                pass
            try:
                insight = recent_error_fix(self.knowledge_service)  # type: ignore[attr-defined]
                if insight:
                    summaries.append(f"{ERROR_FIX} insight: {insight}")
            except Exception:
                pass
        return "\n".join(summaries)

    def _record_memory_outcome(
        self,
        module: str,
        action: str,
        success: bool,
        *,
        tags: Sequence[str] | None = None,
    ) -> None:
        try:
            outcome_tags = [
                f"self_improvement_engine.{action}",
                FEEDBACK,
                IMPROVEMENT_PATH,
                ERROR_FIX,
                INSIGHT,
            ]
            if tags:
                outcome_tags.extend(tags)
            log_with_tags(
                self.gpt_memory,
                f"{action}:{module}",
                "success" if success else "failure",
                tags=outcome_tags,
            )
        except Exception:
            self.logger.exception("memory logging failed", extra=log_record(module=module))

    @radar.track
    def _generate_patch_with_memory(
        self,
        module: str,
        action: str,
        *,
        tags: Sequence[str] | None = None,
    ) -> int | None:
        start = time.perf_counter()
        history = self._memory_summaries(module)
        if history:
            self.logger.info(
                "patch memory context",
                extra=log_record(module=module, history=history, tags=[INSIGHT]),
            )
        client = getattr(self.self_coding_engine, "llm_client", None)
        if client is not None:
            try:
                ask_tags = [ERROR_FIX, IMPROVEMENT_PATH]
                if tags:
                    ask_tags.extend(tags)
                data = ask_with_memory(
                    client,
                    f"self_improvement_engine.{action}",
                    f"{action}:{module}",
                    memory=self.local_knowledge,
                    tags=ask_tags,
                )
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if text:
                    self.logger.info(
                        "gpt_suggestion",
                        extra=log_record(
                            module=module, suggestion=text, tags=[ERROR_FIX]
                        ),
                    )
            except Exception:
                self.logger.exception(
                    "gpt suggestion failed", extra=log_record(module=module)
                )
        patch_id = generate_patch(module, self.self_coding_engine)
        elapsed = time.perf_counter() - start
        if self.metrics_db:
            try:
                self.metrics_db.record(
                    module, elapsed, self.module_index, tags, roi_delta=0.0
                )
            except Exception:
                self.logger.exception(
                    "relevancy metrics record failed", extra=log_record(module=module)
                )
        try:
            log_tags = [
                f"self_improvement_engine.{action}",
                FEEDBACK,
                IMPROVEMENT_PATH,
                ERROR_FIX,
                INSIGHT,
            ]
            if tags:
                log_tags.extend(tags)
            log_with_tags(
                self.gpt_memory,
                f"{action}:{module}",
                f"patch_id={patch_id}",
                tags=log_tags,
            )
        except Exception:
            self.logger.exception(
                "memory logging failed", extra=log_record(module=module)
            )
        self._record_memory_outcome(
            module, action, patch_id is not None, tags=tags
        )
        return patch_id

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
            self.raroi_history = [float(x) for x in data.get("raroi_history", [])]
            self.roi_group_history = {
                int(k): [float(vv) for vv in v]
                for k, v in data.get("roi_group_history", {}).items()
            }
            self.last_run = float(data.get("last_run", self.last_run))
            self.roi_delta_ema = float(data.get("roi_delta_ema", self.roi_delta_ema))
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
                        "raroi_history": self.raroi_history,
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
        self.synergy_weight_maintainability = self.synergy_learner.weights[
            "maintainability"
        ]
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
        self.synergy_learner.weights["antifragility"] = (
            self.synergy_weight_antifragility
        )
        self.synergy_learner.weights["reliability"] = self.synergy_weight_reliability
        self.synergy_learner.weights["maintainability"] = (
            self.synergy_weight_maintainability
        )
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
            conn = GLOBAL_ROUTER.get_connection("synergy_history")
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
            extra=log_record(history_file=str(history_file), interval=float(interval)),
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
        tracker = getattr(self, "tracker", None) or getattr(self, "roi_tracker", None)
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

    def _evaluate_scenario_metrics(self, metrics: dict[str, float]) -> None:
        """Evaluate scenario metrics and trigger remediation when thresholds fail."""
        prev = getattr(self, "_last_scenario_metrics", {})
        trend = {k: float(v) - float(prev.get(k, 0.0)) for k, v in metrics.items()}
        thresholds = {
            "latency_error_rate": {"max": 0.2, "action": "alert"},
            "hostile_failures": {"max": 5.0, "action": "patch"},
            "misuse_failures": {"max": 5.0, "action": "patch"},
            "concurrency_throughput": {"min": 100.0, "action": "rerun"},
        }
        failing: list[str] = []
        for name, val in metrics.items():
            cfg = thresholds.get(name)
            if not cfg:
                continue
            exceed = ("max" in cfg and val > cfg["max"]) or (
                "min" in cfg and val < cfg["min"]
            )
            if exceed:
                failing.append(name)
                action = cfg.get("action")
                if action == "alert":
                    try:
                        dispatch_alert(
                            "scenario_degradation",
                            2,
                            f"{name} degraded",
                            {"value": float(val)},
                        )
                    except Exception:
                        self.logger.exception("alert dispatch failed for %s", name)
                elif action == "patch":
                    try:
                        with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                            src = Path(name)
                            if src.suffix == "":
                                src = src.with_suffix(".py")
                            rel = src.name if src.is_absolute() else src
                            before_target = Path(before_dir) / rel
                            before_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, before_target)
                            self.logger.info(
                                "gpt_suggestion",
                                extra=log_record(
                                    module=name,
                                    suggestion="scenario_patch",
                                    tags=[ERROR_FIX],
                                ),
                            )
                            try:
                                log_with_tags(
                                    self.gpt_memory,
                                    f"scenario_patch:{name}",
                                    "suggested",
                                    tags=[f"self_improvement_engine.scenario_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                                )
                            except Exception:
                                self.logger.exception(
                                    "memory logging failed", extra=log_record(module=name)
                                )
                            patch_id = self._generate_patch_with_memory(
                                name, "scenario_patch"
                            )
                            self.logger.info(
                                "patch result",
                                extra=log_record(
                                    module=name,
                                    patch_id=patch_id,
                                    success=patch_id is not None,
                                    tags=["fix_result"],
                                ),
                            )
                            if patch_id is not None:
                                after_target = Path(after_dir) / rel
                                after_target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src, after_target)
                                diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                                self._pre_commit_alignment_check(diff_data)
                                self._alignment_review_last_commit(
                                    f"scenario_patch_{patch_id}"
                                )
                    except Exception:
                        self.logger.exception("patch generation failed for %s", name)
                elif action == "rerun":
                    self._force_rerun = True
        total = len(metrics)
        passed = total - len(failing)
        frac = passed / total if total else 1.0
        # store negative value when scenarios fail so reward is penalised
        self._scenario_pass_rate = frac - 1.0
        self._last_scenario_trend = trend
        self._last_scenario_metrics = dict(metrics)

    # ------------------------------------------------------------------
    def _alignment_review_last_commit(self, description: str) -> None:
        """Run alignment flagger on the most recent commit."""
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return
        try:
            diff_proc = subprocess.run(
                ["git", "show", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            patch = diff_proc.stdout
            files_proc = subprocess.run(
                ["git", "show", "--pretty=", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            files = [ln.strip() for ln in files_proc.stdout.splitlines() if ln.strip()]
        except Exception:
            return

        try:
            report = self.alignment_flagger.flag_patch(patch, {"files": files})
        except Exception:
            return
        issues = report.get("issues", [])
        max_severity = max((i.get("severity", 0) for i in issues), default=0) / 4.0
        warn_th = settings.alignment_warning_threshold
        fail_th = settings.alignment_failure_threshold
        if max_severity >= warn_th:
            warnings = [i.get("message", "") for i in issues]
            try:
                audit_log_event(
                    "alignment_flag",
                    {"description": description, "warnings": warnings, "files": files},
                )
            except Exception:
                self.logger.exception("alignment audit log failed")
            if max_severity >= fail_th:
                try:
                    dispatch_alert(
                        "alignment_warning",
                        5,
                        "alignment failure detected",
                        {"description": description, "severity": max_severity},
                    )
                except Exception:
                    self.logger.exception("alignment warning dispatch failed")
            else:
                self.logger.warning(
                    "alignment warnings detected",
                    extra={"description": description, "warnings": warnings},
                )

    # ------------------------------------------------------------------
    def _flag_patch_alignment(
        self, patch_id: int | None, context: dict[str, Any]
    ) -> None:
        """Analyse the latest commit for alignment concerns and log findings."""
        if patch_id is None:
            return
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return
        try:
            diff_proc = subprocess.run(
                ["git", "show", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            diff = diff_proc.stdout
        except Exception:
            return
        try:
            commit_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            commit_hash = "unknown"
        try:
            report = self.alignment_flagger.flag_patch(diff, context)
            context["alignment_report"] = report
            score = report.get("score", 0)
            issues = report.get("issues", [])
            max_severity = max((i.get("severity", 0) for i in issues), default=0) / 4.0
            context["alignment_blocked"] = max_severity >= settings.alignment_failure_threshold
            self.logger.info(
                "alignment score computed",
                extra=log_record(patch_id=patch_id, score=score, severity=max_severity),
            )
            warn_th = settings.alignment_warning_threshold
            fail_th = settings.alignment_failure_threshold
            if max_severity < warn_th:
                self.logger.info(
                    "alignment severity below warning threshold",
                    extra=log_record(patch_id=patch_id, severity=max_severity),
                )
                _update_alignment_baseline(settings)
                return
            for idx, issue in enumerate(issues):
                msg = issue.get("message", "")
                sev = int(issue.get("severity", 1))
                file = msg.rsplit(" in ", 1)[-1] if " in " in msg else None
                evidence = {"file": file, "snippet": msg}
                log_violation(
                    f"patch_{patch_id}_{idx}",
                    "alignment_warning",
                    sev,
                    evidence,
                    alignment_warning=True,
                )
            record = {
                "patch_id": patch_id,
                "commit": commit_hash,
                "score": score,
                "severity": max_severity,
                "report": report,
            }
            escalated = False
            if max_severity >= fail_th:
                escalated = True
                try:
                    dispatch_alert(
                        "alignment_review",
                        5,
                        "alignment severity exceeded threshold",
                        {"patch_id": patch_id, "severity": max_severity},
                    )
                except Exception:
                    self.logger.exception("alignment review dispatch failed")
            record["escalated"] = escalated
            try:
                security_auditor.dispatch_alignment_warning(record)
            except Exception:
                self.logger.exception("alignment warning dispatch failed")
            self.cycle_logs.append({"cycle": self._cycle_count, **record})
            if self.event_bus:
                try:
                    self.event_bus.publish("alignment:flag", record)
                except Exception:
                    self.logger.exception("alignment event publish failed")
            try:
                path = Path("sandbox_data") / "alignment_flags.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception:
                self.logger.exception("alignment flag persistence failed")
            if not escalated:
                _update_alignment_baseline(settings)
        except Exception:
            self.logger.exception(
                "alignment flagging failed", extra=log_record(patch_id=patch_id)
            )

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
                "synergy_efficiency": lw.get(
                    "efficiency", self.synergy_weight_efficiency
                ),
                "synergy_resilience": lw.get(
                    "resilience", self.synergy_weight_resilience
                ),
                "synergy_antifragility": lw.get(
                    "antifragility", self.synergy_weight_antifragility
                ),
                "synergy_reliability": lw.get(
                    "reliability", self.synergy_weight_reliability
                ),
                "synergy_maintainability": lw.get(
                    "maintainability", self.synergy_weight_maintainability
                ),
                "synergy_throughput": lw.get(
                    "throughput", self.synergy_weight_throughput
                ),
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
                            weights = {
                                k: base_weights[k] / len(base_weights)
                                for k in base_weights
                            }
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
            syn_adj = sum(norm_delta(name) * weights.get(name, 0.0) for name in weights)
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
        before_weight = self.synergy_weight_roi
        # Create a mutation event up-front so we can record the outcome even if
        # the update step fails.
        event_id = MutationLogger.log_mutation(
            change="synergy_weights_updated",
            reason="roi_delta adjustment",
            trigger="roi_delta",
            performance=0.0,
            workflow_id=0,
            before_metric=before_weight,
            after_metric=before_weight,
            parent_id=self._last_mutation_id,
        )

        after_metric = before_weight
        perf = 0.0
        roi_val = 0.0
        try:
            extra = dict(getattr(self, "_last_orphan_metrics", {}) or {})
            scen = getattr(self, "_last_scenario_trend", None)
            if scen:
                try:
                    extra["avg_roi"] = float(sum(scen.values()) / len(scen))
                except Exception:
                    extra["avg_roi"] = 0.0
            pr = getattr(self, "_scenario_pass_rate", None)
            if pr is not None:
                extra["pass_rate"] = float(pr)
            self.synergy_learner.update(roi_delta, deltas, extra)
            self.logger.info(
                "synergy weights updated",
                extra=log_record(
                    weights=self.synergy_learner.weights,
                    roi_delta=roi_delta,
                    state=self.synergy_learner._state,
                ),
            )
            self.synergy_weight_roi = self.synergy_learner.weights["roi"]
            self.synergy_weight_efficiency = self.synergy_learner.weights["efficiency"]
            self.synergy_weight_resilience = self.synergy_learner.weights["resilience"]
            self.synergy_weight_antifragility = self.synergy_learner.weights[
                "antifragility"
            ]
            self.synergy_weight_reliability = self.synergy_learner.weights["reliability"]
            self.synergy_weight_maintainability = self.synergy_learner.weights[
                "maintainability"
            ]
            self.synergy_weight_throughput = self.synergy_learner.weights["throughput"]
            self.logger.info(
                "synergy weights after update",
                extra=log_record(weights=self.synergy_learner.weights, roi_delta=roi_delta),
            )
            after_metric = self.synergy_weight_roi
            perf = roi_delta
            roi_val = roi_delta
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
        finally:
            MutationLogger.record_mutation_outcome(
                event_id,
                after_metric=after_metric,
                roi=roi_val,
                performance=perf,
            )
            self._last_mutation_id = event_id

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
                        anomaly = float(len(DataBot.detect_anomalies(df, "roi"))) / len(
                            df
                        )
                elif isinstance(df, list) and df:
                    rois = [
                        float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df
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
                    self.logger.exception("patch success rate lookup failed: %s", exc)
                    patch_rate = 0.0
        avg_roi = avg_complex = revert_rate = 0.0
        module_idx = 0
        module_trend = 0.0
        entropy_flag = 0
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
        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)
        if pdb:
            try:
                repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                with pdb._connect() as conn:
                    rows = conn.execute(
                        "SELECT roi_delta, complexity_delta, reverted, filename "
                        "FROM patch_history ORDER BY id DESC LIMIT ?",
                        (10,),
                    ).fetchall()
                if rows:
                    avg_roi = float(sum(r[0] for r in rows) / len(rows))
                    avg_complex = float(sum(r[1] for r in rows) / len(rows))
                    revert_rate = float(sum(1 for r in rows if r[2]) / len(rows))
                    p = Path(rows[0][3])
                    abs_p = p if p.is_absolute() else repo / p
                    try:
                        mod_name = abs_p.resolve().relative_to(repo).as_posix()
                    except Exception:
                        mod_name = p.name
                    module_idx = self.module_index.get(mod_name)
                    mods = [
                        m
                        for m, idx in self.module_clusters.items()
                        if idx == module_idx
                    ]
                    try:
                        if mods:
                            placeholders = ",".join("?" * len(mods))
                            total = conn.execute(
                                f"SELECT SUM(roi_delta) FROM patch_history WHERE filename IN ({placeholders})",
                                [Path(m).name for m in mods],
                            ).fetchone()
                            module_trend = float(total[0] or 0.0)
                        else:
                            module_trend = 0.0
                    except Exception:
                        module_trend = 0.0
                    if self.meta_logger:
                        try:
                            if Path(mod_name).as_posix() in self.entropy_ceiling_modules:
                                entropy_flag = 1
                            if module_trend == 0.0:
                                module_trend = dict(self.meta_logger.rankings()).get(
                                    mod_name, 0.0
                                )
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("meta logger stats failed: %s", exc)
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
                self.logger.exception("evolution history stats failed: %s", exc)
                avg_roi_delta = avg_eff = 0.0
        short_avg = 0.0
        if self.raroi_history:
            n = min(len(self.raroi_history), 5)
            short_avg = float(sum(self.raroi_history[-n:]) / n)
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
            int(entropy_flag),
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
    def _collect_action_features(self) -> list[list[float]]:
        """Return recent ROI deltas and their changes for growth prediction.

        Each history entry is evaluated with ``roi_predictor`` when available.
        Entries classified as ``"marginal"`` are skipped unless the optional
        ``allow_marginal_candidates`` flag is set on the engine.  Skipped
        entries are recorded via :meth:`_log_action` for auditability.
        """

        feats: list[list[float]] = []
        history = self.raroi_history[-5:]
        prev = 0.0
        allow_marginal = getattr(self, "allow_marginal_candidates", False)

        for idx, val in enumerate(history):
            feat = [float(val), float(val - prev)]
            roi_est = float(val)
            category = "unknown"
            if getattr(self, "roi_predictor", None):
                try:
                    try:
                        seq, category, _, _ = self.roi_predictor.predict(
                            [feat], horizon=1
                        )
                        roi_est = float(seq[-1]) if seq else 0.0
                    except TypeError:
                        roi_est, category, _, _ = self.roi_predictor.predict([feat])
                        roi_est = float(roi_est)
                except Exception:
                    roi_est, category = 0.0, "unknown"

            if category == "marginal" and not allow_marginal:
                ctx = getattr(self, "_current_context", {}) or {}
                session_id = ""
                vectors: list[tuple[str, str, float]] = []
                if isinstance(ctx, dict):
                    session_id = ctx.get("retrieval_session_id", "")
                    raw_vecs = ctx.get("retrieval_vectors") or []
                    for item in raw_vecs:
                        if isinstance(item, dict):
                            origin = item.get("origin_db") or item.get("origin")
                            vec_id = item.get("vector_id") or item.get("id")
                            score = item.get("score") or item.get("similarity")
                        else:
                            if len(item) == 3:
                                origin, vec_id, score = item
                            elif len(item) == 2:
                                origin, vec_id = item
                                score = 0.0
                            else:
                                continue
                        if origin is not None and vec_id is not None:
                            vectors.append((str(origin), str(vec_id), float(score or 0.0)))
                if getattr(self, "data_bot", None):
                    try:  # pragma: no cover - best effort
                        self.data_bot.db.log_patch_outcome(
                            f"raroi_history_{idx}",
                            False,
                            [(o, v) for o, v, _ in vectors],
                            session_id=session_id,
                            reverted=False,
                        )
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                pl = getattr(getattr(self, "self_coding_engine", None), "patch_logger", None)
                if pl and vectors and session_id:
                    try:  # pragma: no cover - best effort
                        ids = [(f"{o}:{v}", s) for o, v, s in vectors]
                        pl.track_contributors(ids, False, session_id=session_id)
                    except Exception:
                        self.logger.exception("failed to log patch outcome")
                try:  # pragma: no cover - best effort
                    self._log_action(
                        "skip_candidate", f"raroi_history_{idx}", roi_est, category
                    )
                except Exception:
                    pass
            else:
                feats.append(feat)
            prev = val

        return feats

    def _candidate_features(self, module: str) -> list[list[float]]:
        """Return feature vector aligned with ``adaptive_roi_dataset.build_dataset``.

        The returned sequence follows the column layout used by
        :func:`adaptive_roi_dataset.build_dataset` so the predictor can operate
        on live data. Missing values default to ``0.0`` allowing the engine to
        work even when telemetry is incomplete.  Columns gathered here are::

            [before_metric, after_metric, api_cost_delta, cpu_seconds_delta,
             success_rate_delta, gpt_score, gpt_feedback_score,
             gpt_feedback_tokens, long_term_perf_delta,
             long_term_eval_outcome, resource_cost, resource_cpu_usage,
             resource_gpu_usage, error_count, repair_count]
        """

        # Historical ROI forecast for the module
        hist_roi = 0.0
        if self.pre_roi_bot:
            try:
                res = self.pre_roi_bot.predict_model_roi(module, [])
                hist_roi = float(getattr(res, "roi", 0.0))
            except Exception:
                self.logger.exception("pre ROI forecast failed for %s", module)

        # Recent deltas for core metrics
        perf_delta = self._metric_delta("synergy_roi")
        api_cost_delta = self._metric_delta("api_cost")
        cpu_seconds_delta = self._metric_delta("cpu_seconds")
        success_rate_delta = self._metric_delta("success_rate")

        # GPT score from patch scoring backend (if available)
        gpt_score = 0.0
        if self._score_backend:
            try:  # pragma: no cover - backend optional
                rows = self._score_backend.fetch_recent(1)
                if rows:
                    gpt_score = float(rows[0][-1])
            except Exception:
                pass

        # GPT feedback metrics from evaluation history
        gpt_fb_score = 0.0
        gpt_fb_tokens = 0.0
        try:
            eval_db = EvaluationHistoryDB()
            rows = eval_db.history(module, limit=1)
            if rows:
                gpt_fb_score = float(rows[0][0])
            try:
                cur = eval_db.conn.execute(
                    "SELECT gpt_feedback_tokens FROM evaluation_history "
                    "WHERE engine=? ORDER BY ts DESC LIMIT 1",
                    (module,),
                )
                token_row = cur.fetchone()
                if token_row and token_row[0] is not None:
                    gpt_fb_tokens = float(token_row[0])
            except Exception:
                pass
            try:
                eval_db.conn.close()
            except Exception:
                pass
        except Exception:
            pass

        # Resource usage deltas from ROITracker
        res_cost = res_cpu = res_gpu = 0.0
        tracker = getattr(self, "roi_tracker", None)
        if tracker and getattr(tracker, "resource_metrics", None):
            metrics = tracker.resource_metrics
            if len(metrics) >= 2:
                prev = metrics[-2]
                curr = metrics[-1]

                def _extract(row: Sequence[float]) -> tuple[float, float, float]:
                    if len(row) >= 5:
                        cpu, _mem, _disk, cost, gpu = row[:5]
                    elif len(row) == 3:
                        cost, cpu, gpu = row
                    else:
                        cost = cpu = gpu = 0.0
                    return float(cost), float(cpu), float(gpu)

                p_cost, p_cpu, p_gpu = _extract(prev)
                c_cost, c_cpu, c_gpu = _extract(curr)
                res_cost = c_cost - p_cost
                res_cpu = c_cpu - p_cpu
                res_gpu = c_gpu - p_gpu

        # Error counts from tracker metrics
        err_count = rep_count = 0.0
        if tracker:
            try:
                errs = tracker.metrics_history.get("error_count", [])
                reps = tracker.metrics_history.get("repair_count", [])
                if errs:
                    err_count = float(errs[-1])
                if reps:
                    rep_count = float(reps[-1])
            except Exception:
                pass

        before = hist_roi
        after = hist_roi + perf_delta

        feats = [
            before,
            after,
            api_cost_delta,
            cpu_seconds_delta,
            success_rate_delta,
            gpt_score,
            gpt_fb_score,
            gpt_fb_tokens,
            0.0,  # long_term_perf_delta
            0.0,  # long_term_eval_outcome
            res_cost,
            res_cpu,
            res_gpu,
            err_count,
            rep_count,
        ]
        return [feats]

    def _score_modifications(self, modules: Iterable[str]) -> list[tuple[str, float, str, float]]:
        """Score and rank candidate modules for patching.

        Each candidate is weighted by its predicted ROI and an optional
        growth-type multiplier from :class:`AdaptiveROIPredictor`.  If the
        predictor is unavailable or fails, the candidate receives zero weight.
        Confidence scores from the ROI tracker modulate the risk-adjusted ROI
        to produce the final score used for ranking.
        """

        completed = {Path(m).as_posix() for m in self.entropy_ceiling_modules}
        scored: list[tuple[str, float, str, float]] = []
        for mod in modules:
            if Path(mod).as_posix() in completed:
                continue
            features = self._candidate_features(mod)
            roi_est = 0.0
            category = "unknown"
            if self.roi_predictor:
                try:
                    try:
                        seq, category, _, _ = self.roi_predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, category, _, _ = self.roi_predictor.predict(features)
                        seq = [float(val)]
                    if isinstance(seq, (list, tuple)) and seq:
                        roi_est = float(seq[-1])
                except Exception:
                    roi_est, category = 0.0, "unknown"
            if self.roi_tracker:
                base_roi, raroi, _ = self.roi_tracker.calculate_raroi(
                    roi_est,
                    workflow_type="standard",
                    metrics={},
                    failing_tests=sts.get_failed_critical_tests(),
                )
            else:
                base_roi, raroi, _ = roi_est, roi_est, []
            tracker = self.roi_tracker
            mult = (
                self.growth_multipliers.get(category, 1.0)
                if self.growth_weighting
                else 1.0
            )
            if tracker:
                final_score, needs_review, confidence = tracker.score_workflow(
                    mod, raroi
                )
            else:
                confidence = 1.0
                final_score, needs_review = raroi, False
            weight = final_score * mult
            if needs_review or raroi < self.borderline_raroi_threshold:
                reason = "needs_review" if needs_review else "low_raroi"
                label: str | None = None
                recs: Dict[str, str] = {}
                try:
                    cards = tracker.generate_scorecards() if tracker else []
                    label = tracker.workflow_label if tracker else None
                    recs = {
                        c.scenario: c.recommendation
                        for c in cards
                        if c.recommendation and c.roi_delta < 0.0
                    }
                except Exception:
                    pass
                self.logger.info(
                    "borderline workflow; deferring to review/shadow testing",
                    extra=log_record(
                        module=mod,
                        confidence=confidence,
                        raroi=raroi,
                        weight=weight,
                        threshold=self.tau,
                        shadow_test=True,
                        reason=reason,
                        workflow_label=label,
                        recommendations=recs,
                    ),
                )
                try:
                    self._log_action("review", mod, weight, category, confidence)
                except Exception:
                    pass
                try:
                    self.borderline_bucket.add_candidate(mod, raroi, confidence, reason)
                    settings = SandboxSettings()
                    if getattr(settings, "micropilot_mode", "") == "auto":
                        try:
                            evaluator = getattr(self, "micro_pilot_evaluator", None)
                            self.borderline_bucket.process(
                                evaluator,
                                raroi_threshold=self.borderline_raroi_threshold,
                                confidence_threshold=getattr(
                                    self.roi_tracker, "confidence_threshold", 0.0
                                ),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                continue
            scored.append((mod, base_roi, category, weight))
            self.logger.debug(
                "scored modification",
                extra=log_record(
                    module=mod,
                    base_roi=base_roi,
                    raroi=raroi,
                    confidence=confidence,
                    weight=weight,
                ),
            )
        if self.use_adaptive_roi:
            scored = [s for s in scored if s[3] > 0]
            scored.sort(key=lambda x: -x[3])
        else:
            scored = [s for s in scored if s[1] > 0]
            scored.sort(key=lambda x: -x[1])
        planner = getattr(self, "action_planner", None)
        if planner:
            try:
                planner.update_priorities({m: w for m, _, _, w in scored})
            except Exception:
                self.logger.exception("priority queue update failed")
        return scored

    def _log_action(
        self,
        action: str,
        target: str,
        roi: float,
        growth: str,
        confidence: float | None = None,
    ) -> None:
        """Persist chosen actions and ROI predictions for auditing."""
        try:
            conn = GLOBAL_ROUTER.get_connection("action_audit")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_audit(
                    ts TEXT,
                    engine TEXT,
                    action TEXT,
                    target TEXT,
                    roi REAL,
                    growth TEXT,
                    confidence REAL
                )
                """
            )
            conn.execute(
                "INSERT INTO action_audit(ts, engine, action, target, roi, growth, confidence) VALUES(?,?,?,?,?,?,?)",
                (
                    datetime.utcnow().isoformat(),
                    self.bot_name,
                    action,
                    target,
                    float(roi),
                    growth,
                    float(confidence or 0.0),
                ),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("failed to log action audit: %s", exc)

    # ------------------------------------------------------------------
    def _should_trigger(self) -> bool:
        if getattr(self, "_force_rerun", False):
            self._force_rerun = False
            return True
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
                forecast = self.pre_roi_bot.predict_model_roi(self.bot_name, [])
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
                    self.logger.exception("failed to record error item: %s", exc)

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
                        self.logger.exception("persist_evaluation failed: %s", exc)
            else:
                X, y = self.learning_engine._dataset()  # type: ignore[attr-defined]
                if not X or len(set(y)) < 2:
                    return
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(self.learning_engine.model, X, y, cv=3)
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
                        self.logger.exception("persist_evaluation failed: %s", exc)
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

    def _evaluate_roi_predictor(self) -> None:
        """Evaluate and periodically retrain the adaptive ROI predictor."""
        if not self.roi_predictor:
            return
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return
        mae_threshold = 0.1
        acc_threshold = 0.6
        try:
            acc, mae = self.roi_predictor.evaluate_model(
                tracker,
                mae_threshold=mae_threshold,
                acc_threshold=acc_threshold,
            )
            try:
                self.roi_predictor.record_drift(acc, mae)
            except Exception:
                pass
            self.logger.info(
                "adaptive roi evaluation",
                extra=log_record(accuracy=float(acc), mae=float(mae)),
            )
            drift_metrics = getattr(self.roi_predictor, "drift_metrics", None)
            if drift_metrics:
                try:
                    self.logger.info(
                        "adaptive roi drift",
                        extra=log_record(
                            **{k: float(v) for k, v in drift_metrics.items()}
                        ),
                    )
                except Exception:
                    pass
            try:  # pragma: no cover - best effort telemetry
                prediction_mae.labels(metric="adaptive_roi").set(float(mae))
                prediction_reliability.labels(metric="adaptive_roi").set(float(acc))
            except Exception:
                pass
            if mae > mae_threshold or acc < acc_threshold:
                self.logger.info(
                    "adaptive roi model drift detected",
                    extra=log_record(accuracy=float(acc), mae=float(mae)),
                )
                try:
                    self.roi_predictor.partial_fit()
                except Exception:
                    self.logger.exception("adaptive roi partial_fit failed")
                    try:
                        self.roi_predictor.train()
                    except Exception:
                        self.logger.exception("adaptive roi training failed")
        except Exception as exc:  # pragma: no cover - evaluation failure
            self.logger.exception("adaptive roi evaluation failed: %s", exc)
            acc = mae = 0.0

        now = time.time()
        if (
            len(tracker.roi_history)
            > getattr(self.roi_predictor, "_trained_size", 0)
            and now - self._adaptive_roi_last_train
            >= self.adaptive_roi_train_interval
        ):
            try:
                load_training_data(tracker)
                self.logger.info("adaptive roi training data loaded")
                selected = getattr(self.roi_predictor, "selected_features", None)
                X, y, g, names = build_dataset(
                    evolution_path="evolution_history.db",
                    roi_path="roi.db",
                    evaluation_path="evaluation_history.db",
                    selected_features=selected,
                    return_feature_names=True,
                )
                dataset = (X, y, g)
                self.roi_predictor.train(
                    dataset,
                    cv=self.roi_predictor.cv,
                    param_grid=self.roi_predictor.param_grid,
                    feature_names=names,
                )
                self._adaptive_roi_last_train = now
                msg = f"adaptive roi model retrained on {len(dataset[0])} samples"
                self.logger.info(msg)
                val_scores = getattr(self.roi_predictor, "validation_scores", {}) or {}
                for name, score in val_scores.items():
                    self.logger.info("validation MAE %s: %.4f", name, score)
                best_params = getattr(self.roi_predictor, "best_params", None)
                best_score = getattr(self.roi_predictor, "best_score", None)
                if best_params and best_score is not None:
                    params = {k: v for k, v in best_params.items() if k != "model"}
                    self.logger.info(
                        "best model %s (MAE=%.4f)",
                        best_params.get("model"),
                        best_score,
                    )
                    if params:
                        self.logger.info("best params: %s", params)
            except Exception as exc:  # pragma: no cover - retrain failure
                self.logger.exception(
                    "adaptive roi scheduled retrain failed: %s", exc
                )

    def fit_truth_adapter(self, X: np.ndarray, y: np.ndarray) -> None:
        """Retrain :class:`TruthAdapter` with new data."""
        try:
            self.truth_adapter.fit(X, y)
            self._truth_adapter_needs_retrain = False
        except Exception:
            self.logger.exception("truth adapter fit failed")

    def _record_warning_summary(
        self, delta: float, warnings: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Store high-severity warnings that coincide with positive ROI."""
        if delta <= 0:
            return
        if warnings.get("ethics") or warnings.get("risk_reward"):
            entry = {"roi_delta": delta, "warnings": warnings}
            self.warning_summary.append(entry)
            self.logger.warning(
                "improvement flagged with high severity warnings",
                extra=log_record(**entry),
            )

    def get_warning_summary(self) -> list[dict[str, Any]]:
        """Return summary entries for high-severity warning improvements."""
        return list(self.warning_summary)

    def _log_improvement_warnings(
        self, warnings: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Write individual improvement warnings to the violation log."""
        for category, entries in warnings.items():
            for idx, warn in enumerate(entries):
                sev = int(warn.get("severity", 1))
                evidence: dict[str, Any] = {
                    "category": category,
                    "file": warn.get("file"),
                    "issue": warn.get("issue") or warn.get("message"),
                }
                if "entry" in warn:
                    evidence["entry"] = warn["entry"]
                if "violations" in warn:
                    evidence["violations"] = warn["violations"]
                if "snippet" in warn:
                    evidence["snippet"] = warn["snippet"]
                if "snippets" in warn:
                    evidence["snippets"] = warn["snippets"]
                log_violation(
                    f"improvement_{self._cycle_count}_{category}_{idx}",
                    "alignment_warning",
                    sev,
                    evidence,
                    alignment_warning=True,
                )

    def _pre_commit_alignment_check(
        self, diff_data: dict[str, dict[str, list[str]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Run ``flag_improvement`` on proposed changes before commit."""
        settings = SandboxSettings()
        if not getattr(settings, "enable_alignment_flagger", True):
            return {}
        workflow_changes: list[dict[str, Any]] = []
        for file, diff in diff_data.items():
            added = diff.get("added", [])
            removed = diff.get("removed", [])
            diff_text = "\n".join(["+" + ln for ln in added] + ["-" + ln for ln in removed])
            workflow_changes.append(
                {"file": file, "code": "\n".join(added), "diff": diff_text}
            )
        metrics = getattr(self, "_last_scenario_metrics", {})
        warnings: dict[str, list[dict[str, Any]]] = {}
        try:
            warnings = flag_improvement(workflow_changes, metrics, [], None, settings)
        except Exception:
            warnings = {}
        if any(warnings.values()):
            self._log_improvement_warnings(warnings)
            high = any(
                int(w.get("severity", 1)) >= 3
                for cat in warnings.values()
                for w in cat
            )
            if high:
                try:
                    if self._alignment_agent is None:
                        self._alignment_agent = AlignmentReviewAgent()
                        self._alignment_agent.start()
                    self._alignment_agent.auditor.audit({"warnings": warnings})
                except Exception:
                    self.logger.exception("alignment review agent audit failed")
        return warnings

    @radar.track
    def _optimize_self(self) -> tuple[int | None, bool, float]:
        """Apply a patch to this engine via :class:`SelfCodingEngine`."""
        if not self.self_coding_engine:
            return None, False, 0.0
        # Consult ROI predictor before applying a self optimisation patch.
        try:
            roi_est = 0.0
            growth = "unknown"
            weight = 0.0
            predictor = getattr(self, "roi_predictor", None)
            use_adaptive = getattr(self, "use_adaptive_roi", False)
            tracker = getattr(self, "roi_tracker", None)
            bot_name = getattr(self, "bot_name", "")
            confidence = 1.0
            if predictor and use_adaptive:
                try:
                    features = self._candidate_features(bot_name)
                except Exception:
                    features = [[0.0, 0.0, 0.0]]
                try:
                    try:
                        seq, growth, _conf, _ = predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, growth, _conf, _ = predictor.predict(features)
                        seq = [float(val)]
                    roi_est = float(seq[-1]) if seq else 0.0
                except Exception:
                    roi_est, growth = 0.0, "unknown"
                if tracker:
                    base_roi, raroi, _ = tracker.calculate_raroi(
                        roi_est,
                        workflow_type="standard",
                        metrics={},
                        failing_tests=sts.get_failed_critical_tests(),
                    )
                else:
                    base_roi, raroi, _ = roi_est, roi_est, []
                if tracker:
                    final_score, needs_review, confidence = tracker.score_workflow(
                        bot_name, raroi
                    )
                else:
                    confidence = 1.0
                    final_score, needs_review = raroi, False
                if needs_review or raroi < self.borderline_raroi_threshold:
                    reason = "needs_review" if needs_review else "low_raroi"
                    label: str | None = None
                    recs: Dict[str, str] = {}
                    try:
                        cards = tracker.generate_scorecards() if tracker else []
                        label = tracker.workflow_label if tracker else None
                        recs = {
                            c.scenario: c.recommendation
                            for c in cards
                            if c.recommendation and c.roi_delta < 0.0
                        }
                    except Exception:
                        pass
                    self.logger.info(
                        "self optimisation deferred: borderline",
                        extra=log_record(
                            growth_type=growth,
                            roi_estimate=base_roi,
                            raroi=raroi,
                            confidence=confidence,
                            final_score=final_score,
                            reason=reason,
                            workflow_label=label,
                            recommendations=recs,
                        ),
                    )
                    try:
                        self._log_action("review", bot_name, final_score, growth, confidence)
                    except Exception:
                        pass
                    try:
                        self.borderline_bucket.add_candidate(bot_name, raroi, confidence, reason)
                        settings = SandboxSettings()
                        if getattr(settings, "micropilot_mode", "") == "auto":
                            try:
                                evaluator = getattr(self, "micro_pilot_evaluator", None)
                                self.borderline_bucket.process(
                                    evaluator,
                                    raroi_threshold=self.borderline_raroi_threshold,
                                    confidence_threshold=getattr(
                                        self.roi_tracker, "confidence_threshold", 0.0
                                    ),
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return None, False, 0.0
                mult = (
                    self.growth_multipliers.get(growth, 1.0)
                    if self.growth_weighting
                    else 1.0
                )
                weight = final_score * mult
                if tracker:
                    tracker._next_prediction = base_roi
                    tracker._next_category = growth
                if weight <= 0:
                    self.logger.info(
                        "self optimisation skipped",
                        extra=log_record(
                            growth_type=growth,
                            roi_estimate=base_roi,
                            raroi=raroi,
                            weight=weight,
                            confidence=confidence,
                            final_score=final_score,
                        ),
                    )
                    return None, False, 0.0

            with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
                src = Path(__file__).resolve()
                try:
                    module_rel = src.relative_to(repo).as_posix()
                except Exception:
                    module_rel = src.name
                rel = src.name
                before_target = Path(before_dir) / rel
                before_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, before_target)
                start_time = time.perf_counter()
                patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                    Path(__file__),
                    "self_improvement",
                    parent_patch_id=self._last_patch_id,
                    reason="self_improvement",
                    trigger="optimize_self",
                )
                if patch_id is not None and not reverted:
                    after_target = Path(after_dir) / rel
                    after_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, after_target)
                    diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                    self._pre_commit_alignment_check(diff_data)
            before_metric = 0.0
            after_metric = delta
            if self.self_coding_engine.patch_db and patch_id is not None:
                try:
                    with self.self_coding_engine.patch_db._connect() as conn:
                        row = conn.execute(
                            "SELECT roi_before, roi_after FROM patch_history WHERE id=?",
                            (patch_id,),
                        ).fetchone()
                    if row:
                        before_metric = float(row[0])
                        after_metric = float(row[1])
                except Exception:
                    after_metric = before_metric + delta
            else:
                after_metric = before_metric + delta
            with MutationLogger.log_context(
                change=f"self_opt_patch_{patch_id}",
                reason="self_improvement",
                trigger="optimize_self",
                workflow_id=0,
                before_metric=before_metric,
                parent_id=self._last_mutation_id,
            ) as mutation:
                mutation["after_metric"] = after_metric
                mutation["performance"] = delta
                mutation["roi"] = after_metric
            self._last_mutation_id = int(mutation["event_id"])
            self._last_patch_id = patch_id
            if patch_id is not None and not reverted:
                self._alignment_review_last_commit(f"self_opt_patch_{patch_id}")
                self._flag_patch_alignment(
                    patch_id, {"trigger": "optimize_self", "patch_id": patch_id}
                )
            roi_delta = after_metric - before_metric
            if tracker:
                try:
                    tracker.update(
                        before_metric,
                        after_metric,
                        modules=[module_rel],
                        category=growth,
                        confidence=confidence,
                    )
                except Exception:
                    self.logger.exception("roi tracker update failed")
            if self.metrics_db:
                try:
                    elapsed = time.perf_counter() - start_time
                    self.metrics_db.record(module_rel, elapsed, roi_delta=roi_delta)
                except Exception:
                    self.logger.exception("relevancy metrics record failed")
            if predictor and use_adaptive:
                self._log_action(
                    "self_optimize", bot_name, roi_est, growth, confidence
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
                self.logger.exception("failed to process pathway record: %s", exc)

    def _compress_module(self, module_path: Path) -> Path | None:
        """Compress ``module_path`` into a zip archive.

        The resulting archive is stored under ``SANDBOX_DATA_DIR`` in the
        ``compressed_modules`` directory. ``None`` is returned on failure.
        """

        try:
            if not module_path.exists():
                return None
            out_dir = (
                Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
                / "compressed_modules"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            archive_base = out_dir / module_path.stem
            archive = shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=module_path.parent,
                base_dir=module_path.name,
            )
            return Path(archive)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _test_orphan_modules(self, paths: Iterable[str]) -> set[str]:
        """Run self tests for ``paths`` and return modules that succeed.

        Modules are classified via :func:`orphan_analyzer.classify_module` and
        the classification is stored in ``self.orphan_traces`` as well as the
        ``orphan_classifications.json`` cache. Modules classified as
        ``legacy`` or ``redundant`` are skipped unless
        :attr:`SandboxSettings.test_redundant_modules` is enabled. Remaining
        modules are executed via :class:`SelfTestService` with orphan discovery
        enabled. Basic metrics about the run are logged and stored via
        ``data_bot`` when available. Actual integration of passing modules is
        handled by the caller.
        """

        all_modules = [str(p) for p in paths]
        if not all_modules:
            return set()

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        meta_path = data_dir / "orphan_classifications.json"
        orphan_path = data_dir / "orphan_modules.json"

        try:  # pragma: no cover - dynamic import
            from self_test_service import SelfTestService as _STS
        except Exception:  # pragma: no cover - service unavailable
            _STS = None

        classifications: dict[str, dict[str, Any]] = {}
        candidates: list[str] = []
        legacy: list[str] = []
        redundant: list[str] = []

        traces = getattr(self, "orphan_traces", None)
        if traces is None:
            traces = {}
            setattr(self, "orphan_traces", traces)

        threshold = SandboxSettings().side_effect_threshold

        for m in all_modules:
            start = time.perf_counter()
            path = Path(m)
            abs_path = path if path.is_absolute() else repo / path
            try:
                res = classify_module(abs_path, include_meta=True)
                cls, meta = res if isinstance(res, tuple) else (res, {})
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", abs_path, exc)
                cls, meta = "candidate", {}
            score = float(meta.get("side_effects", 0))
            try:
                rel = abs_path.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = str(abs_path)
            info = traces.setdefault(rel, {"parents": []})
            prev_cls = info.get("classification")
            info["classification"] = cls
            info["redundant"] = cls != "candidate"
            info["side_effects"] = score
            classifications[rel] = {"classification": cls, "side_effects": score}
            elapsed = time.perf_counter() - start
            if getattr(self, "metrics_db", None):
                try:
                    self.metrics_db.record(
                        rel, elapsed, self.module_index, roi_delta=0.0
                    )
                except Exception:
                    self.logger.exception(
                        "relevancy metrics record failed", extra=log_record(module=rel)
                    )
            if score > threshold:
                info["heavy_side_effects"] = True
                self.logger.info(
                    "orphan module skipped due to side effects",
                    extra=log_record(module=rel, side_effects=score),
                )
                try:
                    existing = (
                        json.loads(orphan_path.read_text())
                        if orphan_path.exists()
                        else {}
                    )
                except Exception:
                    existing = {}
                if not isinstance(existing, dict):
                    existing = {}
                entry = existing.get(rel, {})
                entry["reason"] = "heavy_side_effects"
                entry["side_effects"] = score
                existing[rel] = entry
                try:
                    orphan_path.parent.mkdir(parents=True, exist_ok=True)
                    orphan_path.write_text(json.dumps(existing, indent=2))
                except Exception:
                    pass
                continue
            if cls == "legacy":
                legacy.append(rel)
                self.logger.info(
                    "orphan module classified",
                    extra=log_record(module=rel, classification="legacy"),
                )
            elif cls == "redundant":
                redundant.append(rel)
                self.logger.info(
                    "orphan module classified",
                    extra=log_record(module=rel, classification="redundant"),
                )
            else:
                candidates.append(rel)
            try:
                if prev_cls == "legacy" and cls != "legacy":
                    orphan_modules_legacy_total.dec(1)
                elif prev_cls != "legacy" and cls == "legacy":
                    orphan_modules_legacy_total.inc(1)
            except Exception:
                pass

        try:
            existing_meta = (
                json.loads(meta_path.read_text()) if meta_path.exists() else {}
            )
        except Exception:  # pragma: no cover - best effort
            existing_meta = {}
        existing_meta.update(classifications)
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(existing_meta, indent=2))
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan classifications")

        try:
            settings = SandboxSettings()
        except Exception:  # pragma: no cover - fallback for tests
            from sandbox_settings import SandboxSettings as _SS  # type: ignore

            settings = _SS()

        past_results: dict[str, dict[str, Any]] = {}
        if _STS is not None:
            try:
                past_results = _STS.orphan_summary()
            except Exception:
                past_results = {}
        if past_results:

            def _ok(m: str) -> bool:
                return past_results.get(m, {}).get("roi_delta", 0.0) >= 0

            candidates[:] = [m for m in candidates if _ok(m)]
            legacy[:] = [m for m in legacy if _ok(m)]
            redundant[:] = [m for m in redundant if _ok(m)]

        radar_flags: dict[str, str] = {}
        try:  # pragma: no cover - optional dependency
            from relevancy_radar import load_usage_stats, evaluate_relevancy

            usage_stats = load_usage_stats()
            if usage_stats:
                radar_flags = evaluate_relevancy(
                    {m: 1 for m in all_modules}, usage_stats
                )
        except Exception:
            radar_flags = {}

        retire_modules: list[str] = []
        compress_modules: list[str] = []
        replace_modules: list[str] = []
        for mod, flag in radar_flags.items():
            info = self.orphan_traces.setdefault(mod, {"parents": []})
            info["radar_flag"] = flag
            if flag == "retire":
                retire_modules.append(mod)
            elif flag == "compress":
                compress_modules.append(mod)
            elif flag == "replace":
                replace_modules.append(mod)

        audit_fn = globals().get("audit_log_event")
        gen_patch = globals().get("generate_patch")

        for mod in retire_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            try:
                cls = classify_module(path)
            except Exception:
                cls = "error"
            if audit_fn:
                audit_fn("radar_retire", {"module": mod, "classification": cls})

        for mod in compress_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            archive = None
            compress_fn = getattr(self, "_compress_module", None)
            if callable(compress_fn):
                archive = compress_fn(path)
            if audit_fn:
                audit_fn(
                    "radar_compress",
                    {"module": mod, "archive": str(archive) if archive else None},
                )

        for mod in replace_modules:
            path = repo / mod if not Path(mod).is_absolute() else Path(mod)
            try:
                cls = classify_module(path)
            except Exception:
                cls = "error"
            patch_id = None
            if callable(gen_patch):
                try:
                    patch_id = gen_patch(str(path))
                except Exception:
                    patch_id = None
            if audit_fn:
                audit_fn(
                    "radar_replace",
                    {"module": mod, "classification": cls, "patch_id": patch_id},
                )

        if not getattr(settings, "test_redundant_modules", False):
            retired = {m for m, f in radar_flags.items() if f == "retire"}
            if retired:
                candidates[:] = [m for m in candidates if m not in retired]
                legacy[:] = [m for m in legacy if m not in retired]
                redundant[:] = [m for m in redundant if m not in retired]

        modules = (
            [*candidates, *legacy, *redundant]
            if getattr(settings, "test_redundant_modules", False)
            else [*candidates]
        )
        self.logger.info(
            "self test start",
            extra=log_record(
                modules=sorted(modules),
                legacy=sorted(legacy),
                redundant=sorted(redundant),
            ),
        )

        if not modules:
            counts = {
                "orphan_modules_tested": 0,
                "orphan_modules_passed": 0,
                "orphan_modules_failed": 0,
                "orphan_modules_reclassified": 0,
                "orphan_modules_redundant": len(redundant),
                "orphan_modules_legacy": len(legacy),
            }
            self._last_orphan_counts = counts
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.register_metrics(*counts.keys())
                base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                tracker.update(base, base, metrics=counts)
            try:
                orphan_modules_tested_total.inc(0)
                orphan_modules_passed_total.inc(0)
                orphan_modules_failed_total.inc(0)
                orphan_modules_reclassified_total.inc(0)
                orphan_modules_redundant_total.inc(len(redundant))
                orphan_modules_legacy_total.inc(len(legacy))
            except Exception:
                pass
            return set()

        try:
            environment.auto_include_modules(
                sorted(modules), recursive=True, validate=True
            )
        except Exception:
            try:
                environment.auto_include_modules(sorted(modules), recursive=True)
            except Exception:
                pass

        reuse_scores: dict[str, float] = {}
        try:
            from module_index_db import ModuleIndexDB
            from task_handoff_bot import WorkflowDB

            idx = getattr(self, "module_index", None) or ModuleIndexDB()
            wf_db = WorkflowDB(Path(os.getenv("WORKFLOWS_DB", "workflows.db")))
            workflows = wf_db.fetch(limit=1000)
            total_wf = len(workflows)
            grp_counts: dict[int, int] = {}
            if total_wf:
                for wf in workflows:
                    seen: set[int] = set()
                    for step in getattr(wf, "workflow", []):
                        mod = step.split(":")[0]
                        file = repo / (mod.replace(".", "/") + ".py")
                        try:
                            gid = idx.get(file.as_posix())
                        except Exception:
                            try:
                                gid = idx.get(mod)
                            except Exception:
                                continue
                        seen.add(gid)
                    for g in seen:
                        grp_counts[g] = grp_counts.get(g, 0) + 1
            for mod in modules:
                try:
                    gid = idx.get(mod)
                except Exception:
                    reuse_scores[mod] = 0.0
                    continue
                reuse_scores[mod] = (
                    grp_counts.get(gid, 0) / total_wf if total_wf else 0.0
                )
        except Exception:
            reuse_scores = {m: 0.0 for m in modules}

        for mod, score in reuse_scores.items():
            info = self.orphan_traces.setdefault(mod, {"parents": []})
            info["reuse_score"] = score

        try:
            reuse_threshold = float(os.getenv("ORPHAN_REUSE_THRESHOLD", "0"))
        except Exception:
            reuse_threshold = 0.0

        scenario_results: dict[str, dict[str, dict[str, Any]]] = {}
        tracker_wf = None
        try:
            from environment_generator import generate_canonical_presets

            canonical = generate_canonical_presets()
            flat_presets = [p for levels in canonical.values() for p in levels.values()]
            env_map = {m: flat_presets for m in modules}
            tracker_wf, wf_details = environment.run_workflow_simulations(
                env_presets=env_map, return_details=True
            )
            scenario_synergy = getattr(tracker_wf, "scenario_synergy", {})
            for runs in wf_details.values():
                for entry in runs:
                    module = entry.get("module")
                    preset = entry.get("preset", {})
                    scen = preset.get("SCENARIO_NAME")
                    res = entry.get("result", {})
                    if not module or module not in modules or not scen:
                        continue
                    metrics = {
                        k: float(v)
                        for k, v in res.items()
                        if k != "exit_code" and isinstance(v, (int, float))
                    }
                    sy_list = scenario_synergy.get(scen, [])
                    try:
                        scen_roi = (
                            float(sy_list[-1].get("synergy_roi", 0.0))
                            if sy_list
                            else 0.0
                        )
                    except Exception:
                        scen_roi = 0.0
                    scen_map = scenario_results.setdefault(module, {})
                    scen_map[scen] = {
                        "roi": scen_roi,
                        "metrics": metrics,
                        "failed": bool(res.get("exit_code")),
                    }
            for mod, scen_map in scenario_results.items():
                trace = self.orphan_traces.setdefault(mod, {"parents": []})
                trace.setdefault("scenarios", {}).update(scen_map)
                rois = [v.get("roi", 0.0) for v in scen_map.values()]
                trace["workflow_robustness"] = float(min(rois)) if rois else 0.0
                trace["workflow_failed"] = any(v.get("failed") for v in scen_map.values())
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("workflow simulation failed: %s", exc)
            tracker_wf = None

        if _STS is None:
            try:
                from sandbox_runner import run_repo_section_simulations as _sim
            except Exception:  # pragma: no cover - best effort
                return set()
            passed: set[str] = set()
            for m in modules:
                try:
                    tracker_res, details = _sim(
                        str(repo), modules=[m], return_details=True
                    )
                except Exception:
                    continue
                scenarios = details.get(m, {})
                scenario_synergy = getattr(tracker_res, "scenario_synergy", {})
                info = self.orphan_traces.setdefault(m, {"parents": []})
                scen_info = info.setdefault("scenarios", {})
                worst_roi = math.inf
                scenario_failed = False
                try:
                    from environment_generator import CANONICAL_PROFILES
                except Exception:  # pragma: no cover - optional dependency
                    CANONICAL_PROFILES = []  # type: ignore[assignment]

                for scen, runs in scenarios.items():
                    names = [s.strip() for s in scen.split("+") if s.strip()]
                    sy_list = scenario_synergy.get(scen, [])
                    try:
                        scen_roi = (
                            float(sy_list[-1].get("synergy_roi", 0.0))
                            if sy_list
                            else 0.0
                        )
                    except Exception:
                        scen_roi = 0.0
                    failed = any(r.get("result", {}).get("exit_code") for r in runs)
                    worst_roi = min(worst_roi, scen_roi)
                    scenario_failed = scenario_failed or failed
                    recorded = False
                    for name in names:
                        if name in CANONICAL_PROFILES:
                            scen_info[name] = {"roi": scen_roi, "failed": failed}
                            recorded = True
                    if not recorded:
                        scen_info[scen] = {"roi": scen_roi, "failed": failed}

                if worst_roi is math.inf:
                    worst_roi = 0.0
                info["robustness"] = float(worst_roi)
                if scenario_failed or worst_roi < 0.0:
                    self.logger.info(
                        "self tests failed",
                        extra=log_record(module=m, robustness=worst_roi),
                    )
                    continue
                roi_total = 0.0
                try:
                    roi_total = sum(tracker_res.module_deltas.get(m, []))
                except Exception:
                    pass
                if roi_total < 0:
                    continue
                if reuse_scores.get(m, 0.0) < reuse_threshold:
                    continue
                passed.add(m)
            reclassified = {m for m in passed if m in legacy or m in redundant}
            for name in sorted(reclassified):
                classifications[name] = {"classification": "candidate"}
                info = self.orphan_traces.setdefault(name, {"parents": []})
                info["classification"] = "candidate"
                info["redundant"] = False
                if name in legacy:
                    legacy.remove(name)
                if name in redundant:
                    redundant.remove(name)
            failed_mods = sorted(
                m for m in modules if m not in passed and m not in redundant
            )
            self.logger.info(
                "self test summary",
                extra=log_record(
                    tested=sorted(modules),
                    passed=sorted(passed),
                    failed=failed_mods,
                    redundant=sorted(redundant),
                    legacy=sorted(legacy),
                    reuse_scores={m: reuse_scores.get(m, 0.0) for m in modules},
                ),
            )
            counts = {
                "orphan_modules_tested": len(modules),
                "orphan_modules_passed": len(passed),
                "orphan_modules_failed": len(failed_mods),
                "orphan_modules_reclassified": len(reclassified),
                "orphan_modules_redundant": len(redundant),
                "orphan_modules_legacy": len(legacy),
            }
            self._last_orphan_counts = counts
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.register_metrics(*counts.keys())
                base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                tracker.update(base, base, metrics=counts)
            try:
                orphan_modules_tested_total.inc(len(modules))
                orphan_modules_passed_total.inc(len(passed))
                orphan_modules_failed_total.inc(len(failed_mods))
                orphan_modules_reclassified_total.inc(len(reclassified))
                orphan_modules_redundant_total.inc(len(redundant))
                orphan_modules_legacy_total.inc(len(legacy))
            except Exception:
                pass
            try:
                existing_meta = (
                    json.loads(meta_path.read_text()) if meta_path.exists() else {}
                )
            except Exception:  # pragma: no cover - best effort
                existing_meta = {}
            existing_meta.update(classifications)
            try:
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(existing_meta, indent=2))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to write orphan classifications")
            return passed

        settings = SandboxSettings()
        svc = _STS(
            pytest_args=" ".join(modules),
            include_orphans=True,
            discover_orphans=True,
            discover_isolated=True,
            recursive_orphans=settings.recursive_orphan_scan,
            recursive_isolated=settings.recursive_isolated,
            auto_include_isolated=True,
            include_redundant=settings.test_redundant_modules,
            clean_orphans=True,
            disable_auto_integration=True,
        )

        try:
            asyncio.run(svc._run_once())
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("self test execution failed: %s", exc)
            return set()

        results = svc.results or {}
        integration = results.get("integration")
        if isinstance(integration, dict):
            passing = set(integration.get("integrated", []))
            new_redundant = set(integration.get("redundant", []))
        else:
            passed = set(results.get("orphan_passed", []))
            new_redundant = set(results.get("orphan_redundant", []))
            passing = {p for p in passed if p not in new_redundant}

        passing = {p for p in passing if reuse_scores.get(p, 0.0) >= reuse_threshold}

        filtered: set[str] = set()
        tracker_res = tracker_wf
        for p in passing:
            info = self.orphan_traces.setdefault(p, {"parents": []})
            worst_roi = min(
                float(info.get("robustness", 0.0)),
                float(info.get("workflow_robustness", 0.0)),
            )
            scen_failed = bool(info.get("workflow_failed")) or any(
                v.get("failed") for v in info.get("scenarios", {}).values()
            )
            if scen_failed or worst_roi < 0.0:
                continue
            roi_total = 0.0
            if tracker_res:
                for key, vals in tracker_res.module_deltas.items():
                    if key.startswith(p):
                        roi_total += sum(float(v) for v in vals)
            if roi_total < 0:
                continue
            filtered.add(p)
        passing = filtered

        if new_redundant:
            for name in new_redundant:
                classifications[name] = {"classification": "redundant"}
                info = self.orphan_traces.setdefault(name, {"parents": []})
                info["classification"] = "redundant"
                info["redundant"] = True
            redundant.extend(sorted(new_redundant))
            passing -= new_redundant

        reclassified = {m for m in passing if m in legacy or m in redundant}
        for name in sorted(reclassified):
            classifications[name] = {"classification": "candidate"}
            info = self.orphan_traces.setdefault(name, {"parents": []})
            info["classification"] = "candidate"
            info["redundant"] = False
            if name in legacy:
                legacy.remove(name)
            if name in redundant:
                redundant.remove(name)

        failed_mods = [m for m in modules if m not in passing and m not in redundant]
        self.logger.info(
            "self test summary",
            extra=log_record(
                tested=sorted(modules),
                passed=sorted(passing),
                failed=sorted(failed_mods),
                redundant=sorted(redundant),
                legacy=sorted(legacy),
                passed_count=len(passing),
                failed_count=len(failed_mods),
                redundant_count=len(redundant),
                legacy_count=len(legacy),
                reuse_scores={m: reuse_scores.get(m, 0.0) for m in modules},
            ),
        )

        counts = {
            "orphan_modules_tested": len(modules),
            "orphan_modules_passed": len(passing),
            "orphan_modules_failed": len(failed_mods),
            "orphan_modules_reclassified": len(reclassified),
            "orphan_modules_redundant": len(redundant),
            "orphan_modules_legacy": len(legacy),
        }
        self._last_orphan_counts = counts
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            tracker.register_metrics(*counts.keys())
            base = tracker.roi_history[-1] if tracker.roi_history else 0.0
            tracker.update(base, base, metrics=counts)
        try:
            orphan_modules_tested_total.inc(len(modules))
            orphan_modules_passed_total.inc(len(passing))
            orphan_modules_failed_total.inc(len(failed_mods))
            orphan_modules_reclassified_total.inc(len(reclassified))
            orphan_modules_redundant_total.inc(len(redundant))
            orphan_modules_legacy_total.inc(len(legacy))
        except Exception:
            pass

        if self.data_bot and getattr(self.data_bot, "metrics_db", None):
            try:
                cycle = datetime.utcnow().isoformat()
                db = self.data_bot.metrics_db
                db.log_eval(cycle, "self_test_passed", float(len(passing)))
                if redundant:
                    db.log_eval(cycle, "self_test_redundant", float(len(redundant)))
                if legacy:
                    db.log_eval(cycle, "self_test_legacy", float(len(legacy)))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to record self test metrics")

        try:
            existing_meta = (
                json.loads(meta_path.read_text()) if meta_path.exists() else {}
            )
        except Exception:  # pragma: no cover - best effort
            existing_meta = {}
        existing_meta.update(classifications)
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(existing_meta, indent=2))
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan classifications")

        return passing

    # ------------------------------------------------------------------
    def _integrate_orphans(self, paths: Iterable[str]) -> set[str]:
        """Integrate tested orphan modules and refresh module mappings.

        Candidate modules from ``paths`` are merged with isolates discovered by
        :mod:`scripts.discover_isolated_modules` and expanded through
        :func:`collect_local_dependencies`. The resulting module set is passed to
        :func:`environment.auto_include_modules` with ``recursive=True`` and
        ``validate=True``. After inclusion, ``module_index`` and
        ``module_clusters`` are refreshed, orphan tracking files are cleaned and a
        new orphan discovery round is scheduled so that newly uncovered
        dependencies are evaluated automatically.

        Returns
        -------
        set[str]
            Names of modules that were successfully integrated.
        """

        if not self.module_index:
            return set()

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))

        candidates: set[str] = set()
        for p in paths:
            path = Path(p)
            try:
                rel = path.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = path.name
            candidates.add(rel)

        try:
            from scripts.discover_isolated_modules import discover_isolated_modules

            iso = discover_isolated_modules(str(repo), recursive=True)
            candidates.update(iso)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("isolated module discovery failed: %s", exc)

        try:
            from sandbox_runner.dependency_utils import collect_local_dependencies

            expanded = collect_local_dependencies(sorted(candidates))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("dependency expansion failed: %s", exc)
            expanded = set(candidates)

        try:
            settings = SandboxSettings()
        except Exception:
            try:
                from sandbox_settings import SandboxSettings as _SS  # type: ignore

                settings = _SS()  # type: ignore
            except Exception:  # pragma: no cover - last resort
                settings = type("_SS", (), {"test_redundant_modules": False})()

        traces = getattr(self, "orphan_traces", {})
        mods: set[str] = set()
        legacy = 0
        redundant = 0
        for rel in expanded:
            path = repo / rel
            info = traces.get(rel, {})
            cls = info.get("classification")
            if cls is None:
                try:
                    cls = classify_module(path)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("classification failed for %s: %s", path, exc)
                    cls = "candidate"
                info["classification"] = cls
                traces[rel] = info
            if cls == "legacy":
                self.logger.info(
                    "redundant module skipped",
                    extra=log_record(module=rel, classification=cls),
                )
                legacy += 1
                continue
            if cls == "redundant":
                allow = getattr(settings, "test_redundant_modules", False)
                if not allow:
                    self.logger.info(
                        "redundant module skipped",
                        extra=log_record(module=rel, classification=cls),
                    )
                    redundant += 1
                    continue
                redundant += 1
            mods.add(rel)

        try:
            if legacy:
                orphan_modules_legacy_total.inc(legacy)
            if redundant:
                orphan_modules_redundant_total.inc(redundant)
        except Exception:
            pass

        unknown = [m for m in mods if m not in self.module_clusters]
        if not unknown:
            return set()

        try:
            self.module_index.refresh(mods, force=True)
            self.module_index.save()
            self._last_map_refresh = time.time()
            try:
                environment.auto_include_modules(
                    sorted(mods), recursive=True, validate=True
                )
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("auto inclusion failed: %s", exc)
            updated_wfs: list[int] = []
            try:
                updated_wfs = (
                    environment.try_integrate_into_workflows(sorted(mods)) or []
                )
            except Exception:  # pragma: no cover - best effort
                updated_wfs = []
            if updated_wfs:
                try:
                    self.logger.info(
                        "workflows updated",
                        extra=log_record(modules=sorted(mods), workflows=updated_wfs),
                    )
                except Exception:
                    pass
                for m in mods:
                    info = traces.setdefault(m, {})
                    info.setdefault("workflows", [])
                    info["workflows"].extend(updated_wfs)
            counts = getattr(self, "_last_orphan_counts", {})
            counts["workflows_updated"] = len(updated_wfs)
            self._last_orphan_counts = counts

            grp_map = {m: self.module_index.get(m) for m in mods}
            for m, idx in grp_map.items():
                self.module_clusters[m] = idx

            try:
                self._update_orphan_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("recursive orphan update failed: %s", exc)

            data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
            orphan_path = data_dir / "orphan_modules.json"
            meta_path = data_dir / "orphan_classifications.json"
            survivors = [
                m
                for m, info in traces.items()
                if not info.get("redundant") and m not in mods
            ]
            try:
                if orphan_path.exists() or survivors:
                    orphan_path.parent.mkdir(parents=True, exist_ok=True)
                    orphan_path.write_text(json.dumps(sorted(survivors), indent=2))
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to clean orphan modules")
            try:
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                else:
                    meta = {}
                changed = False
                for m in mods:
                    if m in meta:
                        meta.pop(m, None)
                        changed = True
                if changed:
                    if meta:
                        meta_path.write_text(json.dumps(meta, indent=2))
                    else:
                        meta_path.unlink()
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to clean orphan classifications")
            try:
                orphan_modules_reintroduced_total.inc(len(mods))
            except Exception:
                pass
            roi_vals: dict[str, float] = {}
            for m in mods:
                roi_val = 0.0
                pre_bot = getattr(self, "pre_roi_bot", None)
                if pre_bot is not None:
                    try:
                        res = pre_bot.predict_model_roi(m, [])
                        roi_val = float(getattr(res, "roi", 0.0))
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("roi prediction failed for %s", m)
                roi_vals[m] = roi_val
                try:
                    data_bot = getattr(self, "data_bot", None)
                    if data_bot and getattr(data_bot, "metrics_db", None):
                        db = data_bot.metrics_db
                        db.log_eval(m, "orphan_module_roi", roi_val)
                        db.log_eval(m, "orphan_module_pass", 1.0)
                        db.log_eval(m, "orphan_module_fail", 0.0)
                except Exception:  # pragma: no cover - best effort
                    pass
                try:
                    self.logger.info(
                        "orphan integration stats",
                        extra=log_record(
                            module=m, roi=float(roi_val), passed=True, failed=False
                        ),
                    )
                except Exception:
                    pass

            counts = getattr(self, "_last_orphan_counts", {})
            tested = float(counts.get("orphan_modules_tested", len(mods)))
            passed = float(counts.get("orphan_modules_passed", len(mods)))
            pass_rate = passed / tested if tested else 0.0
            avg_roi = sum(roi_vals.values()) / len(roi_vals) if roi_vals else 0.0
            robust_vals = [
                self.orphan_traces.get(m, {}).get("robustness", 0.0) for m in mods
            ]
            worst_robust = min(robust_vals) if robust_vals else 0.0
            self._last_orphan_metrics = {
                "pass_rate": float(pass_rate),
                "avg_roi": float(avg_roi),
                "worst_scenario_roi": float(worst_robust),
            }

            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                try:
                    tracker.register_metrics(
                        "orphan_pass_rate",
                        "orphan_avg_roi",
                        "orphan_worst_scenario_roi",
                    )
                    base = tracker.roi_history[-1] if tracker.roi_history else 0.0
                    tracker.update(
                        base,
                        base,
                        metrics={
                            "orphan_pass_rate": pass_rate,
                            "orphan_avg_roi": avg_roi,
                            "orphan_worst_scenario_roi": worst_robust,
                        },
                    )
                except Exception:  # pragma: no cover - best effort
                    pass
            return mods
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)
            return set()

    def _collect_recursive_modules(self, modules: Iterable[str]) -> set[str]:
        """Return ``modules`` plus any local imports they depend on recursively."""
        from pathlib import Path
        from sandbox_runner.dependency_utils import collect_local_dependencies

        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
        traces = getattr(self, "orphan_traces", None)

        roots: list[str] = []
        initial: dict[str, list[str]] = {}
        for m in modules:
            p = Path(m)
            if not p.is_absolute():
                p = repo / p
            try:
                rel = p.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = p.as_posix()
            roots.append(p.as_posix())
            if traces is not None:
                parents = list(traces.get(rel, {}).get("parents", []))
                initial[rel] = parents
                traces.setdefault(rel, {"parents": parents})

        def _on_module(rel: str, _path: Path, parents: list[str]) -> None:
            if traces is None:
                return
            entry = traces.setdefault(rel, {"parents": []})
            if parents:
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + parents)
                )

        def _on_dep(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
            if traces is None:
                return
            entry = traces.setdefault(dep_rel, {"parents": []})
            if chain:
                entry["parents"] = list(dict.fromkeys(entry.get("parents", []) + chain))

        deps = collect_local_dependencies(
            roots,
            initial_parents=initial if traces is not None else None,
            on_module=_on_module if traces is not None else None,
            on_dependency=_on_dep if traces is not None else None,
        )
        return deps | set(initial.keys())

    # ------------------------------------------------------------------
    def _update_orphan_modules(self, modules: Iterable[str] | None = None) -> None:
        """Discover orphan modules and update the tracking file or integrate ``modules``."""
        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        path = data_dir / "orphan_modules.json"

        if not hasattr(self, "orphan_traces"):
            self.orphan_traces = {}

        if modules:
            modules = list(modules)
            meta_path = data_dir / "orphan_classifications.json"
            try:
                existing_meta = (
                    json.loads(meta_path.read_text()) if meta_path.exists() else {}
                )
            except Exception:  # pragma: no cover - best effort
                existing_meta = {}
            try:
                settings = SandboxSettings()
            except Exception:  # pragma: no cover - fallback for tests
                from sandbox_settings import SandboxSettings as _SS  # type: ignore

                settings = _SS()

            legacy_mods: list[str] = []
            redundant_mods: list[str] = []
            filtered: list[str] = []
            for m in modules:
                p = Path(m)
                try:
                    rel = p.resolve().relative_to(repo).as_posix()
                except Exception:
                    rel = str(p)
                cls = existing_meta.get(rel, {}).get("classification")
                info = self.orphan_traces.setdefault(rel, {"parents": []})
                if cls in {"legacy", "redundant"}:
                    info["classification"] = cls
                    info["redundant"] = True
                    if not getattr(settings, "test_redundant_modules", False):
                        if cls == "legacy":
                            legacy_mods.append(rel)
                        else:
                            redundant_mods.append(rel)
                        continue
                filtered.append(m)
            modules = filtered
            if legacy_mods or redundant_mods:
                try:
                    if legacy_mods:
                        orphan_modules_legacy_total.set(len(legacy_mods))
                    if redundant_mods:
                        orphan_modules_redundant_total.set(len(redundant_mods))
                except Exception:
                    pass
                existing_meta.update(
                    {m: {"classification": "legacy"} for m in legacy_mods}
                )
                existing_meta.update(
                    {m: {"classification": "redundant"} for m in redundant_mods}
                )
                try:
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(json.dumps(existing_meta, indent=2))
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("failed to write orphan classifications")

            try:
                from scripts.discover_isolated_modules import discover_isolated_modules

                iso_mods = discover_isolated_modules(
                    str(repo), recursive=getattr(settings, "recursive_isolated", True)
                )
                modules.extend(sorted(iso_mods))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("isolated module discovery failed: %s", exc)

            try:
                from sandbox_runner import discover_recursive_orphans as _discover

                trace = _discover(str(repo), module_map=data_dir / "module_map.json")
                for k, v in trace.items():
                    info: dict[str, Any] = {
                        "parents": [
                            str(Path(*p.split(".")).with_suffix(".py"))
                            for p in (v.get("parents") if isinstance(v, dict) else v)
                        ]
                    }
                    if isinstance(v, dict) and "redundant" in v:
                        info["redundant"] = bool(v["redundant"])
                    mod_path = str(Path(*k.split(".")).with_suffix(".py"))
                    self.orphan_traces.setdefault(mod_path, info)
                    modules.append(mod_path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("orphan discovery failed: %s", exc)

            try:
                repo_mods = sorted(self._collect_recursive_modules(modules))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("dependency expansion failed: %s", exc)
                repo_mods = sorted(set(modules))

            passing = self._test_orphan_modules(repo_mods)

            if passing:
                metrics = {
                    m: self.orphan_traces.get(m, {}).get("side_effects", 0)
                    for m in passing
                }
                safe: list[str] = []
                threshold = SandboxSettings().side_effect_threshold
                for m in passing:
                    if metrics.get(m, 0) > threshold:
                        info = self.orphan_traces.setdefault(m, {"parents": []})
                        info["heavy_side_effects"] = True
                    else:
                        safe.append(m)
                if safe:
                    try:
                        environment.auto_include_modules(
                            sorted(safe), recursive=True, validate=True
                        )
                        try:
                            kwargs: dict[str, object] = {}
                            try:
                                if (
                                    "side_effects"
                                    in inspect.signature(
                                        environment.try_integrate_into_workflows
                                    ).parameters
                                ):
                                    kwargs["side_effects"] = {
                                        m: metrics.get(m, 0) for m in safe
                                    }
                            except Exception:
                                pass
                            environment.try_integrate_into_workflows(
                                sorted(safe), **kwargs
                            )
                        except Exception:  # pragma: no cover - best effort
                            pass
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)

                    abs_paths = [str(repo / p) for p in safe]
                    integrated: set[str] = set()
                    try:
                        integrated = self._integrate_orphans(abs_paths)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("orphan integration failed: %s", exc)
                    try:
                        self._refresh_module_map(safe)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("module map refresh failed: %s", exc)
                    try:
                        survivors = [
                            m for m in modules if Path(m).name not in integrated
                        ]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(sorted(survivors), indent=2))
                        meta_path = data_dir / "orphan_classifications.json"
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            changed = False
                            for m in integrated:
                                if m in meta:
                                    meta.pop(m, None)
                                    changed = True
                            if changed:
                                if meta:
                                    meta_path.write_text(json.dumps(meta, indent=2))
                                else:
                                    meta_path.unlink()
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("failed to write orphan modules")
            return

        try:
            settings = SandboxSettings()
        except Exception:  # pragma: no cover - fallback for tests
            from sandbox_settings import SandboxSettings as _SS  # type: ignore

            settings = _SS()
        modules: list[str] = []
        try:
            from scripts.discover_isolated_modules import discover_isolated_modules

            iso_mods = discover_isolated_modules(
                str(repo), recursive=getattr(settings, "recursive_isolated", True)
            )
            modules.extend(sorted(iso_mods))
            for m in iso_mods:
                self.orphan_traces.setdefault(m, {"parents": []})
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("isolated module discovery failed: %s", exc)

        try:
            from sandbox_runner import discover_recursive_orphans as _discover

            trace = _discover(str(repo), module_map=data_dir / "module_map.json")
            for mod, info in trace.items():
                parents = [
                    str(Path(*p.split(".")).with_suffix(".py"))
                    for p in (info.get("parents") if isinstance(info, dict) else info)
                ]
                entry: dict[str, Any] = {"parents": parents}
                if isinstance(info, dict):
                    cls = info.get("classification")
                    if cls is not None:
                        entry["classification"] = cls
                        entry["redundant"] = cls != "candidate"
                    elif "redundant" in info:
                        entry["redundant"] = bool(info["redundant"])
                mod_path = str(Path(*mod.split(".")).with_suffix(".py"))
                self.orphan_traces.setdefault(mod_path, entry).update(entry)
                modules.append(mod_path)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan discovery failed: %s", exc)

        try:
            existing = json.loads(path.read_text()) if path.exists() else []
        except Exception:  # pragma: no cover - best effort
            existing = []
        if isinstance(existing, list):
            modules.extend(existing)
            for m in existing:
                self.orphan_traces.setdefault(m, {"parents": []})
        elif isinstance(existing, dict):
            for m, info in existing.items():
                modules.append(m)
                if isinstance(info, dict):
                    self.orphan_traces.setdefault(
                        m, {"parents": info.get("parents", [])}
                    )

        if not modules:
            tester = getattr(self, "_test_orphan_modules", None)
            if tester is not None:
                tester([])
            return

        modules = sorted(set(modules))

        filtered: list[str] = []
        skipped: list[str] = []
        for m in modules:
            p = Path(m)
            info = self.orphan_traces.setdefault(m, {"parents": []})
            try:
                cls = classify_module(p)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", p, exc)
                cls = "candidate"
            try:
                redundant = analyze_redundancy(p)
            except Exception:  # pragma: no cover - best effort
                redundant = False
            is_redundant = redundant or cls in {"legacy", "redundant"}
            if is_redundant and not getattr(settings, "test_redundant_modules", False):
                info["classification"] = cls
                info["redundant"] = True
                skipped.append(m)
                self.logger.info(
                    "redundant module skipped",
                    extra=log_record(module=m, classification=cls),
                )
                try:
                    if cls == "legacy":
                        orphan_modules_legacy_total.inc(1)
                    else:
                        orphan_modules_redundant_total.inc(1)
                except Exception:
                    pass
                continue
            if is_redundant:
                info["classification"] = cls
                info["redundant"] = True
            filtered.append(m)

        if skipped:
            self.logger.info(
                "redundant modules skipped", extra=log_record(modules=sorted(skipped))
            )

        if not filtered:
            tester = getattr(self, "_test_orphan_modules", None)
            if tester is not None:
                tester([])
            return

        try:
            if hasattr(self, "_collect_recursive_modules"):
                repo_mods = sorted(self._collect_recursive_modules(filtered))
            else:  # pragma: no cover - fallback
                from sandbox_runner.dependency_utils import collect_local_dependencies

                repo_mods = sorted(collect_local_dependencies(filtered))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("dependency expansion failed: %s", exc)
            repo_mods = sorted(set(filtered))

        passing = self._test_orphan_modules(repo_mods)
        integrated: set[str] = set()
        if passing:
            metrics = {
                m: self.orphan_traces.get(m, {}).get("side_effects", 0) for m in passing
            }
            safe: list[str] = []
            threshold = SandboxSettings().side_effect_threshold
            for m in passing:
                if metrics.get(m, 0) > threshold:
                    info = self.orphan_traces.setdefault(m, {"parents": []})
                    info["heavy_side_effects"] = True
                else:
                    safe.append(m)
            if safe:
                try:
                    environment.auto_include_modules(
                        sorted(safe), recursive=True, validate=True
                    )
                    try:
                        kwargs: dict[str, object] = {}
                        try:
                            if (
                                "side_effects"
                                in inspect.signature(
                                    environment.try_integrate_into_workflows
                                ).parameters
                            ):
                                kwargs["side_effects"] = {
                                    m: metrics.get(m, 0) for m in safe
                                }
                        except Exception:
                            pass
                        environment.try_integrate_into_workflows(sorted(safe), **kwargs)
                    except Exception:  # pragma: no cover - best effort
                        pass
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("auto inclusion failed: %s", exc)

                repo_paths = [str(repo / p) for p in safe]
                try:
                    integrated = self._integrate_orphans(repo_paths)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                try:
                    if hasattr(self, "_refresh_module_map"):
                        self._refresh_module_map(safe)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("module map refresh failed: %s", exc)

        env_clean = os.getenv("SANDBOX_CLEAN_ORPHANS")
        if env_clean and env_clean.lower() in ("1", "true", "yes"):
            survivors = [m for m in filtered if m not in passing]
        else:
            survivors = [m for m in filtered if m not in passing]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sorted(survivors), indent=2))
            self.logger.info(
                "orphan modules updated", extra=log_record(count=len(survivors))
            )
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to write orphan modules")

    # ------------------------------------------------------------------
    def _load_orphan_candidates(self) -> list[str]:
        """Read orphan module candidates from the tracking file."""
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        path = data_dir / "orphan_modules.json"
        try:
            if path.exists():
                data = json.loads(path.read_text()) or {}
            else:
                return []
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load orphan candidates")
            return []
        if isinstance(data, list):
            return [str(p) for p in data]
        if isinstance(data, dict):
            return [
                m
                for m, info in data.items()
                if not isinstance(info, dict)
                or info.get("classification", "candidate") == "candidate"
            ]
        return []

    # ------------------------------------------------------------------
    def retest_redundant_modules(self) -> None:
        """Re-run self tests for modules previously classified as redundant.

        Modules listed in ``orphan_modules.json`` that have a corresponding
        classification of ``"redundant"`` in ``orphan_classifications.json``
        are re-executed via :func:`environment.auto_include_modules` with
        ``validate=True`` so they can be reintegrated once they start passing
        their self tests again.
        """

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        orphan_path = data_dir / "orphan_modules.json"
        meta_path = data_dir / "orphan_classifications.json"
        try:
            modules = (
                json.loads(orphan_path.read_text()) if orphan_path.exists() else []
            )
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to load orphan modules")
            return
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:  # pragma: no cover - best effort
            meta = {}

        redundant = [
            m
            for m in modules
            if isinstance(m, str)
            and meta.get(m, {}).get("classification") == "redundant"
        ]
        if not redundant:
            return
        try:
            environment.auto_include_modules(
                sorted(redundant), recursive=True, validate=True
            )
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("redundant module recheck failed: %s", exc)

    # ------------------------------------------------------------------
    def _refresh_module_map(self, modules: Iterable[str] | None = None) -> None:
        """Refresh module grouping when new modules appear.

        Modules accepted for integration are auto-included with recursive
        dependency expansion. Redundant or legacy modules identified by
        :func:`classify_module` are skipped. When
        ``SANDBOX_RECURSIVE_ORPHANS`` is enabled, orphan discovery is executed
        again after integration to traverse any newly uncovered dependencies.
        """
        if modules:
            repo_mods = self._collect_recursive_modules(modules)
            passing = self._test_orphan_modules(repo_mods)
            if passing:
                repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                try:
                    environment.auto_include_modules(
                        sorted(passing), recursive=True, validate=True
                    )
                except Exception:
                    try:
                        auto_include_modules(
                            sorted(passing), recursive=True, validate=True
                        )
                    except TypeError:
                        auto_include_modules(sorted(passing), recursive=True)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                integrated: set[str] = set()
                try:
                    abs_paths = [str(repo / p) for p in passing]
                    integrated = self._integrate_orphans(abs_paths)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                try:
                    self._update_orphan_modules()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "post integration orphan update failed: %s", exc
                    )
            return

        if not self.auto_refresh_map or not self.module_index:
            return
        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)
        if not pdb:
            return
        try:
            iso_ts = datetime.utcfromtimestamp(self._last_map_refresh).isoformat()
            with pdb._connect() as conn:
                rows = conn.execute(
                    "SELECT filename FROM patch_history WHERE ts > ?",
                    (iso_ts,),
                ).fetchall()
        except Exception as exc:  # pragma: no cover - database issues
            self.logger.exception("module refresh query failed: %s", exc)
            return
        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        pending: dict[str, Path] = {}
        for r in rows:
            p = Path(r[0])
            abs_p = p if p.is_absolute() else repo / p
            try:
                rel = abs_p.resolve().relative_to(repo).as_posix()
            except Exception:
                continue
            if rel in self.module_clusters or rel in pending:
                continue
            pending[rel] = abs_p
        new_mods: set[str] = set()
        skipped: set[str] = set()
        for rel, path in pending.items():
            try:
                cls = classify_module(path)
                if cls != "candidate":
                    skipped.add(rel)
                    self.logger.info(
                        "redundant module skipped",
                        extra=log_record(module=rel, classification=cls),
                    )
                    try:
                        if cls == "legacy":
                            orphan_modules_legacy_total.inc(1)
                        else:
                            orphan_modules_redundant_total.inc(1)
                    except Exception:
                        pass
                    continue
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("classification failed for %s: %s", path, exc)
            new_mods.add(rel)
        if skipped:
            self.logger.info(
                "redundant modules skipped", extra=log_record(modules=sorted(skipped))
            )
        if not new_mods:
            return
        try:
            exclude_env = os.getenv("SANDBOX_EXCLUDE_DIRS")
            exclude = [e for e in exclude_env.split(",") if e] if exclude_env else None
            mapping = build_module_map(repo, ignore=exclude)
            mapping = {
                (f"{k}.py" if not k.endswith(".py") else k): v
                for k, v in mapping.items()
            }
            if skipped:
                for key in list(mapping.keys()):
                    if key in skipped:
                        mapping.pop(key, None)
            self.module_index.merge_groups(mapping)
            self.module_clusters.update(mapping)
            data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
            out = data_dir / "module_map.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "modules": self.module_index._map,
                        "groups": self.module_index._groups,
                    },
                    fh,
                    indent=2,
                )
            self._last_map_refresh = time.time()
            if self.meta_logger and hasattr(self.meta_logger, "audit"):
                try:
                    self.meta_logger.audit.record(
                        {"event": "module_map_refreshed", "modules": sorted(new_mods)}
                    )
                except Exception:
                    pass
            self.logger.info(
                "module map refreshed",
                extra=log_record(modules=sorted(new_mods)),
            )
            try:
                abs_new = [str(pending[m]) for m in new_mods]
                deps = self._collect_recursive_modules(abs_new)
                abs_deps = [str(repo / p) for p in deps]
                try:
                    environment.auto_include_modules(
                        sorted(deps), recursive=True, validate=True
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("auto inclusion failed: %s", exc)
                deps_integrated: set[str] = set()
                try:
                    deps_integrated = self._integrate_orphans(abs_deps)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("orphan integration failed: %s", exc)
                try:
                    self._update_orphan_modules()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "post integration orphan update failed: %s", exc
                    )
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("orphan integration failed: %s", exc)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("module map refresh failed: %s", exc)

    # ------------------------------------------------------------------
    def enqueue_preventative_fixes(self, modules: Iterable[str]) -> None:
        """Queue modules for preventative patch generation."""
        for mod in modules:
            m = str(mod)
            if m and m not in self._preventative_queue:
                self._preventative_queue.append(m)

    def _process_preventative_queue(self) -> None:
        """Generate patches for queued modules."""
        if not self._preventative_queue or not self.self_coding_engine:
            self._preventative_queue.clear()
            return
        queue = list(self._preventative_queue)
        self._preventative_queue.clear()
        scored = self._score_modifications(queue)
        for mod, roi_est, category, weight in scored:
            self.logger.info(
                "patch candidate",
                extra=log_record(
                    module=mod,
                    roi_category=category,
                    roi_estimate=roi_est,
                    weight=weight,
                ),
            )
            try:
                patch_id = None
                with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                    src = Path(mod)
                    if src.suffix == "":
                        src = src.with_suffix(".py")
                    rel = src.name if src.is_absolute() else src
                    before_target = Path(before_dir) / rel
                    before_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, before_target)
                    self.logger.info(
                        "gpt_suggestion",
                        extra=log_record(
                            module=mod,
                            suggestion="preventative_patch",
                            tags=[ERROR_FIX],
                        ),
                    )
                    try:
                        log_with_tags(
                            self.gpt_memory,
                            f"preventative_patch:{mod}",
                            "suggested",
                            tags=[f"self_improvement_engine.preventative_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                        )
                    except Exception:
                        self.logger.exception(
                            "memory logging failed", extra=log_record(module=mod)
                        )
                    patch_id = self._generate_patch_with_memory(
                        mod, "preventative_patch"
                    )
                    self.logger.info(
                        "patch result",
                        extra=log_record(
                            module=mod,
                            patch_id=patch_id,
                            success=patch_id is not None,
                            tags=["fix_result"],
                        ),
                    )
                    if patch_id is not None:
                        after_target = Path(after_dir) / rel
                        after_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, after_target)
                        diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                        self._pre_commit_alignment_check(diff_data)
                        findings = flag_alignment_issues(diff_data)
                        if findings:
                            log_violation(
                                str(patch_id),
                                "alignment_warning",
                                1,
                                {"findings": findings},
                                alignment_warning=True,
                            )
                        self._alignment_review_last_commit(
                            f"preventative_patch_{patch_id}"
                        )
                        self._flag_patch_alignment(
                            patch_id,
                            {
                                "trigger": "preventative_patch",
                                "module": str(mod),
                                "patch_id": patch_id,
                            },
                        )
                if self.error_bot and hasattr(self.error_bot, "db"):
                    try:
                        self.error_bot.db.add_telemetry(
                            TelemetryEvent(
                                module=str(mod),
                                patch_id=patch_id,
                                resolution_status="attempted",
                            )
                        )
                    except Exception:
                        self.logger.exception(
                            "telemetry record failed", extra=log_record(module=mod)
                        )
                if self.error_predictor:
                    try:
                        self.error_predictor.graph.add_telemetry_event(
                            self.bot_name,
                            "preemptive_patch",
                            str(mod),
                            patch_id=patch_id,
                        )
                    except Exception:
                            self.logger.exception(
                                "graph patch record failed", extra=log_record(module=mod)
                            )
            except Exception:
                self.logger.exception(
                    "preemptive fix failed", extra=log_record(module=mod)
                )
            if self.use_adaptive_roi:
                self._log_action("preventative_patch", mod, roi_est, category)

    def _apply_high_risk_patches(self) -> None:
        """Predict high-risk modules and attempt preemptive fixes."""
        if not (self.error_predictor and self.auto_patch_high_risk):
            return
        try:
            high_risk = self.error_predictor.predict_high_risk_modules()
            self.error_predictor.graph.update_error_stats(self.error_bot.db)
            if not high_risk:
                return
            scored = self._score_modifications(high_risk)
            self.logger.info(
                "high risk modules",
                extra=log_record(modules=[m for m, _, _, _ in scored]),
            )
            for mod, roi_est, category, weight in scored:
                self.logger.info(
                    "patch candidate",
                    extra=log_record(
                        module=mod,
                        roi_category=category,
                        roi_estimate=roi_est,
                        weight=weight,
                    ),
                )
                try:
                    patch_id = None
                    with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
                        src = Path(mod)
                        if src.suffix == "":
                            src = src.with_suffix(".py")
                        rel = src.name if src.is_absolute() else src
                        before_target = Path(before_dir) / rel
                        before_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, before_target)
                        self.logger.info(
                            "gpt_suggestion",
                            extra=log_record(
                                module=mod,
                                suggestion="high_risk_patch",
                                tags=[ERROR_FIX],
                            ),
                        )
                        try:
                            log_with_tags(
                                self.gpt_memory,
                                f"high_risk_patch:{mod}",
                                "suggested",
                                tags=[f"self_improvement_engine.high_risk_patch", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                            )
                        except Exception:
                            self.logger.exception(
                                "memory logging failed", extra=log_record(module=mod)
                            )
                        patch_id = self._generate_patch_with_memory(
                            mod, "high_risk_patch"
                        )
                        self.logger.info(
                            "patch result",
                            extra=log_record(
                                module=mod,
                                patch_id=patch_id,
                                success=patch_id is not None,
                                tags=["fix_result"],
                            ),
                        )
                        if patch_id is not None:
                            after_target = Path(after_dir) / rel
                            after_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, after_target)
                            diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                            self._pre_commit_alignment_check(diff_data)
                            findings = flag_alignment_issues(diff_data)
                            if findings:
                                log_violation(
                                    str(patch_id),
                                    "alignment_warning",
                                    1,
                                    {"findings": findings},
                                    alignment_warning=True,
                                )
                            self._alignment_review_last_commit(
                                f"high_risk_patch_{patch_id}"
                            )
                            self._flag_patch_alignment(
                                patch_id,
                                {
                                    "trigger": "high_risk_patch",
                                    "module": str(mod),
                                    "patch_id": patch_id,
                                },
                            )
                    if self.error_bot and hasattr(self.error_bot, "db"):
                        try:
                            self.error_bot.db.add_telemetry(
                                TelemetryEvent(
                                    module=str(mod),
                                    patch_id=patch_id,
                                    resolution_status="attempted",
                                )
                            )
                        except Exception:
                            self.logger.exception(
                                "telemetry record failed", extra=log_record(module=mod)
                            )
                    if self.error_predictor:
                        try:
                            self.error_predictor.graph.add_telemetry_event(
                                self.bot_name,
                                "preemptive_patch",
                                str(mod),
                                patch_id=patch_id,
                            )
                        except Exception:
                            self.logger.exception(
                                "graph patch record failed", extra=log_record(module=mod)
                            )
                except Exception:
                    self.logger.exception(
                        "preemptive fix failed", extra=log_record(module=mod)
                    )
                if self.use_adaptive_roi:
                    self._log_action("high_risk_patch", mod, roi_est, category)
        except Exception as exc:
            self.logger.exception(
                "high risk module prediction failed: %s", exc
            )

    def _evaluate_module_relevance(self) -> None:
        """Evaluate module usage and record relevancy recommendations."""
        try:
            settings = SandboxSettings()
            replace_threshold = float(getattr(settings, "relevancy_threshold", 20))
            compress_threshold = max(1.0, replace_threshold / 4)
            auto_process = getattr(
                settings, "auto_process_relevancy_flags", True
            )
        except Exception:
            replace_threshold = 20.0
            compress_threshold = 5.0
            auto_process = True
        flags: dict[str, str] = dict(self.relevancy_flags)
        try:
            radar_flags = self.relevancy_radar.evaluate_final_contribution(
                compress_threshold, replace_threshold
            )
            flags.update(radar_flags)
        except Exception:
            self.logger.exception("relevancy evaluation failed")
        try:
            metrics_db = (
                Path(__file__).resolve().parent / "sandbox_data" / "relevancy_metrics.db"
            )
            scan_flags = radar_scan(metrics_db)
            if scan_flags:
                flags.update(scan_flags)
        except Exception:
            self.logger.exception("relevancy scan failed")
        if not flags or flags == self.relevancy_flags:
            return
        self.relevancy_flags = flags
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy_flags", flags)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("relevancy flag publish failed")
        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        for mod, status in flags.items():
            self.logger.info(
                "module flagged", extra=log_record(module=mod, status=status)
            )
            try:
                audit_log_event(
                    "relevancy_flag", {"module": mod, "status": status}
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "relevancy audit log failed", extra=log_record(module=mod)
                )
            try:
                analyze_redundancy(repo / (mod.replace(".", "/") + ".py"))
            except Exception:
                self.logger.exception(
                    "redundancy analysis failed", extra=log_record(module=mod)
                )
            try:
                MutationLogger.log_mutation(
                    change="relevancy",
                    reason=status,
                    trigger="relevancy_radar",
                    performance=0.0,
                    workflow_id=0,
                )
            except Exception:
                self.logger.exception(
                    "mutation logging failed", extra=log_record(module=mod)
                )
        if auto_process:
            try:
                service = ModuleRetirementService(
                    Path(os.getenv("SANDBOX_REPO_PATH", "."))
                )
                results = service.process_flags(flags)
            except Exception:
                self.logger.exception("relevancy flag processing failed")
            else:
                for mod, action in results.items():
                    self.logger.info(
                        "relevancy action",
                        extra=log_record(module=mod, action=action),
                    )
                    if self.event_bus and action != "skipped":
                        try:
                            self.event_bus.publish(
                                f"relevancy:{action}", {"module": mod}
                            )
                        except Exception:
                            self.logger.exception(
                                "relevancy action publish failed",
                                extra=log_record(module=mod, action=action),
                            )
                if self.event_bus:
                    try:
                        self.event_bus.publish("relevancy_actions", results)
                    except Exception:
                        self.logger.exception(
                            "relevancy actions publish failed"
                        )

    def get_relevancy_flags(self) -> dict[str, str]:
        """Return latest module relevancy flags for downstream pipelines."""
        return dict(self.relevancy_flags)

    def _handle_relevancy_flags(self, flags: dict[str, str]) -> None:
        """Process relevancy radar flags and trigger follow-up actions."""
        service = ModuleRetirementService(Path(os.getenv("SANDBOX_REPO_PATH", ".")))
        try:
            results = service.process_flags(flags)
        except Exception:
            self.logger.exception("flag processing failed")
            results = {}
        for mod, action in results.items():
            self.logger.info(
                "relevancy action", extra=log_record(module=mod, action=action)
            )
            if self.event_bus and action != "skipped":
                try:
                    self.event_bus.publish(f"relevancy:{action}", {"module": mod})
                except Exception:
                    self.logger.exception(
                        "relevancy action publish failed",
                        extra=log_record(module=mod, action=action),
                    )
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy_actions", results)
            except Exception:
                self.logger.exception("relevancy actions publish failed")
        if append_orphan_classifications:
            repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
            retire_entries = {
                m: {"classification": "retired"}
                for m, status in flags.items()
                if status == "retire"
            }
            if retire_entries:
                try:
                    append_orphan_classifications(repo, retire_entries)
                except Exception:
                    self.logger.exception("orphan classification failed")
        replace_mods = [m for m, status in flags.items() if status == "replace"]
        for mod in replace_mods:
            task_id: int | None = None
            if self.self_coding_engine:
                try:
                    task_id = generate_patch(mod, self.self_coding_engine)
                except Exception:
                    self.logger.exception(
                        "replacement generation failed",
                        extra=log_record(module=mod),
                    )
            if self.event_bus:
                try:
                    self.event_bus.publish(
                        "relevancy:replace", {"module": mod, "task_id": task_id}
                    )
                except Exception:
                    self.logger.exception(
                        "relevancy replace event publish failed",
                        extra=log_record(module=mod),
                    )
        if self.event_bus:
            try:
                self.event_bus.publish("relevancy:scan", flags)
            except Exception:
                self.logger.exception("relevancy scan event publish failed")

    @radar.track
    def run_cycle(self, energy: int = 1) -> AutomationResult:
        """Execute a self-improvement cycle.

        The ``workflow_id`` used for foresight tracking is derived from the
        current sandbox workflow context.
        """
        self._cycle_running = True
        self._cycle_count += 1
        cid = f"cycle-{self._cycle_count}"
        set_correlation_id(cid)
        try:
            if self.meta_logger:
                try:
                    settings = SandboxSettings()
                    thr = settings.entropy_ceiling_threshold
                    consecutive = settings.entropy_ceiling_consecutive or 3
                except Exception:
                    thr = None
                    consecutive = 3
                if thr is not None:
                    try:
                        flagged = self.meta_logger.ceiling(thr, consecutive=consecutive)
                    except Exception:
                        flagged = []
                    if flagged:
                        norm = [Path(m).as_posix() for m in flagged]
                        self.entropy_ceiling_modules.update(norm)
                        for m in norm:
                            self.logger.info(
                                "module flagged", extra=log_record(module=m, status="entropy_ceiling")
                            )
                            try:
                                audit_log_event(
                                    "entropy_ceiling",
                                    {"module": m, "status": "entropy_ceiling"},
                                )
                            except Exception:  # pragma: no cover - best effort
                                self.logger.exception(
                                    "entropy ceiling audit log failed",
                                    extra=log_record(module=m),
                                )
                        try:
                            service = ModuleRetirementService(
                                Path(os.getenv("SANDBOX_REPO_PATH", "."))
                            )
                            pending = {m: "retire" for m in flagged}
                            results = service.process_flags(pending)
                            remaining = [
                                m for m, action in results.items() if action == "skipped"
                            ]
                            if remaining:
                                pending = {m: "compress" for m in remaining}
                                results.update(service.process_flags(pending))
                                remaining = [
                                    m
                                    for m in remaining
                                    if results.get(m) == "skipped"
                                ]
                            if remaining:
                                service.process_flags(
                                    {m: "replace" for m in remaining}
                                )
                        except Exception:
                            self.logger.exception(
                                "ceiling flag processing failed",
                                extra=log_record(modules=flagged),
                            )
            try:
                self.retest_redundant_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("redundant module check failed: %s", exc)
            # refresh orphan data so new modules are considered before policy evaluation
            self._update_orphan_modules()
            orphans = self._load_orphan_candidates()
            if orphans:
                passing = self._test_orphan_modules(orphans)
                if passing:
                    try:
                        environment.auto_include_modules(
                            sorted(passing), recursive=True, validate=True
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("auto inclusion failed: %s", exc)
                    repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                    abs_paths = [str(repo / p) for p in passing]
                    integrated: set[str] = set()
                    try:
                        integrated = self._integrate_orphans(abs_paths)
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.exception("orphan integration failed: %s", exc)
            try:
                self._update_orphan_modules()
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "post integration orphan update failed: %s", exc
                )
            self._refresh_module_map()
            now = time.time()
            if now - self._last_relevancy_eval >= self.relevancy_eval_interval:
                self._evaluate_module_relevance()
                self._last_relevancy_eval = now
            self._process_preventative_queue()
            if self.error_bot:
                try:
                    predictions = (
                        self.error_bot.predict_errors()
                        if hasattr(self.error_bot, "predict_errors")
                        else []
                    )
                    clusters = (
                        self.error_bot.get_error_clusters()
                        if hasattr(self.error_bot, "get_error_clusters")
                        else {}
                    )
                    baseline = {
                        item.get("error_type", ""): float(item.get("count", 0.0))
                        for item in (
                            self.error_bot.summarize_telemetry(limit=10)
                            if hasattr(self.error_bot, "summarize_telemetry")
                            else []
                        )
                    }
                    affected_modules: set[str] = set()
                    pred_clusters: set[int] = set()
                    for err in predictions:
                        cid = clusters.get(err)
                        if cid is None:
                            continue
                        pred_clusters.add(cid)
                        affected_modules.update(
                            m for m, idx in self.module_clusters.items() if idx == cid
                        )
                    if pred_clusters and affected_modules:
                        self.logger.info(
                            "predicted error clusters",
                            extra=log_record(
                                predicted_clusters=sorted(pred_clusters),
                                modules=sorted(affected_modules),
                            ),
                        )
                        self.error_bot.auto_patch_recurrent_errors()
                        if self.error_predictor:
                            try:
                                self.error_predictor.graph.update_error_stats(
                                    self.error_bot.db
                                )
                            except Exception:
                                self.logger.exception(
                                    "knowledge graph update failed",
                                    extra=log_record(action="auto_patch_recurrent"),
                                )
                        after = {
                            item.get("error_type", ""): float(item.get("count", 0.0))
                            for item in (
                                self.error_bot.summarize_telemetry(limit=10)
                                if hasattr(self.error_bot, "summarize_telemetry")
                                else []
                            )
                        }
                        prevented = [
                            err
                            for err in predictions
                            if baseline.get(err, 0.0) > 0
                            and baseline.get(err) == after.get(err)
                        ]
                        if prevented:
                            self.logger.info(
                                "proactive patch prevented faults",
                                extra=log_record(errors=prevented),
                            )
                except Exception as exc:
                    self.logger.exception(
                        "proactive prediction patching failed: %s", exc
                    )
            self._apply_high_risk_patches()
            state = self._policy_state() if self.policy else (0,) * POLICY_STATE_LEN
            predicted = self.policy.score(state) if self.policy else 0.0
            roi_pred: float | None = None
            self.logger.info(
                "cycle start",
                extra=log_record(energy=energy, predicted_roi=predicted, state=state),
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
                    self.logger.info("initial ROI", extra=log_record(roi=before_roi))
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
                    self.logger.info("available energy", extra=log_record(value=energy))
                except Exception as exc:
                    self.logger.exception("energy calculation failed: %s", exc)
                    energy = 1
            if self.policy:
                try:
                    energy = max(1, int(round(energy * (1 + max(0.0, predicted)))))
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
                    self.logger.exception("policy energy adjustment failed: %s", exc)
            if self.pre_roi_bot:
                try:
                    forecast = self.pre_roi_bot.predict_model_roi(self.bot_name, [])
                    roi_pred = float(getattr(forecast, "roi", 0.0))
                    scale = (
                        1 + max(0.0, roi_pred + self.pre_roi_bias) * self.pre_roi_scale
                    )
                    if self.pre_roi_cap:
                        scale = min(scale, self.pre_roi_cap)
                    energy = max(1, int(round(energy * scale)))
                    self.logger.info(
                        "pre_roi adjusted energy",
                        extra=log_record(value=energy, roi_prediction=roi_pred),
                    )
                except Exception as exc:
                    self.logger.exception("pre ROI energy adjustment failed: %s", exc)
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
                    self.logger.exception("synergy energy adjustment failed: %s", exc)
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
            self.logger.info("model bootstrapped", extra=log_record(model_id=model_id))
            self.info_db.set_current_model(model_id)
            self._record_state()
            if self.learning_engine:
                try:
                    self.logger.info("training learning engine")
                    self.learning_engine.train()
                    self._evaluate_learning()
                except Exception as exc:
                    self.logger.exception("learning engine run failed: %s", exc)
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
            actions = getattr(getattr(result, "package", None), "actions", None)
            self.logger.info(
                "selected actions",
                extra=log_record(actions=actions, growth_type=self._last_growth_type),
            )
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
                    mod_path = Path("auto_helpers.py")
                    start_patch = time.perf_counter()
                    patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                        mod_path,
                        "helper",
                        trending_topic=trending_topic,
                        parent_patch_id=self._last_patch_id,
                        reason="helper_patch",
                        trigger="automation_cycle",
                    )
                    before_metric = 0.0
                    after_metric = delta
                    if self.self_coding_engine.patch_db and patch_id is not None:
                        try:
                            with self.self_coding_engine.patch_db._connect() as conn:
                                row = conn.execute(
                                    "SELECT roi_before, roi_after FROM patch_history WHERE id=?",
                                    (patch_id,),
                                ).fetchone()
                            if row:
                                before_metric = float(row[0])
                                after_metric = float(row[1])
                        except Exception:
                            after_metric = before_metric + delta
                    else:
                        after_metric = before_metric + delta
                    with MutationLogger.log_context(
                        change=f"helper_patch_{patch_id}",
                        reason="self-improvement helper patch",
                        trigger="automation_cycle",
                        workflow_id=0,
                        before_metric=before_metric,
                        parent_id=self._last_mutation_id,
                    ) as mutation:
                        mutation["after_metric"] = after_metric
                        mutation["performance"] = delta
                        mutation["roi"] = after_metric
                    self._last_mutation_id = int(mutation["event_id"])
                    self._last_patch_id = patch_id
                    if patch_id is not None and not reverted:
                        self._alignment_review_last_commit(f"helper_patch_{patch_id}")
                        self._flag_patch_alignment(
                            patch_id,
                            {"trigger": "automation_cycle", "patch_id": patch_id},
                        )
                    roi_delta = after_metric - before_metric
                    tracker = getattr(self, "roi_tracker", None)
                    if tracker:
                        try:
                            tracker.update(
                                before_metric,
                                after_metric,
                                modules=[mod_path.as_posix()],
                            )
                        except Exception:
                            self.logger.exception("roi tracker update failed")
                    if self.metrics_db:
                        try:
                            elapsed = time.perf_counter() - start_patch
                            self.metrics_db.record(
                                mod_path.as_posix(),
                                elapsed,
                                roi_delta=roi_delta,
                            )
                        except Exception:
                            self.logger.exception("relevancy metrics record failed")
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
                                    self.logger.exception("policy save failed: %s", exc)
                        except Exception as exc:
                            self.logger.exception("policy patch update failed: %s", exc)
                    if self.optimize_self_flag:
                        self._optimize_self()
                except Exception as exc:
                    self.logger.exception("helper patch failed: %s", exc)
                    patch_id = None
                    reverted = False
            if self.error_bot:
                try:
                    self.error_bot.auto_patch_recurrent_errors()
                    if self.error_predictor:
                        try:
                            self.error_predictor.graph.update_error_stats(
                                self.error_bot.db
                            )
                        except Exception:
                            self.logger.exception(
                                "knowledge graph update failed",
                                extra=log_record(action="auto_patch_recurrent"),
                            )
                    self.logger.info("error auto-patching complete")
                except Exception as exc:
                    self.logger.exception("auto patch recurrent errors failed: %s", exc)
            after_roi = before_roi
            if self.capital_bot:
                try:
                    after_roi = self.capital_bot.profit()
                    self.logger.info(
                        "post-cycle ROI",
                        extra=log_record(before=before_roi, after=after_roi),
                    )
                except Exception as exc:
                    self.logger.exception("post-cycle profit lookup failed: %s", exc)
                    after_roi = before_roi
            roi_value = result.roi.roi if result.roi else 0.0
            roi_realish = roi_value
            pred_realish = predicted
            try:
                features = np.array([[float(roi_value)]], dtype=np.float64)
                drift = self.truth_adapter.check_drift(features)
                preds, low_conf = self.truth_adapter.predict(features)
                roi_realish = float(preds[0])
                if predicted is not None:
                    p_arr = np.array([[float(predicted)]], dtype=np.float64)
                    p_preds, _ = self.truth_adapter.predict(p_arr)
                    pred_realish = float(p_preds[0])
                if drift or low_conf:
                    self.logger.warning(
                        "truth adapter low confidence; scheduling retrain"
                    )
                    self._truth_adapter_needs_retrain = True
            except Exception:
                self.logger.exception("truth adapter predict failed")
            self.logger.info(
                "roi realish", extra=log_record(roi_realish=roi_realish)
            )
            if self.roi_tracker and predicted is not None:
                try:
                    self.roi_tracker.record_roi_prediction(
                        [float(pred_realish)],
                        [float(roi_realish)],
                        predicted_class=self._last_growth_type,
                        workflow_id="self_improvement",
                    )
                except Exception:
                    self.logger.exception("roi tracker record failed")
                self.logger.info(
                    "cycle roi", extra=log_record(predicted=pred_realish, actual=roi_realish)
                )
            if self.roi_tracker:
                try:
                    cards = self.roi_tracker.generate_scorecards()
                except Exception:
                    cards = []
                scorecard = {
                    "decision": "rollback" if reverted else "ship",
                    "alignment": "pass",
                    "raroi_increase": sum(
                        1 for c in cards if getattr(c, "raroi_delta", 0.0) > 0
                    ),
                    "raroi": getattr(self.roi_tracker, "last_raroi", None),
                    "confidence": (
                        self.roi_tracker.confidence_history[-1]
                        if getattr(self.roi_tracker, "confidence_history", [])
                        else None
                    ),
                }
                workflow_id = "self_improvement"
                gov_result: Dict[str, Any] | None = None
                forecast_info: Dict[str, Any] | None = None
                reasons: List[str] = []
                try:
                    wf_ctx = getattr(environment, "current_context", None)
                    try:
                        ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                        workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
                    except Exception:
                        pass
                    metrics = {
                        "raroi": scorecard.get("raroi"),
                        "confidence": scorecard.get("confidence"),
                        "sandbox_roi": roi_realish,
                        "adapter_roi": pred_realish,
                    }
                    gov_result = deployment_evaluate(
                        scorecard,
                        metrics,
                        patch=str(patch_id) if patch_id is not None else None,
                        foresight_tracker=self.foresight_tracker,
                        workflow_id=workflow_id,
                        borderline_bucket=self.borderline_bucket,
                    )
                except Exception:
                    self.logger.exception("deployment evaluation failed")
                if gov_result:
                    verdict = str(gov_result.get("verdict"))
                    reasons = list(gov_result.get("reasons", []))
                    forecast_info = gov_result.get("foresight")
                    risk: dict[str, object] | None = None
                    try:
                        if self.foresight_tracker and workflow_id:
                            risk = self.foresight_tracker.predict_roi_collapse(workflow_id)
                    except Exception:
                        self.logger.exception("foresight risk check failed")
                    risk_cls = risk.get("risk") if isinstance(risk, Mapping) else None
                    brittle = bool(risk.get("brittle")) if isinstance(risk, Mapping) else False
                    high = bool(risk and (risk_cls != "Stable" or brittle))
                    self.workflow_risk = risk
                    self.workflow_high_risk = high
                    try:
                        audit_log_event(
                            "foresight_risk",
                            {
                                "workflow_id": workflow_id,
                                "risk": risk_cls,
                                "brittle": brittle,
                            },
                        )
                    except Exception:
                        self.logger.exception("risk audit log failed")
                    try:
                        self.logger.info(
                            "foresight risk classification",
                            extra=log_record(
                                workflow_id=workflow_id,
                                risk=risk_cls,
                                brittle=brittle,
                            ),
                        )
                    except Exception:
                        self.logger.exception("risk classification logging failed")
                    if high:
                        verdict = "no_go"
                        if "roi_collapse_risk" not in reasons:
                            reasons.append("roi_collapse_risk")
                        try:
                            self.enqueue_preventative_fixes([workflow_id])
                        except Exception:
                            self.logger.exception("risk queue enqueue failed")
                    if verdict == "promote" and self.foresight_tracker:
                        logger_obj: ForecastLogger | None = None
                        forecast_info: Dict[str, Any] | None = None
                        try:
                            forecaster = UpgradeForecaster(self.foresight_tracker)
                            graph = WorkflowGraph()
                            logger_obj = ForecastLogger("forecast_records/foresight.log")
                            decision = is_foresight_safe_to_promote(
                                workflow_id,
                                str(patch_id) if patch_id is not None else "",
                                forecaster,
                                graph,
                            )
                            if not isinstance(decision, ForesightDecision):
                                decision = ForesightDecision(*decision)
                            forecast_info = decision.forecast
                            decision_label = (
                                decision.recommendation
                                if not decision.safe
                                else "promote"
                            )
                            log_forecast_record(
                                logger_obj,
                                workflow_id,
                                forecast_info,
                                decision_label,
                                decision.reasons,
                            )
                            if not decision.safe:
                                verdict = decision.recommendation
                                reasons.extend(decision.reasons)
                        except Exception:
                            self.logger.exception("foresight gate check failed")
                        finally:
                            try:
                                if logger_obj is not None:
                                    logger_obj.close()
                            except Exception:
                                pass
                    scorecard["forecast"] = forecast_info
                    scorecard["reasons"] = list(reasons)
                    try:
                        payload = {
                            "verdict": verdict,
                            "reasons": reasons,
                            "forecast": forecast_info,
                        }
                        if verdict in {"borderline", "pilot"}:
                            payload["downgrade_type"] = verdict
                        audit_log_event("deployment_verdict", payload)
                    except Exception:
                        self.logger.exception("audit log failed")
                    if self.event_bus:
                        try:
                            self.event_bus.publish(
                                "deployment_verdict",
                                {
                                    "verdict": verdict,
                                    "reasons": reasons,
                                    "forecast": forecast_info,
                                },
                            )
                        except Exception:
                            self.logger.exception("event bus publish failed")
                    try:
                        self.logger.info(
                            "deployment verdict",
                            extra=log_record(
                                verdict=verdict,
                                reasons=";".join(reasons),
                                forecast=forecast_info,
                            ),
                        )
                    except Exception:
                        self.logger.exception("deployment verdict logging failed")
                    scorecard["deployment_verdict"] = verdict
                    if verdict == "promote":
                        self.workflow_ready = True
                    elif verdict == "demote":
                        self.workflow_ready = False
                        if (
                            patch_id is not None
                            and self.self_coding_engine
                            and not reverted
                        ):
                            try:
                                self.self_coding_engine.rollback_patch(str(patch_id))
                                reverted = True
                                scorecard["decision"] = "rollback"
                            except Exception:
                                self.logger.exception("patch rollback failed")
                    elif verdict in {"micro_pilot", "borderline", "pilot"}:
                        try:
                            self.borderline_bucket.add_candidate(
                                self.bot_name,
                                scorecard.get("raroi"),
                                scorecard.get("confidence"),
                                ";".join(reasons),
                            )
                            settings = SandboxSettings()
                            if getattr(settings, "micropilot_mode", "") == "auto":
                                try:
                                    evaluator = getattr(
                                        self, "micro_pilot_evaluator", None
                                    )
                                    self.borderline_bucket.process(
                                        evaluator,
                                        raroi_threshold=self.borderline_raroi_threshold,
                                        confidence_threshold=getattr(
                                            self.roi_tracker,
                                            "confidence_threshold",
                                            0.0,
                                        ),
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            self.logger.exception("borderline enqueue failed")
                        self.workflow_ready = False
                    else:
                        self.workflow_ready = False
                vetoes: List[str] = []
                try:
                    vetoes = check_veto(scorecard, load_rules())
                except Exception:
                    self.logger.exception("governance check failed")
                try:
                    append_governance_result(scorecard, vetoes, forecast_info, reasons)
                except Exception:
                    self.logger.exception("governance logging failed")
                if vetoes and patch_id is not None and self.self_coding_engine and not reverted:
                    try:
                        self.self_coding_engine.rollback_patch(str(patch_id))
                        reverted = True
                        self.logger.warning(
                            "patch rolled back due to governance veto",
                            extra=log_record(patch_id=patch_id, veto=";".join(vetoes)),
                        )
                    except Exception:
                        self.logger.exception("patch rollback failed")
            if self.evolution_history:
                try:
                    from .evolution_history_db import EvolutionEvent

                    event_id = self.evolution_history.add(
                        EvolutionEvent(
                            action="self_improvement",
                            before_metric=before_roi,
                            after_metric=after_roi,
                            roi=roi_realish,
                            predicted_roi=pred_realish,
                            trending_topic=trending_topic,
                            reason="self improvement cycle",
                            trigger="run_cycle",
                            performance=after_roi - before_roi,
                            parent_event_id=self._last_mutation_id,
                        )
                    )
                    self._last_mutation_id = event_id
                except Exception as exc:
                    self.logger.exception("evolution history logging failed: %s", exc)
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
                        patch_rate = self.self_coding_engine.patch_db.success_rate()
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
                            df_anom["roi"] = df_anom["revenue"] - df_anom["expense"]
                            anomaly = float(
                                len(DataBot.detect_anomalies(df_anom, "roi"))
                            ) / len(df_anom)
                    elif isinstance(df_anom, list) and df_anom:
                        rois = [
                            float(r.get("revenue", 0.0) - r.get("expense", 0.0))
                            for r in df_anom
                        ]
                        df_list = [{"roi": r} for r in rois]
                        anomaly = float(
                            len(DataBot.detect_anomalies(df_list, "roi"))
                        ) / len(rois)
                except Exception as exc:
                    self.logger.exception("anomaly calculation failed: %s", exc)
                    anomaly = 0.0
                try:
                    self.data_bot.log_evolution_cycle(
                        "self_improvement",
                        before_roi,
                        after_roi,
                          roi_realish,
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
                        reason="cycle complete",
                        trigger="run_cycle",
                        parent_event_id=self._last_mutation_id,
                    )
                    scenario_metrics: dict[str, float] = {}
                    tracker = getattr(self, "tracker", None)
                    if tracker is not None:
                        db = getattr(self.data_bot, "db", None)
                        for name in (
                            "latency_error_rate",
                            "hostile_failures",
                            "misuse_failures",
                            "concurrency_throughput",
                        ):
                            vals = tracker.metrics_history.get(name)
                            if vals:
                                val = float(vals[-1])
                                scenario_metrics[name] = val
                                if db is not None:
                                    try:
                                        db.log_eval("self_improvement", name, val)
                                    except Exception:
                                        pass
                    if scenario_metrics:
                        self._evaluate_scenario_metrics(scenario_metrics)
                        self.logger.info(
                            "scenario metrics",
                            extra=log_record(**scenario_metrics),
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
                    self.logger.exception("data_bot evolution logging failed: %s", exc)
            self.last_run = time.time()
            if self._alignment_agent is None:
                try:
                    self._alignment_agent = AlignmentReviewAgent()
                    self._alignment_agent.start()
                except Exception:
                    self.logger.exception("alignment review agent failed to start")
            delta = after_roi - before_roi
            warnings: dict[str, list[dict[str, Any]]] = {}
            if delta > 0:
                try:
                    metrics = result.roi.__dict__ if result.roi else None
                    settings = SandboxSettings()
                    agent = HumanAlignmentAgent(settings=settings)
                    logs: list[dict[str, Any]] | None = getattr(result, "logs", None)
                    if logs is None:
                        try:
                            logs = get_recent_events(limit=20)
                        except Exception:
                            logs = None
                    try:
                        out = subprocess.check_output(
                            ["git", "show", "-s", "--format=%an,%s"],
                            text=True,
                        ).strip()
                        author, message = out.split(",", 1)
                        commit_info = {"author": author, "message": message}
                    except Exception:
                        commit_info = None
                    warnings = agent.evaluate_changes(actions, metrics, logs, commit_info)
                    if any(warnings.values()):
                        result.warnings = warnings
                    _update_alignment_baseline(settings)
                except Exception as exc:
                    self.logger.exception("improvement flagging failed: %s", exc)
            self.logger.info(
                "roi delta",
                extra=log_record(roi_delta=delta, warnings=warnings),
            )
            self._record_warning_summary(delta, warnings)
            self.roi_delta_ema = (
                1 - self.roi_ema_alpha
            ) * self.roi_delta_ema + self.roi_ema_alpha * delta
            group_idx = None
            if self.patch_db:
                try:
                    repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                    with self.patch_db._connect() as conn:
                        row = conn.execute(
                            "SELECT filename FROM patch_history ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        p = Path(row[0])
                        abs_p = p if p.is_absolute() else repo / p
                        try:
                            mod_name = abs_p.resolve().relative_to(repo).as_posix()
                        except Exception:
                            mod_name = p.name
                        group_idx = self.module_clusters.get(mod_name)
                        if group_idx is None and self.module_index:
                            group_idx = self.module_index.get(mod_name)
                            self.module_clusters[mod_name] = group_idx
                except Exception as exc:
                    self.logger.exception("group index lookup failed: %s", exc)
            if group_idx is not None:
                self.roi_group_history.setdefault(int(group_idx), []).append(delta)
            tracker = getattr(self, "tracker", None)
            raroi_delta = 0.0
            if tracker is not None and len(tracker.raroi_history) >= 2:
                raroi_delta = tracker.raroi_history[-1] - tracker.raroi_history[-2]
            workflow_id = "self_improvement"
            wf_ctx = getattr(environment, "current_context", None)
            try:
                ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
            except Exception:
                pass
            try:
                profile_map = getattr(self.foresight_tracker, "workflow_profiles", None)
                if not isinstance(profile_map, Mapping):
                    profile_map = getattr(self.foresight_tracker, "profile_map", {})
                profile = profile_map.get(workflow_id, workflow_id)
                self.foresight_tracker.capture_from_roi(tracker, workflow_id, profile)
            except Exception:
                self.logger.exception("foresight tracker record failed")
            risk_info: dict[str, object] | None = None
            try:
                if self.foresight_tracker:
                    risk_info = self.foresight_tracker.predict_roi_collapse(workflow_id)
            except Exception:
                self.logger.exception("foresight risk check failed")
            prior_high = self.workflow_high_risk
            self.workflow_risk = risk_info
            high_risk = bool(
                risk_info
                and (
                    risk_info.get("risk") in {"Immediate collapse risk", "Volatile"}
                    or bool(risk_info.get("brittle"))
                )
            )
            self.workflow_high_risk = high_risk
            if high_risk:
                self.workflow_ready = False
                if not prior_high:
                    try:
                        self.enqueue_preventative_fixes([workflow_id])
                    except Exception:
                        self.logger.exception("risk queue enqueue failed")
            if result.warnings is None:
                result.warnings = {}
            result.warnings.setdefault("foresight_risk", [])
            if risk_info:
                result.warnings["foresight_risk"].append(risk_info)
            result.warnings["workflow_high_risk"] = [{"value": high_risk}]
            self.roi_history.append(delta)
            self.raroi_history.append(raroi_delta)
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
                    growth_type=self._last_growth_type,
                ),
            )
            self._evaluate_roi_predictor()
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
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("policy save failed: %s", exc)
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
            with MutationLogger.log_context(
                change="self_improvement_cycle",
                reason="cycle complete",
                trigger="run_cycle",
                workflow_id=0,
                before_metric=before_roi,
                parent_id=self._last_mutation_id,
            ) as mutation:
                mutation["after_metric"] = after_roi
                mutation["performance"] = delta
                mutation["roi"] = roi_realish
            self._last_mutation_id = int(mutation["event_id"])
            try:
                flags = radar_scan()
                if flags:
                    self._handle_relevancy_flags(flags)
            except Exception:
                self.logger.exception("relevancy radar scan failed")
            if self.foresight_tracker and self.workflow_risk is None:
                workflow_id = "self_improvement"
                wf_ctx = getattr(environment, "current_context", None)
                try:
                    ctx_obj = wf_ctx() if callable(wf_ctx) else wf_ctx
                    workflow_id = getattr(ctx_obj, "workflow_id", workflow_id)
                except Exception:
                    pass
                risk_info: dict[str, object] | None = None
                try:
                    risk_info = self.foresight_tracker.predict_roi_collapse(workflow_id)
                except Exception:
                    self.logger.exception("foresight risk check failed")
                prior_high = self.workflow_high_risk
                self.workflow_risk = risk_info
                high_risk = bool(
                    risk_info
                    and (
                        risk_info.get("risk") in {"Immediate collapse risk", "Volatile"}
                        or bool(risk_info.get("brittle"))
                    )
                )
                self.workflow_high_risk = high_risk
                if high_risk:
                    self.workflow_ready = False
                    if not prior_high:
                        try:
                            self.enqueue_preventative_fixes([workflow_id])
                        except Exception:
                            self.logger.exception("risk queue enqueue failed")
                if result.warnings is None:
                    result.warnings = {}
                result.warnings.setdefault("foresight_risk", [])
                if risk_info:
                    result.warnings["foresight_risk"].append(risk_info)
                result.warnings["workflow_high_risk"] = [{"value": high_risk}]
            self.logger.info(
                "cycle complete",
                extra=log_record(roi=roi_realish, predicted_roi=pred_realish),
            )
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
            if self.roi_predictor and self.use_adaptive_roi:
                features = self._collect_action_features()
                try:
                    try:
                        seq, growth_type, _, _ = self.roi_predictor.predict(
                            features, horizon=len(features)
                        )
                    except TypeError:
                        val, growth_type, _, _ = self.roi_predictor.predict(features)
                        seq = [float(val)]
                    roi_estimate = float(seq[-1]) if seq else 0.0
                except Exception:
                    roi_estimate, growth_type = 0.0, "unknown"
                self.logger.info(
                    "growth prediction",
                    extra=log_record(
                        growth_type=growth_type,
                        roi_estimate=roi_estimate,
                        features=features,
                    ),
                )
                self._last_growth_type = growth_type
                if growth_type == "exponential":
                    current_energy *= 1.2
                elif growth_type == "marginal":
                    current_energy *= 0.8
            else:
                self._last_growth_type = None
            if current_energy >= self.energy_threshold and not self._cycle_running:
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

    def status(self) -> dict[str, object]:
        """Expose the latest workflow risk evaluation."""
        return {
            "workflow_ready": self.workflow_ready,
            "workflow_high_risk": self.workflow_high_risk,
            "workflow_risk": self.workflow_risk,
        }


from typing import Any, Callable, Optional, Type, Iterable, Sequence, Dict, List


class ImprovementEngineRegistry:
    """Register and run multiple :class:`SelfImprovementEngine` instances."""

    def __init__(self) -> None:
        self.engines: dict[str, SelfImprovementEngine] = {}

    def register_engine(self, name: str, engine: SelfImprovementEngine) -> None:
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

        tasks = [asyncio.create_task(_run(n, e)) for n, e in self.engines.items()]
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
            energy <= remove_energy or trend <= roi_threshold or projected_roi <= 0.0
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
        rows = (
            GLOBAL_ROUTER.get_connection("synergy_history")
            .execute("SELECT entry FROM synergy_history ORDER BY id")
            .fetchall()
        )
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

    p_dash = sub.add_parser("synergy-dashboard", help="start synergy metrics dashboard")
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

    p_fit = sub.add_parser(
        "fit-truth-adapter", help="retrain TruthAdapter with live and shadow data"
    )
    p_fit.add_argument("live", help="NPZ file with live data")
    p_fit.add_argument("shadow", help="NPZ file with shadow data")

    p_update = sub.add_parser(
        "update-truth-adapter",
        help="incrementally update TruthAdapter or reset if retraining required",
    )
    p_update.add_argument("live", help="NPZ file with live data")
    p_update.add_argument("shadow", help="NPZ file with shadow data")

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

    if args.cmd == "fit-truth-adapter":
        live = np.load(args.live)
        shadow = np.load(args.shadow)
        X = np.vstack([live["X"], shadow["X"]])
        y = np.concatenate([live["y"], shadow["y"]])
        engine = SelfImprovementEngine()
        engine.fit_truth_adapter(X, y)
        return

    if args.cmd == "update-truth-adapter":
        live = np.load(args.live)
        shadow = np.load(args.shadow)
        X = np.vstack([live["X"], shadow["X"]])
        y = np.concatenate([live["y"], shadow["y"]])
        engine = SelfImprovementEngine()
        adapter = engine.truth_adapter
        if adapter.metadata.get("retraining_required"):
            adapter.reset()
            engine.fit_truth_adapter(X, y)
        else:
            adapter.partial_fit(X, y)
        return

    parser.error("unknown command")


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
