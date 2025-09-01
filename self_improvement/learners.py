"""Learning components for the self improvement engine.

This module contains the standalone learner classes that were previously
implemented directly inside :mod:`self_improvement.engine`.  They are now
exposed separately so that other parts of the system can depend on them
without importing the rather hefty engine module.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Dict
from types import SimpleNamespace

from sandbox_settings import SandboxSettings

try:  # pragma: no cover - simplified environments
    from ..logging_utils import get_logger, log_record
except Exception:  # pragma: no cover - fallback when logging helpers missing
    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

    def log_record(**fields: object) -> dict[str, object]:  # type: ignore
        return fields

from .init import _atomic_write, get_default_synergy_weights
from ..self_improvement_policy import (
    ActorCriticStrategy,
    DQNStrategy,
    DoubleDQNStrategy,
    SelfImprovementPolicy,
    torch as sip_torch,
)


logger = get_logger(__name__)


class TorchReplayStrategy:
    """Minimal DQN-style learner with replay buffer and configurable optimiser."""

    def __init__(
        self,
        *,
        net_factory: Callable[[int, int, Any], Any],
        optimizer_cls: Any,
        lr: float,
        train_interval: int,
        replay_size: int,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        net_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if sip_torch is None:  # pragma: no cover - torch not available
            raise RuntimeError("PyTorch required for TorchReplayStrategy")
        self.model = net_factory(7, 7, **(net_kwargs or {}))
        self.optimizer = optimizer_cls(
            self.model.parameters(), lr=lr, **(optimizer_kwargs or {})
        )
        self.loss_fn = sip_torch.nn.MSELoss()
        self.train_interval = max(1, int(train_interval))
        self.buffer: deque[tuple[Any, float]] = deque(maxlen=int(replay_size))
        self.steps = 0
        self.eval_loss = 0.0

    def update(
        self,
        state: Sequence[float],
        reward: float,
        extra: Mapping[str, float] | None = None,
    ) -> list[float]:
        self.steps += 1
        if extra:
            reward *= 1.0 + float(extra.get("avg_roi", 0.0))
            reward *= 1.0 + float(extra.get("pass_rate", 0.0))
        state_tensor = sip_torch.tensor(state, dtype=sip_torch.float32)
        self.buffer.append((state_tensor, float(reward)))
        if self.steps % self.train_interval == 0 and self.buffer:
            states, rewards = zip(*self.buffer)
            states_batch = sip_torch.stack(states)
            rewards_batch = (
                sip_torch.tensor(rewards, dtype=sip_torch.float32).unsqueeze(1)
            )
            targets = states_batch * rewards_batch
            preds = self.model(states_batch)
            loss = self.loss_fn(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.eval_loss = float(loss.item())
            self.buffer.clear()
        with sip_torch.no_grad():
            q_vals = self.model(state_tensor).clamp(0.0, 10.0).tolist()
        return [float(v) for v in q_vals]

    def save(self, base: str) -> bool:
        try:
            buf = io.BytesIO()
            sip_torch.save(self.model.state_dict(), buf)
            _atomic_write(Path(base + ".model.pt"), buf.getvalue(), binary=True)
            buf = io.BytesIO()
            sip_torch.save(self.optimizer.state_dict(), buf)
            _atomic_write(Path(base + ".optim.pt"), buf.getvalue(), binary=True)
            _atomic_write(
                Path(base + ".replay.pkl"),
                pickle.dumps(list(self.buffer)),
                binary=True,
            )
        except Exception as exc:  # pragma: no cover - disk errors
            logger.exception("failed to save strategy state: %s", exc)
            return False
        try:
            sip_torch.load(base + ".model.pt")
            sip_torch.load(base + ".optim.pt")
        except Exception as exc:
            logger.warning("failed to validate strategy save: %s", exc)
            return False
        return True

    def load(self, base: str) -> None:
        try:
            model_file = Path(base + ".model.pt")
            optim_file = Path(base + ".optim.pt")
            if model_file.exists():
                self.model.load_state_dict(sip_torch.load(model_file))
            if optim_file.exists():
                self.optimizer.load_state_dict(sip_torch.load(optim_file))
            replay_file = Path(base + ".replay.pkl")
            if replay_file.exists():
                with open(replay_file, "rb") as fh:
                    items = pickle.load(fh)
                self.buffer = deque(items, maxlen=self.buffer.maxlen)
        except Exception as exc:  # pragma: no cover - disk errors
            logger.warning("failed to load strategy state: %s", exc)


class SynergyWeightLearner:
    """Learner adjusting synergy weights using a simple RL policy.

    All tunable parameters are loaded from :class:`SandboxSettings` allowing
    configuration via environment variables.
    """

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        settings: SandboxSettings | None = None,
        strategy_factory: Callable[..., Any] | None = None,
        net_factory: Callable[..., Any] | None = None,
        hyperparams: Mapping[str, Any] | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        sy = getattr(settings, "synergy", None)
        if sy is None:
            sy = SimpleNamespace(
                weights_lr=getattr(settings, "synergy_weights_lr", 0.1),
                train_interval=getattr(settings, "synergy_train_interval", 10),
                replay_size=getattr(settings, "synergy_replay_size", 100),
                checkpoint_interval=getattr(
                    settings, "synergy_checkpoint_interval", 50
                ),
                python_fallback=getattr(settings, "synergy_python_fallback", True),
                python_max_replay=getattr(settings, "synergy_python_max_replay", 1000),
                hidden_size=getattr(settings, "synergy_hidden_size", 32),
                layers=getattr(settings, "synergy_layers", 1),
                optimizer=getattr(settings, "synergy_optimizer", "adam"),
            )
        if lr is None:
            lr = float(sy.weights_lr)
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.train_interval = int(sy.train_interval)
        self.replay_size = int(sy.replay_size)
        self.path = Path(path) if path else Path(settings.synergy_weight_file)
        self.lr = lr
        self.weights = get_default_synergy_weights()
        hp: Dict[str, Any] = dict(hyperparams or {})
        strat_factory = strategy_factory or (lambda **kw: ActorCriticStrategy(**kw))
        self.strategy = strat_factory(**hp)
        self._state: tuple[float, ...] = (0.0,) * 7
        self._steps = 0
        self.target_sync = int(hp.get("target_sync", 10))
        self.eval_loss = 0.0
        self.checkpoint_interval = int(sy.checkpoint_interval)
        self._save_count = 0
        self.checkpoint_path = self.path.with_suffix(self.path.suffix + ".bak")
        allow_py = bool(sy.python_fallback)
        max_replay = int(sy.python_max_replay)
        if sip_torch is None:
            if not allow_py or self.replay_size > max_replay:
                raise RuntimeError(
                    "PyTorch is required for SynergyWeightLearner; pure-Python "
                    "fallback is disabled or too slow",
                )
            logger.warning(
                "PyTorch not installed - using %s strategy",
                type(self.strategy).__name__,
            )
            self.buffer: deque[tuple[Any, float]] = deque(maxlen=self.replay_size)
        else:
            hidden = int(sy.hidden_size)
            layers = int(sy.layers)
            opt_name = str(sy.optimizer).lower()
            opt_cls_default = {
                "sgd": sip_torch.optim.SGD,
                "adam": sip_torch.optim.Adam,
            }.get(opt_name, sip_torch.optim.Adam)

            def default_net(
                i: int,
                o: int,
                *,
                hidden_size: int = hidden,
                layers: int = layers,
            ) -> Any:
                modules: list[Any] = []
                in_dim = i
                for _ in range(max(0, layers)):
                    modules.append(sip_torch.nn.Linear(in_dim, hidden_size))
                    modules.append(sip_torch.nn.ReLU())
                    in_dim = hidden_size
                modules.append(sip_torch.nn.Linear(in_dim, o))
                return sip_torch.nn.Sequential(*modules)

            net_factory = net_factory or default_net
            opt_kwargs = dict(hp.get("optimizer_kwargs", {}))
            net_kwargs = dict(hp.get("net_kwargs", {}))
            opt_cls = hp.get("optimizer_cls", opt_cls_default)
            self.nn_strategy = TorchReplayStrategy(
                net_factory=net_factory,
                optimizer_cls=opt_cls,
                lr=self.lr,
                train_interval=self.train_interval,
                replay_size=self.replay_size,
                optimizer_kwargs=opt_kwargs,
                net_kwargs=net_kwargs,
            )
            self.buffer = self.nn_strategy.buffer
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data: dict[str, Any] | None = None
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("failed to load synergy weights %s: %s", self.path, exc)
            if self.checkpoint_path.exists():
                try:
                    with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        logger.warning(
                            "recovered synergy weights from checkpoint %s",
                            self.checkpoint_path,
                        )
                except Exception as exc2:
                    logger.warning(
                        "failed to load checkpoint %s: %s",
                        self.checkpoint_path,
                        exc2,
                    )
        if isinstance(data, dict):
            self.weights.update({k: float(v) for k, v in data.items() if k in self.weights})

    def save(self) -> None:
        if not self.path:
            return
        try:
            _atomic_write(self.path, json.dumps(self.weights))
            self._save_count += 1
            if self._save_count % self.checkpoint_interval == 0:
                _atomic_write(self.checkpoint_path, json.dumps(self.weights))
        except Exception as exc:  # pragma: no cover - disk errors
            logger.warning("failed to save synergy weights %s: %s", self.path, exc)
        base = os.path.splitext(self.path)[0]
        if sip_torch is not None:
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
                    _atomic_write(Path(base + ".target.pt"), buf.getvalue(), binary=True)
            except Exception as exc:  # pragma: no cover - disk errors
                logger.exception("failed to save DQN models: %s", exc)
            try:
                pkl = Path(base + ".policy.pkl")
                _atomic_write(pkl, pickle.dumps(self.strategy), binary=True)
            except Exception as exc:  # pragma: no cover - disk errors
                logger.exception("failed to save strategy pickle: %s", exc)

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


class DQNSynergyLearner(SynergyWeightLearner):
    """Synergy learner using a configurable DQN style strategy.

    ``synergy_strategy`` selects the DQN variant while ``synergy_target_sync``
    controls how often the target network is synchronised.
    """

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        strategy: str | None = None,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        sy = getattr(settings, "synergy", None)
        if sy is None:
            sy = SimpleNamespace(
                strategy=getattr(settings, "synergy_strategy", "dqn"),
                target_sync=getattr(settings, "synergy_target_sync", 10),
            )
        strategy = strategy or sy.strategy
        target_sync = int(target_sync or sy.target_sync)
        strat_map = {
            "dqn": DQNStrategy,
            "double_dqn": DoubleDQNStrategy,
            "sac": ActorCriticStrategy,
            "td3": ActorCriticStrategy,
        }
        strategy_factory = strat_map.get(strategy, DQNStrategy)
        super().__init__(
            path=path,
            lr=lr,
            settings=settings,
            strategy_factory=strategy_factory,
            hyperparams={"target_sync": target_sync},
        )


class DoubleDQNSynergyLearner(DQNSynergyLearner):
    """Synergy learner using a Double DQN strategy."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        super().__init__(
            path,
            lr,
            strategy="double_dqn",
            target_sync=target_sync,
            settings=settings,
        )


class SACSynergyLearner(DQNSynergyLearner):
    """Synergy learner using a simplified SAC strategy."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        super().__init__(
            path,
            lr,
            strategy="sac",
            target_sync=target_sync,
            settings=settings,
        )


class TD3SynergyLearner(DQNSynergyLearner):
    """Synergy learner using a simplified TD3 strategy."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        super().__init__(
            path,
            lr,
            strategy="td3",
            target_sync=target_sync,
            settings=settings,
        )


__all__ = [
    "TorchReplayStrategy",
    "SynergyWeightLearner",
    "DQNSynergyLearner",
    "DoubleDQNSynergyLearner",
    "SACSynergyLearner",
    "TD3SynergyLearner",
]

