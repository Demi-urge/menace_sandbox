"""Learning components for the self improvement engine.

This module contains the standalone learner classes that were previously
implemented directly inside :mod:`self_improvement.engine`.  They are now
exposed separately so that other parts of the system can depend on them
without importing the rather hefty engine module.  In addition to the simple
DQN-based learners it also defines an abstract reinforcement learning
interface, :class:`_BaseRLSynergyLearner`, with lightweight SAC and TD3
implementations.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Dict
from types import SimpleNamespace
from abc import ABC, abstractmethod

from sandbox_settings import SandboxSettings

try:  # pragma: no cover - simplified environments
    from menace_sandbox.logging_utils import get_logger, log_record
except Exception:  # pragma: no cover - fallback when logging helpers missing
    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

    def log_record(**fields: object) -> dict[str, object]:  # type: ignore
        return fields

from .init import _atomic_write, get_default_synergy_weights
from menace_sandbox.self_improvement_policy import (
    ActorCriticStrategy,
    DQNStrategy,
    DoubleDQNStrategy,
    SelfImprovementPolicy,
    torch as sip_torch,
)
try:  # pragma: no cover - simplified environments
    from menace_sandbox.dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when helper missing
    from dynamic_path_router import resolve_path  # type: ignore


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
        gamma: float,
        batch_size: int,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        net_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if sip_torch is None:  # pragma: no cover - torch not available
            raise RuntimeError("PyTorch required for TorchReplayStrategy")
        dim = len(get_default_synergy_weights())
        self.dim = dim
        self.model = net_factory(dim, dim, **(net_kwargs or {}))
        self.optimizer = optimizer_cls(
            self.model.parameters(), lr=lr, **(optimizer_kwargs or {})
        )
        self.loss_fn = sip_torch.nn.MSELoss()
        self.train_interval = max(1, int(train_interval))
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.buffer: deque[tuple[Any, float, Any, bool]] = deque(
            maxlen=int(replay_size)
        )
        self.target_model = net_factory(dim, dim, **(net_kwargs or {}))
        self.target_model.load_state_dict(self.model.state_dict())
        self.steps = 0
        self.eval_loss = 0.0

    def update(
        self,
        state: Sequence[float],
        reward: float,
        next_state: Sequence[float],
        done: bool = False,
        extra: Mapping[str, float] | None = None,
    ) -> list[float]:
        self.steps += 1
        if extra:
            reward *= 1.0 + float(extra.get("avg_roi", 0.0))
            reward *= 1.0 + float(extra.get("pass_rate", 0.0))
        state_tensor = sip_torch.tensor(state, dtype=sip_torch.float32)
        next_tensor = sip_torch.tensor(next_state, dtype=sip_torch.float32)
        self.buffer.append((state_tensor, float(reward), next_tensor, bool(done)))
        if (
            self.steps % self.train_interval == 0
            and len(self.buffer) >= self.batch_size
        ):
            batch = random.sample(self.buffer, self.batch_size)
            states, rewards, next_states, dones = zip(*batch)
            states_batch = sip_torch.stack(states)
            rewards_batch = sip_torch.tensor(rewards, dtype=sip_torch.float32).unsqueeze(1)
            next_batch = sip_torch.stack(next_states)
            dones_batch = sip_torch.tensor(dones, dtype=sip_torch.float32).unsqueeze(1)
            with sip_torch.no_grad():
                next_q = self.target_model(next_batch).max(dim=1, keepdim=True)[0]
                targets = rewards_batch + self.gamma * next_q * (1 - dones_batch)
            preds = self.model(states_batch)
            loss = self.loss_fn(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.target_model.load_state_dict(self.model.state_dict())
            self.eval_loss = float(loss.item())
        with sip_torch.no_grad():
            q_vals = self.model(state_tensor).clamp(0.0, 10.0).tolist()
        return [float(v) for v in q_vals]

    def save(self, base: str) -> bool:
        try:
            base_path = Path(resolve_path(base)).with_suffix("")
            buf = io.BytesIO()
            sip_torch.save(self.model.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".model.pt"), buf.getvalue(), binary=True)
            buf = io.BytesIO()
            sip_torch.save(self.target_model.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".target.pt"), buf.getvalue(), binary=True)
            buf = io.BytesIO()
            sip_torch.save(self.optimizer.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".optim.pt"), buf.getvalue(), binary=True)
            data = [
                {
                    "state": s.tolist(),
                    "reward": r,
                    "next_state": n.tolist(),
                    "done": d,
                }
                for s, r, n, d in self.buffer
            ]
            _atomic_write(
                base_path.with_suffix(".replay.json"),
                json.dumps(data),
            )
        except Exception as exc:  # pragma: no cover - disk errors
            logger.exception("failed to save strategy state: %s", exc)
            return False
        try:
            sip_torch.load(base_path.with_suffix(".model.pt"))
            sip_torch.load(base_path.with_suffix(".optim.pt"))
        except Exception as exc:
            logger.warning("failed to validate strategy save: %s", exc)
            return False
        return True

    def load(self, base: str) -> None:
        try:
            base_path = Path(resolve_path(base)).with_suffix("")
            model_file = base_path.with_suffix(".model.pt")
            target_file = base_path.with_suffix(".target.pt")
            optim_file = base_path.with_suffix(".optim.pt")
            if model_file.exists():
                state = sip_torch.load(model_file)
                try:
                    first = next(iter(state.values()))
                    if first.shape[-1] != self.dim:
                        raise ValueError("incompatible checkpoint dimensions")
                    self.model.load_state_dict(state)
                except Exception as exc:
                    logger.warning("skipping model checkpoint: %s", exc)
            if target_file.exists():
                try:
                    self.target_model.load_state_dict(sip_torch.load(target_file))
                except Exception as exc:
                    logger.warning("skipping target checkpoint: %s", exc)
            if optim_file.exists():
                try:
                    self.optimizer.load_state_dict(sip_torch.load(optim_file))
                except Exception as exc:
                    logger.warning("skipping optimizer checkpoint: %s", exc)
            replay_file = base_path.with_suffix(".replay.json")
            if replay_file.exists():
                try:
                    items = json.loads(replay_file.read_text("utf-8"))
                    buf = deque(maxlen=self.buffer.maxlen)
                    if not isinstance(items, list):
                        raise ValueError("invalid replay data")
                    for it in items:
                        if not isinstance(it, dict):
                            raise ValueError("invalid replay item")
                        state = it.get("state")
                        reward = it.get("reward")
                        next_state = it.get("next_state")
                        done = it.get("done")
                        if not (
                            isinstance(state, list)
                            and isinstance(reward, (int, float))
                            and isinstance(next_state, list)
                            and isinstance(done, bool)
                        ):
                            raise ValueError("invalid replay item")
                        buf.append(
                            (
                                sip_torch.tensor(state, dtype=sip_torch.float32),
                                float(reward),
                                sip_torch.tensor(next_state, dtype=sip_torch.float32),
                                bool(done),
                            )
                        )
                    self.buffer = buf
                except Exception as exc:
                    logger.warning("skipping replay buffer: %s", exc)
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
                batch_size=getattr(settings, "synergy_batch_size", 32),
                gamma=getattr(settings, "synergy_gamma", 0.99),
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
        self.metric_names = list(self.weights.keys())
        self.state_names = [f"synergy_{n}" for n in self.metric_names]
        self.dim = len(self.metric_names)
        hp: Dict[str, Any] = dict(hyperparams or {})
        strat_factory = strategy_factory or (lambda **kw: ActorCriticStrategy(**kw))
        self.strategy = strat_factory(**hp)
        self._state: tuple[float, ...] = (0.0,) * self.dim
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
                gamma=float(sy.gamma),
                batch_size=int(sy.batch_size),
                optimizer_kwargs=opt_kwargs,
                net_kwargs=net_kwargs,
            )
            self.buffer = self.nn_strategy.buffer
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data: dict[str, Any] | None = None
        self._checkpoint_compatible = True
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
            keys = [k for k in data.keys() if k in self.weights]
            if len(keys) != self.dim:
                self._checkpoint_compatible = False
                logger.warning(
                    "synergy weight checkpoint incompatible: %s metrics (expected %s)",
                    len(keys),
                    self.dim,
                )
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
        base_path = Path(resolve_path(str(self.path))).with_suffix("")
        if sip_torch is not None:
            try:
                if hasattr(self.strategy, "model") and self.strategy.model is not None:
                    buf = io.BytesIO()
                    sip_torch.save(self.strategy.model.state_dict(), buf)
                    _atomic_write(base_path.with_suffix(".pt"), buf.getvalue(), binary=True)
                if (
                    hasattr(self.strategy, "target_model")
                    and self.strategy.target_model is not None
                ):
                    buf = io.BytesIO()
                    sip_torch.save(self.strategy.target_model.state_dict(), buf)
                    _atomic_write(base_path.with_suffix(".target.pt"), buf.getvalue(), binary=True)
            except Exception as exc:  # pragma: no cover - disk errors
                logger.exception("failed to save DQN models: %s", exc)
            try:
                pkl = base_path.with_suffix(".policy.pkl")
                _atomic_write(pkl, pickle.dumps(self.strategy), binary=True)
            except Exception as exc:  # pragma: no cover - disk errors
                logger.exception("failed to save strategy pickle: %s", exc)

    def update(
        self,
        roi_delta: float,
        deltas: dict[str, float],
        extra: dict[str, float] | None = None,
    ) -> None:
        names = self.state_names
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
        for idx, key in enumerate(self.metric_names):
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


class _ReplayBuffer:
    """Simple replay buffer storing state, action and reward."""

    def __init__(self, size: int) -> None:
        self.data: deque[tuple[sip_torch.Tensor, sip_torch.Tensor, sip_torch.Tensor]] = deque(
            maxlen=size
        )

    def add(self, state: sip_torch.Tensor, action: sip_torch.Tensor, reward: sip_torch.Tensor) -> None:
        self.data.append((state, action, reward))

    def sample(
        self, batch_size: int
    ) -> tuple[sip_torch.Tensor, sip_torch.Tensor, sip_torch.Tensor]:
        import random

        batch = random.sample(self.data, min(batch_size, len(self.data)))
        states, actions, rewards = zip(*batch)
        return sip_torch.stack(states), sip_torch.stack(actions), sip_torch.stack(rewards)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)


class _BaseRLSynergyLearner(ABC):
    """Abstract reinforcement learning based synergy weight learner.

    Subclasses must implement :meth:`update` to adjust ``self.weights`` based on
    the provided ROI delta and per-metric deltas.  Optional hooks
    :meth:`_save_networks` and :meth:`_load_networks` allow persistence of model
    state; by default they are no-ops to provide a safe baseline behaviour.
    """

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        settings: SandboxSettings | None = None,
        target_sync: int | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        sy = getattr(settings, "synergy", None)
        if sy is None:
            sy = {
                "weights_lr": getattr(settings, "synergy_weights_lr", 0.1),
                "train_interval": getattr(settings, "synergy_train_interval", 10),
                "replay_size": getattr(settings, "synergy_replay_size", 100),
                "checkpoint_interval": getattr(
                    settings, "synergy_checkpoint_interval", 50
                ),
                "target_sync": getattr(settings, "synergy_target_sync", 10),
            }
        self.lr = float(lr if lr is not None else sy["weights_lr"])
        if self.lr <= 0:  # pragma: no cover - sanity check
            raise ValueError("learning rate must be positive")
        self.train_interval = int(sy["train_interval"])
        self.replay_size = int(sy["replay_size"])
        self.path = Path(path) if path else Path(settings.synergy_weight_file)
        self.weights = get_default_synergy_weights()
        self.metric_names = list(self.weights.keys())
        self.state_names = [f"synergy_{n}" for n in self.metric_names]
        self.names = self.state_names
        self.dim = len(self.metric_names)
        self._state: tuple[float, ...] = (0.0,) * self.dim
        self._steps = 0
        if target_sync is None:
            target_sync = (
                sy["target_sync"]
                if isinstance(sy, dict)
                else getattr(sy, "target_sync", 10)
            )
        self.target_sync = int(target_sync)
        self.buffer = _ReplayBuffer(self.replay_size)
        self.checkpoint_interval = int(sy["checkpoint_interval"])
        self._save_count = 0
        self.checkpoint_path = self.path.with_suffix(self.path.suffix + ".bak")
        self._checkpoint_compatible = True
        self.load()

    # Weight persistence -------------------------------------------------
    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        data: Dict[str, float] | None = None
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # pragma: no cover - disk issues
            logger.warning("failed to load synergy weights %s: %s", self.path, exc)
            if self.checkpoint_path.exists():
                try:
                    with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    data = None
        if isinstance(data, dict):
            keys = [k for k in data.keys() if k in self.weights]
            if len(keys) != self.dim:
                self._checkpoint_compatible = False
                logger.warning(
                    "synergy weight checkpoint incompatible: %s metrics (expected %s)",
                    len(keys),
                    self.dim,
                )
            self.weights.update({k: float(v) for k, v in data.items() if k in self.weights})
        if self._checkpoint_compatible:
            self._load_networks()
        else:
            logger.warning("skipping network load due to metric mismatch")

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
        self._save_networks()

    # Hooks for subclasses -----------------------------------------------
    def _save_networks(self) -> None:
        """Persist policy and target networks to disk."""

    def _load_networks(self) -> None:
        """Load policy and target networks from disk if present."""

    # API compatible with SynergyWeightLearner ---------------------------
    @abstractmethod
    def update(
        self,
        roi_delta: float,
        deltas: Mapping[str, float],
        extra: Mapping[str, float] | None = None,
    ) -> None:
        """Update synergy weights based on observed deltas and reward."""


class SACSynergyLearner(_BaseRLSynergyLearner):
    """Concrete learner using a tiny SAC-like actor-critic setup."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        hidden_sizes: Sequence[int] | None = None,
        noise: float | None = None,
        batch_size: int | None = None,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        sy = getattr(settings, "synergy", None)
        if hidden_sizes is None:
            hidden_size = (
                getattr(sy, "hidden_size", None)
                if sy is not None
                else getattr(settings, "synergy_hidden_size", 32)
            )
            layers = (
                getattr(sy, "layers", None)
                if sy is not None
                else getattr(settings, "synergy_layers", 1)
            )
            hidden_sizes = [int(hidden_size)] * int(layers)
        hidden_list = [int(h) for h in hidden_sizes]
        if noise is None:
            noise = (
                getattr(sy, "noise", None)
                if sy is not None
                else getattr(settings, "synergy_noise", 0.1)
            )
        if batch_size is None:
            batch_size = (
                getattr(sy, "batch_size", None)
                if sy is not None
                else getattr(settings, "synergy_batch_size", 32)
            )
        if target_sync is None:
            target_sync = (
                getattr(sy, "target_sync", None)
                if sy is not None
                else getattr(settings, "synergy_target_sync", 10)
            )
        super().__init__(path, lr, settings=settings, target_sync=int(target_sync))

        def _build_net(in_dim: int, out_dim: int) -> sip_torch.nn.Sequential:
            layers_list: list[sip_torch.nn.Module] = []
            last = in_dim
            for h in hidden_list:
                layers_list.append(sip_torch.nn.Linear(last, h))
                layers_list.append(sip_torch.nn.ReLU())
                last = h
            layers_list.append(sip_torch.nn.Linear(last, out_dim))
            return sip_torch.nn.Sequential(*layers_list)

        self.actor = _build_net(self.dim, self.dim)
        self.critic = _build_net(self.dim * 2, self.dim)
        self.target_critic = _build_net(self.dim * 2, self.dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = sip_torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = sip_torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.noise = float(noise)
        self.batch_size = int(batch_size)
        self._load_networks()

    def _save_networks(self) -> None:
        base_path = Path(resolve_path(str(self.path))).with_suffix("")
        try:
            buf = io.BytesIO()
            sip_torch.save(self.actor.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".policy.pkl"), buf.getvalue(), binary=True)
            buf = io.BytesIO()
            sip_torch.save(self.target_critic.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".target.pt"), buf.getvalue(), binary=True)
        except Exception as exc:  # pragma: no cover - disk errors
            logger.exception("failed to save SAC models: %s", exc)

    def _load_networks(self) -> None:
        base_path = Path(resolve_path(str(self.path))).with_suffix("")
        pol = base_path.with_suffix(".policy.pkl")
        tgt = base_path.with_suffix(".target.pt")
        try:
            if pol.exists():
                state = sip_torch.load(pol)
                try:
                    first = next(iter(state.values()))
                    if first.shape[-1] != self.dim:
                        raise ValueError("incompatible checkpoint dimensions")
                    self.actor.load_state_dict(state)
                except Exception as exc:
                    logger.warning("skipping actor checkpoint: %s", exc)
            if tgt.exists():
                state = sip_torch.load(tgt)
                try:
                    first = next(iter(state.values()))
                    if first.shape[-1] != self.dim * 2:
                        raise ValueError("incompatible checkpoint dimensions")
                    self.target_critic.load_state_dict(state)
                except Exception as exc:
                    logger.warning("skipping target checkpoint: %s", exc)
        except Exception as exc:  # pragma: no cover - disk errors
            logger.warning("failed to load SAC models: %s", exc)

    def update(
        self,
        roi_delta: float,
        deltas: Mapping[str, float],
        extra: Mapping[str, float] | None = None,
    ) -> None:
        state = sip_torch.tensor(
            [float(deltas.get(n, 0.0)) for n in self.state_names],
            dtype=sip_torch.float32,
        )
        self._state = tuple(float(s) for s in state.tolist())
        action = self.actor(state)
        action = (action + sip_torch.randn_like(action) * self.noise).clamp(0.0, 10.0)
        reward = sip_torch.tensor([float(roi_delta)], dtype=sip_torch.float32)
        self.buffer.add(state, action.detach(), reward)
        if (
            self._steps % self.train_interval == 0
            and len(self.buffer) >= self.batch_size
        ):
            states, actions, rewards = self.buffer.sample(self.batch_size)
            q_pred = self.critic(sip_torch.cat([states, actions], dim=1))
            target = rewards.repeat(1, self.dim)
            critic_loss = sip_torch.nn.functional.mse_loss(q_pred, target)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_loss = -self.critic(
                sip_torch.cat([states, self.actor(states)], dim=1)
            ).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            if self._steps % self.target_sync == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())
        self._steps += 1
        for idx, key in enumerate(self.metric_names):
            self.weights[key] = float(action[idx].item())
        self.save()


class TD3SynergyLearner(_BaseRLSynergyLearner):
    """Concrete learner using a tiny TD3-style algorithm."""

    def __init__(
        self,
        path: Path | None = None,
        lr: float | None = None,
        *,
        hidden_sizes: Sequence[int] | None = None,
        noise: float | None = None,
        batch_size: int | None = None,
        target_sync: int | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        sy = getattr(settings, "synergy", None)
        if hidden_sizes is None:
            hidden_size = (
                getattr(sy, "hidden_size", None)
                if sy is not None
                else getattr(settings, "synergy_hidden_size", 32)
            )
            layers = (
                getattr(sy, "layers", None)
                if sy is not None
                else getattr(settings, "synergy_layers", 1)
            )
            hidden_sizes = [int(hidden_size)] * int(layers)
        hidden_list = [int(h) for h in hidden_sizes]
        if noise is None:
            noise = (
                getattr(sy, "noise", None)
                if sy is not None
                else getattr(settings, "synergy_noise", 0.1)
            )
        if batch_size is None:
            batch_size = (
                getattr(sy, "batch_size", None)
                if sy is not None
                else getattr(settings, "synergy_batch_size", 32)
            )
        if target_sync is None:
            target_sync = (
                getattr(sy, "target_sync", None)
                if sy is not None
                else getattr(settings, "synergy_target_sync", 10)
            )
        super().__init__(path, lr, settings=settings, target_sync=int(target_sync))

        def _build_net(in_dim: int, out_dim: int) -> sip_torch.nn.Sequential:
            layers_list: list[sip_torch.nn.Module] = []
            last = in_dim
            for h in hidden_list:
                layers_list.append(sip_torch.nn.Linear(last, h))
                layers_list.append(sip_torch.nn.ReLU())
                last = h
            layers_list.append(sip_torch.nn.Linear(last, out_dim))
            return sip_torch.nn.Sequential(*layers_list)

        self.actor = _build_net(self.dim, self.dim)
        self.target_actor = _build_net(self.dim, self.dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic1 = _build_net(self.dim * 2, self.dim)
        self.critic2 = _build_net(self.dim * 2, self.dim)
        self.target_c1 = _build_net(self.dim * 2, self.dim)
        self.target_c2 = _build_net(self.dim * 2, self.dim)
        self.target_c1.load_state_dict(self.critic1.state_dict())
        self.target_c2.load_state_dict(self.critic2.state_dict())
        self.actor_opt = sip_torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.c1_opt = sip_torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.c2_opt = sip_torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.noise = float(noise)
        self.batch_size = int(batch_size)
        self.policy_delay = 2
        self._load_networks()

    def _save_networks(self) -> None:
        base_path = Path(resolve_path(str(self.path))).with_suffix("")
        try:
            buf = io.BytesIO()
            sip_torch.save(self.actor.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".policy.pkl"), buf.getvalue(), binary=True)
            buf = io.BytesIO()
            sip_torch.save(self.target_actor.state_dict(), buf)
            _atomic_write(base_path.with_suffix(".target.pt"), buf.getvalue(), binary=True)
        except Exception as exc:  # pragma: no cover - disk errors
            logger.exception("failed to save TD3 models: %s", exc)

    def _load_networks(self) -> None:
        base_path = Path(resolve_path(str(self.path))).with_suffix("")
        pol = base_path.with_suffix(".policy.pkl")
        tgt = base_path.with_suffix(".target.pt")
        try:
            if pol.exists():
                state = sip_torch.load(pol)
                try:
                    first = next(iter(state.values()))
                    if first.shape[-1] != self.dim:
                        raise ValueError("incompatible checkpoint dimensions")
                    self.actor.load_state_dict(state)
                except Exception as exc:
                    logger.warning("skipping actor checkpoint: %s", exc)
            if tgt.exists():
                state = sip_torch.load(tgt)
                try:
                    first = next(iter(state.values()))
                    if first.shape[-1] != self.dim:
                        raise ValueError("incompatible checkpoint dimensions")
                    self.target_actor.load_state_dict(state)
                except Exception as exc:
                    logger.warning("skipping target checkpoint: %s", exc)
        except Exception as exc:  # pragma: no cover - disk errors
            logger.warning("failed to load TD3 models: %s", exc)

    def update(
        self,
        roi_delta: float,
        deltas: Mapping[str, float],
        extra: Mapping[str, float] | None = None,
    ) -> None:
        state = sip_torch.tensor(
            [float(deltas.get(n, 0.0)) for n in self.state_names],
            dtype=sip_torch.float32,
        )
        self._state = tuple(float(s) for s in state.tolist())
        action = self.actor(state)
        action = (action + sip_torch.randn_like(action) * self.noise).clamp(0.0, 10.0)
        reward = sip_torch.tensor([float(roi_delta)], dtype=sip_torch.float32)
        self.buffer.add(state, action.detach(), reward)
        if len(self.buffer) >= self.batch_size:
            states, actions, rewards = self.buffer.sample(self.batch_size)
            # critic update
            target_actions = self.target_actor(states)
            target_q1 = self.target_c1(sip_torch.cat([states, target_actions], dim=1))
            target_q2 = self.target_c2(sip_torch.cat([states, target_actions], dim=1))
            target_q = rewards.repeat(1, self.dim)
            c1_loss = sip_torch.nn.functional.mse_loss(
                self.critic1(sip_torch.cat([states, actions], dim=1)), target_q
            )
            c2_loss = sip_torch.nn.functional.mse_loss(
                self.critic2(sip_torch.cat([states, actions], dim=1)), target_q
            )
            self.c1_opt.zero_grad()
            c1_loss.backward()
            self.c1_opt.step()
            self.c2_opt.zero_grad()
            c2_loss.backward()
            self.c2_opt.step()
            # actor update with delay
            if self._steps % self.policy_delay == 0:
                actor_loss = -self.critic1(
                    sip_torch.cat([states, self.actor(states)], dim=1)
                ).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                if self._steps % self.target_sync == 0:
                    self.target_actor.load_state_dict(self.actor.state_dict())
            if self._steps % self.target_sync == 0:
                self.target_c1.load_state_dict(self.critic1.state_dict())
                self.target_c2.load_state_dict(self.critic2.state_dict())
        self._steps += 1
        for idx, key in enumerate(self.metric_names):
            self.weights[key] = float(action[idx].item())
        self.save()


__all__ = [
    "TorchReplayStrategy",
    "SynergyWeightLearner",
    "DQNSynergyLearner",
    "DoubleDQNSynergyLearner",
    "SACSynergyLearner",
    "TD3SynergyLearner",
]

