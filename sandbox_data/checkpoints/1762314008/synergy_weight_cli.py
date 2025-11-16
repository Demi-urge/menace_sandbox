from __future__ import annotations

"""Command line tool for inspecting and training synergy weight settings.

Refer to ``docs/synergy_learning.md`` for an overview of how synergy
weights are learned and how predictions feed into ROI calculations.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from menace.metrics_exporter import (
    synergy_weight_update_failures_total,
    synergy_weight_update_alerts_total,
)
from alert_dispatcher import dispatch_alert
from dynamic_path_router import resolve_path
from context_builder_util import create_context_builder

LOG_PATH = resolve_path("sandbox_data/synergy_weights.log")


def _load_engine(path: str | None):
    """Initialise SelfImprovementEngine with the given weights path."""
    from menace.self_improvement.engine import (
        SelfImprovementEngine,
        SynergyWeightLearner,
        DQNSynergyLearner,
        DoubleDQNSynergyLearner,
        SACSynergyLearner,
        TD3SynergyLearner,
    )

    mapping = {
        "dqn": DQNSynergyLearner,
        "double": DoubleDQNSynergyLearner,
        "double_dqn": DoubleDQNSynergyLearner,
        "ddqn": DoubleDQNSynergyLearner,
        "sac": SACSynergyLearner,
        "td3": TD3SynergyLearner,
    }
    env_name = os.getenv("SYNERGY_LEARNER", "").lower()
    cls = mapping.get(env_name, SynergyWeightLearner)

    builder = create_context_builder()
    return SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        synergy_weights_path=path,
        synergy_learner_cls=cls,
    )


def train_from_history(
    history: Sequence[dict[str, float]], path: str | Path | None
) -> dict[str, float]:
    """Train synergy weights using ``history`` and save them to ``path``."""

    from menace.self_improvement.engine import (
        SynergyWeightLearner,
        DQNSynergyLearner,
        DoubleDQNSynergyLearner,
        SACSynergyLearner,
        TD3SynergyLearner,
    )
    from sandbox_settings import SandboxSettings

    mapping = {
        "dqn": DQNSynergyLearner,
        "double": DoubleDQNSynergyLearner,
        "double_dqn": DoubleDQNSynergyLearner,
        "ddqn": DoubleDQNSynergyLearner,
        "sac": SACSynergyLearner,
        "td3": TD3SynergyLearner,
    }
    env_name = os.getenv("SYNERGY_LEARNER", "").lower()
    cls = mapping.get(env_name, SynergyWeightLearner)

    settings = SandboxSettings()
    weights_path = (
        Path(path)
        if path is not None
        else resolve_path(settings.sandbox_data_dir) / "synergy_weights.json"
    )
    learner = cls(weights_path, lr=settings.synergy_weights_lr)

    try:
        for entry in history:
            if not isinstance(entry, dict):
                continue
            roi_delta = float(entry.get("synergy_roi", 0.0))
            learner.update(roi_delta, entry)
        learner.save()
    except Exception:
        try:
            synergy_weight_update_failures_total.inc()
        except Exception:
            pass
        try:
            dispatch_alert(
                "synergy_weight_update_failure",
                2,
                "Weight update failed",
                {"path": str(weights_path)},
            )
            synergy_weight_update_alerts_total.inc()
        except Exception:
            pass
        raise
    _log_weights(LOG_PATH, learner.weights)
    return learner.weights


def _log_weights(path: Path, weights: dict[str, float]) -> None:
    """Append weight values to ``path`` as a JSON line."""
    entry = {"timestamp": time.time(), **weights}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _plot_history(path: Path) -> None:
    """Display weight history from ``path`` using matplotlib."""
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available", file=sys.stderr)
        return

    history: list[dict[str, float]] = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue

    if not history:
        print("no history", file=sys.stderr)
        return

    keys = [k for k in history[0] if k != "timestamp"]
    xs = range(len(history))
    for key in keys:
        ys = [float(h.get(key, 0.0)) for h in history]
        plt.plot(xs, ys, label=key)

    plt.xlabel("entry")
    plt.ylabel("weight")
    plt.legend()
    plt.tight_layout()
    plt.show()


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", dest="path", default=None, help="Synergy weights JSON file"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="Display current synergy weights")

    p_exp = sub.add_parser("export", help="Export weights to JSON")
    p_exp.add_argument("--out", default="-", help="Output file or '-' for stdout")

    p_imp = sub.add_parser("import", help="Import weights from JSON")
    p_imp.add_argument("file", help="JSON file to read")

    p_train = sub.add_parser("train", help="Train weights from synergy history")
    p_train.add_argument("history", help="Synergy history JSON")

    sub.add_parser("reset", help="Reset weights to defaults")

    p_hist = sub.add_parser("history", help="Display weight history")
    p_hist.add_argument("--log", default=LOG_PATH, help="history log file")
    p_hist.add_argument("--plot", action="store_true", help="show matplotlib plot")

    args = parser.parse_args(argv)

    engine = _load_engine(args.path)
    learner = engine.synergy_learner

    if args.cmd == "show":
        data = json.dumps(learner.weights, indent=2)
        sys.stdout.write(data)
        return 0

    if args.cmd == "export":
        data = json.dumps(learner.weights, indent=2)
        if args.out == "-":
            sys.stdout.write(data)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(data)
        return 0

    if args.cmd == "import":
        with open(args.file, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in learner.weights:
            if key in data:
                learner.weights[key] = float(data[key])
        learner.save()
        _log_weights(LOG_PATH, learner.weights)
        return 0

    if args.cmd == "train":
        with open(args.history, encoding="utf-8") as fh:
            hist = json.load(fh)
        train_from_history(hist, args.path)
        return 0

    if args.cmd == "reset":
        for key in list(learner.weights):
            learner.weights[key] = 1.0
        learner.save()
        _log_weights(LOG_PATH, learner.weights)
        return 0

    if args.cmd == "history":
        log = Path(args.log)
        if args.plot:
            _plot_history(log)
        else:
            if log.exists():
                sys.stdout.write(log.read_text())
        return 0

    parser.error("unknown command")
    return 1


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


__all__ = ["train_from_history", "cli", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
