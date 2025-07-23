from __future__ import annotations

"""Command line tool for inspecting and training synergy weight settings.

Refer to ``docs/synergy_learning.md`` for an overview of how synergy
weights are learned and how predictions feed into ROI calculations.
"""

import argparse
import json
import sys
import time
from pathlib import Path



LOG_PATH = Path("sandbox_data/synergy_weights.log")


def _load_engine(path: str | None):
    """Initialise SelfImprovementEngine with the given weights path."""
    from menace.self_improvement_engine import SelfImprovementEngine

    return SelfImprovementEngine(interval=0, synergy_weights_path=path)


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
        for entry in hist:
            if not isinstance(entry, dict):
                continue
            roi_delta = float(entry.get("synergy_roi", 0.0))
            learner.update(roi_delta, entry)
        learner.save()
        _log_weights(LOG_PATH, learner.weights)
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


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
