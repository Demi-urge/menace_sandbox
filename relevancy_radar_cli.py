from __future__ import annotations

"""Command line interface for inspecting relevancy metrics and annotations.

Use ``--retire``, ``--compress`` or ``--replace`` to annotate modules for
subsequent review."""

import argparse
import json
from pathlib import Path
from typing import Iterable

# Path to the relevancy metrics file produced by :class:`RelevancyRadar`.
_METRICS_FILE = Path(__file__).resolve().parent / "sandbox_data" / "relevancy_metrics.json"


# ---------------------------------------------------------------------------
def _load_metrics() -> dict:
    """Return parsed metrics data or an empty dictionary if unavailable."""

    if _METRICS_FILE.exists():
        try:
            with _METRICS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    return {}


def _save_metrics(metrics: dict) -> None:
    """Persist ``metrics`` to :data:`_METRICS_FILE`."""

    _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Score below which modules are displayed",
    )
    parser.add_argument(
        "--retire",
        nargs="*",
        default=[],
        help="Modules to annotate for retirement",
    )
    parser.add_argument(
        "--compress",
        nargs="*",
        default=[],
        help="Modules to annotate for compression",
    )
    parser.add_argument(
        "--replace",
        nargs="*",
        default=[],
        help="Modules to annotate for replacement",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    metrics = _load_metrics()
    changed = False
    for mod in args.retire:
        entry = metrics.setdefault(mod, {"imports": 0, "executions": 0})
        if entry.get("annotation") != "retire":
            entry["annotation"] = "retire"
            changed = True
    for mod in args.compress:
        entry = metrics.setdefault(mod, {"imports": 0, "executions": 0})
        if entry.get("annotation") != "compress":
            entry["annotation"] = "compress"
            changed = True
    for mod in args.replace:
        entry = metrics.setdefault(mod, {"imports": 0, "executions": 0})
        if entry.get("annotation") != "replace":
            entry["annotation"] = "replace"
            changed = True
    if changed:
        _save_metrics(metrics)

    rows = []
    for mod, counts in metrics.items():
        imports = int(counts.get("imports", 0))
        executions = int(counts.get("executions", 0))
        score = imports + executions
        annotation = str(counts.get("annotation", ""))
        if score < args.threshold or annotation:
            rows.append((mod, score, annotation))

    if not rows:
        print("No modules below threshold.")
        return 0

    for mod, score, annotation in sorted(rows, key=lambda r: r[1]):
        if annotation:
            print(f"{mod}: {score} ({annotation})")
        else:
            print(f"{mod}: {score}")
    return 0


# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    import sys

    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
