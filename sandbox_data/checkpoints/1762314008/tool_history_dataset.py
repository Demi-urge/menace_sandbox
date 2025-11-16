"""Export training data for the tool predictor.

This module inspects historical bot interactions alongside snapshots of the
capability registry to derive training pairs of
``{"bot_description": ..., "added_tool": ...}``.

The goal is to provide a small language model with examples of how bots evolve
as new tools are introduced. Each record represents the state of a bot just
before a capability was added and the capability that subsequently appeared.

The capability registry is expected to be serialised as JSON files mapping a
capability name to a list of implementing bots. When multiple snapshots are
provided the files are processed in order and differences between consecutive
snapshots yield new training examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - allow execution from repo root
    from ..bot_registry import BotRegistry
    from ..capability_registry import CapabilityRegistry  # noqa: F401 - imported for type hints
except Exception:  # pragma: no cover - allow running as a script
    from bot_registry import BotRegistry  # type: ignore
    from capability_registry import CapabilityRegistry  # type: ignore  # noqa: F401


# ---------------------------------------------------------------------------
# helpers


def _bot_descriptions(registry: BotRegistry) -> Dict[str, str]:
    """Return a simple text description for each bot based on interactions."""

    descriptions: Dict[str, str] = {}
    for bot in registry.graph.nodes:
        connections = [n for n, _ in registry.connections(bot, depth=1)]
        if connections:
            desc = f"Bot {bot} interacts with {', '.join(sorted(connections))}."
        else:
            desc = f"Bot {bot}."
        descriptions[bot] = desc
    return descriptions


def _snapshot_to_bot_caps(snapshot: Dict[str, Sequence[str]]) -> Dict[str, set[str]]:
    """Convert ``capability -> [bots]`` mapping to ``bot -> {capabilities}``."""

    result: Dict[str, set[str]] = {}
    for cap, bots in snapshot.items():
        for bot in bots:
            result.setdefault(bot, set()).add(cap)
    return result


def build_rows(
    registry: BotRegistry, snapshots: Iterable[Dict[str, Sequence[str]]]
) -> List[Dict[str, str]]:
    """Yield training rows from ``snapshots`` using ``registry`` for context."""

    descriptions = _bot_descriptions(registry)
    rows: List[Dict[str, str]] = []
    previous: Dict[str, set[str]] | None = None
    for snap in snapshots:
        current = _snapshot_to_bot_caps(snap)
        if previous is not None:
            for bot, caps in current.items():
                old_caps = previous.get(bot, set())
                added = caps - old_caps
                if not added:
                    continue
                desc = descriptions.get(bot, f"Bot {bot}.")
                if old_caps:
                    desc += " Current capabilities: " + ", ".join(sorted(old_caps)) + "."
                for cap in sorted(added):
                    rows.append({"bot_description": desc, "added_tool": cap})
        previous = current
    return rows


def export_dataset(
    registry_path: Path, snapshot_paths: Sequence[Path], output: Path
) -> int:
    """Load data sources and write the resulting dataset to ``output``.

    Parameters
    ----------
    registry_path:
        Location of the persisted :class:`BotRegistry` data.
    snapshot_paths:
        Ordered paths to JSON files representing :class:`CapabilityRegistry`
        snapshots.
    output:
        Destination file that will receive newline delimited JSON records.

    Returns
    -------
    int
        Number of rows written.
    """

    registry = BotRegistry(persist=registry_path)
    snapshots: List[Dict[str, Sequence[str]]] = []
    for path in snapshot_paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                snapshots.append(json.load(fh))
        except Exception:
            # corrupt snapshot; skip silently – dataset generation is best effort
            continue
    rows = build_rows(registry, snapshots)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh)
            fh.write("\n")
    return len(rows)


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Export tool predictor dataset")
    parser.add_argument(
        "--registry", type=Path, required=True, help="Path to bot registry DB"
    )
    parser.add_argument(
        "--caps", type=Path, nargs="+", required=True, help="Capability registry snapshots"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tool_history_dataset.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()
    count = export_dataset(args.registry, args.caps, args.out)
    print(f"wrote {count} records to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

