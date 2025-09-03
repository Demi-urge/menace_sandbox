"""CLI for summarising failure regions and escalation levels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Iterable

from failure_fingerprint_store import FailureFingerprintStore


def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="failure_fingerprints.jsonl",
        help="Path to failure fingerprint log",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    store = FailureFingerprintStore(path=args.path)
    region_counts: Counter[str] = Counter()
    escalation: dict[str, int] = defaultdict(int)
    for fp in store._cache.values():
        region = fp.target_region or "unknown"
        region_counts[region] += fp.count
        if fp.escalation_level > escalation[region]:
            escalation[region] = fp.escalation_level

    if not region_counts:
        print("No failures recorded.")
        return 0

    print("region\tcount\tescalation")
    for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{region}\t{count}\t{escalation.get(region, 0)}")
    return 0


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    import sys

    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
