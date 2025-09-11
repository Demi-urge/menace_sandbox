import argparse
import json
from pathlib import Path
from dynamic_path_router import resolve_path


def purge(log_path: str) -> None:
    path = Path(log_path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(entry.get("event_type", "")).startswith("visual_agent"):
                continue
            fh.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Purge visual agent events from an audit log",
    )
    parser.add_argument(
        "log",
        nargs="?",
        default=str(resolve_path("logs/audit_log.jsonl")),
        help="path to JSONL audit log",
    )
    args = parser.parse_args()
    purge(args.log)
