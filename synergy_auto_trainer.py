import json
import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
import time

from . import synergy_weight_cli
from . import synergy_history_db as shd


class SynergyAutoTrainer:
    """Periodically train synergy weights from history."""

    def __init__(
        self,
        *,
        history_file: str | Path = "synergy_history.db",
        weights_file: str | Path = "synergy_weights.json",
        interval: float = 600.0,
        progress_file: str | Path = "last_ts.json",
    ) -> None:
        self.history_file = Path(history_file)
        self.weights_file = Path(weights_file)
        self.interval = float(interval)
        self.progress_file = Path(progress_file)
        self._last_id = 0
        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text())
                self._last_id = int(data.get("last_id", 0))
            except Exception:
                self._last_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --------------------------------------------------------------
    def _load_history(self) -> list[tuple[int, dict[str, float]]]:
        if not self.history_file.exists():
            return []
        try:
            with sqlite3.connect(self.history_file) as conn:
                return shd.fetch_after(conn, self._last_id)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("failed to load history: %s", exc)
            return []

    # --------------------------------------------------------------
    def _train_once(self) -> None:
        hist = self._load_history()
        if not hist:
            return
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        try:
            json.dump([h[1] for h in hist], tmp)
            tmp.close()
            try:
                synergy_weight_cli.cli([
                    "--path",
                    str(self.weights_file),
                    "train",
                    tmp.name,
                ])
            except SystemExit:
                pass
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training failed: %s", exc)
            else:
                self._last_id = hist[-1][0]
                try:
                    self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                    self.progress_file.write_text(json.dumps({"last_id": self._last_id}))
                except Exception:
                    pass
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._train_once()
            self._stop.wait(self.interval)

    # --------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

# --------------------------------------------------------------
def cli(argv: list[str] | None = None) -> int:
    """Run the auto trainer as a standalone process."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-file",
        default="synergy_history.db",
        help="SQLite history database",
    )
    parser.add_argument(
        "--weights-file",
        default="synergy_weights.json",
        help="Synergy weights JSON file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=600.0,
        help="training interval in seconds",
    )
    parser.add_argument(
        "--progress-file",
        default="last_ts.json",
        help="file tracking the last processed history id",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="perform a single training cycle and exit",
    )
    args = parser.parse_args(argv)

    trainer = SynergyAutoTrainer(
        history_file=args.history_file,
        weights_file=args.weights_file,
        interval=args.interval,
        progress_file=args.progress_file,
    )

    if args.run_once:
        trainer._train_once()
        return 0

    trainer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        trainer.stop()
    return 0


def main(argv: list[str] | None = None) -> None:
    import sys

    sys.exit(cli(argv))


__all__ = ["SynergyAutoTrainer", "cli", "main"]
