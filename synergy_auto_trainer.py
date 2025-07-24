import json
import logging
import os
import tempfile
import threading
from pathlib import Path
import time
import asyncio
from filelock import FileLock

from .metrics_exporter import Gauge

# Prometheus gauges tracking trainer state
synergy_trainer_iterations = Gauge(
    "synergy_trainer_iterations",
    "Total number of synergy training cycles executed",
)
synergy_trainer_last_id = Gauge(
    "synergy_trainer_last_id",
    "ID of the last processed synergy history entry",
)

from . import synergy_weight_cli
from . import synergy_history_db as shd


class SynergyWeightCliError(RuntimeError):
    """Raised when ``synergy_weight_cli`` exits with a non-zero status."""

    def __init__(self, code: int) -> None:
        super().__init__(f"synergy_weight_cli exited with code {code}")
        self.code = code


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
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))

        if not self.history_file.exists():
            new_path = data_dir / self.history_file.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.touch()
            logging.getLogger(self.__class__.__name__).warning(
                "history file %s missing - created empty file at %s",
                self.history_file,
                new_path,
            )
            self.history_file = new_path

        weights_path = self.weights_file
        if not weights_path.exists():
            weights_path = data_dir / self.weights_file.name
        lock = FileLock(str(weights_path) + ".lock")
        with lock:
            if not weights_path.exists():
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                weights_path.touch()
                logging.getLogger(self.__class__.__name__).warning(
                    "weights file %s missing - created empty file at %s",
                    self.weights_file,
                    weights_path,
                )
            self.weights_file = weights_path

        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text())
                self._last_id = int(data.get("last_id", 0))
            except Exception:
                self._last_id = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._task: asyncio.Task | None = None

    # --------------------------------------------------------------
    def _load_history(self) -> list[tuple[int, dict[str, float]]]:
        if not self.history_file.exists():
            return []
        try:
            with shd.connect_locked(self.history_file) as conn:
                return shd.fetch_after(conn, self._last_id)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception(
                "failed to load history from %s: %s", self.history_file, exc
            )
            raise

    # --------------------------------------------------------------
    def _train_once(self) -> None:
        hist = self._load_history()
        if not hist:
            return
        # update metrics for each successful cycle
        try:
            synergy_trainer_iterations.inc()
        except Exception:
            pass
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        try:
            json.dump([h[1] for h in hist], tmp)
            tmp.close()
            rc = 0
            lock = FileLock(str(self.weights_file) + ".lock")
            try:
                with lock:
                    rc = synergy_weight_cli.cli([
                        "--path",
                        str(self.weights_file),
                        "train",
                        tmp.name,
                    ])
            except SystemExit as exc:
                rc = int(getattr(exc, "code", 1) or 0)
                self.logger.info("synergy_weight_cli requested exit (%s)", rc)
            except Exception as exc:  # pragma: no cover - runtime issues
                rc = 1
                self.logger.warning("synergy_weight_cli failed: %s", exc)
            else:
                if rc != 0:
                    self.logger.error(
                        "synergy_weight_cli returned non-zero exit code %s", rc
                    )
                    raise SynergyWeightCliError(rc)
            finally:
                self._last_id = hist[-1][0]
                try:
                    synergy_trainer_last_id.set(float(self._last_id))
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.exception("training failed: %s", exc)
            raise
        finally:
            try:
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                self.progress_file.write_text(json.dumps({"last_id": self._last_id}))
            except Exception as exc:
                self.logger.exception("failed to update progress: %s", exc)
            try:
                os.unlink(tmp.name)
            except Exception as exc:
                self.logger.warning("failed to remove temp file %s: %s", tmp.name, exc)

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._train_once()
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training cycle failed: %s", exc)
            if self._stop.wait(self.interval):
                break

    # --------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(self._last_id))
        except Exception:
            pass

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self._loop()
                except Exception as exc:  # pragma: no cover - runtime issues
                    self.logger.exception("trainer loop crashed: %s", exc)
                if self._stop.wait(self.interval):
                    break

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # --------------------------------------------------------------
    async def _async_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._train_once)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.exception("training cycle failed: %s", exc)
            if await asyncio.to_thread(self._stop.wait, self.interval):
                break

    def start_async(self) -> None:
        if self._task:
            return
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(self._last_id))
        except Exception:
            pass
        self._task = asyncio.create_task(self._async_loop())

    async def stop_async(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

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
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="run trainer within asyncio event loop",
    )
    args = parser.parse_args(argv)

    trainer = SynergyAutoTrainer(
        history_file=args.history_file,
        weights_file=args.weights_file,
        interval=args.interval,
        progress_file=args.progress_file,
    )

    if args.run_once:
        try:
            synergy_trainer_iterations.set(0.0)
            synergy_trainer_last_id.set(float(trainer._last_id))
        except Exception:
            pass
        trainer._train_once()
        return 0

    if args.async_mode:
        async def runner() -> int:
            trainer.start_async()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                await trainer.stop_async()
            return 0

        return asyncio.run(runner())

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


__all__ = [
    "SynergyAutoTrainer",
    "cli",
    "main",
    "synergy_trainer_iterations",
    "synergy_trainer_last_id",
    "SynergyWeightCliError",
]
