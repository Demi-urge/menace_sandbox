from __future__ import annotations
"""Basic backup and recovery helpers."""


import os
import tarfile
import time
import logging
import subprocess
from pathlib import Path
from typing import Iterable, Sequence
from threading import Event

try:  # pragma: no cover - optional dependency
    import boto3
except Exception:  # pragma: no cover - boto3 may be missing
    boto3 = None  # type: ignore

from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler


class DisasterRecovery:
    """Create and restore backups of data directories."""

    def __init__(
        self,
        data_dirs: Iterable[str] | None = None,
        *,
        backup_dir: str = "backups",
        backup_hosts: Sequence[str] | None = None,
    ) -> None:
        self.data_dirs = [Path(p) for p in (data_dirs or [])]
        self.backup_dir = Path(backup_dir)
        env_hosts = os.getenv("BACKUP_HOSTS", "").split(",")
        self.backup_hosts = [h.strip() for h in (backup_hosts or env_hosts) if h.strip()]
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduler: object | None = None

        self._s3_enabled = boto3 is not None
        if not self._s3_enabled and any(h.startswith("s3://") for h in self.backup_hosts):
            self.logger.warning("boto3 not available; skipping S3 backups")

    def backup(self) -> Path:
        """Archive data directories and return the path."""
        ts = time.strftime("%Y%m%d%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        archive = self.backup_dir / f"backup_{ts}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for d in self.data_dirs:
                if d.exists():
                    tar.add(d, arcname=d.name)
        if self.backup_hosts:
            self._sync_remote(archive)
        return archive

    # ------------------------------------------------------------------
    def _sync_remote(self, archive: Path) -> None:
        for host in self.backup_hosts:
            if host.startswith("s3://"):
                if not self._s3_enabled:
                    self.logger.info("Skipping %s because boto3 is unavailable", host)
                    continue
                bucket_key = host[5:]
                bucket, _, prefix = bucket_key.partition("/")
                key = f"{prefix.rstrip('/')}/{archive.name}" if prefix else archive.name
                try:
                    boto3.client("s3").upload_file(str(archive), bucket, key)
                except Exception as exc:  # pragma: no cover - network issues
                    self.logger.error("failed uploading to %s: %s", host, exc)
            else:
                try:
                    subprocess.run(["rsync", str(archive), host], check=True)
                except Exception as exc:  # pragma: no cover - rsync may fail
                    self.logger.error("rsync to %s failed: %s", host, exc)

    def restore(self, archive: Path) -> None:
        """Extract the given backup archive."""
        if not archive.exists():
            return
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall()

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: float = 3600.0,
        *,
        nodes: Iterable[str] | None = None,
        stop_event: Event | None = None,
    ) -> None:
        """Schedule periodic backups and restores."""
        if self.scheduler:
            return

        env_nodes = os.getenv("NODES", "").split(",")
        self.nodes = [n.strip() for n in (nodes or env_nodes) if n.strip()]
        self._stop = stop_event or Event()

        def _cycle() -> None:
            archive = self.backup()
            for node in self.nodes:
                try:
                    subprocess.run(["rsync", str(archive), f"{node}:{archive.name}"], check=True)
                    subprocess.run(["ssh", node, "tar", "xzf", archive.name], check=True)
                except Exception as exc:
                    self.logger.error("restore on %s failed: %s", node, exc)

        if BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(_cycle, "interval", seconds=interval, id="disaster_recovery")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(_cycle, interval, "disaster_recovery")
            self.scheduler = sched

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self.scheduler:
            return
        if hasattr(self, "_stop") and self._stop:
            self._stop.set()
        if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
            self.scheduler.shutdown(wait=False)
        else:
            self.scheduler.shutdown()
        self.scheduler = None


__all__ = ["DisasterRecovery"]

