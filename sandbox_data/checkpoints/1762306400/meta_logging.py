from __future__ import annotations

"""Meta logging utilities for Menace."""

import json
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    from kafka import KafkaProducer
except Exception:  # pragma: no cover - optional dependency
    KafkaProducer = None  # type: ignore

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore


@dataclass
class LogEvent:
    """Generic event structure for Kafka."""

    event_type: str
    payload: Dict[str, Any]
    ts: str = datetime.utcnow().isoformat()


class KafkaMetaLogger:
    """Publish Menace events to Kafka topics."""

    def __init__(self, brokers: str = "localhost:9092", topic_prefix: str = "menace.events", producer: KafkaProducer | None = None) -> None:
        if producer:
            self.producer = producer
        elif KafkaProducer:
            self.producer = KafkaProducer(
                bootstrap_servers=brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        else:  # pragma: no cover - fallback
            self.producer = None
        self.topic_prefix = topic_prefix

    def log(self, event: LogEvent) -> None:
        if not self.producer:
            return
        topic = f"{self.topic_prefix}.{event.event_type}"
        self.producer.send(topic, event.__dict__)

    def flush(self) -> None:
        if self.producer:
            self.producer.flush()


@dataclass
class SecureLog:
    """Ed25519-based tamper evidence for logs."""

    path: Path
    private_key: Ed25519PrivateKey = field(default_factory=Ed25519PrivateKey.generate)
    hashes: List[str] = field(default_factory=list)

    @property
    def public_key_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def append(self, line: str) -> None:
        sig = self.private_key.sign(line.encode("utf-8"))
        sig_b64 = base64.b64encode(sig).decode()
        self.hashes.append(sig_b64)
        with self.path.open("a") as fh:
            fh.write(f"{sig_b64} {line}\n")

    def export(self, dest: Path) -> None:
        """Verify signatures and write a clean copy to ``dest``."""
        pub = Ed25519PublicKey.from_public_bytes(self.public_key_bytes)
        lines = []
        with self.path.open() as src:
            for idx, raw in enumerate(src):
                sig_b64, text = raw.rstrip("\n").split(" ", 1)
                if idx >= len(self.hashes) or sig_b64 != self.hashes[idx]:
                    raise RuntimeError("hash mismatch")
                try:
                    pub.verify(base64.b64decode(sig_b64), text.encode("utf-8"))
                except Exception as exc:  # pragma: no cover - unlikely
                    raise RuntimeError("signature check failed") from exc
                lines.append(text)
        with dest.open("w") as out:
            for line in lines:
                out.write(f"{line}\n")


class LogCompactor:
    """Compact old JSON logs to Parquet and upload to S3."""

    def __init__(self, bucket: str, prefix: str = "menace", retention_days: int = 7) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.retention_days = retention_days
        self.s3 = boto3.client("s3") if boto3 else None

    def compact(self, files: Iterable[Path]) -> None:
        if not self.s3:
            return
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        rows = []
        keep: list[Path] = []
        for fp in files:
            try:
                ts = datetime.utcfromtimestamp(fp.stat().st_mtime)
            except Exception:
                keep.append(fp)
                continue
            if ts < cutoff:
                try:
                    with open(fp) as fh:
                        rows.extend(json.loads(line) for line in fh)
                except Exception:
                    continue
            else:
                keep.append(fp)
        if not rows or pd is None:
            return
        df = pd.DataFrame(rows)
        tmp = Path("/tmp/compact.parquet")
        df.to_parquet(tmp)
        key = f"{self.prefix}/logs_{int(datetime.utcnow().timestamp())}.parquet"
        try:
            self.s3.upload_file(str(tmp), self.bucket, key)
            for fp in files:
                if fp not in keep:
                    fp.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("failed to upload logs: %s", exc)


__all__ = ["LogEvent", "KafkaMetaLogger", "SecureLog", "LogCompactor"]
