from __future__ import annotations

"""Meta logging utilities for Menace."""

import json
import base64
import hmac
import hashlib
import secrets
try:  # pragma: no cover - optional dependency
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _CRYPTO_AVAILABLE = True
except Exception:  # pragma: no cover - cryptography may be unavailable
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    serialization = None  # type: ignore
    _CRYPTO_AVAILABLE = False
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
class _HMACKey:
    """Lightweight signing helper used when ``cryptography`` is unavailable."""

    def __init__(self, key: bytes | None = None) -> None:
        self.key = key or secrets.token_bytes(32)

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self.key, data, hashlib.blake2b).digest()

    def verify(self, signature: bytes, data: bytes) -> None:
        expected = self.sign(data)
        if not hmac.compare_digest(signature, expected):
            raise RuntimeError("signature check failed")

    @property
    def public_key_bytes(self) -> bytes:
        return self.key


def _generate_signing_key() -> Any:
    if _CRYPTO_AVAILABLE:
        return Ed25519PrivateKey.generate()  # type: ignore[call-arg]
    return _HMACKey()


class SecureLog:
    """Ed25519-based tamper evidence for logs."""

    path: Path
    private_key: Any = field(default_factory=_generate_signing_key)
    hashes: List[str] = field(default_factory=list)

    @property
    def public_key_bytes(self) -> bytes:
        if _CRYPTO_AVAILABLE:
            return self.private_key.public_key().public_bytes(  # type: ignore[union-attr]
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        return self.private_key.public_key_bytes

    def _sign(self, line: str) -> bytes:
        payload = line.encode("utf-8")
        if _CRYPTO_AVAILABLE:
            return self.private_key.sign(payload)  # type: ignore[union-attr]
        return self.private_key.sign(payload)

    def _verify(self, signature: bytes, line: str) -> None:
        payload = line.encode("utf-8")
        if _CRYPTO_AVAILABLE:
            pub = self.private_key.public_key()  # type: ignore[union-attr]
            pub.verify(signature, payload)
            return
        self.private_key.verify(signature, payload)

    def append(self, line: str) -> None:
        sig = self._sign(line)
        sig_b64 = base64.b64encode(sig).decode()
        self.hashes.append(sig_b64)
        with self.path.open("a") as fh:
            fh.write(f"{sig_b64} {line}\n")

    def export(self, dest: Path) -> None:
        """Verify signatures and write a clean copy to ``dest``."""
        lines = []
        with self.path.open() as src:
            for idx, raw in enumerate(src):
                sig_b64, text = raw.rstrip("\n").split(" ", 1)
                if idx >= len(self.hashes) or sig_b64 != self.hashes[idx]:
                    raise RuntimeError("hash mismatch")
                try:
                    self._verify(base64.b64decode(sig_b64), text)
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
