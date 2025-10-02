"""Append-only audit trail with optional Ed25519 signatures.

If instantiated without a private key the trail accepts entries without
signatures. Such unsigned entries begin with ``"-"`` and are ignored during
verification.
"""
from __future__ import annotations

import base64
import json
import shutil
from typing import Iterable

import logging


LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _CRYPTO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    Ed25519PrivateKey = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc


def _warn_once(message: str, *, level: int = logging.WARNING) -> None:
    """Emit ``message`` once per interpreter session at ``level`` severity."""

    if message in _warn_once._seen:  # type: ignore[attr-defined]
        return
    _warn_once._seen.add(message)  # type: ignore[attr-defined]
    LOGGER.log(level, message)


_warn_once._seen = set()  # type: ignore[attr-defined]


class AuditTrail:
    def __init__(
        self,
        path: str,
        private_key: bytes | None = None,
        handler: logging.Handler | None = None,
    ) -> None:
        """Create an audit trail at *path*.

        If *private_key* is ``None`` no signing is performed and entries are
        prefixed with ``"-"``. A warning is logged in this case.
        When ``handler`` is provided entries are written via the handler,
        enabling external log rotation mechanisms.
        """
        self.path = path
        self.handler = handler
        self._crypto_enabled = (
            Ed25519PrivateKey is not None and serialization is not None
        )
        if not self._crypto_enabled and _CRYPTO_IMPORT_ERROR:
            _warn_once(
                "cryptography library unavailable; audit trail signatures disabled "
                f"({_CRYPTO_IMPORT_ERROR})",
                level=logging.INFO,
            )
        if private_key and not self._crypto_enabled:
            _warn_once(
                "Private key provided but cryptography is unavailable; signatures disabled",
                level=logging.WARNING,
            )
            private_key = None
        if private_key and self._crypto_enabled:
            try:
                # Allow both raw and DER-encoded private keys. The latter is
                # produced by ``openssl pkey`` when following the README
                # instructions.
                if len(private_key) != 32:
                    key_obj = serialization.load_der_private_key(
                        private_key, password=None
                    )
                    if not isinstance(key_obj, Ed25519PrivateKey):
                        raise ValueError("Not an Ed25519 private key")
                    private_key = key_obj.private_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PrivateFormat.Raw,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                self.private_key = Ed25519PrivateKey.from_private_bytes(private_key)
            except Exception as exc:
                raise ValueError("Invalid Ed25519 private key format") from exc
        else:
            if private_key and not self._crypto_enabled:
                self.private_key = None
            else:
                LOGGER.info(
                    "AuditTrail created without signing key; entries will not be signed"
                )
                self.private_key = None

    @property
    def public_key_bytes(self) -> bytes:
        if not self.private_key:
            raise ValueError("Private key not loaded")
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def record(self, message: str | object) -> None:
        """Append ``message`` to the trail, signing when possible."""
        if not isinstance(message, str):
            try:
                message = json.dumps(message, sort_keys=True)
            except Exception:
                message = str(message)
        if self.private_key:
            sig = self.private_key.sign(message.encode())
            prefix = base64.b64encode(sig).decode()
        else:
            prefix = "-"
        line = f"{prefix} {message}\n"
        if self.handler:
            record = logging.LogRecord(
                name="AuditTrail",
                level=logging.INFO,
                pathname=self.path,
                lineno=0,
                msg=line,
                args=(),
                exc_info=None,
            )
            self.handler.emit(record)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    def verify(self, public_key: bytes) -> bool:
        if not self._crypto_enabled:
            _warn_once(
                "cryptography unavailable; assuming audit log is valid",
                level=logging.INFO,
            )
            return True
        pub = Ed25519PublicKey.from_public_bytes(public_key)
        with open(self.path, "rb") as f:
            for line in f:
                try:
                    sig_b64, msg = line.decode().split(" ", 1)
                    if sig_b64 == "-":
                        # unsigned entry
                        continue
                    sig = base64.b64decode(sig_b64)
                    pub.verify(sig, msg.rstrip().encode())
                except Exception:
                    return False
        return True

    def export(self, dest: str, public_key: bytes) -> None:
        """Verify the log and copy it to *dest*."""
        if not self.verify(public_key):
            raise RuntimeError("Audit log verification failed")
        shutil.copyfile(self.path, dest)


__all__ = ["AuditTrail"]
