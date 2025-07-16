from __future__ import annotations

"""Simple secrets manager with rotation support."""

import os
import json
import logging
import base64
import hashlib
import time
from pathlib import Path
from typing import Dict


class SecretsManager:
    """Manage secrets generation and rotation."""

    def __init__(self, path: str = "secrets.json", *, rotation_days: int = 7) -> None:
        self.path = Path(path)
        self.rotation_days = rotation_days
        self.logger = logging.getLogger(self.__class__.__name__)
        self.secrets: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    self.secrets = {str(k): str(v) for k, v in data.items()}
            except Exception:
                self.logger.exception("failed to load secrets")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.secrets, indent=2))
        except Exception:
            self.logger.exception("failed to save secrets")

    def _new_secret(self) -> str:
        token = os.urandom(32)
        return base64.urlsafe_b64encode(token).decode()

    def _hash_secret(self, token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    # ------------------------------------------------------------------
    def get(self, name: str, *, rotate: bool = True) -> str:
        token = self.secrets.get(name)
        if token is None:
            token = self._new_secret()
            self.secrets[name] = token
            self._save()
        elif rotate:
            env_var = f"{name.upper()}_ROTATE_EVERY"
            days = int(os.getenv(env_var, self.rotation_days))
            meta_var = f"{name}_updated"
            updated = float(self.secrets.get(meta_var, "0"))
            if updated + days * 86400 < time.time():
                token = self._new_secret()
                self.secrets[name] = token
                self.secrets[meta_var] = str(time.time())
                self._save()
        return token

    def set(self, name: str, token: str) -> None:
        self.secrets[name] = token
        self.secrets[f"{name}_updated"] = str(time.time())
        self._save()

    # ------------------------------------------------------------------
    def export_env(self, name: str) -> None:
        token = self.get(name)
        os.environ[name.upper()] = token


__all__ = ["SecretsManager"]
