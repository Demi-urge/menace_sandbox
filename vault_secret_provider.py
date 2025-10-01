from __future__ import annotations

"""Fetch secrets from a remote vault with local fallback."""

import logging
import os
from typing import Optional

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:  # pragma: no cover - support execution without package context
    from .secrets_manager import SecretsManager
except ImportError:  # pragma: no cover - bootstrap scripts import as top-level modules
    from secrets_manager import SecretsManager  # type: ignore


class VaultSecretProvider:
    """Retrieve secrets from a remote vault then cache locally."""

    def __init__(self, url: str | None = None, manager: SecretsManager | None = None) -> None:
        self.url = url or os.getenv("SECRET_VAULT_URL")
        self.manager = manager or SecretsManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[object] = None
        self._cache: dict[str, str] = {}
        if self.url and requests:
            try:
                self.session = requests.Session()
            except Exception:  # pragma: no cover - best effort
                self.session = None

    # ------------------------------------------------------------------
    def get(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        if self.session and self.url:
            try:
                resp = self.session.get(f"{self.url.rstrip('/')}/{name}", timeout=5)
                if resp.status_code == 200 and resp.text.strip():
                    token = resp.text.strip()
                    self.manager.set(name, token)
                    self._cache[name] = token
                    return token
            except Exception as exc:  # pragma: no cover - log only
                self.logger.error("vault fetch failed for %s: %s", name, exc)
        token = self.manager.get(name)
        if token:
            self._cache[name] = token
        return token

    # ------------------------------------------------------------------
    def export_env(self, name: str) -> None:
        token = self.get(name)
        os.environ[name.upper()] = token


__all__ = ["VaultSecretProvider"]
