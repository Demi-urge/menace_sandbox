from __future__ import annotations
"""Provision local infrastructure via docker compose."""

import logging
import secrets
import subprocess
import time
from pathlib import Path

DEFAULT_COMPOSE_TEMPLATE = """
services:
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: {postgres_password}
      POSTGRES_DB: menace
    ports:
      - "5432:5432"
  vault:
    image: hashicorp/vault:1.13
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: root
    ports:
      - "8200:8200"
"""


class LocalInfrastructureProvisioner:
    """Create a default compose file and start containers."""

    def __init__(self, compose_file: str = "docker-compose.yml") -> None:
        self.compose_path = Path(compose_file)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.postgres_password = secrets.token_urlsafe(16)
        self.logger.info("generated postgres password: %s", self.postgres_password)

    # ------------------------------------------------------------------
    def ensure_compose_file(self) -> Path:
        self.logger.info("ensuring compose file at %s", self.compose_path)
        compose_text = DEFAULT_COMPOSE_TEMPLATE.format(
            postgres_password=self.postgres_password
        )
        if not self.compose_path.exists() or self.compose_path.read_text() != compose_text:
            self.compose_path.write_text(compose_text)
            self.logger.info("wrote compose file")
        return self.compose_path

    # ------------------------------------------------------------------
    def _containers_running(self) -> bool:
        services = ["rabbitmq", "postgres", "vault"]
        for svc in services:
            try:
                cid = subprocess.check_output(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(self.compose_path),
                        "ps",
                        "-q",
                        svc,
                    ],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                if not cid:
                    self.logger.debug("%s container missing", svc)
                    return False
                running = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", cid],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                if running != "true":
                    self.logger.debug("%s not running", svc)
                    return False
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("status check failed for %s: %s", svc, exc)
                return False
        return True

    # ------------------------------------------------------------------
    def up(self, retries: int = 3, delay: float = 2.0) -> None:
        cfg = self.ensure_compose_file()
        for attempt in range(1, retries + 1):
            try:
                self.logger.info("starting containers (attempt %s)", attempt)
                subprocess.check_call(
                    ["docker", "compose", "-f", str(cfg), "up", "-d"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if self._containers_running():
                    self.logger.info("containers are running")
                    return
                self.logger.warning("containers not ready")
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("local infrastructure failed: %s", exc)
            if attempt < retries:
                time.sleep(delay)
        self.logger.error("containers failed to start after %s attempts", retries)

    # ------------------------------------------------------------------
    def down(self) -> None:
        cfg = self.compose_path
        try:
            self.logger.info("stopping containers")
            subprocess.check_call(
                ["docker", "compose", "-f", str(cfg), "down"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.logger.info("containers stopped")
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("failed to stop containers: %s", exc)


__all__ = ["LocalInfrastructureProvisioner"]
