"""Distributed service supervisor for launching and monitoring remote nodes."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, Iterable

BACKENDS = {"ssh", "docker", "k8s"}

from .service_supervisor import ServiceSupervisor


class ClusterServiceSupervisor(ServiceSupervisor):
    """Extend :class:`ServiceSupervisor` to manage remote supervisors."""

    def __init__(
        self,
        hosts: Iterable[str] | None = None,
        check_interval: float = 5.0,
        *,
        context_builder: "ContextBuilder",
    ) -> None:
        super().__init__(check_interval=check_interval, context_builder=context_builder)
        env_hosts = os.getenv("CLUSTER_HOSTS", "").split(",")
        self.hosts = [h.strip() for h in (hosts or env_hosts) if h.strip()]
        failover = os.getenv("FAILOVER_HOSTS", "").split(",")
        self.failover_hosts = [h.strip() for h in failover if h.strip()]
        self.backend = os.getenv("CLUSTER_BACKEND", "ssh").lower()
        if self.backend not in BACKENDS:
            self.backend = "ssh"
        self.docker_image = os.getenv("CLUSTER_DOCKER_IMAGE", "menace:latest")
        self.k8s_namespace = os.getenv("CLUSTER_K8S_NAMESPACE", "default")
        self.remote_procs: Dict[str, subprocess.Popen] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _start_remote(self, host: str) -> None:
        try:
            if self.backend == "ssh":
                cmd = ["ssh", host, "python3", "-m", "menace.service_supervisor"]
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.remote_procs[host] = proc
            elif self.backend == "docker":
                cmd = ["ssh", host, "docker", "run", "-d", "--name", "menace_supervisor", self.docker_image]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:  # k8s
                cmd = ["kubectl", "-n", self.k8s_namespace, "rollout", "restart", f"deployment/{host}"]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.logger.info("started remote supervisor on %s", host)
        except Exception as exc:  # pragma: no cover - remote call may fail
            self.logger.error("failed starting remote supervisor on %s: %s", host, exc)

    # ------------------------------------------------------------------
    def add_hosts(self, hosts: Iterable[str]) -> None:
        """Add new hosts and start supervisors on them."""
        for host in hosts:
            h = host.strip()
            if not h or h in self.hosts:
                continue
            self.hosts.append(h)
            self._start_remote(h)

    def start_all(self) -> None:  # type: ignore[override]
        super().start_all()
        env_new = os.getenv("NEW_HOSTS", "").split(",")
        self.add_hosts([h.strip() for h in env_new if h.strip()])
        os.environ["NEW_HOSTS"] = ""
        for host in self.hosts:
            self._start_remote(host)

    # ------------------------------------------------------------------
    def _check_remote(self, host: str) -> bool:
        if self.backend == "ssh":
            proc = self.remote_procs.get(host)
            healthy = proc is not None and proc.poll() is None
            cmd_extra = os.getenv("CLUSTER_HEALTH_CHECK_CMD")
            if healthy and cmd_extra:
                try:
                    res = subprocess.run(
                        ["ssh", host, cmd_extra],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    healthy = res.returncode == 0
                except Exception:
                    healthy = False
            return healthy
        elif self.backend == "docker":
            cmd = ["ssh", host, "docker", "ps", "-q", "-f", "name=menace_supervisor"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return bool(res.stdout.strip())
        else:  # k8s
            cmd = ["kubectl", "-n", self.k8s_namespace, "get", "pod", host, "-o", "jsonpath={.status.phase}"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return res.stdout.strip() == b"Running"

    # ------------------------------------------------------------------
    def _monitor(self) -> None:  # override
        while True:
            super()._monitor()
            env_new = os.getenv("NEW_HOSTS", "").split(",")
            if env_new and any(h.strip() for h in env_new):
                self.add_hosts([h.strip() for h in env_new if h.strip()])
                os.environ["NEW_HOSTS"] = ""
            for host in list(self.hosts):
                if not self._check_remote(host):
                    self.logger.warning("remote supervisor on %s unhealthy", host)
                    self._start_remote(host)
                    if not self._check_remote(host) and self.failover_hosts:
                        replacement = self.failover_hosts.pop(0)
                        self.logger.info("failover: replacing %s with %s", host, replacement)
                        self.hosts.remove(host)
                        self.hosts.append(replacement)
                        self._start_remote(replacement)


