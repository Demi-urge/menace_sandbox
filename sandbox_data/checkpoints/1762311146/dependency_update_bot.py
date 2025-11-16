from __future__ import annotations

"""Automatic dependency updater for Menace."""

import json
import logging
import os
import subprocess
import sys
import tempfile
import tomllib
import venv
import uuid

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Dict, Generator, Optional


class DependencyUpdater:
    """Check for outdated packages and upgrade them automatically."""

    def __init__(
        self,
        pyproject_path: str | Path = "pyproject.toml",
        *,
        interval: int = 86400,
        deployer: Optional[object] = None,
        container_image: str | None = None,
        orchestrator_url: str | None = None,
    ) -> None:
        self.pyproject_path = Path(pyproject_path)
        self.interval = interval
        self.deployer = deployer
        self.container_image = container_image
        self.orchestrator_url = orchestrator_url or os.getenv("ORCHESTRATOR_URL", "")
        self.logger = logging.getLogger("DependencyUpdater")

    # ------------------------------------------------------------------
    # Pyproject helpers
    # ------------------------------------------------------------------

    def _parse_dependencies(self) -> List[str]:
        """Return dependency names from ``pyproject.toml``."""
        try:
            data = tomllib.loads(self.pyproject_path.read_text())
        except Exception:
            return []
        deps: List[str] = []
        proj = data.get("project", {})
        if isinstance(proj, dict):
            for dep in proj.get("dependencies", []):
                if isinstance(dep, str):
                    name = dep.split(" ")[0].split("<")[0].split("=")[0]
                    deps.append(name)
        poetry = data.get("tool", {}).get("poetry", {}) if isinstance(data.get("tool"), dict) else {}
        if isinstance(poetry, dict):
            for name in poetry.get("dependencies", {}):
                if name.lower() != "python":
                    deps.append(name)
        return deps

    def _outdated(self) -> List[dict[str, str]]:
        try:
            res = subprocess.run([
                "pip",
                "list",
                "--outdated",
                "--format=json",
            ], capture_output=True, text=True, check=True)
        except Exception:
            return []
        try:
            return json.loads(res.stdout)
        except Exception:
            return []

    def _install(self, packages: Iterable[str], *, python: str = sys.executable, upgrade: bool = True) -> bool:
        cmd = [python, "-m", "pip", "install"]
        if upgrade:
            cmd.append("-U")
        cmd.extend(list(packages))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error("install failed: %s", exc)
            return False

    def _update_os(self) -> bool:
        """Update system packages using apt when available."""
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "upgrade", "-y"], check=True)
            return True
        except Exception as exc:  # pragma: no cover - platform specific
            self.logger.error("os update failed: %s", exc)
            return False

    def _run_tests(self, python: str = sys.executable) -> bool:
        try:
            subprocess.run([python, "-m", "pytest", "-q"], check=True)
            return True
        except subprocess.CalledProcessError as exc:
            self.logger.error("tests failed: %s", exc)
            return False

    def _freeze(self, python: str) -> Dict[str, str]:
        try:
            res = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
            out: Dict[str, str] = {}
            for line in res.stdout.splitlines():
                if "==" in line:
                    name, ver = line.split("==", 1)
                    out[name.lower()] = ver
            return out
        except Exception:
            return {}

    @contextmanager
    def _temp_env(self, *, container_image: str | None = None) -> Generator[str, None, None]:
        if not container_image:
            with tempfile.TemporaryDirectory() as td:
                venv.create(td, with_pip=True)
                python = Path(td) / ("Scripts" if os.name == "nt" else "bin") / "python"
                yield str(python)
        else:
            name = f"depupd-{uuid.uuid4().hex[:8]}"
            subprocess.run(
                ["docker", "run", "-d", "--name", name, container_image, "sleep", "infinity"],
                check=True,
            )
            try:
                yield f"docker exec {name} python"
            finally:
                subprocess.run(["docker", "rm", "-f", name], check=False)

    def run_cycle(self, *, deploy: bool = False, update_os: bool = False) -> List[str]:
        if update_os:
            self._update_os()
        deps = set(self._parse_dependencies())
        outdated = [d for d in self._outdated() if d.get("name") in deps]
        if not outdated:
            return []
        packages = [d["name"] for d in outdated if d.get("name")]
        with self._temp_env(container_image=self.container_image) as py:
            if not self._install(packages, python=py, upgrade=True):
                return []
            versions = self._freeze(py)
            if not self._run_tests(py):
                return []
        pinned = [f"{p}=={versions.get(p.lower())}" for p in packages if versions.get(p.lower())]
        if not pinned:
            return []
        if not self._install(pinned, upgrade=False):
            return []
        self.logger.info("updated packages: %s", ",".join(packages))
        if self.orchestrator_url and requests:
            try:
                url = self.orchestrator_url.rstrip("/") + "/deploy"
                requests.post(url, json={"packages": packages}, timeout=5)
            except Exception:
                self.logger.exception("orchestrator deployment failed")
        if deploy and self.deployer is not None:
            try:
                from .deployment_bot import DeploymentSpec
                self.deployer.deploy("dependency_update", [], DeploymentSpec(name="deps", resources={}, env={}))
            except Exception:
                self.logger.exception("deployment failed")
        return packages


__all__ = ["DependencyUpdater"]
