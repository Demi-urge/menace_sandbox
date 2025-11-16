#!/usr/bin/env python3
"""Fully automated setup for the autonomous sandbox."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path

from auto_env_setup import ensure_env, interactive_setup
from auto_resource_setup import ensure_proxies, ensure_accounts
from setup_dependencies import check_and_install
from sandbox_settings import SandboxSettings
from environment_bootstrap import EnvironmentBootstrapper
from environment_generator import generate_presets, adapt_presets
from roi_tracker import ROITracker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_tests() -> None:
    """Execute the project's test suite."""
    logger.info("running tests")
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"])


def discover_hardware() -> dict[str, list[str]]:
    """Return available GPUs and network interfaces."""
    gpus: list[str] = []
    try:
        import GPUtil  # type: ignore

        gpus = [g.name for g in GPUtil.getGPUs()]
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            gpus = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            gpus = []

    interfaces: list[str] = []
    try:
        import psutil  # type: ignore

        interfaces = list(psutil.net_if_addrs().keys())
    except Exception:
        interfaces = []

    return {"gpus": gpus, "network_interfaces": interfaces}


def main() -> None:
    logger.info("ensuring environment")
    ensure_env()
    os.environ.setdefault("MENACE_NON_INTERACTIVE", "1")
    interactive_setup()

    logger.info("ensuring resources")
    ensure_proxies(resolve_path("proxies.json"))
    ensure_accounts(resolve_path("accounts.json"))

    logger.info("installing dependencies")
    check_and_install(SandboxSettings())

    run_tests()

    logger.info("bootstrapping environment")
    EnvironmentBootstrapper().bootstrap()

    logger.info("generating presets")
    presets = generate_presets()
    presets = adapt_presets(ROITracker(), presets)
    data_dir = resolve_path("sandbox_data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "latest_presets.json").write_text(
        json.dumps(presets, indent=2)
    )

    logger.info("discovering hardware")
    hw = discover_hardware()
    try:
        hw_path = resolve_path("hardware.json")
    except FileNotFoundError:
        hw_path = resolve_path("") / "hardware.json"
    hw_path.write_text(json.dumps(hw, indent=2))

    logger.info("setup complete")


if __name__ == "__main__":
    main()
