from __future__ import annotations

"""Setup script to install required packages and system tools."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sandbox_settings import SandboxSettings
from menace.startup_checks import verify_project_dependencies
from menace.dependency_installer import install_packages

logger = logging.getLogger(__name__)

SETUP_MARKER = Path(".autonomous_setup_complete")


def _apt_install(pkg: str, offline: bool) -> bool:
    if offline:
        logger.info("offline mode; skipping apt-get install %s", pkg)
        return False
    if shutil.which("apt-get") is None:
        return False
    try:
        subprocess.run(["apt-get", "update"], check=False)
        subprocess.run(["apt-get", "install", "-y", pkg], check=True)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("apt-get install %s failed: %s", pkg, exc)
        return False


def _pip_install(pkg: str, offline: bool, wheel_dir: str | None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"]
    if offline:
        if wheel_dir:
            cmd += ["--no-index", "--find-links", wheel_dir]
        else:
            logger.info("offline mode; skipping pip install %s", pkg)
            return False
    cmd.append(pkg)
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("pip install %s failed: %s", pkg, exc)
        return False


def check_and_install(settings: SandboxSettings) -> None:
    missing: list[str] = []
    offline = settings.menace_offline_install
    wheel_dir = settings.menace_wheel_dir

    if sys.version_info[:2] < (3, 10):
        raise RuntimeError(
            f"Python >=3.10 required, found {sys.version_info.major}.{sys.version_info.minor}"
        )

    if shutil.which("docker") is None:
        if _apt_install("docker.io", offline) and shutil.which("docker") is not None:
            logger.info("installed docker successfully")
        else:
            missing.append("docker")

    try:  # pragma: no cover - optional
        import docker  # type: ignore
    except Exception:
        if _pip_install("docker", offline, wheel_dir):
            try:
                import docker  # type: ignore
                logger.info("installed docker python package")
            except Exception:
                missing.append("docker python package")
        else:
            missing.append("docker python package")

    if shutil.which("qemu-system-x86_64") is None:
        if _apt_install("qemu-system-x86", offline) and shutil.which("qemu-system-x86_64") is not None:
            logger.info("installed qemu-system-x86_64 successfully")
        else:
            missing.append("qemu-system-x86_64")

    if shutil.which("git") is None:
        if _apt_install("git", offline) and shutil.which("git") is not None:
            logger.info("installed git successfully")
        else:
            missing.append("git")

    if shutil.which("pytest") is None:
        if _pip_install("pytest", offline, wheel_dir) and shutil.which("pytest") is not None:
            logger.info("installed pytest successfully")
        else:
            missing.append("pytest")

    if shutil.which("docker") is not None:
        try:  # pragma: no cover - optional
            import grp
            import getpass

            user = getpass.getuser()
            docker_grp = grp.getgrnam("docker")
            if user not in docker_grp.gr_mem and os.getgid() != docker_grp.gr_gid:
                missing.append("docker group")
        except Exception:  # pragma: no cover - platform dependent
            missing.append("docker group")

    missing_pkgs = verify_project_dependencies()
    if missing_pkgs:
        errors = install_packages(missing_pkgs, offline=offline, wheel_dir=wheel_dir)
        if offline and missing_pkgs:
            logger.info(
                "offline install mode enabled; skipping installation for: %s",
                ", ".join(missing_pkgs),
            )
        for pkg, err in errors.items():
            logger.error("failed installing %s: %s", pkg, err)
        failed_pkgs = [p for p in missing_pkgs if p in errors]
        if failed_pkgs:
            missing.extend(failed_pkgs)

    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing))

    SETUP_MARKER.touch()
    logger.info("All dependencies satisfied")


def auto_env_setup(
    env_path: str = ".env",
    *,
    settings: SandboxSettings | None = None,
    run_interactive_setup: bool = False,
) -> None:
    """Ensure the environment file exists before installing dependencies.

    Historically :mod:`setup_dependencies` exposed an ``auto_env_setup`` helper
    that callers imported to provision both the environment file and package
    dependencies.  Some integration scripts – and a few external tools – still
    rely on that behaviour.  The implementation vanished when the installer
    module was refactored which surfaced ``ImportError`` exceptions for those
    consumers.  Reintroduce the helper as a thin wrapper around
    :mod:`auto_env_setup` and :func:`check_and_install` so legacy entry points
    keep working without duplicating logic.

    Parameters
    ----------
    env_path:
        Optional override for the generated ``.env`` file location.
    settings:
        Pre-configured :class:`SandboxSettings` instance.  A new instance is
        created when omitted so existing call sites continue to function.
    run_interactive_setup:
        When ``True`` also triggers ``auto_env_setup.interactive_setup`` after
        generating the environment file.  The default ``False`` preserves the
        previous non-interactive behaviour that scripts expect.
    """

    try:
        from auto_env_setup import ensure_env, interactive_setup  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("auto_env_setup module is unavailable") from exc

    logger.info("ensuring environment configuration via auto_env_setup")
    ensure_env(env_path)
    if run_interactive_setup:
        logger.info("running interactive environment setup")
        interactive_setup()

    check_and_install(settings or SandboxSettings())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = SandboxSettings()
    check_and_install(settings)


if __name__ == "__main__":
    main()
