from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import MutableMapping, Sequence

# === BEGIN PATH SETUP ===

def _ensure_package_on_path() -> Path:
    script_path = Path(__file__).resolve()
    package_root = script_path.parent
    marker = package_root / "__init__.py"
    if marker.exists():
        parent = package_root.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
    else:
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    return package_root

_REPO_ROOT = _ensure_package_on_path()
_DEFAULT_DATA_DIR = _REPO_ROOT / "sandbox_data"

_DYNAMIC_PATH_ROUTER = importlib.import_module("menace_sandbox.dynamic_path_router")
sys.modules["dynamic_path_router"] = _DYNAMIC_PATH_ROUTER

for _alias in ("unified_event_bus", "resilience"):
    try:
        module = importlib.import_module(f"menace_sandbox.{_alias}")
    except ModuleNotFoundError:
        continue
    sys.modules[_alias] = module

_menace_pkg = importlib.import_module("menace")
if not getattr(_menace_pkg, "RAISE_ERRORS", None):
    _menace_pkg = importlib.reload(_menace_pkg)

# === END PATH SETUP ===

from menace_sandbox.environment_bootstrap import EnvironmentBootstrapper
from menace_sandbox.sandbox_runner.bootstrap import bootstrap_environment
from menace_sandbox.sandbox_settings import SandboxSettings

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Menace sandbox bootstrap tasks manually")
    parser.add_argument("--skip-sandbox", action="store_true", help="Skip sandbox-specific checks")
    parser.add_argument("--skip-environment", action="store_true", help="Skip system/environment bootstrap")
    parser.add_argument("--auto-install", action="store_true", help="Allow bootstrap to install missing packages")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO, WARNING)")
    return parser.parse_args(argv)

def _setup_logging(level: str) -> logging.Logger:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=numeric_level)
    else:
        root.setLevel(numeric_level)
    return logging.getLogger("manual_bootstrap")

def _configure_environment(repo_root: Path, env: MutableMapping[str, str] | None = None) -> None:
    target = env if env is not None else os.environ
    target.setdefault("SANDBOX_REPO_PATH", str(repo_root))
    target.setdefault("SANDBOX_DATA_DIR", str(_DEFAULT_DATA_DIR))

def _describe_settings(logger: logging.Logger, settings: SandboxSettings) -> None:
    try:
        env_file = Path(settings.menace_env_file)
    except Exception:
        env_file = Path(".env")
    logger.info(
        "Sandbox configuration ready (repo=%s data_dir=%s env_file=%s)",
        getattr(settings, "sandbox_repo_path", "<unknown>"),
        getattr(settings, "sandbox_data_dir", "<unknown>"),
        env_file.resolve(),
    )

def _register_balolos_coder():
    from menace.self_coding_engine import SelfCodingEngine
    from menace.data_bot import DataBot
    from menace.bot_registry import BotRegistry, internalize_coding_bot
    from menace.model_automation_pipeline import ModelAutomationPipeline

    engine = SelfCodingEngine()
    data_bot = DataBot()
    registry = BotRegistry()
    pipeline = ModelAutomationPipeline()

    internalize_coding_bot(
        name="Balolos Coder",
        engine=engine,
        data_bot=data_bot,
        pipeline=pipeline,
        registry=registry
    )

    print("Bots now in registry:", registry.get_all_bots())

def main(argv: Sequence[str] | None = None, env: MutableMapping[str, str] | None = None) -> int:
    args = _parse_args(argv)
    logger = _setup_logging(args.log_level)
    _configure_environment(_REPO_ROOT, env)

    exit_code = 0
    settings: SandboxSettings | None = None

    if not args.skip_sandbox:
        try:
            settings = bootstrap_environment(auto_install=args.auto_install)
        except SystemExit as exc:
            logger.error(str(exc) or "sandbox dependency verification failed")
            return exc.code if isinstance(exc.code, int) else 1
        except KeyboardInterrupt:
            logger.info("sandbox bootstrap interrupted")
            return 130
        except Exception:
            logger.exception("sandbox bootstrap failed")
            return 1
        else:
            if settings is not None:
                _describe_settings(logger, settings)
            logger.info("sandbox bootstrap completed successfully")
    else:
        logger.info("sandbox bootstrap completed successfully (skipped)")

    if not args.skip_environment:
        try:
            EnvironmentBootstrapper().bootstrap()
        except KeyboardInterrupt:
            logger.info("environment bootstrap interrupted")
            return 130
        except Exception:
            logger.exception("environment bootstrap failed")
            return 1
        else:
            logger.info("environment bootstrap completed successfully")

    # Register the Balolos Coder after environment is ready
    _register_balolos_coder()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
