from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from logging_utils import get_logger, setup_logging
from dynamic_path_router import resolve_path, path_for_prompt

logger = get_logger(__name__)

MetricsFunc = Callable[[float, float, Optional[Dict[str, float]]], Dict[str, float]]


def _load_plugin_dirs_from_file(path: str | Path) -> List[str]:
    """Return plugin directories listed in ``path``."""
    p = Path(resolve_path(str(path)))
    if not p.exists():
        return []
    try:
        if p.suffix.lower() in {".json", ".jsn"}:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            import yaml  # type: ignore
            with open(p, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
    except Exception:
        logger.exception("failed to load metrics config %s", path_for_prompt(p))
        return []
    dirs = data.get("plugin_dirs", []) if isinstance(data, dict) else []
    if isinstance(dirs, str):
        return [str(resolve_path(dirs))]
    if isinstance(dirs, list):
        return [str(resolve_path(str(d))) for d in dirs]
    return []


def load_metrics_plugins(directories: str | Path | Sequence[str] | None) -> List[MetricsFunc]:
    """Load metric collector callbacks from ``directories``."""
    if not directories:
        return []
    dirs = [directories] if isinstance(directories, (str, Path)) else list(directories)
    plugins: List[MetricsFunc] = []
    for d in dirs:
        path = Path(resolve_path(str(d)))
        if not path.is_dir():
            logger.warning(
                "metrics plugin directory %s does not exist", path_for_prompt(path)
            )
            continue
        for file in path.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, "collect_metrics", None)
                    if callable(func):
                        plugins.append(func)
                    else:
                        logger.warning(
                            "plugin %s missing collect_metrics",
                            path_for_prompt(file),
                        )
            except Exception:
                logger.exception(
                    "failed to load metrics plugin %s", path_for_prompt(file)
                )
    return plugins


def discover_metrics_plugins(env: dict | None = None) -> List[MetricsFunc]:
    """Discover plugins based on environment variables and config files."""
    env = env or os.environ
    dirs: List[str] = []
    env_dir = env.get("SANDBOX_METRICS_PLUGIN_DIR")
    if env_dir:
        dirs.extend(str(resolve_path(d)) for d in env_dir.split(os.pathsep))
    cfg_file = env.get("SANDBOX_METRICS_FILE")
    if cfg_file:
        dirs.extend(_load_plugin_dirs_from_file(resolve_path(cfg_file)))
    return load_metrics_plugins(dirs)


def collect_plugin_metrics(
    plugins: List[MetricsFunc],
    prev_roi: float,
    roi: float,
    resources: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Merge metrics returned by ``plugins``."""
    merged: Dict[str, float] = {}
    for func in plugins:
        try:
            res = func(prev_roi, roi, resources)
            if isinstance(res, dict):
                for k, v in res.items():
                    try:
                        merged[k] = float(v)
                    except Exception:
                        merged[k] = 0.0
        except Exception:
            logger.exception("metrics plugin %s failed", getattr(func, "__name__", "?"))
    return merged


if __name__ == "__main__":  # pragma: no cover - manual invocation
    setup_logging()
