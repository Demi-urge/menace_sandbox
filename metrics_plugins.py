import importlib.util
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

logger = logging.getLogger(__name__)

MetricsFunc = Callable[[float, float, Optional[Dict[str, float]]], Dict[str, float]]


def load_metrics_plugins(directory: str | Path | None) -> List[MetricsFunc]:
    """Return plugin callbacks found in ``directory``."""
    if not directory:
        return []
    path = Path(directory)
    plugins: List[MetricsFunc] = []
    if not path.is_dir():
        logger.warning("metrics plugin directory %s does not exist", path)
        return plugins
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
                    logger.warning("plugin %s missing collect_metrics", file)
        except Exception:
            logger.exception("failed to load metrics plugin %s", file)
    return plugins


def collect_plugin_metrics(
    plugins: List[MetricsFunc],
    prev_roi: float,
    roi: float,
    resources: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Return merged metrics from ``plugins``."""
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


def fetch_retrieval_stats(path: str | Path | None = None) -> Dict[str, float]:
    """Return aggregated win/regret statistics for ranking algorithms.

    Parameters
    ----------
    path:
        Optional path to a JSONL file containing retrieval outcome records.  If
        omitted, ``analytics/retrieval_outcomes.jsonl`` relative to this module
        is used.

    Returns
    -------
    dict
        Mapping containing average ``win_rate`` and ``regret_rate`` values as
        well as a ``count`` of how many records were processed.  Missing or
        unreadable datasets yield zeroed metrics so callers can safely adjust
        ranking weights without additional checks.
    """

    if path is None:
        path = resolve_path("analytics/retrieval_outcomes.jsonl")
    else:
        path = Path(path)
    if not path.exists():
        return {"win_rate": 0.0, "regret_rate": 0.0, "count": 0.0}

    wins: List[float] = []
    regrets: List[float] = []
    try:
        with path.open("r", encoding="utf8") as fh:
            for line in fh:
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                try:
                    wins.append(float(data.get("win_rate", 0.0)))
                except Exception:
                    pass
                try:
                    regrets.append(float(data.get("regret_rate", 0.0)))
                except Exception:
                    pass
    except Exception:
        logger.exception("failed reading retrieval stats from %s", path)
        return {"win_rate": 0.0, "regret_rate": 0.0, "count": 0.0}

    return {
        "win_rate": mean(wins) if wins else 0.0,
        "regret_rate": mean(regrets) if regrets else 0.0,
        "count": float(len(wins)),
    }


__all__ = ["load_metrics_plugins", "collect_plugin_metrics", "fetch_retrieval_stats"]
