from __future__ import annotations

"""Retrain retrieval ranker from metrics and reload services."""

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json
import time

from dynamic_path_router import resolve_path

# ``retrieval_ranker_dataset`` historically exposed ``load_training_data``.  The
# helper may not be available in lightweight environments so we fall back to
# ``build_dataset`` if necessary.
try:  # pragma: no cover - best effort import
    from .retrieval_ranker_dataset import load_training_data  # type: ignore
except Exception:  # pragma: no cover - alias when helper missing
    from .retrieval_ranker_dataset import build_dataset as load_training_data  # type: ignore

from retrieval_ranker import train, save_model

MODEL_PATH = resolve_path("analytics/retrieval_ranker.model")
REGISTRY_PATH = Path("retrieval_ranker.json")


def _import_obj(path: str) -> Any:
    """Return object referenced by ``module:attr`` path."""

    mod_name, attr = path.rsplit(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)


def _persist_model_path(path: Path, cfg_path: Path = REGISTRY_PATH) -> None:
    """Record the latest model *path* in ``retrieval_ranker.json``."""

    data = {}
    if cfg_path.exists():
        try:
            loaded = json.loads(cfg_path.read_text())
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass

    history = data.get("history", []) or []
    current = data.get("current")
    if current and current != str(path):
        history.insert(0, current)
    data["current"] = str(path)
    data["history"] = history[:2]
    try:
        cfg_path.write_text(json.dumps(data))
    except Exception:
        pass


def retrain_and_reload(
    services: Sequence[Any] = (),
    *,
    vector_db: Path | str = "vector_metrics.db",
    patch_db: Path | str = "metrics.db",
    model_dir: Path | str = MODEL_PATH.parent,
) -> None:
    """Retrain ranker using latest metrics and notify services."""

    df = load_training_data(vec_db_path=vector_db, roi_path=patch_db)  # type: ignore[arg-type]
    tm, _ = train(df)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"retrieval_ranker_{int(time.time())}.json"
    save_model(tm, model_path)
    _persist_model_path(model_path)

    for svc in services:
        try:
            if hasattr(svc, "reload_ranker_model"):
                svc.reload_ranker_model(model_path)
        except Exception:
            continue


def retrain(
    services: Sequence[Any] = (),
    *,
    vector_db: Path | str = "vector_metrics.db",
    patch_db: Path | str = "roi.db",
    model_dir: Path | str = MODEL_PATH.parent,
) -> None:
    """Backward compatible wrapper around :func:`retrain_and_reload`."""

    retrain_and_reload(
        services, vector_db=vector_db, patch_db=patch_db, model_dir=model_dir
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrain vector ranker and reload services")
    p.add_argument("--vector-db", default="vector_metrics.db")
    p.add_argument("--patch-db", default="roi.db")
    p.add_argument("--model-dir", default=str(MODEL_PATH.parent))
    p.add_argument("--interval", type=int, default=0, help="Seconds between retraining runs")
    p.add_argument(
        "--service",
        action="append",
        default=[],
        help="Import path to service exposing reload_ranker_model",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    services = [_import_obj(s) for s in args.service]
    interval = max(int(args.interval), 0)
    while True:
        retrain_and_reload(
            services,
            vector_db=args.vector_db,
            patch_db=args.patch_db,
            model_dir=args.model_dir,
        )
        if interval <= 0:
            break
        time.sleep(interval)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
