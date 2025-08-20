from __future__ import annotations

"""Retrain retrieval ranker from metrics and reload services."""

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse

from retrieval_ranker import load_training_data, train, save_model

MODEL_PATH = Path("analytics/retrieval_ranker.model")


def _import_obj(path: str) -> Any:
    """Return object referenced by ``module:attr`` path."""

    mod_name, attr = path.rsplit(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)


def retrain_and_reload(
    services: Sequence[Any] = (),
    *,
    vector_db: Path | str = "vector_metrics.db",
    patch_db: Path | str = "metrics.db",
    model_path: Path | str = MODEL_PATH,
) -> None:
    """Retrain ranker using latest metrics and notify services."""

    df = load_training_data(vector_db=vector_db, patch_db=patch_db)
    tm, _ = train(df)
    save_model(tm, model_path)

    for svc in services:
        try:
            if hasattr(svc, "reload_ranker_model"):
                svc.reload_ranker_model(model_path)
        except Exception:
            continue


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrain vector ranker and reload services")
    p.add_argument("--vector-db", default="vector_metrics.db")
    p.add_argument("--patch-db", default="metrics.db")
    p.add_argument("--model-path", default=str(MODEL_PATH))
    p.add_argument(
        "--service",
        action="append",
        default=[],
        help="Import path to service exposing reload_ranker_model",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    services = [_import_obj(s) for s in args.service]
    retrain_and_reload(services, vector_db=args.vector_db, patch_db=args.patch_db, model_path=args.model_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
