from __future__ import annotations

"""Rebuild retrieval ranker dataset, retrain model and reload services."""

from importlib import import_module
from pathlib import Path
from typing import Sequence, Any, Iterable
import argparse

from .retrieval_ranker_dataset import build_dataset
from .retrieval_ranker_model import train, MODEL_PATH


# ---------------------------------------------------------------------------

def _import_service(path: str) -> Any:
    """Return object referred to by ``module:attr`` path."""
    mod_name, attr = path.rsplit(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)


# ---------------------------------------------------------------------------

def retrain_and_reload(services: Sequence[Any] = (), *, model_path: Path | str = MODEL_PATH) -> None:
    """Rebuild dataset, retrain model and ask services to reload."""

    build_dataset()  # rebuild dataset from latest metrics
    train(save_path=model_path)

    for svc in services:
        try:
            if hasattr(svc, "reload_ranker_model"):
                svc.reload_ranker_model(model_path)
        except Exception:
            continue


# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrain retrieval ranker and reload services")
    p.add_argument("--model-path", default=str(MODEL_PATH))
    p.add_argument(
        "--service",
        action="append",
        default=[],
        help="Import path to service exposing reload_ranker_model",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    services = [_import_service(s) for s in args.service]
    retrain_and_reload(services, model_path=args.model_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
