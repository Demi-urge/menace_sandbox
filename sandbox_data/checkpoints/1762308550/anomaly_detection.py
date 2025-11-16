from __future__ import annotations

from typing import Iterable, List
from statistics import median

import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    KMeans = None  # type: ignore
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .data_bot import MetricsDB  # type: ignore
except Exception:  # pragma: no cover - optional
    MetricsDB = None  # type: ignore


if nn is not None:  # pragma: no cover - optional
    class _AutoEncoder(nn.Module):
        """Very small autoencoder for one-dimensional data."""

        def __init__(self, hidden: int = 2) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
            self.decoder = nn.Linear(hidden, 1)

        def forward(self, x):  # type: ignore[override]
            z = self.encoder(x)
            return self.decoder(z)
else:
    _AutoEncoder = None  # type: ignore


def _ae_scores(values: List[float]) -> List[float]:
    if torch is None or nn is None or _AutoEncoder is None:  # pragma: no cover - dependency missing
        raise RuntimeError("PyTorch not available")
    x = torch.tensor(values, dtype=torch.float32).view(-1, 1)
    model = _AutoEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(20):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec = model(x)
        err = ((rec - x) ** 2).view(-1)
    return err.cpu().numpy().tolist()


def _cluster_scores(values: List[float]) -> List[float]:
    if KMeans is None or np is None:  # pragma: no cover - dependency missing
        raise RuntimeError("scikit-learn not available")
    arr = np.array(values).reshape(-1, 1)
    model = KMeans(n_clusters=1, n_init=1)
    model.fit(arr)
    center = float(model.cluster_centers_[0][0])
    return [abs(v - center) for v in values]


def _mad_scores(values: List[float]) -> List[float]:
    """Robust anomaly scores based on Median Absolute Deviation."""
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations)
    if mad == 0:
        return [0.0 for _ in values]
    return [d / mad for d in deviations]


def anomaly_scores(
    values: Iterable[float],
    *,
    metrics_db: "MetricsDB" | None = None,
    field: str = "value",
) -> List[float]:
    """Return anomaly scores for a sequence of values.

    When provided, ``metrics_db`` receives all calculated scores via
    :meth:`MetricsDB.log_eval`.
    """

    vals = list(values)
    if not vals:
        return []

    scores: List[float]
    if torch is not None and nn is not None:
        try:
            scores = _ae_scores(vals)
            if metrics_db:
                for s in scores:
                    metrics_db.log_eval("anomaly", field, float(s))
            return scores
        except Exception:  # pragma: no cover - runtime issues
            logger.exception("_ae_scores failed, falling back to alternative method")
    if KMeans is not None and np is not None:
        try:
            scores = _cluster_scores(vals)
            if metrics_db:
                for s in scores:
                    metrics_db.log_eval("anomaly", field, float(s))
            return scores
        except Exception:  # pragma: no cover - runtime issues
            logger.exception("_cluster_scores failed, falling back to MAD")

    scores = _mad_scores(vals)
    if metrics_db:
        for s in scores:
            metrics_db.log_eval("anomaly", field, float(s))
    return scores


__all__ = ["anomaly_scores"]
