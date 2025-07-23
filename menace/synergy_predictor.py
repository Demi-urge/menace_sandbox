from __future__ import annotations

"""Synergy prediction helpers used by ROITracker and PredictionManager."""

from typing import Iterable, Optional, Tuple

import os

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore


def _pick_best_order(
    series: Iterable[float],
    *,
    max_ar: int = 3,
    max_ma: int = 3,
    ic: str = "bic",
) -> Tuple[int, int, int]:
    """Return best ``(p, d, q)`` order for ``series``."""
    if np is None:
        return (1, 1, 1)
    try:  # pragma: no cover - optional dependency
        from statsmodels.tsa.stattools import arma_order_select_ic  # type: ignore

        arr = np.array(list(series), dtype=float)
        res = arma_order_select_ic(
            arr, max_ar=max_ar, max_ma=max_ma, ic=[ic]
        )
        if ic == "aic":
            p, q = res.aic_min_order
        else:
            p, q = res.bic_min_order
        return (int(p), 1, int(q))
    except Exception:  # pragma: no cover - fallback
        return (1, 1, 1)


class ARIMASynergyPredictor:
    """ARIMA based synergy forecaster."""

    def __init__(self, order: Optional[Tuple[int, int, int]] = (1, 1, 1)) -> None:
        self.order = order

    def predict(self, history: Iterable[float]) -> float:
        values = [float(v) for v in history]
        if len(values) < 2:
            return float(values[-1]) if values else 0.0
        order = self.order
        if order is None and len(values) > 4:
            order = _pick_best_order(values)
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore

            model = ARIMA(values, order=order or (1, 1, 1)).fit()
            res = model.get_forecast(steps=1)
            return float(res.predicted_mean[0])
        except Exception:
            return float(values[-1])


class LSTMSynergyPredictor:
    """Minimal LSTM forecaster for synergy metrics."""

    def __init__(self, *, seq_len: int = 5, hidden: int = 16, epochs: int = 5) -> None:
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.use_torch = torch is not None and nn is not None
        self.avg = 0.0
        if self.use_torch:
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
            params = list(self.lstm.parameters()) + list(self.fc.parameters())
            self.optim = torch.optim.Adam(params, lr=0.01)
            self.loss = nn.MSELoss()
        else:
            self.lstm = None
            self.fc = None
            self.optim = None
            self.loss = None

    def _train(self, series: list[float]) -> None:
        if len(series) <= self.seq_len:
            self.avg = float(series[-1]) if series else 0.0
            return
        if not self.use_torch or np is None:
            self.avg = float(sum(series)) / len(series)
            return
        seqs = []
        targets = []
        for i in range(len(series) - self.seq_len):
            seqs.append([[series[j]] for j in range(i, i + self.seq_len)])
            targets.append(series[i + self.seq_len])
        x = torch.tensor(np.array(seqs), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
        for _ in range(self.epochs):
            out, _ = self.lstm(x)
            pred = self.fc(out[:, -1, :])
            loss = self.loss(pred, y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def predict(self, history: Iterable[float]) -> float:
        series = [float(v) for v in history]
        if not series:
            return 0.0
        self._train(series)
        if not self.use_torch or np is None:
            return self.avg
        seq = torch.tensor([[ [v] for v in series[-self.seq_len:]]], dtype=torch.float32)
        with torch.no_grad():
            out, _ = self.lstm(seq)
            pred = self.fc(out[:, -1, :])[0, 0]
            return float(pred.item())


__all__ = ["ARIMASynergyPredictor", "LSTMSynergyPredictor", "_pick_best_order"]
