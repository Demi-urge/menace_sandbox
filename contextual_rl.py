from __future__ import annotations

from typing import Dict, Tuple

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


class ContextualRL:
    """Simple contextual RL using optional PyTorch for value approximation."""

    def __init__(self, state_size: int = 4, *, alpha: float = 0.5) -> None:
        self.state_size = state_size
        self.alpha = alpha
        self.values: Dict[Tuple[int, ...], float] = {}
        self.use_torch = torch is not None
        if self.use_torch:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(state_size, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )
            self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.loss = torch.nn.MSELoss()

    def _key(self, state: Tuple[float, ...]) -> Tuple[int, ...]:
        return tuple(int(round(s)) for s in state)

    def update(self, state: Tuple[float, ...], reward: float) -> float:
        if self.use_torch:
            assert torch is not None
            t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([[reward]], dtype=torch.float32)
            pred = self.model(t_state)
            loss = self.loss(pred, target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            value = float(pred.item())
        else:
            key = self._key(state)
            q = self.values.get(key, 0.0)
            q += self.alpha * (reward - q)
            self.values[key] = q
            value = q
        return value

    def score(self, state: Tuple[float, ...]) -> float:
        if self.use_torch:
            assert torch is not None
            with torch.no_grad():
                t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return float(self.model(t_state).item())
        key = self._key(state)
        return self.values.get(key, 0.0)


__all__ = ["ContextualRL"]
