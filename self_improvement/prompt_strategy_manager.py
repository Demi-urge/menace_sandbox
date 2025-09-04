"""Utilities for rotating prompt strategies with persistent state."""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
from typing import Callable, Sequence

try:  # pragma: no cover - optional helper for dynamic paths
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback
    def resolve_path(p: str | Path) -> str | Path:  # type: ignore
        return p


class PromptStrategyManager:
    """Maintain an ordered list of prompt strategies and rotate on failures.

    The manager keeps track of the currently preferred strategy index and
    persists it to ``state_path`` so that rotation state survives across runs.
    When selecting a strategy the provided ``selector`` callable is invoked with
    the strategies in their current rotation order, allowing the caller to apply
    custom scoring (for example :meth:`SelfImprovementEngine._select_prompt_strategy`).
    """

    def __init__(
        self,
        strategies: Sequence[str] | None = None,
        state_path: Path | str | None = None,
    ) -> None:
        self.strategies: list[str] = list(strategies or [])
        if state_path is None:
            try:  # local import to avoid heavy dependency during import
                from .init import _data_dir

                state_path = _data_dir() / "prompt_strategy_state.json"
            except Exception:  # pragma: no cover - fallback when init unavailable
                state_path = Path(resolve_path("prompt_strategy_state.json"))
        if not isinstance(state_path, Path):
            state_path = Path(resolve_path(state_path))
        self.state_path = state_path
        self.index: int = 0
        self._load_state()

    # ------------------------------------------------------------------
    def _rotated(self) -> list[str]:
        if not self.strategies:
            return []
        return self.strategies[self.index:] + self.strategies[: self.index]

    # ------------------------------------------------------------------
    def select(self, selector: Callable[[Sequence[str]], str | None]) -> str | None:
        """Return the next strategy using ``selector`` for scoring."""

        ordered = self._rotated()
        if not ordered:
            return None
        return selector(ordered)

    # ------------------------------------------------------------------
    def record_failure(self) -> None:
        """Advance to the next strategy and persist state."""

        if not self.strategies:
            return
        self.index = (self.index + 1) % len(self.strategies)
        self._save_state()

    # ------------------------------------------------------------------
    def set_strategies(self, strategies: Sequence[str]) -> None:
        """Replace the managed strategies preserving rotation index."""

        self.strategies = list(strategies)
        if self.strategies:
            self.index %= len(self.strategies)
        else:
            self.index = 0
        self._save_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not self.strategies:
                self.strategies = list(data.get("strategies", []))
            self.index = int(data.get("index", 0))
            if self.strategies:
                self.index %= len(self.strategies)
            else:
                self.index = 0
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=self.state_path.parent, encoding="utf-8"
            ) as fh:
                json.dump({"index": self.index, "strategies": self.strategies}, fh)
                tmp = Path(fh.name)
            os.replace(tmp, self.state_path)
        except Exception:  # pragma: no cover - best effort persistence
            pass


__all__ = ["PromptStrategyManager"]
