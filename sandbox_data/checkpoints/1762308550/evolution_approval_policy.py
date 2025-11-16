from __future__ import annotations

"""Policy for approving major structural changes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from .structural_evolution_bot import EvolutionRecord


@dataclass
class EvolutionApprovalPolicy:
    """Simple threshold-based approval policy."""

    impact_threshold: float = 25.0

    def approve(self, rec: EvolutionRecord) -> bool:
        """Return ``True`` if ``rec`` should be applied."""
        try:
            return rec.impact <= self.impact_threshold
        except Exception:
            return False


__all__ = ["EvolutionApprovalPolicy"]
