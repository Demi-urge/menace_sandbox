from __future__ import annotations

"""Backward compatibility wrapper for :mod:`prompt_evolution_memory`."""

try:  # pragma: no cover - optional dependency
    from llm_interface import Prompt  # noqa: F401
except Exception:  # pragma: no cover - llm_interface unavailable
    try:  # pragma: no cover - optional dependency
        from prompt_types import Prompt  # noqa: F401
    except Exception as exc:  # pragma: no cover - explicit failure
        raise ImportError(
            "Prompt dataclass is required for PromptEvolutionLogger. "
            "Install 'prompt_types' or ensure 'llm_interface' is available."
        ) from exc

from prompt_evolution_memory import PromptEvolutionMemory as PromptEvolutionLogger

__all__ = ["PromptEvolutionLogger"]
