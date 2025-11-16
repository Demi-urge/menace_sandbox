from __future__ import annotations

"""Token pricing information for supported language models.

Rates are expressed in USD per token.  Helper functions allow providers to
retrieve input and output token prices, optionally applying overrides supplied
via :mod:`llm_config`.
"""

from typing import Dict, Tuple

# Default per-token pricing for models.  Values are USD per token.
DEFAULT_RATES: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.000005, "output": 0.000015},
    "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
}


def get_rates(
    model: str, overrides: Dict[str, Dict[str, float]] | None = None
) -> Tuple[float, float]:
    """Return ``(input_rate, output_rate)`` for *model*.

    Overrides are applied on top of :data:`DEFAULT_RATES` when provided.
    Unknown models default to zero cost.
    """

    rates = DEFAULT_RATES.get(model, {"input": 0.0, "output": 0.0}).copy()
    if overrides and overrides.get(model):
        rates.update(overrides[model])
    return rates["input"], rates["output"]


def get_input_rate(
    model: str, overrides: Dict[str, Dict[str, float]] | None = None
) -> float:
    """Return the input token rate for *model*."""

    return get_rates(model, overrides)[0]


def get_output_rate(
    model: str, overrides: Dict[str, Dict[str, float]] | None = None
) -> float:
    """Return the output token rate for *model*."""

    return get_rates(model, overrides)[1]


__all__ = [
    "DEFAULT_RATES",
    "get_rates",
    "get_input_rate",
    "get_output_rate",
]
