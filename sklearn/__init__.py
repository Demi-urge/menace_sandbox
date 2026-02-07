"""Minimal sklearn stubs for local usage."""

from . import metrics, model_selection, pipeline, preprocessing
from .linear_model import LinearRegression

__all__ = [
    "LinearRegression",
    "metrics",
    "model_selection",
    "pipeline",
    "preprocessing",
]
