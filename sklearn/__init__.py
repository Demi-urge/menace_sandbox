"""Minimal sklearn stubs for local usage."""

from . import model_selection, pipeline, preprocessing
from .linear_model import LinearRegression

__all__ = ["LinearRegression", "model_selection", "pipeline", "preprocessing"]
