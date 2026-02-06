"""Minimal sklearn stubs for local usage."""

from . import model_selection, preprocessing
from .linear_model import LinearRegression

__all__ = ["LinearRegression", "model_selection", "preprocessing"]
