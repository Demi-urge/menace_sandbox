"""Compatibility wrapper exposing classes from :mod:`vector_service.stack_ingest`."""

from __future__ import annotations

from .stack_ingest import (  # re-export for backwards compatibility
    StackChunk,
    StackDatasetStream,
    StackFile,
    StackIngestor,
    StackMetadataStore,
)

__all__ = [
    "StackChunk",
    "StackDatasetStream",
    "StackFile",
    "StackIngestor",
    "StackMetadataStore",
]

