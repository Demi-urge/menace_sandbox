"""Smoke tests covering vector_service import side-effects."""

from __future__ import annotations

import importlib


def test_vector_service_bootstraps_embeddable_db_mixin_module() -> None:
    """Importing vector_service should register mixin module aliases."""

    import vector_service  # noqa: F401  # ensures __init__ side-effects run

    qualified = importlib.import_module("menace_sandbox.embeddable_db_mixin")
    flat = importlib.import_module("embeddable_db_mixin")

    assert qualified is flat
    assert hasattr(flat, "EmbeddableDBMixin")
