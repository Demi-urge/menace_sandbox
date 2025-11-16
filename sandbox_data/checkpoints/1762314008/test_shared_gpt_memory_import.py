"""Tests for the shared GPT memory module import behaviour."""

import importlib
import sys


def test_shared_gpt_memory_imports_canonical_module() -> None:
    """Importing via the package keeps flat aliases pointing to one module."""

    importlib.import_module("menace_sandbox.shared_gpt_memory")

    assert "menace_sandbox.gpt_memory" in sys.modules
    assert "gpt_memory" in sys.modules
    assert sys.modules["menace_sandbox.gpt_memory"] is sys.modules["gpt_memory"]
