import os, sys, types
pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "neurosales"))
sys.path.insert(0, pkg_path)
# Provide minimal neurosales package stubs for tests
neuro_pkg = types.ModuleType("neurosales")
neuro_pkg.__path__ = [pkg_path]
sys.modules.setdefault("neurosales", neuro_pkg)
sys.modules.setdefault(
    "neurosales.user_preferences", types.ModuleType("neurosales.user_preferences")
)
sys.modules.setdefault("neurosales.sentiment", types.ModuleType("neurosales.sentiment"))
import pytest

@pytest.fixture(autouse=True)
def restore_modules():
    yield
