import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import neurosales
import importlib

REAL_USER_PREFS = importlib.import_module("neurosales.user_preferences")
REAL_SENTIMENT = importlib.import_module("neurosales.sentiment")

import pytest

# Import candidate_response_scorer early so its stubs don't interfere
try:
    import tests.test_candidate_response_scorer  # noqa: F401
finally:
    sys.modules["neurosales.user_preferences"] = REAL_USER_PREFS
    sys.modules["neurosales.sentiment"] = REAL_SENTIMENT

@pytest.fixture(autouse=True)
def restore_modules():
    yield
    import sys
    sys.modules["neurosales.user_preferences"] = REAL_USER_PREFS
    sys.modules["neurosales.sentiment"] = REAL_SENTIMENT
