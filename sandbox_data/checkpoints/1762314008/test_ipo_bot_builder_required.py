import importlib
import sys
import types
import pytest


def test_ipobot_requires_builder():
    vs = types.ModuleType("vector_service")
    cb_mod = types.ModuleType("vector_service.context_builder")

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *args, **kwargs):
            return []

    cb_mod.ContextBuilder = DummyBuilder
    vs.context_builder = cb_mod
    sys.modules.setdefault("vector_service", vs)
    sys.modules.setdefault("vector_service.context_builder", cb_mod)

    ipb = importlib.import_module("menace.ipo_bot")

    with pytest.raises(TypeError):
        ipb.IPOBot()
