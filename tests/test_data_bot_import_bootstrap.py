from __future__ import annotations

import importlib
import sys


def test_data_bot_imports_without_menace_namespace() -> None:
    """``data_bot`` should import with or without the ``menace`` shim."""

    namespace_snapshot = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "menace" or name.startswith("menace.")
    }
    databot_snapshot = {
        name: sys.modules.pop(name)
        for name in ("data_bot", "menace_sandbox.data_bot")
        if name in sys.modules
    }

    try:
        flat_module = importlib.import_module("data_bot")
        package_module = importlib.import_module("menace_sandbox.data_bot")

        assert hasattr(flat_module, "DataBot")
        assert hasattr(package_module, "DataBot")
        assert flat_module.__spec__ and package_module.__spec__
        assert flat_module.__spec__.origin == package_module.__spec__.origin
    finally:
        for name in list(sys.modules):
            if name == "menace" or name.startswith("menace."):
                if name not in namespace_snapshot:
                    sys.modules.pop(name, None)
        sys.modules.update(namespace_snapshot)

        for name in ("data_bot", "menace_sandbox.data_bot"):
            if name in databot_snapshot:
                sys.modules[name] = databot_snapshot[name]
            else:
                sys.modules.pop(name, None)
