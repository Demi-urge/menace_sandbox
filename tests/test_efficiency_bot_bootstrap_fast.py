import logging
import sys
from pathlib import Path
import importlib.util
import types


PACKAGE_NAME = "menace_sandbox"


def _load_efficiency_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parent.parent)]
    sys.modules.setdefault(PACKAGE_NAME, package)

    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.efficiency_bot",
        Path(__file__).resolve().parent.parent / "efficiency_bot.py",
        submodule_search_locations=[str(Path(__file__).resolve().parent.parent)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eb = _load_efficiency_module()


def test_efficiency_data_bot_bootstrap_fast(monkeypatch, caplog):
    monkeypatch.setenv("MENACE_BOOTSTRAP_FAST", "1")
    created: dict[str, dict] = {}

    class StubSettings:
        def __init__(self, *_, **kwargs):
            created["settings"] = kwargs

    class StubDataBot:
        def __init__(self, *_, **kwargs):
            created["data_bot"] = kwargs

    monkeypatch.setattr(eb, "SandboxSettings", StubSettings)
    monkeypatch.setattr(eb, "DataBot", StubDataBot)
    monkeypatch.setattr(eb, "_DATA_BOT", None, raising=False)

    with caplog.at_level(logging.INFO):
        instance = eb._get_data_bot()

    assert instance is eb._get_data_bot()
    assert created["settings"]["bootstrap_fast"] is True
    assert created["data_bot"]["bootstrap"] is True
    assert created["data_bot"].get("start_server") is False
    assert any(
        "bootstrap_fast enabled" in record.getMessage() for record in caplog.records
    )
