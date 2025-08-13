import logging
import importlib

import menace.config as config
from menace.unified_event_bus import UnifiedEventBus


def test_setup_logging_debug(monkeypatch):
    config.CONFIG = None
    config._OVERRIDES = {"logging": {"verbosity": "DEBUG"}}
    import logging_utils as lu
    importlib.reload(lu)
    lu.setup_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_logging_updates_on_config_reload(monkeypatch):
    bus = UnifiedEventBus()
    config.set_event_bus(bus)
    config.CONFIG = None
    config._OVERRIDES = {"logging": {"verbosity": "INFO"}}
    import logging_utils as lu
    importlib.reload(lu)
    lu.setup_logging()
    assert logging.getLogger().level == logging.INFO
    config._OVERRIDES = {"logging": {"verbosity": "ERROR"}}
    config.reload()
    assert logging.getLogger().level == logging.ERROR

