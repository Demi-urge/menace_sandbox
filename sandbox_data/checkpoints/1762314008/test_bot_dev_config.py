import os
import sys
import types
import importlib
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("menace", types.ModuleType("menace")).RAISE_ERRORS = False

cfg_mod = importlib.import_module("menace.bot_dev_config")


def test_validate_rejects_visual_agents():
    cfg = cfg_mod.BotDevConfig()
    cfg.visual_agents = []  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        cfg.validate()
