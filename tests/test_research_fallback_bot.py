import os
import types
import sys

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

# provide minimal stubs for optional dependencies before importing menace
sys.modules.setdefault("bs4", types.SimpleNamespace(BeautifulSoup=None))
sys.modules.setdefault("playwright.async_api", types.SimpleNamespace(async_playwright=None))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
primitives = types.ModuleType("primitives")
sys.modules.setdefault("cryptography.hazmat.primitives", primitives)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
ed_mod = types.ModuleType("ed25519")
ed_mod.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed_mod.Ed25519PublicKey = object
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", ed_mod)
ser_mod = types.ModuleType("serialization")
primitives.serialization = ser_mod
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", ser_mod)
env_mod = types.ModuleType("env_config")
env_mod.DATABASE_URL = "sqlite:///:memory:"
sys.modules.setdefault("env_config", env_mod)

# provide minimal stubs for heavy optional dependencies so ResearchFallbackBot can be imported
mods = {
    "menace.chatgpt_enhancement_bot": ["EnhancementDB", "ChatGPTEnhancementBot", "Enhancement"],
    "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
    "menace.text_research_bot": ["TextResearchBot"],
    "menace.video_research_bot": ["VideoResearchBot"],
    "menace.chatgpt_research_bot": ["ChatGPTResearchBot", "Exchange", "summarise_text"],
    "menace.database_manager": ["get_connection", "DB_PATH"],
    "menace.capital_management_bot": ["CapitalManagementBot"],
}
for name, attrs in mods.items():
    module = types.ModuleType(name)
    for attr in attrs:
        if attr == "summarise_text":
            setattr(module, attr, lambda text, ratio=0.2: text[:10])
        elif attr == "get_connection":
            setattr(module, attr, lambda path: None)
        elif attr == "DB_PATH":
            setattr(module, attr, ":memory:")
        else:
            setattr(module, attr, type(attr, (), {}))
    sys.modules.setdefault(name, module)

import menace.research_fallback_bot as rfb
import menace.research_aggregator_bot as rab
from pathlib import Path


def test_process(monkeypatch, tmp_path: Path):
    bot = rfb.ResearchFallbackBot(info_db=rab.InfoDB(tmp_path / "info.db"))
    html = "<html><body><p>Some detailed answer about python.</p></body></html>"
    monkeypatch.setattr(bot, "fetch_page", lambda url: html)
    results = bot.process("python error")
    assert results
    assert results[0].summary
    assert results[0].embedding is not None
    assert all(isinstance(x, float) for x in results[0].embedding)
