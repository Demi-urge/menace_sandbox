import pytest
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types
sys.modules["menace.db_router"] = types.SimpleNamespace(DBRouter=object)
sys.modules["menace.chatgpt_idea_bot"] = types.SimpleNamespace(ChatGPTClient=object)
import menace.chatgpt_research_bot as crb


def test_summarise_text_fallback(monkeypatch):
    monkeypatch.setattr(crb, "summarize", None)
    text = (
        "Common words repeat repeat. Unique phrase stands alone. Another repeat repeat sentence."
    )
    summary = crb.summarise_text(text, ratio=0.34)
    assert "Unique phrase stands alone" in summary
    assert summary.count(".") <= 2
