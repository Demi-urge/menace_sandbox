import os
import pytest
from unittest import mock

pytest.importorskip("tkinter")


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requires display")
def test_gui_init(monkeypatch):
    from menace import menace_gui as mg
    builder = mock.MagicMock()
    monkeypatch.setattr(mg, "OPENAI_API_KEY", "")
    gui = mg.MenaceGUI(context_builder=builder)
    names = [gui.notebook.tab(i, "text") for i in gui.notebook.tabs()]
    assert names == [
        "Communication",
        "Activity Log",
        "Statistics",
        "Overview",
        "Forecast Chains",
    ]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requires display")
def test_gui_uses_context_builder(monkeypatch):
    from menace import menace_gui as mg

    builder = mock.MagicMock()
    monkeypatch.setattr(mg.ChatGPTClient, "__post_init__", lambda self: None)
    monkeypatch.setattr(mg, "OPENAI_API_KEY", "key")

    gui = mg.MenaceGUI(context_builder=builder)

    assert builder.refresh_db_weights.called
    assert gui.context_builder is builder
    assert gui.conv_bot is not None
    assert gui.conv_bot.client.context_builder is builder
    assert gui.error_bot.context_builder is builder


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requires display")
def test_refresh_failure_disables_prompts(monkeypatch):
    from menace import menace_gui as mg

    builder = mock.MagicMock()
    builder.refresh_db_weights.side_effect = RuntimeError("boom")
    monkeypatch.setattr(mg.ChatGPTClient, "__post_init__", lambda self: None)
    monkeypatch.setattr(mg, "OPENAI_API_KEY", "key")

    gui = mg.MenaceGUI(context_builder=builder)

    assert gui.conv_bot is None
    assert not gui.chatgpt_enabled
