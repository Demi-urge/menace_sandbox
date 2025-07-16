import os
import pytest

pytest.importorskip("tkinter")

import tkinter as tk
import importlib


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="requires display")
def test_gui_init(monkeypatch):
    from menace import menace_gui as mg
    gui = mg.MenaceGUI()
    names = [gui.notebook.tab(i, "text") for i in gui.notebook.tabs()]
    assert names == [
        "Communication",
        "Activity Log",
        "Statistics",
        "Overview",
        "Forecast Chains",
    ]

