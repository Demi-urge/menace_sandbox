import importlib
import os
import platform
import shutil
import sys
import types

import pytest
from tests.test_persistent_queue import _stub_deps
from tests.test_visual_agent_auto_recover import _setup_va


@pytest.mark.usefixtures("tmp_path")
def test_configure_tesseract_linux(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    for name in ["cv2", "numpy", "mss", "pyautogui"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    # Provide a minimal pytesseract implementation
    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")  # path-ignore
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tok")
    monkeypatch.setenv("VA_DATASET_DIR", str(tmp_path / "ds"))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path / "data"))

    orig_getenv = os.getenv
    monkeypatch.setattr(os, "getenv", lambda n, d=None: None if n == "TESSERACT_CMD" else orig_getenv(n, d))
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    orig_exists = os.path.exists

    def fake_exists(p):
        if p == "/usr/local/bin/tesseract":
            return True
        if p == "/usr/bin/tesseract":
            return False
        return orig_exists(p)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(shutil, "which", lambda n: None)

    va_mod = _setup_va(monkeypatch, tmp_path)

    assert va_mod.pytesseract.pytesseract.tesseract_cmd == "/usr/local/bin/tesseract"  # path-ignore

