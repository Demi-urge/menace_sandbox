import pytest
import menace.vision_utils as vu


def _disable_backends(monkeypatch):
    monkeypatch.setattr(vu, "vision", None)
    monkeypatch.setattr(vu, "pytesseract", None)
    monkeypatch.setattr(vu, "Image", None)


def test_detect_text_warns(monkeypatch, caplog):
    caplog.set_level("WARNING")
    _disable_backends(monkeypatch)
    assert vu.detect_text(b"x") == ""
    assert "text detection failed" in caplog.text


def test_detect_text_raises_param(monkeypatch):
    _disable_backends(monkeypatch)
    with pytest.raises(vu.OCRError):
        vu.detect_text(b"x", critical=True)


def test_detect_text_raises_env(monkeypatch):
    monkeypatch.setenv("OCR_CRITICAL", "1")
    _disable_backends(monkeypatch)
    with pytest.raises(vu.OCRError):
        vu.detect_text(b"x")
