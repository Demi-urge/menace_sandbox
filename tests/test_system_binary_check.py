import pytest

from system_binary_check import assert_required_system_binaries, DEFAULT_REQUIREMENTS


def test_assert_required_system_binaries_reports_missing(monkeypatch):
    monkeypatch.setattr(
        "system_binary_check.shutil.which",
        lambda name: "/usr/bin/tool" if name == "ffmpeg" else None,
    )

    with pytest.raises(RuntimeError) as excinfo:
        assert_required_system_binaries(DEFAULT_REQUIREMENTS)

    message = str(excinfo.value)
    assert "ffmpeg" not in message
    assert "tesseract" in message
    assert "qemu-system-x86_64" in message


def test_assert_required_system_binaries_passes_when_present(monkeypatch):
    monkeypatch.setattr("system_binary_check.shutil.which", lambda name: f"/usr/bin/{name}")

    missing = assert_required_system_binaries(DEFAULT_REQUIREMENTS, exit_on_missing=False)

    assert missing == []
