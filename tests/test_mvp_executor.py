import os
import shutil
import time

import mvp_executor


def test_run_untrusted_code_timeout():
    start = time.monotonic()
    stdout, stderr = mvp_executor.run_untrusted_code("while True:\n    pass")
    elapsed = time.monotonic() - start
    assert stdout == ""
    assert "Execution timed out after 5s" in stderr
    assert elapsed < 6.5


def test_run_untrusted_code_blocks_imports():
    stdout, stderr = mvp_executor.run_untrusted_code("import os\nprint('hi')")
    assert stdout == ""
    assert "Blocked import: os" in stderr
    assert "Execution was not attempted" in stderr


def test_run_untrusted_code_reports_syntax_error():
    stdout, stderr = mvp_executor.run_untrusted_code("def bad(:\n    pass")
    assert stdout == ""
    assert "SyntaxError:" in stderr
    assert "line" in stderr


def test_run_untrusted_code_cleans_temp_dir(monkeypatch, tmp_path):
    created_paths = []

    class TrackingTempDir:
        def __init__(self, prefix: str):
            self._path = tmp_path / f"{prefix}tracked"

        def __enter__(self):
            self._path.mkdir(parents=True, exist_ok=True)
            created_paths.append(self._path)
            return str(self._path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(self._path, ignore_errors=True)
            return False

    monkeypatch.setattr(mvp_executor.tempfile, "TemporaryDirectory", TrackingTempDir)

    stdout, stderr = mvp_executor.run_untrusted_code("print('ok')")

    assert stdout.strip() == "ok"
    assert stderr == ""
    assert created_paths
    assert not os.path.exists(created_paths[0])


def test_run_untrusted_code_empty_returns_blank():
    assert mvp_executor.run_untrusted_code("  ") == ("", "")
