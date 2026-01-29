import types
import tempfile

from mvp_executor import execute_untrusted


def test_timeout_infinite_loop_triggers_timeout():
    stdout, stderr = execute_untrusted("""
while True:
    pass
""")

    assert stdout == ""
    assert "execution timed out" in stderr


def test_syntax_error_returns_error_marker():
    stdout, stderr = execute_untrusted("def broken(")

    assert stdout == ""
    assert stderr.startswith("error: syntax error:")


def test_import_blocking_prevents_execution():
    stdout, stderr = execute_untrusted("import os\nprint('hi')")

    assert stdout == ""
    assert "import of 'os' is not allowed" in stderr


def test_temp_dir_cleanup(tmp_path, monkeypatch):
    temp_root = tmp_path / "mvp_temp_root"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    stdout, stderr = execute_untrusted("print('hi')")

    assert stdout.strip() == "hi"
    assert stderr == ""
    assert list(temp_root.iterdir()) == []


def test_output_decoding_normalizes_newlines(monkeypatch):
    result = types.SimpleNamespace(
        stdout=b"line1\r\nline2\rline3\xff",
        stderr=b"bad\xff\r\n",
        returncode=0,
    )

    def fake_run(*_args, **_kwargs):
        return result

    monkeypatch.setattr("mvp_executor.subprocess.run", fake_run)

    stdout, stderr = execute_untrusted("print('ignored')")

    assert stdout == "line1\nline2\nline3�"
    assert stderr == "bad�\n"
