import types
import tempfile

import mvp_executor
from mvp_executor import execute_untrusted


def test_timeout_infinite_loop_triggers_timeout():
    stdout, stderr = execute_untrusted("""
while True:
    pass
""")

    assert stdout == ""
    assert "execution timed out" in stderr


def test_execute_untrusted_rejects_unsupported_platform(monkeypatch):
    def fail_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(mvp_executor.os, "name", "java", raising=False)
    monkeypatch.setattr("mvp_executor.subprocess.run", fail_run)

    stdout, stderr = execute_untrusted("print('hi')")

    assert stdout == ""
    assert stderr == "error: unsupported platform for sandboxed execution"


def test_execute_untrusted_uses_job_object_on_windows(monkeypatch):
    called = {"job": False, "resume": False, "close": False}

    class DummyProc:
        def __init__(self):
            self._handle = 123
            self.pid = 456
            self.args = ["python", "runner.py"]
            self.returncode = 0

        def communicate(self, timeout=None):
            return (b"ok\n", b"")

        def kill(self):
            self.returncode = 1

    def fake_popen(*_args, **_kwargs):
        return DummyProc()

    def fake_job_object(handle):
        assert handle == 123
        called["job"] = True
        return 999

    def fake_resume(pid):
        assert pid == 456
        called["resume"] = True

    def fake_close(handle):
        assert handle == 999
        called["close"] = True

    monkeypatch.setattr(mvp_executor.os, "name", "nt", raising=False)
    monkeypatch.setattr(mvp_executor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mvp_executor, "_apply_windows_job_object", fake_job_object)
    monkeypatch.setattr(mvp_executor, "_resume_windows_process", fake_resume)
    monkeypatch.setattr(mvp_executor, "_close_windows_handle", fake_close)

    stdout, stderr = execute_untrusted("print('hi')")

    assert stdout.strip() == "ok"
    assert stderr == ""
    assert called["job"] is True
    assert called["resume"] is True
    assert called["close"] is True


def test_syntax_error_returns_error_marker():
    stdout, stderr = execute_untrusted("def broken(")

    assert stdout == ""
    assert stderr.startswith("error: syntax error:")


def test_import_blocking_prevents_execution():
    stdout, stderr = execute_untrusted("import os\nprint('hi')")

    assert stdout == ""
    assert "import of 'os' is not allowed" in stderr


def test_runtime_import_blocking_for_io_module():
    stdout, stderr = execute_untrusted("import io\nio.open('/etc/passwd')")

    assert stdout == ""
    assert "import of 'io' is not allowed" in stderr
    assert "error:" in stderr


def test_builtins_open_access_is_blocked():
    stdout, stderr = execute_untrusted("__builtins__['open']('/etc/passwd')")

    assert stdout == ""
    assert "not allowed" in stderr


def test_getattr_builtins_open_access_is_blocked():
    stdout, stderr = execute_untrusted('getattr(__builtins__, "open")("/etc/passwd")')

    assert stdout == ""
    assert "not allowed" in stderr


def test_import_builtins_open_access_is_blocked():
    stdout, stderr = execute_untrusted("import builtins\nbuiltins.open('x')")

    assert stdout == ""
    assert "not allowed" in stderr


def test_path_traversal_open_is_blocked():
    stdout, stderr = execute_untrusted("open('../somefile')")

    assert stdout == ""
    assert "error:" in stderr


def test_module_open_access_is_blocked():
    stdout, stderr = execute_untrusted("import io\nio.open('x')")

    assert stdout == ""
    assert "import of 'io' is not allowed" in stderr

    stdout, stderr = execute_untrusted("import codecs\ncodecs.open('x')")

    assert stdout == ""
    assert "import of 'codecs' is not allowed" in stderr


def test_open_allows_temp_directory_access():
    code = """
with open("note.txt", "w") as handle:
    handle.write("ok")
with open("note.txt", "r") as handle:
    print(handle.read())
"""
    stdout, stderr = execute_untrusted(code)

    assert stdout.strip() == "ok"
    assert stderr == ""


def test_open_rejects_outside_temp_directory():
    stdout, stderr = execute_untrusted("open('/etc/passwd')")

    assert stdout == ""
    assert "error:" in stderr


def test_import_multiprocessing_is_blocked():
    stdout, stderr = execute_untrusted("import multiprocessing")

    assert stdout == ""
    assert "not allowed" in stderr


def test_import_concurrent_futures_is_blocked():
    stdout, stderr = execute_untrusted("from concurrent.futures import ProcessPoolExecutor")

    assert stdout == ""
    assert "not allowed" in stderr


def test_import_asyncio_subprocess_is_blocked():
    stdout, stderr = execute_untrusted("import asyncio.subprocess")

    assert stdout == ""
    assert "import of 'asyncio.subprocess' is not allowed" in stderr


def test_from_asyncio_subprocess_is_blocked():
    stdout, stderr = execute_untrusted("from asyncio import subprocess")

    assert stdout == ""
    assert "import of 'asyncio.subprocess' is not allowed" in stderr


def test_multiprocessing_process_call_is_blocked():
    stdout, stderr = execute_untrusted("multiprocessing.Process(target=lambda: None)")

    assert stdout == ""
    assert "call to 'multiprocessing.Process' is not allowed" in stderr


def test_multiprocessing_pool_call_is_blocked():
    stdout, stderr = execute_untrusted("multiprocessing.Pool()")

    assert stdout == ""
    assert "call to 'multiprocessing.Pool' is not allowed" in stderr


def test_concurrent_futures_process_pool_executor_call_is_blocked():
    stdout, stderr = execute_untrusted("concurrent.futures.ProcessPoolExecutor()")

    assert stdout == ""
    assert "call to 'concurrent.futures.ProcessPoolExecutor' is not allowed" in stderr


def test_asyncio_create_subprocess_exec_call_is_blocked():
    stdout, stderr = execute_untrusted("asyncio.create_subprocess_exec('ls')")

    assert stdout == ""
    assert "call to 'asyncio.create_subprocess_exec' is not allowed" in stderr


def test_asyncio_create_subprocess_shell_call_is_blocked():
    stdout, stderr = execute_untrusted("asyncio.create_subprocess_shell('ls')")

    assert stdout == ""
    assert "call to 'asyncio.create_subprocess_shell' is not allowed" in stderr


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
