import os
import sys
import subprocess
import time
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from neurosales.db_manager import DatabaseConnectionManager


def _start_memcached(port: int):
    if shutil.which("memcached") is None:
        pytest.skip("memcached not available")
    proc = subprocess.Popen(["memcached", "-p", str(port), "-u", "root"])
    time.sleep(0.5)
    return proc


def test_memcached_env(monkeypatch):
    port = 11224
    proc = _start_memcached(port)
    monkeypatch.setenv("NEURO_MEMCACHED_SERVERS", f"127.0.0.1:{port}")
    mgr = DatabaseConnectionManager()
    client = mgr.get_memcached()
    try:
        assert client is not None
        client.set("foo", "bar")
        assert client.get("foo") in (b"bar", "bar")
    finally:
        proc.terminate()
        proc.wait()

