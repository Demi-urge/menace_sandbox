import os
import types
from unittest import mock

import bootstrap_conflict_check as bcc


def test_detect_conflicts_skips_when_psutil_missing():
    with mock.patch.dict("sys.modules", {"psutil": None}):
        bootstraps, watchers = bcc.detect_conflicts(logger=None)

    assert bootstraps == []
    assert watchers == []


def test_detect_conflicts_flags_bootstrap_and_watchers():
    current_pid = os.getpid()

    class _Proc:
        def __init__(self, pid, name, cmdline):
            self.info = {"pid": pid, "name": name, "cmdline": cmdline}

    fake_psutil = types.SimpleNamespace()

    def _iter(fields):
        yield _Proc(current_pid, "python", ["start_autonomous_sandbox.py"])  # skipped self
        yield _Proc(101, "python", ["/opt/run_autonomous.py"])
        yield _Proc(202, "watchman", ["--no-pretty", "sandbox_data"])
        yield _Proc(303, "watchman", ["/tmp/unrelated"])

    fake_psutil.process_iter = _iter

    with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
        bootstraps, watchers = bcc.detect_conflicts(logger=None)

    assert bootstraps == ["pid 101: python /opt/run_autonomous.py"]
    assert watchers == ["pid 202: watchman --no-pretty sandbox_data"]
