import inspect
import traceback
from pathlib import Path

import inspect
import traceback
from pathlib import Path

from error_parser import extract_target_region
from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner


def _inner_fail():
    raise RuntimeError("boom")


def _outer_call():
    _inner_fail()


def test_extract_target_region_identifies_innermost_frame():
    try:
        _outer_call()
    except Exception:
        trace = traceback.format_exc()
    region = extract_target_region(trace)
    assert region is not None
    assert Path(region.filename) == Path(inspect.getsourcefile(_inner_fail))
    src, start = inspect.getsourcelines(_inner_fail)
    assert region.function == "_inner_fail"
    assert region.start_line == start
    assert region.end_line == start + len(src) - 1


def failing_func():
    raise ValueError("boom")


def test_runner_records_traceback_frames():
    runner = WorkflowSandboxRunner()
    metrics = runner.run(failing_func, safe_mode=True)
    mod = metrics.modules[0]
    assert not mod.success
    assert mod.frames
    frame_file, frame_line, frame_func = mod.frames[-1]
    assert frame_func == "failing_func"
    assert Path(frame_file) == Path(inspect.getsourcefile(failing_func))
    start = inspect.getsourcelines(failing_func)[1]
    assert frame_line == start + 1
