from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from error_parser import ErrorParser


def _build_trace(file: Path) -> str:
    return (
        "Traceback (most recent call last):\n"
        f"  File \"{file}\", line 2, in divide\n"
        "    return a/0\n"
        "ZeroDivisionError: division by zero\n"
    )


class EngineStub:
    def __init__(self):
        self.called_with: list | None = []

    def apply_patch_with_retry(self, path: Path, description: str, **kwargs):
        self.called_with.append(kwargs.get("target_region"))
        return None, False, 0.0


class ImproverStub:
    def __init__(self):
        self.self_coding_engine = EngineStub()
        self.received: list | None = []

    def run_cycle(self, *, target_region=None):
        self.received.append(target_region)
        if target_region is not None:
            self.self_coding_engine.apply_patch_with_retry(
                Path("buggy.py"), "fix", target_region=target_region  # path-ignore
            )
        return SimpleNamespace(roi=None)


class TesterStub:
    def __init__(self, traces):
        self.traces = traces
        self.results = {}
        self.idx = 0

    def run_once(self):
        trace = self.traces[self.idx]
        self.results = {"stdout": trace, "stderr": ""}
        self.idx += 1


def test_target_region_flows(tmp_path):
    buggy = tmp_path / "buggy.py"  # path-ignore
    buggy.write_text("def divide(a,b):\n    return a/0\n")
    trace = _build_trace(buggy)

    ctx = SimpleNamespace(
        improver=ImproverStub(),
        tester=TesterStub([trace, ""]),
        target_region=None,
    )

    for _ in range(2):
        ctx.improver.run_cycle(target_region=getattr(ctx, "target_region", None))
        ctx.tester.run_once()
        results = getattr(ctx.tester, "results", {}) or {}
        trace_out = (results.get("stdout", "") + results.get("stderr", "")).strip()
        if trace_out:
            parsed = ErrorParser.parse(trace_out)
            region = parsed.get("target_region") if isinstance(parsed, dict) else None
            if region:
                ctx.target_region = region

    improver = ctx.improver
    assert improver.received == [None, ctx.target_region]
    assert improver.self_coding_engine.called_with == [ctx.target_region]
