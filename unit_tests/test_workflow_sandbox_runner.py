import io
import urllib.request
import urllib.parse
import tempfile
import pathlib
import builtins
from unittest import mock


class WorkflowSandboxRunner:
    def _resolve(self, root, path):
        root = pathlib.Path(root).resolve()
        p = pathlib.Path(path)
        if p.is_absolute():
            try:
                p = p.relative_to(root)
            except ValueError:
                p = pathlib.Path(*p.parts[1:])
        candidate = (root / p).resolve()
        if not candidate.is_relative_to(root):
            raise RuntimeError("path escapes sandbox")
        return candidate

    def run(self, workflow, *, safe_mode=False, test_data=None, network_mocks=None, fs_mocks=None):
        test_data = test_data or {}
        network_mocks = network_mocks or {}
        network_data = {}
        for key, value in test_data.items():
            if urllib.parse.urlparse(key).scheme in {"http", "https"}:
                network_data[key] = value
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.open") as popen:
            root = pathlib.Path(tmp).resolve()

            orig_open = builtins.open

            def sandbox_open(file, mode="r", *a, **kw):
                if any(m in mode for m in ("w", "a", "x", "+")):
                    fn = (fs_mocks or {}).get("open")
                    if fn:
                        return fn(file, mode, *a, **kw)
                    if safe_mode:
                        raise RuntimeError("file write disabled in safe_mode")
                return orig_open(self._resolve(root, file), mode, *a, **kw)

            popen.side_effect = sandbox_open

            orig_urlopen = urllib.request.urlopen

            def fake_urlopen(url, *a, **kw):
                u = url if isinstance(url, str) else url.get_full_url()
                if u in network_data:
                    data = network_data[u]
                    return io.BytesIO(data.encode() if isinstance(data, str) else data)
                fn = (network_mocks or {}).get(u)
                if fn:
                    return fn(url, *a, **kw)
                if safe_mode:
                    raise RuntimeError("network access disabled in safe_mode")
                return orig_urlopen(url, *a, **kw)

            with mock.patch("urllib.request.urlopen", fake_urlopen):
                try:
                    workflow()
                    success = True
                    error = None
                except Exception as exc:
                    success = False
                    error = str(exc)

        class Module:
            def __init__(self, success, error):
                self.success = success
                self.exception = error

        metrics = type(
            "RunMetrics",
            (),
            {"modules": [Module(success, error)], "crash_count": 0 if success else 1},
        )
        self.metrics = metrics
        return metrics


def test_safe_mode_blocks_network_access():
    runner = WorkflowSandboxRunner()

    def wf():
        urllib.request.urlopen("http://example.com")

    metrics = runner.run(wf, safe_mode=True)
    assert not metrics.modules[0].success
    assert "network access disabled" in metrics.modules[0].exception


def test_safe_mode_blocks_fs_write():
    runner = WorkflowSandboxRunner()

    def wf():
        with open("foo.txt", "w") as fh:
            fh.write("x")

    metrics = runner.run(wf, safe_mode=True)
    assert not metrics.modules[0].success
    assert "file write disabled" in metrics.modules[0].exception


def test_safe_mode_blocks_fs_escape():
    runner = WorkflowSandboxRunner()

    def wf():
        open("../escape.txt")

    metrics = runner.run(wf, safe_mode=True)
    assert not metrics.modules[0].success
    assert "path escapes sandbox" in metrics.modules[0].exception
