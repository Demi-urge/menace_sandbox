from pathlib import Path
import ast
import types


class DummyLogger:
    def __init__(self):
        self.records = []

    def exception(self, msg, *args, **kwargs):
        self.records.append((msg % args if args else msg, kwargs.get("extra")))


def _load_async_track_usage():
    src = Path("sandbox_runner/cycle.py").read_text()
    tree = ast.parse(src)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_async_track_usage")
    module = ast.Module([func], type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns: dict[str, object] = {}

    class DummyThread:
        def __init__(self, target, daemon=True):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    ns.update(
        {
            "_ENABLE_RELEVANCY_RADAR": True,
            "_radar_track_usage": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            "record_output_impact": lambda *a, **k: None,
            "logger": DummyLogger(),
            "threading": types.SimpleNamespace(Thread=DummyThread),
            "time": types.SimpleNamespace(sleep=lambda *a, **k: None),
            "log_record": lambda **kw: kw,
        }
    )

    exec(compile(module, "<ast>", "exec"), ns)
    return ns["_async_track_usage"], ns["logger"]


def test_async_track_usage_logs_failure():
    fn, logger = _load_async_track_usage()
    fn("test.mod", 1.0)
    assert len(logger.records) == 3
    assert all(r[1].get("module") == "test.mod" for r in logger.records)
    assert [r[1].get("attempt") for r in logger.records] == [1, 2, 3]
