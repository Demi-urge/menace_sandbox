import ast
import types
from dynamic_path_router import resolve_path


class DummyLogger:
    def __init__(self):
        self.records = []

    def exception(self, msg, *args, **kwargs):
        self.records.append(("exception", kwargs.get("extra")))

    def error(self, msg, *args, **kwargs):
        self.records.append(("error", kwargs.get("extra")))


class DummyQueue:
    def __init__(self):
        self.items = []

    def put(self, item, *args, **kwargs):
        self.items.append(item)

    def get(self, *args, **kwargs):
        return self.items.pop(0)

    def task_done(self):
        pass


def _load_async_track_usage():
    src = resolve_path("sandbox_runner/cycle.py").read_text()
    tree = ast.parse(src)
    async_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_async_track_usage"
    )
    worker_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_usage_worker"
    )
    module = ast.Module([async_fn, worker_fn], type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns: dict[str, object] = {
        "_ENABLE_RELEVANCY_RADAR": True,
        "_radar_track_usage": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        "record_output_impact": lambda *a, **k: None,
        "logger": DummyLogger(),
        "_usage_queue": DummyQueue(),
        "_usage_stop_event": types.SimpleNamespace(is_set=lambda: False),
        "dispatch_alert": lambda *a, **k: None,
        "time": types.SimpleNamespace(sleep=lambda *a, **k: None),
        "log_record": lambda **kw: kw,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns["_async_track_usage"], ns["_usage_worker"], ns["_usage_queue"], ns["logger"]


def test_async_track_usage_logs_failure():
    track, worker, q, logger = _load_async_track_usage()
    track("test.mod", 1.0)
    q.put(None)
    worker()
    exceptions = [r for r in logger.records if r[0] == "exception"]
    assert [r[1]["attempt"] for r in exceptions] == [1, 2, 3, 4, 5]
    errors = [r for r in logger.records if r[0] == "error"]
    assert errors and errors[0][1]["attempts"] == 5
