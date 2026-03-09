import ast
from pathlib import Path


def _load_symbols():
    src = Path("self_improvement/engine.py").read_text()
    tree = ast.parse(src)
    targets = {
        "_SI2F_LOCAL_RETRY_BASE_SECONDS",
        "_SI2F_LOCAL_RETRY_MAX_SECONDS",
        "_SI2F_LOCAL_RETRY_JITTER_SECONDS",
        "_SI2F_LOCAL_BREAKER_ATTEMPTS",
        "_SI2F_LOCAL_BREAKER_WINDOW_SECONDS",
        "_SI2F_LOCAL_BREAKER_PAUSE_SECONDS",
        "_SI2F_LOCAL_ATTEMPT_TIMESTAMPS",
        "_SI2F_LOCAL_BREAKER_STATE",
        "_SI2F_LOCAL_BREAKER_OPENED_AT",
        "_SI2F_LOCAL_LAST_SUCCESS_AT",
        "_SI2F_LOCAL_LAST_EXCEPTION",
        "_SI2F_LOCAL_LAST_ROOT_CAUSE",
        "_SI2F_LOCAL_RETRY_TOTAL",
    }
    fns = {
        "_si2f_local_metrics_snapshot",
        "_si2f_local_retry_delay_seconds",
        "_si2f_local_record_failure",
        "_si2f_local_probe_allowed",
        "_si2f_local_reset_breaker",
        "_si2f_local_record_retry",
        "_si2f_local_record_success",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in targets for n in names):
                body.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in targets:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in fns:
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])

    class DummyLogger:
        def __init__(self):
            self.errors = []

        def error(self, msg, *, extra):
            self.errors.append((msg, extra))

    ns = {
        "Any": object,
        "BaseException": BaseException,
        "time": __import__("time"),
        "random": __import__("random"),
        "logger": DummyLogger(),
        "log_record": lambda **kw: kw,
    }
    exec(compile(mod, "<ast>", "exec"), ns)
    return ns


def test_si2f_retry_delay_exponential():
    ns = _load_symbols()
    ns["random"].uniform = lambda a, b: 0.0
    f = ns["_si2f_local_retry_delay_seconds"]
    assert f(1) == 1.0
    assert f(2) == 2.0
    assert f(3) == 4.0


def test_si2f_breaker_opens_and_records_error():
    ns = _load_symbols()
    now = 100.0
    for _ in range(ns["_SI2F_LOCAL_BREAKER_ATTEMPTS"]):
        ns["_si2f_local_record_failure"](RuntimeError("x"), now=now)
        now += 1
    assert ns["_SI2F_LOCAL_BREAKER_STATE"] == "open"
    assert ns["logger"].errors
    _, payload = ns["logger"].errors[-1]
    assert payload["event"] == "si-2f.local-breaker-open"
    assert payload["attempts_in_window"] >= ns["_SI2F_LOCAL_BREAKER_ATTEMPTS"]


def test_si2f_probe_requires_recovery_or_timeout():
    ns = _load_symbols()
    ns["_SI2F_LOCAL_BREAKER_STATE"] = "open"
    ns["_SI2F_LOCAL_BREAKER_OPENED_AT"] = 10.0
    allowed = ns["_si2f_local_probe_allowed"](explicit_recovery=False, now=20.0)
    assert allowed is False
    allowed = ns["_si2f_local_probe_allowed"](explicit_recovery=True, now=20.0)
    assert allowed is True
    assert ns["_SI2F_LOCAL_BREAKER_STATE"] == "half-open"
