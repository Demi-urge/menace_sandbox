import ast
from pathlib import Path


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(("info", msg % args if args else msg))

    def warning(self, msg, *args, **kwargs):
        self.messages.append(("warning", msg % args if args else msg))

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", msg % args if args else msg))

    def debug(self, msg, *args, **kwargs):
        self.messages.append(("debug", msg % args if args else msg))


class EnvProxy:
    def __init__(self, namespace: dict[str, object]) -> None:
        super().__setattr__("_ns", namespace)

    def __getattr__(self, name: str):
        try:
            return self._ns[name]
        except KeyError as exc:  # pragma: no cover - diagnostic aid
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        self._ns[name] = value


def _load_env_subset() -> EnvProxy:
    path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "environment.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    ensure_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_docker_client":
            ensure_node = node
            break
    if ensure_node is None:  # pragma: no cover - guard against refactor
        raise AssertionError("ensure_docker_client not found")

    subset = ast.Module(body=[ensure_node], type_ignores=[])
    code = compile(subset, str(path), "exec")

    logger = DummyLogger()
    namespace: dict[str, object] = {
        "docker": object(),
        "DockerException": Exception,
        "logger": logger,
        "_fallback_logger": lambda: logger,
        "_DOCKER_CLIENT": None,
        "_DOCKER_PING_TIMEOUT": 1.0,
        "_close_docker_client": lambda client: None,
        "_create_docker_client": lambda: (None, None),
        "_ping_docker": lambda client, timeout=None: (True, None),
        "_suspend_cleanup_workers": lambda reason=None: None,
    }
    exec(code, namespace)
    env = EnvProxy(namespace)
    env.logger = logger
    return env


def test_reconnect_when_ping_fails(monkeypatch):
    class DummyErr(Exception):
        pass

    env = _load_env_subset()
    good = object()
    env._DOCKER_CLIENT = object()

    monkeypatch.setattr(
        env,
        "_ping_docker",
        lambda client, timeout=None: (False, DummyErr("boom")),
        raising=False,
    )
    monkeypatch.setattr(env, "_create_docker_client", lambda: (good, None), raising=False)
    suspended: list[str | None] = []
    monkeypatch.setattr(
        env,
        "_suspend_cleanup_workers",
        lambda reason=None: suspended.append(reason),
        raising=False,
    )

    env.ensure_docker_client()

    assert env._DOCKER_CLIENT is good
    assert not suspended


def test_no_reconnect_when_ping_ok(monkeypatch):
    env = _load_env_subset()
    client = object()
    env._DOCKER_CLIENT = client

    monkeypatch.setattr(
        env,
        "_ping_docker",
        lambda client, timeout=None: (True, None),
        raising=False,
    )

    called = []
    monkeypatch.setattr(env, "_create_docker_client", lambda: called.append(True), raising=False)

    env.ensure_docker_client()

    assert env._DOCKER_CLIENT is client
    assert not called
