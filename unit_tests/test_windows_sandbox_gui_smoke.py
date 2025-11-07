"""Smoke tests for the Windows sandbox launcher GUI helpers."""

import queue
import threading
from unittest import mock

import pytest

from tools import windows_sandbox_launcher_gui as gui


class _DummyLogger:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.removed: list[object] = []
        self.exceptions: list[str] = []

    def addHandler(self, handler: object) -> None:  # pragma: no cover - trivial
        self.added.append(handler)

    def removeHandler(self, handler: object) -> None:  # pragma: no cover - defensive
        self.removed.append(handler)

    def exception(self, message: str, *args) -> None:  # pragma: no cover - defensive
        if args:
            message = message % args
        self.exceptions.append(message)


class _DummyText:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.inserts: list[tuple[str, str, str]] = []
        self.seen: list[str] = []

    def configure(self, **kwargs) -> None:
        state = kwargs.get("state")
        if state is not None:
            self.states.append(state)

    def insert(self, index: str, message: str, tag: str) -> None:
        self.inserts.append((index, message, tag))

    def see(self, index: str) -> None:
        self.seen.append(index)


@pytest.fixture(autouse=True)
def _restore_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure any modifications to the module logger are reverted."""

    original_logger = gui.logger
    yield
    monkeypatch.setattr(gui, "logger", original_logger)


def _make_stub_instance() -> gui.SandboxLauncherGUI:
    inst = gui.SandboxLauncherGUI.__new__(gui.SandboxLauncherGUI)
    inst._file_log_queue = None
    inst._file_queue_handler = None
    inst._file_handler = None
    inst._file_listener = None
    inst._log_file_path = gui.LOG_FILE_PATH
    inst._pause_dialog_window = None
    inst._pause_dialog_message_var = None
    inst._pause_dialog_step_name = None
    inst._pause_dialog_presented = False
    inst._resume_after_pause = False
    inst._abort_after_pause = False
    inst._pause_user_decision = None
    return inst


def test_initialise_file_logging_wires_queue_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _make_stub_instance()

    fake_queue = mock.Mock(name="log_queue")
    fake_handler = mock.Mock(name="file_handler")
    fake_queue_handler = mock.Mock(name="queue_handler")
    fake_listener = mock.Mock(name="listener")

    monkeypatch.setattr(gui.queue, "Queue", mock.Mock(return_value=fake_queue))
    monkeypatch.setattr(gui, "RotatingFileHandler", mock.Mock(return_value=fake_handler))
    monkeypatch.setattr(gui, "QueueHandler", mock.Mock(return_value=fake_queue_handler))
    monkeypatch.setattr(gui, "QueueListener", mock.Mock(return_value=fake_listener))

    dummy_logger = _DummyLogger()
    monkeypatch.setattr(gui, "logger", dummy_logger)

    inst._initialise_file_logging()

    assert inst._file_log_queue is fake_queue
    assert inst._file_queue_handler is fake_queue_handler
    assert inst._file_handler is fake_handler
    assert inst._file_listener is fake_listener
    assert fake_listener.start.call_count == 1
    assert dummy_logger.added == [fake_queue_handler]
    gui.RotatingFileHandler.assert_called_once_with(
        gui.REPO_ROOT / "menace_gui_logs.txt",
        maxBytes=1_048_576,
        backupCount=5,
        encoding="utf-8",
    )
    gui.QueueHandler.assert_called_once_with(fake_queue)


def test_drain_log_queue_flushes_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _make_stub_instance()
    inst.log_queue = queue.Queue()
    inst.log_queue.put(("info", "first line\n"))
    inst.log_queue.put(("error", "second line\n"))

    text = _DummyText()
    inst.log_text = text
    inst._drain_running = True

    scheduled: list[bool] = []
    monkeypatch.setattr(inst, "_schedule_log_drain", lambda: scheduled.append(True))

    inst._drain_log_queue()

    assert inst.log_queue.empty()
    assert text.inserts == [
        (gui.tk.END, "first line\n", "info"),
        (gui.tk.END, "second line\n", "error"),
    ]
    assert text.states == [gui.tk.NORMAL, gui.tk.DISABLED]
    assert text.seen == [gui.tk.END]
    assert scheduled == [True]


def test_handle_pause_prompt_prompts_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _make_stub_instance()
    inst.pause_event = threading.Event()
    inst.pause_event.set()
    inst.abort_event = threading.Event()
    inst.abort_event.set()
    inst.decision_queue = queue.Queue()
    context = {"step": "demo", "exception": "boom"}
    inst.decision_queue.put(("Failure", "Step failed", context))

    captured: dict[str, str] = {}

    def _fake_prompt(*, title: str, message: str, step_name: str | None) -> str:
        captured["title"] = title
        captured["message"] = message
        captured["step_name"] = step_name or ""
        return "continue"

    monkeypatch.setattr(inst, "_prompt_pause_decision", _fake_prompt)

    inst._handle_pause_prompt()

    assert inst._pause_user_decision == "continue"
    assert inst._resume_after_pause is True
    assert inst._abort_after_pause is False
    assert not inst.pause_event.is_set()
    assert not inst.abort_event.is_set()
    assert inst._latest_pause_context == context
    assert inst._latest_pause_context_trace
    assert "Details" in captured["message"]
    assert "boom" in captured["message"]
    assert captured["step_name"] == "demo"


def test_handle_pause_prompt_prompts_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _make_stub_instance()
    inst.pause_event = threading.Event()
    inst.pause_event.set()
    inst.abort_event = threading.Event()
    inst.decision_queue = queue.Queue()
    context = {"step": "demo", "exception": "explosion"}
    inst.decision_queue.put(("Failure", "Step failed", context))

    monkeypatch.setattr(inst, "_prompt_pause_decision", lambda **_: "abort")

    inst._handle_pause_prompt()

    assert inst._pause_user_decision == "abort"
    assert inst._resume_after_pause is False
    assert inst._abort_after_pause is True
    assert inst.pause_event.is_set()
    assert inst.abort_event.is_set()
    assert inst._latest_pause_context == context
    assert inst._latest_pause_context_trace
