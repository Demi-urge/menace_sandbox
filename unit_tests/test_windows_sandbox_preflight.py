import contextlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from dependency_health import DependencyMode

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

auto_env_stub = types.ModuleType("auto_env_setup")
auto_env_stub.ensure_env = lambda *args, **kwargs: None
sys.modules.setdefault("auto_env_setup", auto_env_stub)

bootstrap_stub = types.ModuleType("bootstrap_self_coding")
bootstrap_stub.bootstrap_self_coding = lambda *args, **kwargs: None
bootstrap_stub.purge_stale_files = lambda: None
bootstrap_stub._iter_cleanup_targets = lambda: []
sys.modules.setdefault("bootstrap_self_coding", bootstrap_stub)

heavy_stub = types.ModuleType("neurosales.scripts.setup_heavy_deps")
heavy_stub.main = lambda download_only=True: None
scripts_pkg = types.ModuleType("neurosales.scripts")
scripts_pkg.setup_heavy_deps = heavy_stub
neurosales_pkg = types.ModuleType("neurosales")
neurosales_pkg.scripts = scripts_pkg
sys.modules.setdefault("neurosales", neurosales_pkg)
sys.modules.setdefault("neurosales.scripts", scripts_pkg)
sys.modules.setdefault("neurosales.scripts.setup_heavy_deps", heavy_stub)

prime_stub = types.ModuleType("prime_registry")
prime_stub.main = lambda: None
sys.modules.setdefault("prime_registry", prime_stub)

sandbox_runner_pkg = sys.modules.setdefault(
    "sandbox_runner", types.ModuleType("sandbox_runner")
)

def _healthy_sandbox_snapshot() -> dict:
    return {
        "databases_accessible": True,
        "database_errors": {},
        "dependency_health": {"missing": []},
    }


bootstrap_module = types.ModuleType("sandbox_runner.bootstrap")
bootstrap_module.sandbox_health = staticmethod(_healthy_sandbox_snapshot)
setattr(sandbox_runner_pkg, "bootstrap", bootstrap_module)
sys.modules.setdefault("sandbox_runner.bootstrap", bootstrap_module)


class _StubSharedVectorService:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass

    def vectorise(self, *_args, **_kwargs) -> list[float]:  # pragma: no cover - trivial
        return []


vector_service_pkg = types.ModuleType("vector_service")
vectorizer_stub = types.ModuleType("vector_service.vectorizer")
vectorizer_stub.SharedVectorService = _StubSharedVectorService
vector_service_pkg.vectorizer = vectorizer_stub
sys.modules.setdefault("vector_service", vector_service_pkg)
sys.modules.setdefault("vector_service.vectorizer", vectorizer_stub)

stripe_pkg = types.ModuleType("stripe")
stripe_pkg.__path__ = []  # mark as package
sys.modules.setdefault("stripe", stripe_pkg)

stripe_client_stub = types.ModuleType("stripe._stripe_client")
stripe_client_stub.StripeClient = object
sys.modules.setdefault("stripe._stripe_client", stripe_client_stub)

stripe_services_stub = types.ModuleType("stripe._v1_services")
stripe_services_stub.V1Services = object
sys.modules.setdefault("stripe._v1_services", stripe_services_stub)

stripe_payment_stub = types.ModuleType("stripe._payment_intent_service")
stripe_payment_stub.PaymentIntentService = object
sys.modules.setdefault("stripe._payment_intent_service", stripe_payment_stub)

sys.modules.setdefault("datasets", types.ModuleType("datasets"))
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

from tools import windows_sandbox_launcher_gui as gui


class DummyLogger:
    """Capture log messages for assertions."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, message: str, *args) -> None:  # pragma: no cover - debug support
        self.records.append(("debug", message % args if args else message))

    def info(self, message: str, *args) -> None:
        self.records.append(("info", message % args if args else message))

    def warning(self, message: str, *args) -> None:  # pragma: no cover - unused
        self.records.append(("warning", message % args if args else message))

    def error(self, message: str, *args) -> None:  # pragma: no cover - error support
        self.records.append(("error", message % args if args else message))

    def exception(self, message: str, *args) -> None:
        self.records.append(("exception", message % args if args else message))


class StepImplementationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = DummyLogger()

    def test_git_sync_runs_expected_commands(self) -> None:
        with mock.patch("tools.windows_sandbox_launcher_gui._run_command") as run_mock:
            gui._git_sync(self.logger)

        expected_calls = [
            mock.call(self.logger, ["git", "fetch", "origin"], cwd=gui.REPO_ROOT),
            mock.call(self.logger, ["git", "reset", "--hard", "origin/main"], cwd=gui.REPO_ROOT),
        ]
        run_mock.assert_has_calls(expected_calls)
        self.assertEqual(run_mock.call_count, 2)

    def test_purge_stale_files_invokes_bootstrap_cleanup(self) -> None:
        class _DummyCandidate:
            def __init__(self, name: str) -> None:
                self.name = name
                self._exists = True

            def exists(self) -> bool:
                return self._exists

            def remove(self) -> None:
                self._exists = False

            def __str__(self) -> str:  # pragma: no cover - logging helper
                return self.name

        candidates = [_DummyCandidate("one"), _DummyCandidate("two"), _DummyCandidate("three")]

        def _iter_cleanup() -> list[_DummyCandidate]:
            return list(candidates)

        def _purge() -> None:
            candidates[0].remove()
            candidates[2].remove()

        with mock.patch(
            "bootstrap_self_coding._iter_cleanup_targets", side_effect=_iter_cleanup
        ) as iter_mock, mock.patch(
            "bootstrap_self_coding.purge_stale_files", side_effect=_purge
        ) as purge_mock:
            gui._purge_stale_files(self.logger)

        iter_mock.assert_called_once()
        purge_mock.assert_called_once()
        self.assertIn(("info", "Purged 2 stale sandbox artefacts."), self.logger.records)

    def test_cleanup_lock_and_model_artifacts_removes_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as home_dir:
            repo_path = Path(repo_dir)
            sandbox_dir = repo_path / "sandbox_data"
            sandbox_dir.mkdir()
            (sandbox_dir / "sandbox.lock").write_text("")
            (sandbox_dir / "sandbox.lock.tmp").write_text("")

            home_path = Path(home_dir)
            hf_root = home_path / ".cache" / "huggingface" / "transformers"
            (hf_root / "model").mkdir(parents=True)
            (hf_root / "model" / "weights.lock").write_text("")
            (hf_root / "model" / "nested.tmp").mkdir()
            (hf_root / "model" / "nested.tmp" / "artifact").write_text("")
            (hf_root / "orphan.incomplete").mkdir(parents=True)
            (hf_root / "empty_download").mkdir()

            with mock.patch(
                "tools.windows_sandbox_launcher_gui.REPO_ROOT", repo_path
            ), mock.patch(
                "tools.windows_sandbox_launcher_gui.Path.home", return_value=home_path
            ):
                gui._cleanup_lock_and_model_artifacts(self.logger)

            self.assertFalse((sandbox_dir / "sandbox.lock").exists())
            self.assertFalse((sandbox_dir / "sandbox.lock.tmp").exists())
            self.assertFalse((hf_root / "model" / "weights.lock").exists())
            self.assertFalse((hf_root / "model" / "nested.tmp").exists())
            self.assertFalse((hf_root / "orphan.incomplete").exists())
            self.assertFalse((hf_root / "empty_download").exists())
            self.assertIn(
                ("info", "Removed 3 stale files and 4 stale directories."),
                self.logger.records,
            )

    def test_install_heavy_dependencies_invokes_setup(self) -> None:
        with mock.patch("neurosales.scripts.setup_heavy_deps.main") as main_mock:
            gui._install_heavy_dependencies(self.logger)

        main_mock.assert_called_once_with(download_only=True)

    def test_warm_shared_vector_service_instantiates_service(self) -> None:
        with mock.patch("vector_service.vectorizer.SharedVectorService") as svc_mock:
            instance = svc_mock.return_value
            instance.vectorise.return_value = []
            gui._warm_shared_vector_service(self.logger)

        svc_mock.assert_called_once_with()
        instance.vectorise.assert_called_once_with("text", {"text": "warmup"})

    def test_ensure_env_flags_sets_expected_variables(self) -> None:
        with mock.patch("auto_env_setup.ensure_env") as ensure_mock:
            prev_bootstrap = os.environ.get("SANDBOX_ENABLE_BOOTSTRAP")
            prev_self_coding = os.environ.get("SANDBOX_ENABLE_SELF_CODING")
            os.environ.pop("SANDBOX_ENABLE_BOOTSTRAP", None)
            os.environ.pop("SANDBOX_ENABLE_SELF_CODING", None)

            def _restore() -> None:
                if prev_bootstrap is None:
                    os.environ.pop("SANDBOX_ENABLE_BOOTSTRAP", None)
                else:
                    os.environ["SANDBOX_ENABLE_BOOTSTRAP"] = prev_bootstrap
                if prev_self_coding is None:
                    os.environ.pop("SANDBOX_ENABLE_SELF_CODING", None)
                else:
                    os.environ["SANDBOX_ENABLE_SELF_CODING"] = prev_self_coding

            self.addCleanup(_restore)
            gui._ensure_env_flags(self.logger)

        ensure_mock.assert_called_once_with(str(gui.REPO_ROOT / ".env"))
        self.assertEqual(os.environ["SANDBOX_ENABLE_BOOTSTRAP"], "1")
        self.assertEqual(os.environ["SANDBOX_ENABLE_SELF_CODING"], "1")

    def test_prime_registry_invoked(self) -> None:
        with mock.patch("prime_registry.main") as main_mock:
            gui._prime_registry(self.logger)

        main_mock.assert_called_once_with()

    def test_install_python_dependencies_runs_pip(self) -> None:
        with mock.patch("tools.windows_sandbox_launcher_gui._run_command") as run_mock:
            gui._install_python_dependencies(self.logger)

        expected_calls = [
            mock.call(
                self.logger,
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=gui.REPO_ROOT,
            ),
            mock.call(
                self.logger,
                [sys.executable, "-m", "pip", "install", "jsonschema"],
                cwd=gui.REPO_ROOT,
            ),
        ]
        run_mock.assert_has_calls(expected_calls)
        self.assertEqual(run_mock.call_count, 2)

    def test_bootstrap_self_coding_invoked(self) -> None:
        with mock.patch("bootstrap_self_coding.bootstrap_self_coding") as bootstrap_mock:
            gui._bootstrap_self_coding(self.logger)

        bootstrap_mock.assert_called_once_with("AICounterBot")

    def test_run_command_reports_failures(self) -> None:
        completed = subprocess.CompletedProcess(["echo"], 0, stdout="output", stderr="warn")
        with mock.patch("subprocess.run", return_value=completed) as run_mock:
            result = gui._run_command(self.logger, ["echo"], cwd="/")

        self.assertIs(result, completed)
        run_mock.assert_called_once()
        self.assertIn(("info", "Command stdout: output"), self.logger.records)
        self.assertIn(("warning", "Command stderr: warn"), self.logger.records)

        error_process = subprocess.CalledProcessError(1, ["fail"], "bad", "worse")
        with mock.patch("subprocess.run", side_effect=error_process):
            with self.assertRaises(RuntimeError):
                gui._run_command(self.logger, ["fail"])

        self.assertIn(("error", "Command stdout: bad"), self.logger.records)
        self.assertIn(("error", "Command stderr: worse"), self.logger.records)


class PreflightWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = DummyLogger()
        self.pause_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str, Optional[dict[str, object]]]]" = (
            queue.Queue()
        )
        self.abort_event = threading.Event()
        self.debug_queue: "queue.Queue[str]" = queue.Queue()

    def _patch_step(self, name: str, calls: list[str], *, side_effect=None):
        def _runner(_logger):
            calls.append(name)
            if side_effect is not None:
                raise side_effect

        return mock.patch.object(gui, name, side_effect=_runner)

    def test_steps_execute_in_order(self) -> None:
        calls: list[str] = []

        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            self._patch_step("_cleanup_lock_and_model_artifacts", calls),
            self._patch_step("_install_heavy_dependencies", calls),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_env_flags", calls),
            self._patch_step("_prime_registry", calls),
            self._patch_step("_install_python_dependencies", calls),
            self._patch_step("_bootstrap_self_coding", calls),
            mock.patch(
                "sandbox_runner.bootstrap.sandbox_health",
                side_effect=lambda: calls.append("sandbox_health")
                or {
                    "databases_accessible": True,
                    "database_errors": {},
                    "dependency_health": {"missing": []},
                },
            ),
        ]

        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            result = gui.run_full_preflight(
                logger=self.logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=self.debug_queue,
            )

        expected_steps = [
            "_git_sync",
            "_purge_stale_files",
            "_cleanup_lock_and_model_artifacts",
            "_install_heavy_dependencies",
            "_warm_shared_vector_service",
            "_ensure_env_flags",
            "_prime_registry",
            "_install_python_dependencies",
            "_bootstrap_self_coding",
        ]
        self.assertEqual(calls[: len(expected_steps)], expected_steps)
        self.assertEqual(calls[len(expected_steps):], ["sandbox_health"])
        self.assertIsInstance(result, dict)
        self.assertIn("snapshot", result)
        self.assertEqual(
            result["snapshot"],
            {
                "databases_accessible": True,
                "database_errors": {},
                "dependency_health": {"missing": []},
            },
        )
        self.assertFalse(self.pause_event.is_set())
        self.assertTrue(self.decision_queue.empty())

    def test_pause_triggered_on_failure(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("lock cleanup failed")
        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            self._patch_step(
                "_cleanup_lock_and_model_artifacts",
                calls,
                side_effect=failing_error,
            ),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_env_flags", calls),
            self._patch_step("_prime_registry", calls),
            self._patch_step("_install_heavy_dependencies", calls),
            self._patch_step("_install_python_dependencies", calls),
            self._patch_step("_bootstrap_self_coding", calls),
            mock.patch(
                "sandbox_runner.bootstrap.sandbox_health",
                side_effect=lambda: calls.append("sandbox_health")
                or {
                    "databases_accessible": True,
                    "dependency_health": {"missing": []},
                },
            ),
        ]

        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)

            thread = threading.Thread(
                target=gui.run_full_preflight,
                kwargs={
                    "logger": self.logger,
                    "pause_event": self.pause_event,
                    "decision_queue": self.decision_queue,
                    "abort_event": self.abort_event,
                    "debug_queue": self.debug_queue,
                },
                daemon=True,
            )
            thread.start()

            deadline = time.time() + 2
            while not self.pause_event.is_set() and time.time() < deadline:
                time.sleep(0.01)

            self.assertTrue(self.pause_event.is_set())
            self.assertFalse(self.decision_queue.empty())
            title, message, context = self.decision_queue.get_nowait()
            self.assertIn("Lock and model cleanup failed", title)
            self.assertIn("Removing stale lock files and model caches", message)
            self.assertIsInstance(context, dict)
            self.assertEqual(context.get("step"), "_cleanup_lock_and_model_artifacts")
            self.assertIn("lock cleanup failed", context.get("exception", ""))

            self.pause_event.clear()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(
            calls,
            [
                "_git_sync",
                "_purge_stale_files",
                "_cleanup_lock_and_model_artifacts",
            ],
        )
        self.assertFalse(self.abort_event.is_set())
        self.assertFalse(self.debug_queue.empty())
        self.assertIn("lock cleanup failed", self.debug_queue.get_nowait())

    def test_abort_event_short_circuits_execution(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("stop here")
        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            self._patch_step(
                "_cleanup_lock_and_model_artifacts",
                calls,
                side_effect=failing_error,
            ),
        ]

        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)

            self.abort_event.set()

            thread = threading.Thread(
                target=gui.run_full_preflight,
                kwargs={
                    "logger": self.logger,
                    "pause_event": self.pause_event,
                    "decision_queue": self.decision_queue,
                    "abort_event": self.abort_event,
                    "debug_queue": self.debug_queue,
                },
                daemon=True,
            )
            thread.start()

            deadline = time.time() + 2
            while not self.pause_event.is_set() and time.time() < deadline:
                time.sleep(0.01)

            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(
            calls,
            [],
        )
        self.assertTrue(self.abort_event.is_set())
        self.assertTrue(self.decision_queue.empty())

    def test_run_full_preflight_marks_abort_after_failure(self) -> None:
        calls: list[str] = []

        def _failing_step(_logger) -> None:
            calls.append("_git_sync")
            self.abort_event.set()
            raise RuntimeError("simulated abort")

        with mock.patch.object(gui, "_git_sync", side_effect=_failing_step):
            result = gui.run_full_preflight(
                logger=self.logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=self.debug_queue,
            )

        self.assertEqual(calls, ["_git_sync"])
        self.assertTrue(result.get("paused"))
        self.assertTrue(result.get("aborted"))
        self.assertEqual(result.get("failed_step"), "_git_sync")
        self.assertTrue(self.abort_event.is_set())
        self.assertTrue(self.pause_event.is_set())
        self.pause_event.clear()


class PreflightGuiDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gui = gui.SandboxLauncherGUI.__new__(gui.SandboxLauncherGUI)
        self.gui.pause_event = threading.Event()
        self.gui.abort_event = threading.Event()
        self.gui.run_button = types.SimpleNamespace(state=mock.Mock())
        self.gui.retry_button = types.SimpleNamespace(
            state=mock.Mock(),
            grid=mock.Mock(),
            grid_remove=mock.Mock(),
        )
        self.gui.launch_button = types.SimpleNamespace(state=mock.Mock())
        self.gui._start_preflight_thread = mock.Mock()
        self.gui._ensure_elapsed_timer_running = mock.Mock()
        self.gui._finish_elapsed_timer = mock.Mock()
        self.gui._update_debug_panel = mock.Mock()
        self.gui._format_health_failure_message = mock.Mock(return_value="details")
        self.gui._preflight_start_time = time.monotonic()
        self.gui._pause_dialog_presented = False
        self.gui._paused_step_index = None
        self.gui._last_failed_step = None
        self.gui._latest_pause_context_trace = "context trace"
        self.gui._pause_user_decision = None

    def test_continue_decision_restarts_next_step(self) -> None:
        payload = {
            "paused": True,
            "failed_index": 1,
            "failed_step": "_purge_stale_files",
            "start_index": 0,
            "debug_traces": ["traceback"],
        }
        self.gui._pause_user_decision = "continue"

        with mock.patch.object(gui, "logger") as logger_mock, mock.patch.object(
            gui.messagebox, "showwarning"
        ) as warn_mock:
            self.gui._handle_preflight_completion(payload)

        self.assertEqual(self.gui._start_preflight_thread.call_count, 1)
        resume_index = self.gui._start_preflight_thread.call_args[0][0]
        self.assertEqual(resume_index, 2)

        self.assertGreaterEqual(len(self.gui.run_button.state.call_args_list), 2)
        self.assertEqual(self.gui.run_button.state.call_args_list[0][0][0], ["!disabled"])
        self.assertEqual(self.gui.run_button.state.call_args_list[-1][0][0], ["disabled"])

        self.gui.retry_button.state.assert_called_with(["disabled"])
        self.gui.retry_button.grid_remove.assert_called_once()
        self.assertFalse(self.gui._pause_dialog_presented)
        self.assertIsNone(self.gui._paused_step_index)
        self.assertIsNone(self.gui._pause_user_decision)
        self.assertIsNone(self.gui._latest_pause_context_trace)
        self.gui._ensure_elapsed_timer_running.assert_called_once()
        self.gui._finish_elapsed_timer.assert_not_called()
        self.assertFalse(self.gui.abort_event.is_set())
        self.assertFalse(self.gui.pause_event.is_set())

        self.assertIn(mock.call(["traceback"]), self.gui._update_debug_panel.call_args_list)
        self.assertIn(mock.call(["context trace"]), self.gui._update_debug_panel.call_args_list)
        warn_mock.assert_not_called()

        self.assertEqual(logger_mock.info.call_count, 1)
        log_message = logger_mock.info.call_args[0][0]
        self.assertIn("continue preflight", log_message)

    def test_abort_decision_marks_run_aborted(self) -> None:
        payload = {
            "paused": True,
            "failed_index": 0,
            "failed_step": "_git_sync",
            "start_index": 0,
        }
        self.gui._pause_user_decision = "abort"
        self.gui.pause_event.set()

        with mock.patch.object(gui, "logger") as logger_mock, mock.patch.object(
            gui.messagebox, "showwarning"
        ) as warn_mock:
            self.gui._handle_preflight_completion(payload)

        self.assertFalse(self.gui.pause_event.is_set())
        self.assertTrue(self.gui.abort_event.is_set())
        self.assertEqual(self.gui.retry_button.state.call_args[0][0], ["disabled"])
        self.gui.retry_button.grid_remove.assert_called_once()
        self.gui.launch_button.state.assert_called_with(["disabled"])
        self.assertIsNone(self.gui._pause_user_decision)
        self.assertIsNone(self.gui._latest_pause_context_trace)
        self.gui._finish_elapsed_timer.assert_called_once_with("Preflight aborted")
        self.gui._start_preflight_thread.assert_not_called()
        self.gui._ensure_elapsed_timer_running.assert_not_called()
        warn_mock.assert_not_called()

        self.assertEqual(self.gui._update_debug_panel.call_args_list, [mock.call(["context trace"])])
        self.gui._format_health_failure_message.assert_called_once()
        logger_mock.warning.assert_called_once()
        self.assertIn("aborted", logger_mock.warning.call_args[0][0])



class HealthEvaluationTests(unittest.TestCase):
    def test_health_snapshot_success(self) -> None:
        health = {
            "databases_accessible": True,
            "dependency_health": {"missing": []},
        }

        healthy, failures = gui._evaluate_health_snapshot(
            health, dependency_mode=DependencyMode.STRICT
        )

        self.assertTrue(healthy)
        self.assertEqual(failures, [])

    def test_health_snapshot_database_failure(self) -> None:
        health = {
            "databases_accessible": False,
            "database_errors": {"metrics.db": "permission denied"},
        }

        healthy, failures = gui._evaluate_health_snapshot(
            health, dependency_mode=DependencyMode.STRICT
        )

        self.assertFalse(healthy)
        self.assertTrue(any("databases inaccessible" in msg for msg in failures))

    def test_health_snapshot_optional_dependencies_respect_mode(self) -> None:
        health = {
            "databases_accessible": True,
            "dependency_health": {
                "missing": [
                    {"name": "foo", "optional": True},
                    {"name": "bar", "optional": False},
                ]
            },
        }

        strict_healthy, strict_failures = gui._evaluate_health_snapshot(
            health, dependency_mode=DependencyMode.STRICT
        )
        minimal_healthy, minimal_failures = gui._evaluate_health_snapshot(
            health, dependency_mode=DependencyMode.MINIMAL
        )

        self.assertFalse(strict_healthy)
        self.assertTrue(strict_failures)
        self.assertFalse(minimal_healthy)
        # optional dependency should not appear when minimal mode ignores it
        self.assertTrue(all("foo" not in failure for failure in minimal_failures))
        self.assertTrue(any("bar" in failure for failure in minimal_failures))


if __name__ == "__main__":  # pragma: no cover - manual execution
    unittest.main()
