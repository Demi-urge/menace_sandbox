import contextlib
import os
import queue
import sys
import threading
import time
import types
import unittest
from unittest import mock

from dependency_health import DependencyMode

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

auto_env_stub = types.ModuleType("auto_env_setup")
auto_env_stub.ensure_env = lambda *args, **kwargs: None
sys.modules.setdefault("auto_env_setup", auto_env_stub)

bootstrap_stub = types.ModuleType("bootstrap_self_coding")
bootstrap_stub.bootstrap_self_coding = lambda *args, **kwargs: None
bootstrap_stub.purge_stale_files = lambda: None
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


vector_service_stub = types.ModuleType("vector_service")
vector_service_stub.SharedVectorService = _StubSharedVectorService
sys.modules.setdefault("vector_service", vector_service_stub)

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

    def info(self, message: str, *args) -> None:
        self.records.append(("info", message % args if args else message))

    def warning(self, message: str, *args) -> None:  # pragma: no cover - unused
        self.records.append(("warning", message % args if args else message))

    def exception(self, message: str, *args) -> None:
        self.records.append(("exception", message % args if args else message))


class PreflightWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = DummyLogger()
        self.pause_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.abort_event = threading.Event()
        self.retry_event = threading.Event()
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
            self._patch_step("_delete_lock_files", calls),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_env_flags", calls),
            self._patch_step("_prime_registry", calls),
            self._patch_step("_install_dependencies", calls),
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
            result = gui.run_full_preflight(
                logger=self.logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                retry_event=self.retry_event,
                debug_queue=self.debug_queue,
            )

        self.assertEqual(
            calls,
            [
                "_git_sync",
                "_purge_stale_files",
                "_delete_lock_files",
                "_warm_shared_vector_service",
                "_ensure_env_flags",
                "_prime_registry",
                "_install_dependencies",
                "_bootstrap_self_coding",
                "sandbox_health",
            ],
        )
        self.assertIsInstance(result, dict)
        self.assertFalse(self.pause_event.is_set())
        self.assertTrue(self.decision_queue.empty())

    def test_pause_triggered_on_failure(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("lock cleanup failed")
        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            self._patch_step("_delete_lock_files", calls, side_effect=failing_error),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_env_flags", calls),
            self._patch_step("_prime_registry", calls),
            self._patch_step("_install_dependencies", calls),
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
                    "retry_event": self.retry_event,
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
            title, message = self.decision_queue.get_nowait()
            self.assertIn("Lock artefact removal failed", title)
            self.assertIn("Removing stale lock files", message)

            self.pause_event.clear()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(
            calls,
            [
                "_git_sync",
                "_purge_stale_files",
                "_delete_lock_files",
                "_warm_shared_vector_service",
                "_ensure_env_flags",
                "_prime_registry",
                "_install_dependencies",
                "_bootstrap_self_coding",
                "sandbox_health",
            ],
        )
        self.assertFalse(self.abort_event.is_set())
        self.assertFalse(self.debug_queue.empty())
        self.assertIn("lock cleanup failed", self.debug_queue.get_nowait())

    def test_retry_event_retries_failed_step(self) -> None:
        calls: list[str] = []
        attempts = {"count": 0}

        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            mock.patch.object(
                gui,
                "_delete_lock_files",
                side_effect=lambda _logger: self._retryable_step(calls, attempts),
            ),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_env_flags", calls),
            self._patch_step("_prime_registry", calls),
            self._patch_step("_install_dependencies", calls),
            self._patch_step("_bootstrap_self_coding", calls),
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
                    "retry_event": self.retry_event,
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

            self.retry_event.set()

            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(calls.count("_delete_lock_files"), 2)
        self.assertFalse(self.abort_event.is_set())
        self.assertFalse(self.pause_event.is_set())

    def _retryable_step(
        self, calls: list[str], attempts: dict[str, int]
    ) -> None:
        calls.append("_delete_lock_files")
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise RuntimeError("retry me")

    def test_abort_event_short_circuits_execution(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("stop here")
        patchers = [
            self._patch_step("_git_sync", calls),
            self._patch_step("_purge_stale_files", calls),
            self._patch_step("_delete_lock_files", calls, side_effect=failing_error),
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
                    "retry_event": self.retry_event,
                    "debug_queue": self.debug_queue,
                },
                daemon=True,
            )
            thread.start()

            deadline = time.time() + 2
            while not self.pause_event.is_set() and time.time() < deadline:
                time.sleep(0.01)

            self.abort_event.set()
            self.pause_event.clear()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        self.assertEqual(
            calls,
            [
                "_git_sync",
                "_purge_stale_files",
                "_delete_lock_files",
            ],
        )
        self.assertTrue(self.abort_event.is_set())


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
