import contextlib
import os
import queue
import sys
import threading
import time
import types
import unittest
from unittest import mock

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

    def _patch_step(self, name: str, calls: list[str], *, side_effect=None):
        def _runner(_logger):
            calls.append(name)
            if side_effect is not None:
                raise side_effect

        return mock.patch.object(gui, name, side_effect=_runner)

    def test_steps_execute_in_order(self) -> None:
        calls: list[str] = []

        patchers = [
            self._patch_step("_git_fetch_and_reset", calls),
            self._patch_step("_purge_stale_state", calls),
            self._patch_step("_remove_lock_artifacts", calls),
            self._patch_step("_prefetch_heavy_dependencies", calls),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_environment", calls),
            self._patch_step("_prime_self_coding_registry", calls),
            self._patch_step("_run_pip_commands", calls),
            self._patch_step("_bootstrap_ai_counter_bot", calls),
        ]

        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            gui.run_full_preflight(
                logger=self.logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
            )

        self.assertEqual(
            calls,
            [
                "_git_fetch_and_reset",
                "_purge_stale_state",
                "_remove_lock_artifacts",
                "_prefetch_heavy_dependencies",
                "_warm_shared_vector_service",
                "_ensure_environment",
                "_prime_self_coding_registry",
                "_run_pip_commands",
                "_bootstrap_ai_counter_bot",
            ],
        )
        self.assertFalse(self.pause_event.is_set())
        self.assertTrue(self.decision_queue.empty())

    def test_pause_triggered_on_failure(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("lock cleanup failed")
        patchers = [
            self._patch_step("_git_fetch_and_reset", calls),
            self._patch_step("_purge_stale_state", calls),
            self._patch_step(
                "_remove_lock_artifacts", calls, side_effect=failing_error
            ),
            self._patch_step("_prefetch_heavy_dependencies", calls),
            self._patch_step("_warm_shared_vector_service", calls),
            self._patch_step("_ensure_environment", calls),
            self._patch_step("_prime_self_coding_registry", calls),
            self._patch_step("_run_pip_commands", calls),
            self._patch_step("_bootstrap_ai_counter_bot", calls),
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
                "_git_fetch_and_reset",
                "_purge_stale_state",
                "_remove_lock_artifacts",
                "_prefetch_heavy_dependencies",
                "_warm_shared_vector_service",
                "_ensure_environment",
                "_prime_self_coding_registry",
                "_run_pip_commands",
                "_bootstrap_ai_counter_bot",
            ],
        )
        self.assertFalse(self.abort_event.is_set())

    def test_abort_event_short_circuits_execution(self) -> None:
        calls: list[str] = []

        failing_error = RuntimeError("stop here")
        patchers = [
            self._patch_step("_git_fetch_and_reset", calls),
            self._patch_step("_purge_stale_state", calls),
            self._patch_step(
                "_remove_lock_artifacts", calls, side_effect=failing_error
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
                "_git_fetch_and_reset",
                "_purge_stale_state",
                "_remove_lock_artifacts",
            ],
        )
        self.assertTrue(self.abort_event.is_set())


if __name__ == "__main__":  # pragma: no cover - manual execution
    unittest.main()
