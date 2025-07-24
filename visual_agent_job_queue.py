from __future__ import annotations

"""Simple job queue serialising access to :class:`VisualAgentClient`."""

from typing import Iterable, Dict, Any, Callable
from concurrent.futures import Future
import threading
import queue


class VisualAgentJobQueue:
    """Queue that runs visual agent requests sequentially."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._queue: "queue.Queue[tuple[Callable, tuple, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                func, args, kwargs, fut = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = func(*args, **kwargs)
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    # ------------------------------------------------------------------
    def ask_async(self, messages: Iterable[Dict[str, str]]) -> Future:
        """Enqueue an ``ask`` call and return a :class:`Future`."""
        fut: Future = Future()
        self._queue.put((self.client.ask, (messages,), {}, fut))
        return fut

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        return self.ask_async(messages).result()

    def revert_async(self) -> Future:
        fut: Future = Future()
        self._queue.put((self.client.revert, tuple(), {}, fut))
        return fut

    def revert(self) -> bool:
        return self.revert_async().result()


__all__ = ["VisualAgentJobQueue"]
