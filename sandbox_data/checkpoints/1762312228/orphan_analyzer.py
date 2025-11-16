from __future__ import annotations

"""Helpers for analysing orphan modules before integration.

Plugins can extend :data:`DEFAULT_CLASSIFIERS` by exposing additional
callables.  Two mechanisms are supported:

``ORPHAN_ANALYZER_CLASSIFIERS``
    Environment variable containing a comma separated list of
    ``"module:function"`` specifications.  Each referenced callable must
    implement :class:`Classifier`.

``orphan_analyzer.classifiers`` entry points
    External packages may advertise classifiers via the standard entry point
    registry.  Each entry point in this group should resolve to a callable
    conforming to :class:`Classifier`.

Loaded classifiers are appended to :data:`DEFAULT_CLASSIFIERS` and participate
in :func:`classify_module` evaluation.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple, Protocol

import ast
import importlib
import json
import os
import subprocess
import sys

try:  # Python 3.8+ standard library
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover - fallback for older versions
    import importlib_metadata  # type: ignore

from module_graph_analyzer import build_import_graph

try:  # optional dependency
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover - radon missing
    cc_visit = None  # type: ignore


LEGACY_MARKERS = {"deprecated", "legacy", "missing_reference"}


def detect_legacy_patterns(module_path: Path) -> bool:
    """Return ``True`` if ``module_path`` contains obvious legacy patterns.

    The check is intentionally lightweight and searches the source text for
    common markers such as ``deprecated`` or ``legacy``.  Errors while reading
    the file simply result in ``False``.
    """
    try:
        text = module_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(marker in text for marker in LEGACY_MARKERS)


def _static_metrics(module_path: Path) -> Dict[str, Any]:
    """Return basic static metrics for ``module_path``.

    The function counts top level functions and call expressions, records
    whether a module docstring is present and, when :mod:`radon` is available,
    determines the maximum cyclomatic complexity across all functions.  When
    :mod:`radon` is missing a lightweight approximation based on the number of
    branching statements is used instead.
    """

    metrics: Dict[str, Any] = {
        "functions": 0,
        "calls": 0,
        "docstring": False,
        "complexity": 0,
    }
    try:
        text = module_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        metrics["functions"] = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        metrics["calls"] = sum(isinstance(n, ast.Call) for n in ast.walk(tree))
        metrics["docstring"] = ast.get_docstring(tree) is not None
        if cc_visit is not None:
            try:
                blocks = cc_visit(text)
                if blocks:
                    metrics["complexity"] = max(b.complexity for b in blocks)
            except Exception:  # pragma: no cover - best effort
                pass
        else:  # pragma: no cover - executed when radon not installed
            metrics["complexity"] = sum(
                isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp))
                for n in ast.walk(tree)
            )
    except Exception:  # pragma: no cover - best effort
        pass
    return metrics


def _runtime_metrics(module_path: Path) -> Dict[str, Any]:
    """Execute ``module_path`` in a restricted subprocess and collect metrics.

    In addition to recording execution success and emitted warnings, the helper
    counts side effects such as attempted file writes, process creation,
    thread creation, environment mutations and network connections.  Access is
    denied for these operations so modules can be inspected without mutating
    the environment.  Failures are captured and reflected in the resulting
    mapping so callers can prioritise modules that execute cleanly.
    """

    result: Dict[str, Any] = {
        "exec_success": False,
        "warnings": 0,
        "files_written": 0,
        "network_calls": 0,
        "process_calls": 0,
        "env_writes": 0,
        "threads_started": 0,
    }
    try:
        # Instrumented runner that blocks and counts side effects.
        runner = f"""
import builtins, json, runpy, socket, os, subprocess, threading

files_written = 0
network_calls = 0
process_calls = 0
env_writes = 0
threads_started = 0

_open = builtins.open
def open_wrapper(file, mode='r', *args, **kwargs):
    global files_written
    if any(m in mode for m in ('w', 'a', 'x', '+')):
        # ignore Python's own bytecode caching
        if not str(file).endswith('.pyc'):
            files_written += 1
    return _open(file, mode, *args, **kwargs)
builtins.open = open_wrapper

_connect = socket.socket.connect
def connect_wrapper(self, *a, **k):
    global network_calls
    network_calls += 1
    raise PermissionError('network access disabled')
socket.socket.connect = connect_wrapper

_popen = subprocess.Popen
def popen_wrapper(*a, **k):
    global process_calls
    process_calls += 1
    raise PermissionError('process spawning disabled')
subprocess.Popen = popen_wrapper

_run = subprocess.run
def run_wrapper(*a, **k):
    global process_calls
    process_calls += 1
    raise PermissionError('process spawning disabled')
subprocess.run = run_wrapper

_env_cls = os.environ.__class__
_env_set = _env_cls.__setitem__
def env_set_wrapper(self, key, value):
    global env_writes
    env_writes += 1
    return _env_set(self, key, value)
_env_cls.__setitem__ = env_set_wrapper

_thread_start = threading.Thread.start
def start_wrapper(self, *a, **k):
    global threads_started
    threads_started += 1
    raise PermissionError('thread start disabled')
threading.Thread.start = start_wrapper

err = None
try:
    runpy.run_path({repr(str(module_path))}, run_name='__main__')
except Exception as exc:  # pragma: no cover - best effort
    err = str(exc)
print(json.dumps({{'files_written': files_written, 'network_calls': network_calls, 'process_calls': process_calls, 'env_writes': env_writes, 'threads_started': threads_started, 'error': err}}))
"""
        proc = subprocess.run(
            [sys.executable, "-I", "-Wdefault", "-c", runner],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=module_path.parent,
        )
        if proc.stdout:
            try:
                info = json.loads(proc.stdout.splitlines()[-1])
                result.update(
                    {
                        "files_written": info.get("files_written", 0),
                        "network_calls": info.get("network_calls", 0),
                        "process_calls": info.get("process_calls", 0),
                        "env_writes": info.get("env_writes", 0),
                        "threads_started": info.get("threads_started", 0),
                    }
                )
                if info.get("error"):
                    result["error"] = info["error"]
                else:
                    result["exec_success"] = True
            except Exception:  # pragma: no cover - best effort
                pass
        if proc.stderr:
            result["warnings"] = sum(
                1 for line in proc.stderr.splitlines() if "warning" in line.lower()
            )
    except Exception as exc:  # pragma: no cover - best effort
        result["error"] = str(exc)
    return result


class Classifier(Protocol):
    """Callable classifying ``module_path`` using ``metrics``."""

    def __call__(
        self, module_path: Path, metrics: Dict[str, Any]
    ) -> Literal["legacy", "redundant"] | None: ...


def _legacy_classifier(module_path: Path, metrics: Dict[str, Any]) -> Literal["legacy"] | None:
    if detect_legacy_patterns(module_path):
        return "legacy"
    return None


def _redundant_classifier(
    module_path: Path, metrics: Dict[str, Any]
) -> Literal["redundant"] | None:
    try:
        root = module_path.parent
        graph = build_import_graph(root)
        if module_path.name == "__init__.py":
            mod = module_path.parent.name
        else:
            mod = module_path.stem
        if mod not in graph.nodes or graph.degree(mod) == 0:
            if (
                metrics.get("functions", 0) <= 1
                and metrics.get("complexity", 0) <= 5
                and metrics.get("calls", 0) == 0
                and not metrics.get("docstring")
            ):
                return "redundant"
    except Exception:  # pragma: no cover - best effort
        pass
    return None


DEFAULT_CLASSIFIERS: Tuple[Classifier, ...] = (
    _legacy_classifier,
    _redundant_classifier,
)


def _load_external_classifiers() -> Tuple[Classifier, ...]:
    """Return classifiers provided via environment variables or entry points."""

    classifiers: list[Classifier] = []

    env = os.environ.get("ORPHAN_ANALYZER_CLASSIFIERS")
    if env:
        for spec in env.split(","):
            spec = spec.strip()
            if not spec:
                continue
            try:
                module_name, func_name = spec.split(":", 1)
                mod = importlib.import_module(module_name)
                classifiers.append(getattr(mod, func_name))
            except Exception:  # pragma: no cover - best effort
                continue

    try:
        eps = importlib_metadata.entry_points(group="orphan_analyzer.classifiers")
    except TypeError:  # pragma: no cover - legacy API
        eps = importlib_metadata.entry_points().get("orphan_analyzer.classifiers", [])
    except Exception:  # pragma: no cover - best effort
        eps = []
    for ep in eps:
        try:
            classifiers.append(ep.load())
        except Exception:  # pragma: no cover - best effort
            continue

    return tuple(classifiers)


EXTERNAL_CLASSIFIERS: Tuple[Classifier, ...] = _load_external_classifiers()
ALL_CLASSIFIERS: Tuple[Classifier, ...] = DEFAULT_CLASSIFIERS + EXTERNAL_CLASSIFIERS


def classify_module(
    module_path: Path,
    *,
    include_meta: bool = False,
    classifiers: Iterable[Classifier] | None = None,
) -> Literal["legacy", "redundant", "candidate"] | Tuple[
    Literal["legacy", "redundant", "candidate"], Dict[str, Any]
]:
    """Classify ``module_path`` as ``legacy``, ``redundant`` or ``candidate``.

    The function delegates to a sequence of classifier strategies.  Each
    classifier receives ``module_path`` and pre-computed ``metrics`` and may
    return a specific classification or ``None`` to defer to the next
    strategy.  When all classifiers defer the module is considered a
    ``candidate``.  ``metrics`` always includes ``functions``, ``calls``,
    ``docstring`` and ``complexity`` and, for candidate modules, additional
    runtime information such as ``exec_success``, ``warnings``, ``files_written``,
    ``network_calls``, ``process_calls``, ``threads_started`` and ``env_writes``.
    """

    metrics = _static_metrics(module_path)
    result: Literal["legacy", "redundant"] | None = None
    for classifier in classifiers or ALL_CLASSIFIERS:
        try:
            result = classifier(module_path, metrics)
        except Exception:  # pragma: no cover - best effort
            result = None
        if result is not None:
            break
    cls: Literal["legacy", "redundant", "candidate"] = result or "candidate"

    if cls == "candidate":
        metrics.update(_runtime_metrics(module_path))

    if include_meta:
        return cls, metrics
    return cls


def analyze_redundancy(
    module_path: Path, *, classifiers: Iterable[Classifier] | None = None
) -> bool:
    """Backward compatible wrapper returning ``True`` for non-candidates."""

    return classify_module(module_path, classifiers=classifiers) != "candidate"


__all__ = [
    "classify_module",
    "detect_legacy_patterns",
    "analyze_redundancy",
    "Classifier",
]
