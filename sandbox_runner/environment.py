from __future__ import annotations

import ast
import asyncio
import json
import os
import re

try:
    import resource
except Exception:  # pragma: no cover - not available on some platforms
    resource = None  # type: ignore
import shutil
import subprocess
import tempfile
import textwrap
import logging
import multiprocessing
import time
import inspect
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Callable, get_origin, get_args

try:
    from menace.diagnostic_manager import DiagnosticManager, ResolutionRecord
except Exception:  # pragma: no cover - optional dependency
    DiagnosticManager = None  # type: ignore
    ResolutionRecord = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Faker = None  # type: ignore

try:
    from hypothesis import strategies as _hyp_strats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _hyp_strats = None  # type: ignore

_FAKER = Faker() if Faker is not None else None

logger = logging.getLogger(__name__)

from .config import SANDBOX_REPO_URL, SANDBOX_REPO_PATH

ROOT = Path(__file__).resolve().parents[1]

if DiagnosticManager is not None:
    try:
        _DIAGNOSTIC = DiagnosticManager()
    except Exception:  # pragma: no cover - diagnostics optional
        _DIAGNOSTIC = None
else:
    _DIAGNOSTIC = None


def _log_diagnostic(issue: str, success: bool) -> None:
    """Record a resolution attempt with ``DiagnosticManager`` if available."""
    if _DIAGNOSTIC is None:
        return
    try:
        _DIAGNOSTIC.log.add(ResolutionRecord(issue, "retry", success))
        if not success:
            _DIAGNOSTIC.error_bot.handle_error(issue)
    except Exception as exc:
        logger.exception("diagnostic logging failed: %s", exc)


# ----------------------------------------------------------------------
class _DangerVisitor(ast.NodeVisitor):
    """Collect suspicious patterns without executing code."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.files_written: List[str] = []
        self.flags: List[str] = []
        self.imports: List[str] = []
        self.attributes: List[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._name(node.func)
        if name:
            self.calls.append(name)
        lowered = name.lower() if name else ""
        if lowered in {"eval", "exec"}:
            self.flags.append(f"dangerous call {name}")
        if lowered.startswith("subprocess") or lowered.startswith("os.system"):
            self.flags.append(f"process call {name}")
        if lowered == "open":
            mode = "r"
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(m in mode for m in ("w", "a", "+")):
                path = self._literal_arg(node.args[0]) if node.args else "?"
                self.files_written.append(str(path))
                self.flags.append("file write")
        if lowered.startswith(("requests", "socket")):
            self.flags.append(f"network call {name}")
        if lowered in {
            "os.setuid",
            "os.setgid",
            "os.seteuid",
            "os.setegid",
        }:
            self.flags.append(f"privilege escalation {name}")
        if lowered.startswith("subprocess") or lowered.startswith("os.system"):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and "sudo" in arg.value
                ):
                    self.flags.append("privilege escalation sudo")
                    break
        if "reward" in lowered:
            self.flags.append(f"reward manipulation {name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        name = self._name(node)
        self.attributes.append(name)
        if name in {
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "requests.get",
            "requests.post",
        }:
            self.flags.append(f"risky attribute {name}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            mod = alias.name.split(".")[0]
            self.imports.append(alias.name)
            if mod in {"socket", "requests", "subprocess", "ctypes"}:
                self.flags.append(f"import dangerous module {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module or ""
        self.imports.append(mod)
        if mod.split(".")[0] in {"socket", "requests", "subprocess", "ctypes"}:
            self.flags.append(f"import dangerous module {mod}")
        self.generic_visit(node)

    def _name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value = self._name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def _literal_arg(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        return ""


# ----------------------------------------------------------------------
def static_behavior_analysis(code_str: str) -> Dict[str, Any]:
    """Return dictionary describing risky constructs in ``code_str``."""
    logger.debug("starting static analysis of code (%d chars)", len(code_str))
    result: Dict[str, Any] = {
        "calls": [],
        "files_written": [],
        "flags": [],
        "imports": [],
        "attributes": [],
    }
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        result["flags"].append(f"syntax error: {exc}")
        return result
    visitor = _DangerVisitor()
    visitor.visit(tree)
    result["calls"] = visitor.calls
    result["files_written"] = visitor.files_written
    result["flags"] = visitor.flags
    result["imports"] = visitor.imports
    result["attributes"] = visitor.attributes

    patterns = [r"\bsubprocess\b", r"\bos\.system\b", r"eval\(", r"exec\("]
    if any(re.search(p, code_str) for p in patterns):
        result.setdefault("regex_flags", []).append("raw_dangerous_pattern")

    # optional Bandit integration
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code_str)
            tmp_path = tmp.name
        proc = subprocess.run(
            ["bandit", "-f", "json", "-q", tmp_path],
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            data = json.loads(proc.stdout)
            issues = [
                {
                    "line": i.get("line_number"),
                    "severity": i.get("issue_severity"),
                    "text": i.get("issue_text"),
                }
                for i in data.get("results", [])
            ]
            if issues:
                result["bandit"] = issues
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("bandit failed: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    logger.debug(
        "static analysis result: %s",
        {k: v for k, v in result.items() if v},
    )
    return result


# ----------------------------------------------------------------------
# Docker container pooling support
try:  # pragma: no cover - optional dependency
    import docker  # type: ignore
    _DOCKER_CLIENT = docker.from_env()
except Exception as exc:  # pragma: no cover - docker may be unavailable
    logger.warning("docker import failed: %s", exc)
    docker = None  # type: ignore
    _DOCKER_CLIENT = None

_CONTAINER_POOL_SIZE = int(os.getenv("SANDBOX_CONTAINER_POOL_SIZE", "2"))
_CONTAINER_POOLS: Dict[str, List[Any]] = {}
_CONTAINER_DIRS: Dict[str, str] = {}

def _create_pool_container(image: str) -> tuple[Any, str]:
    """Create a long-lived container running ``sleep infinity``."""
    assert _DOCKER_CLIENT is not None
    td = tempfile.mkdtemp(prefix="pool_")
    container = _DOCKER_CLIENT.containers.run(
        image,
        ["sleep", "infinity"],
        detach=True,
        network_disabled=True,
        volumes={td: {"bind": "/code", "mode": "rw"}},
    )
    _CONTAINER_DIRS[container.id] = td
    return container, td


def _get_pooled_container(image: str) -> tuple[Any, str]:
    """Return a container for ``image`` from the pool, creating if needed."""
    pool = _CONTAINER_POOLS.setdefault(image, [])
    if pool:
        c = pool.pop()
        return c, _CONTAINER_DIRS[c.id]
    container, td = _create_pool_container(image)
    return container, td


def _release_container(image: str, container: Any) -> None:
    """Return ``container`` to the pool for ``image``."""
    _CONTAINER_POOLS.setdefault(image, []).append(container)


def _cleanup_pools() -> None:
    """Stop and remove pooled containers."""
    if _DOCKER_CLIENT is None:
        return
    for pool in list(_CONTAINER_POOLS.values()):
        for c in list(pool):
            try:
                c.stop(timeout=0)
            except Exception:
                pass
            try:
                c.remove()
            except Exception:
                pass
            td = _CONTAINER_DIRS.pop(c.id, None)
            if td:
                shutil.rmtree(td, ignore_errors=True)
    _CONTAINER_POOLS.clear()

import atexit

if _DOCKER_CLIENT is not None:
    default_img = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
    for _ in range(_CONTAINER_POOL_SIZE):
        try:
            c, _ = _create_pool_container(default_img)
            _release_container(default_img, c)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to pre-pull container: %s", exc)
            break
    atexit.register(_cleanup_pools)



# ----------------------------------------------------------------------
def _execute_in_container(
    code_str: str,
    env: Dict[str, Any],
    *,
    mounts: Dict[str, str] | None = None,
    network_disabled: bool = True,
    workdir: str | None = None,
) -> Dict[str, float]:
    """Return runtime metrics after executing ``code_str`` in a container.

    If Docker is unavailable or repeatedly fails, the snippet is executed
    locally with the same environment variables and resource limits.
    """

    def _execute_locally() -> Dict[str, float]:
        """Fallback local execution with basic metrics."""
        with tempfile.TemporaryDirectory(prefix="sim_local_") as td:
            path = Path(td) / "snippet.py"
            path.write_text(code_str, encoding="utf-8")

            env_vars = os.environ.copy()
            env_vars.update({k: str(v) for k, v in env.items()})

            rlimit_ok = _rlimits_supported()

            def _limits() -> None:
                if not rlimit_ok or resource is None:
                    return
                cpu = env.get("CPU_LIMIT")
                mem = env.get("MEMORY_LIMIT")
                try:
                    if cpu:
                        sec = int(float(cpu)) * 10
                        resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
                except Exception as exc:
                    logger.warning("failed to set CPU limit: %s", exc)
                try:
                    if mem:
                        size = _parse_size(mem)
                        if size:
                            resource.setrlimit(resource.RLIMIT_AS, (size, size))
                except Exception as exc:
                    logger.warning("failed to set memory limit: %s", exc)

            start = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
            try:
                proc = subprocess.Popen(
                    ["python", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env_vars,
                    cwd=workdir or td,
                    preexec_fn=_limits if rlimit_ok else None,
                )
                p = psutil.Process(proc.pid) if psutil else None
                proc.communicate(timeout=int(env.get("TIMEOUT", "30")))
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = -1
            end = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None

            if p:
                try:
                    cpu_total = p.cpu_times().user + p.cpu_times().system
                    mem_usage = p.memory_info().rss
                    io = p.io_counters()
                    disk_io = io.read_bytes + io.write_bytes
                except Exception:
                    cpu_total = 0.0
                    mem_usage = 0.0
                    disk_io = 0.0
            else:
                if start is not None and end is not None:
                    cpu_total = (end.ru_utime + end.ru_stime) - (
                        start.ru_utime + start.ru_stime
                    )
                    mem_usage = float(end.ru_maxrss - start.ru_maxrss) * 1024
                    disk_io = (
                        float(
                            (end.ru_inblock - start.ru_inblock)
                            + (end.ru_oublock - start.ru_oublock)
                        )
                        * 512
                    )
                else:
                    cpu_total = 0.0
                    mem_usage = 0.0
                    disk_io = 0.0

            return {
                "exit_code": float(exit_code),
                "cpu": float(cpu_total),
                "memory": float(mem_usage),
                "disk_io": float(disk_io),
            }

    try:
        import docker  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("docker import failed: %s", exc)
        return _execute_locally()

    client = _DOCKER_CLIENT or docker.from_env()
    if _DOCKER_CLIENT is None:
        _globals = globals()
        _globals["_DOCKER_CLIENT"] = client

    def _run_ephemeral() -> Dict[str, float]:
        """Run snippet using a one-off container (legacy behaviour)."""
        nonlocal client
        attempt = 0
        delay = 0.5
        while True:
            try:
                with tempfile.TemporaryDirectory(prefix="sim_cont_") as td:
                    path = Path(td) / "snippet.py"
                    path.write_text(code_str, encoding="utf-8")

                    image = env.get("CONTAINER_IMAGE")
                    if not image:
                        image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
                        os_type = env.get("OS_TYPE")
                        if os_type:
                            image = os.getenv(
                                f"SANDBOX_CONTAINER_IMAGE_{os_type.upper()}", image
                            )

                    volumes = {td: {"bind": "/code", "mode": "rw"}}
                    if mounts:
                        for host, dest in mounts.items():
                            volumes[host] = {"bind": dest, "mode": "rw"}

                    kwargs: Dict[str, Any] = {
                        "volumes": volumes,
                        "environment": {k: str(v) for k, v in env.items()},
                        "detach": True,
                        "network_disabled": network_disabled,
                    }

                    mem = env.get("MEMORY_LIMIT")
                    if mem:
                        kwargs["mem_limit"] = str(mem)

                    cpu = env.get("CPU_LIMIT")
                    if cpu:
                        try:
                            kwargs["cpu_quota"] = int(float(cpu) * 100000)
                        except Exception as exc:
                            logger.warning("invalid CPU limit %s: %s", cpu, exc)

                    disk = env.get("DISK_LIMIT")
                    if disk:
                        kwargs["storage_opt"] = {"size": str(disk)}

                    gpu = env.get("GPU_LIMIT")
                    if gpu:
                        try:
                            from docker.types import DeviceRequest

                            kwargs["device_requests"] = [
                                DeviceRequest(count=int(float(gpu)), capabilities=[["gpu"]])
                            ]
                        except Exception as exc:
                            logger.warning("GPU limit ignored: %s", exc)

                    if workdir:
                        kwargs["working_dir"] = workdir

                    container = client.containers.run(
                        image,
                        ["python", "/code/snippet.py"],
                        **kwargs,
                    )

                    result = container.wait()
                    stats = container.stats(stream=False)
                    container.remove()

                    blk = (
                        stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
                    )
                    disk_io = float(sum(x.get("value", 0) for x in blk))

                    cpu_total = float(
                        stats.get("cpu_stats", {})
                        .get("cpu_usage", {})
                        .get("total_usage", 0)
                    )
                    mem_usage = float(stats.get("memory_stats", {}).get("max_usage", 0))

                    if attempt:
                        _log_diagnostic("container_failure", True)

                    return {
                        "exit_code": float(result.get("StatusCode", 0)),
                        "cpu": cpu_total,
                        "memory": mem_usage,
                        "disk_io": disk_io,
                    }
            except Exception as exc:  # pragma: no cover - runtime failures
                logger.exception("container execution failed: %s", exc)
                _log_diagnostic(str(exc), False)
                if attempt >= 2:
                    logger.warning(
                        "docker repeatedly failed; falling back to local execution"
                    )
                    return _execute_locally()
                attempt += 1
                time.sleep(delay)
                delay *= 2

    # use legacy container mode when advanced features are requested
    if (
        mounts
        or not network_disabled
        or workdir
        or any(k in env for k in ("CPU_LIMIT", "MEMORY_LIMIT", "DISK_LIMIT", "GPU_LIMIT"))
    ):
        return _run_ephemeral()

    # pooled execution path
    attempt = 0
    delay = 0.5
    while True:
        try:
            image = env.get("CONTAINER_IMAGE")
            if not image:
                image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
                os_type = env.get("OS_TYPE")
                if os_type:
                    image = os.getenv(
                        f"SANDBOX_CONTAINER_IMAGE_{os_type.upper()}", image
                    )

            container, td = _get_pooled_container(image)
            path = Path(td) / "snippet.py"
            path.write_text(code_str, encoding="utf-8")

            result = container.exec_run(
                ["python", "/code/snippet.py"],
                environment={k: str(v) for k, v in env.items()},
                workdir=workdir,
            )

            stats = container.stats(stream=False)
            _release_container(image, container)

            blk = (
                stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
            )
            disk_io = float(sum(x.get("value", 0) for x in blk))

            cpu_total = float(
                stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            )
            mem_usage = float(stats.get("memory_stats", {}).get("max_usage", 0))

            if attempt:
                _log_diagnostic("container_failure", True)

            exit_code = getattr(result, "exit_code", 0)
            if isinstance(result, tuple):
                exit_code = result[0]

            return {
                "exit_code": float(exit_code),
                "cpu": cpu_total,
                "memory": mem_usage,
                "disk_io": disk_io,
            }
        except Exception as exc:  # pragma: no cover - runtime failures
            logger.exception("container execution failed: %s", exc)
            _log_diagnostic(str(exc), False)
            if attempt >= 2:
                logger.warning(
                    "docker repeatedly failed; falling back to local execution"
                )
                return _execute_locally()
            attempt += 1
            time.sleep(delay)
            delay *= 2


# ----------------------------------------------------------------------
def simulate_execution_environment(
    code_str: str,
    input_stub: Dict[str, Any] | None = None,
    *,
    container: bool | None = None,
) -> Dict[str, Any]:
    """Mock runtime environment and optionally execute code in a container."""

    logger.debug(
        "simulate_execution_environment called with input stub=%s container=%s",
        bool(input_stub),
        container,
    )

    analysis = static_behavior_analysis(code_str)
    env_result = {
        "functions_called": analysis.get("calls", []),
        "files_accessed": analysis.get("files_written", []),
        "risk_flags_triggered": analysis.get("flags", []),
    }

    if input_stub:
        env_result["input_stub"] = input_stub

    if analysis.get("regex_flags"):
        env_result["risk_flags_triggered"].extend(analysis["regex_flags"])

    if container is None:
        container = str(os.getenv("SANDBOX_DOCKER", "0")).lower() not in {
            "0",
            "false",
            "no",
            "",
        }

    runtime_metrics: Dict[str, float] = {}
    if container:
        try:
            runtime_metrics = _execute_in_container(code_str, input_stub or {})
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("container execution failed: %s", exc)

    if runtime_metrics:
        env_result["runtime_metrics"] = runtime_metrics

    logger.debug("environment simulation result: %s", env_result)
    return env_result


# ----------------------------------------------------------------------
def generate_sandbox_report(analysis_result: Dict[str, Any], output_path: str) -> None:
    """Write ``analysis_result`` to ``output_path`` as JSON with timestamp."""
    logger.debug("writing sandbox report to %s", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = dict(analysis_result)
    data["timestamp"] = datetime.utcnow().isoformat()
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    logger.debug("sandbox report written: %s", output_path)


# ----------------------------------------------------------------------
def _parse_size(value: str | int | float) -> int:
    """Return ``value`` interpreted as bytes."""
    try:
        s = str(value).strip().lower()
        if s.endswith("mi"):
            return int(float(s[:-2])) * 1024 * 1024
        if s.endswith("gi"):
            return int(float(s[:-2])) * 1024 * 1024 * 1024
        return int(float(s))
    except Exception:
        return 0


def _rlimits_supported() -> bool:
    """Return ``True`` if ``resource`` limits appear usable."""
    if resource is None:
        return False
    try:
        resource.getrlimit(resource.RLIMIT_CPU)
        resource.getrlimit(resource.RLIMIT_AS)
        return True
    except Exception:
        return False


def _parse_failure_modes(value: Any) -> set[str]:
    """Return a normalized set of failure modes from ``value``."""
    if not value:
        return set()
    modes: set[str] = set()
    if isinstance(value, str):
        for part in value.split(","):
            part = part.strip()
            if part:
                modes.add(part)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            modes.update(_parse_failure_modes(item))
    else:
        modes.add(str(value))
    return modes


def _inject_failure_modes(snippet: str, modes: set[str]) -> str:
    """Return ``snippet`` with failure mode hooks prepended."""

    parts: list[str] = []
    if "disk" in modes or "disk_corruption" in modes:
        corruption = ""
        if "disk_corruption" in modes:
            corruption = (
                "            if isinstance(data, bytes):\n"
                "                data = b'CORRUPTED' + data\n"
                "            else:\n"
                "                data = 'CORRUPTED' + data\n"
            )
        delay = "            time.sleep(0.05)\n" if "disk" in modes else ""
        parts.append(
            "import builtins, time\n"
            "_orig_open = builtins.open\n"
            'def _open(f, mode="r", *a, **k):\n'
            "    file = _orig_open(f, mode, *a, **k)\n"
            '    if "w" in mode:\n'
            "        orig = file.write\n"
            "        def _write(data, *aa, **kk):\n"
            f"{delay}"
            f"{corruption}"
            "            return orig(data, *aa, **kk)\n"
            "        file.write = _write\n"
            "    return file\n"
            "builtins.open = _open\n"
        )

    if "network" in modes or "network_partition" in modes:
        parts.append(
            "import socket\n"
            "class _BlockSocket(socket.socket):\n"
            "    def connect(self, *a, **k):\n"
            "        raise OSError('network blocked')\n"
            "socket.socket = _BlockSocket\n"
        )

    if "cpu_spike" in modes:
        parts.append(
            "import threading, time\n"
            "def _burn():\n"
            "    end = time.time() + 0.2\n"
            "    while time.time() < end:\n"
            "        pass\n"
            "threading.Thread(target=_burn, daemon=True).start()\n"
        )

    if "memory" in modes:
        parts.append(" _mem_fail = bytearray(10_000_000)\n")

    if "timeout" in modes:
        parts.append(
            "import threading, os, time\n"
            "def _abort():\n"
            "    time.sleep(0.05)\n"
            "    os._exit(1)\n"
            "threading.Thread(target=_abort, daemon=True).start()\n"
        )

    if not parts:
        return snippet

    return "\n".join(parts) + "\n" + snippet


async def _section_worker(
    snippet: str, env_input: Dict[str, Any], threshold: float
) -> tuple[Dict[str, Any], list[tuple[float, float, Dict[str, float]]]]:
    """Execute ``snippet`` with resource limits and return results."""

    def _run_snippet() -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="run_") as td:
            path = Path(td) / "snippet.py"
            modes = _parse_failure_modes(env_input.get("FAILURE_MODES"))
            snip = _inject_failure_modes(snippet, modes)
            path.write_text(snip, encoding="utf-8")
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_input.items()})
            if "memory" in modes and "MEMORY_LIMIT" not in env_input:
                env["MEMORY_LIMIT"] = "32Mi"
                env_input["MEMORY_LIMIT"] = "32Mi"
            if "cpu_spike" in modes and "CPU_LIMIT" not in env_input:
                env["CPU_LIMIT"] = "0.1"
                env_input["CPU_LIMIT"] = "0.1"

            netem_args = []
            latency = env_input.get("NETWORK_LATENCY_MS")
            if latency:
                netem_args += ["delay", f"{latency}ms"]
                jitter = env_input.get("NETWORK_JITTER_MS")
                if jitter:
                    netem_args.append(f"{jitter}ms")
            loss = env_input.get("PACKET_LOSS")
            if loss:
                netem_args += ["loss", f"{loss}%"]
            dup = env_input.get("PACKET_DUPLICATION")
            if dup:
                netem_args += ["duplicate", f"{dup}%"]

            if {"network", "network_partition"} & modes:
                if "loss" not in netem_args:
                    netem_args += ["loss", "100%"]
            _use_netem = False
            if netem_args and shutil.which("tc"):
                try:
                    subprocess.run(
                        [
                            "tc",
                            "qdisc",
                            "replace",
                            "dev",
                            "lo",
                            "root",
                            "netem",
                            *netem_args,
                        ],
                        check=True,
                    )
                    _use_netem = True
                except Exception:
                    _use_netem = False
            elif netem_args:
                try:
                    code = "import subprocess\n" + (
                        f"subprocess.run(['tc','qdisc','replace','dev','eth0','root','netem', {', '.join(repr(a) for a in netem_args)}], check=False)\n"
                    )
                    code += (
                        "try:\n"
                        + textwrap.indent(snip, "    ")
                        + "\nfinally:\n    subprocess.run(['tc','qdisc','del','dev','eth0','root','netem'], check=False)\n"
                    )
                    metrics = _execute_in_container(
                        code,
                        env_input,
                        network_disabled={
                            "network" in modes or "network_partition" in modes
                        },
                    )
                    return {
                        "stdout": "",
                        "stderr": "",
                        "exit_code": int(metrics.get("exit_code", 0)),
                    }
                except Exception as exc:
                    logger.warning("tc missing and docker netem failed: %s", exc)

            rlimit_ok = _rlimits_supported()

            def _limits() -> None:
                if not rlimit_ok or resource is None:
                    return
                cpu = env_input.get("CPU_LIMIT")
                mem = env_input.get("MEMORY_LIMIT")
                try:
                    if cpu:
                        sec = int(float(cpu)) * 10
                        resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
                except Exception as exc:
                    logger.warning("failed to set CPU limit: %s", exc)
                try:
                    if mem:
                        size = _parse_size(mem)
                        if size:
                            resource.setrlimit(resource.RLIMIT_AS, (size, size))
                except Exception as exc:
                    logger.warning("failed to set memory limit: %s", exc)

            def _run_psutil() -> Dict[str, Any]:
                proc = subprocess.Popen(
                    ["python", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                p = psutil.Process(proc.pid) if psutil else None
                cpu_lim = (
                    int(float(env_input.get("CPU_LIMIT", 0))) * 10
                    if env_input.get("CPU_LIMIT")
                    else None
                )
                mem_lim = (
                    _parse_size(env_input.get("MEMORY_LIMIT", 0))
                    if env_input.get("MEMORY_LIMIT")
                    else None
                )
                timeout = int(env_input.get("TIMEOUT", "30"))
                start = time.monotonic()
                reason = ""
                while proc.poll() is None:
                    if time.monotonic() - start > timeout:
                        reason = "timeout"
                        proc.kill()
                        break
                    try:
                        if p is not None:
                            times = p.cpu_times()
                            cpu = times.user + times.system
                            mem = p.memory_info().rss
                            if cpu_lim and cpu > cpu_lim:
                                reason = "cpu"
                                proc.kill()
                                break
                            if mem_lim and mem > mem_lim:
                                reason = "memory"
                                proc.kill()
                                break
                    except Exception:
                        logger.warning("psutil metrics collection failed", exc_info=True)
                    time.sleep(0.1)
                out, err = proc.communicate()
                if reason == "timeout":
                    err = err or "timeout"
                    code = -1
                elif reason:
                    code = -1
                else:
                    code = proc.returncode
                return {"stdout": out, "stderr": err, "exit_code": code}

            try:
                if rlimit_ok:
                    proc = subprocess.run(
                        ["python", str(path)],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=int(env_input.get("TIMEOUT", "30")),
                        preexec_fn=_limits,
                    )
                    return {
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "exit_code": proc.returncode,
                    }
                if psutil is not None:
                    return _run_psutil()
                metrics = _execute_in_container(
                    snip,
                    env_input,
                    network_disabled={
                        "network" in modes or "network_partition" in modes
                    },
                )
                return {
                    "stdout": "",
                    "stderr": "",
                    "exit_code": int(metrics.get("exit_code", 0)),
                }
            except subprocess.TimeoutExpired as exc:
                return {
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "timeout",
                    "exit_code": -1,
                }
            except Exception as exc:  # pragma: no cover - unexpected failure
                return {"stdout": "", "stderr": str(exc), "exit_code": -1}
            finally:
                if _use_netem:
                    subprocess.run(
                        ["tc", "qdisc", "del", "dev", "lo", "root", "netem"],
                        check=False,
                    )

    async def _run() -> Dict[str, Any]:
        return await asyncio.to_thread(_run_snippet)

    updates: list[tuple[float, float, Dict[str, float]]] = []
    prev = 0.0
    attempt = 0
    delay = 0.5
    retried = False
    while True:
        try:
            result = await _run()
        except Exception as exc:  # pragma: no cover - runtime failures
            logger.exception("section execution failed: %s", exc)
            _log_diagnostic(str(exc), False)
            if attempt >= 2:
                raise
            attempt += 1
            retried = True
            await asyncio.sleep(delay)
            delay *= 2
            continue
        attempt = 0
        delay = 0.5
        if result.get("exit_code", 0) < 0:
            _log_diagnostic(str(result.get("stderr", "error")), False)
            if attempt >= 2:
                return result, updates
            attempt += 1
            retried = True
            await asyncio.sleep(delay)
            delay *= 2
            continue

        actual = 1.0 if result.get("exit_code") == 0 else 0.0
        metrics = {
            "exit_code": float(result.get("exit_code", 0)),
            "stdout_len": float(len(result.get("stdout", ""))),
            "stderr_len": float(len(result.get("stderr", ""))),
        }
        if SANDBOX_EXTRA_METRICS:
            metrics.update(SANDBOX_EXTRA_METRICS)
        updates.append((prev, actual, metrics))
        if abs(actual - prev) <= threshold:
            if retried:
                _log_diagnostic("section_worker_retry", True)
            return result, updates
        prev = actual


# ----------------------------------------------------------------------
def _load_metrics_file(path: str | Path) -> Dict[str, float]:
    """Return metrics specified in a YAML or JSON file as a dictionary."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        if p.suffix.lower() in {".json", ".jsn"}:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            import yaml  # type: ignore

            with open(p, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
    except Exception:
        logger.exception("failed to load metrics file: %s", p)
        return {}

    metrics = data.get("extra_metrics", data) if isinstance(data, dict) else data
    if isinstance(metrics, list):
        return {str(m): 0.0 for m in metrics}
    if isinstance(metrics, dict):
        out = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                out[str(k)] = 0.0
        return out
    return {}


SANDBOX_EXTRA_METRICS: Dict[str, float] = _load_metrics_file(
    os.getenv("SANDBOX_METRICS_FILE", str(ROOT / "sandbox_metrics.yaml"))
)

_preset_env = os.getenv("SANDBOX_ENV_PRESETS", "[]")
try:
    SANDBOX_ENV_PRESETS: List[Dict[str, Any]] = json.loads(_preset_env)
    if isinstance(SANDBOX_ENV_PRESETS, dict):
        SANDBOX_ENV_PRESETS = [SANDBOX_ENV_PRESETS]
    SANDBOX_ENV_PRESETS = [dict(p) for p in SANDBOX_ENV_PRESETS]
except Exception:
    SANDBOX_ENV_PRESETS = [{}]
if not SANDBOX_ENV_PRESETS:
    SANDBOX_ENV_PRESETS = [{}]

_stub_env = os.getenv("SANDBOX_INPUT_STUBS", "")
try:
    SANDBOX_INPUT_STUBS: List[Dict[str, Any]] = (
        json.loads(_stub_env) if _stub_env else []
    )
    if isinstance(SANDBOX_INPUT_STUBS, dict):
        SANDBOX_INPUT_STUBS = [SANDBOX_INPUT_STUBS]
    SANDBOX_INPUT_STUBS = [dict(s) for s in SANDBOX_INPUT_STUBS]
except Exception:
    SANDBOX_INPUT_STUBS = []

from .stub_providers import discover_stub_providers, StubProvider


def _load_templates(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        logger.exception("failed to load input templates: %s", p)
        return []
    if isinstance(data, dict):
        data = data.get("templates", [])
    if isinstance(data, list):
        return [dict(d) for d in data if isinstance(d, dict)]
    return []


def _load_history(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        if p.suffix.lower() in {".json", ".jsn"}:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records.extend(dict(r) for r in data if isinstance(r, dict))
        else:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        if isinstance(obj, dict):
                            records.append(dict(obj))
                    except Exception:
                        continue
    except Exception:
        logger.exception("failed to load input history: %s", p)
    return records


def _random_strategy(count: int, conf: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    conf = conf or {}
    modes = conf.get("modes", ["default", "alt", "stress"])
    level_range = conf.get("level_range", [1, 5])
    flags = conf.get("flags", ["A", "B", "C"])
    flag_prob = float(conf.get("flag_prob", 0.3))
    stubs: List[Dict[str, Any]] = []
    for _ in range(count):
        stub = {
            "mode": random.choice(modes),
            "level": random.randint(int(level_range[0]), int(level_range[1])),
        }
        if random.random() < flag_prob and flags:
            stub["flag"] = random.choice(flags)
        stubs.append(stub)
    return stubs


def _smart_value(name: str, hint: Any) -> Any:
    """Return a realistic value for ``name`` with type ``hint``."""
    val = None
    if _FAKER is not None:
        if hint is str:
            lowered = name.lower()
            if "email" in lowered:
                val = _FAKER.email()
            elif "name" in lowered:
                val = _FAKER.name()
            elif "url" in lowered:
                val = _FAKER.url()
            else:
                val = _FAKER.word()
        elif hint is int:
            val = _FAKER.random_int(min=0, max=1000)
        elif hint is float:
            val = float(_FAKER.pyfloat(left_digits=2, right_digits=2, positive=True))
        elif hint is bool:
            val = _FAKER.pybool()
        elif hint is datetime:
            val = _FAKER.date_time()
    if val is None and _hyp_strats is not None and hint is not inspect._empty:
        try:
            val = _hyp_strats.from_type(hint).example()
        except Exception:
            val = None
    return val


def _stub_from_signature(func: Callable[..., Any], *, smart: bool = False) -> Dict[str, Any]:
    """Return an input stub derived from ``func`` signature."""
    stub: Dict[str, Any] = {}
    try:
        sig = inspect.signature(func)
    except Exception:
        return stub
    for name, param in sig.parameters.items():
        if param.default is not inspect._empty:
            stub[name] = param.default
            continue
        hint = param.annotation
        val: Any = None
        if smart:
            val = _smart_value(name, hint)
        if hint is not inspect._empty and val is None:
            origin = get_origin(hint)
            if origin is list or hint is list:
                val = []
            elif origin is dict or hint is dict:
                val = {}
            elif origin is tuple or hint is tuple:
                val = []
            elif origin is set or hint is set:
                val = set()
            elif hint in (int, float):
                val = 0
            elif hint is bool:
                val = False
            elif hint is str:
                val = ""
        stub[name] = val
    return stub


def generate_input_stubs(
    count: int | None = None,
    *,
    target: Callable[..., Any] | None = None,
    strategy: str | None = None,
    providers: List[StubProvider] | None = None,
) -> List[Dict[str, Any]]:
    """Return example input dictionaries.

    ``SANDBOX_INPUT_STUBS`` overrides all other behaviour. When unset the
    generator consults ``providers`` discovered via ``SANDBOX_STUB_PLUGINS``.
    The built-in strategies ``templates``, ``history``, ``random`` and ``smart``
    can be selected via ``strategy`` or the ``SANDBOX_STUB_STRATEGY`` environment
    variable. The ``smart`` strategy attempts to generate realistic values using
    ``faker`` or ``hypothesis`` when available.
    """

    if SANDBOX_INPUT_STUBS:
        stubs = [dict(s) for s in SANDBOX_INPUT_STUBS]
        providers = providers or discover_stub_providers()
        for prov in providers:
            try:
                new = prov(stubs, {"strategy": "env", "target": target})
                if new:
                    stubs = [dict(s) for s in new if isinstance(s, dict)]
            except Exception:
                logger.exception(
                    "stub provider %s failed", getattr(prov, "__name__", "?")
                )
        return stubs

    num = 2 if count is None else max(0, count)

    providers = providers or discover_stub_providers()
    stubs: List[Dict[str, Any]] | None = None

    strat = strategy or os.getenv("SANDBOX_STUB_STRATEGY", "templates")
    if strat == "history":
        history = _load_history(os.getenv("SANDBOX_INPUT_HISTORY"))
        if history:
            stubs = [dict(random.choice(history)) for _ in range(num)]
        else:
            stubs = None

    if strat == "smart" and target is not None:
        base = _stub_from_signature(target, smart=True)
        stubs = [dict(base) for _ in range(num)]

    if strat in {"templates", "history"}:
        templates = _load_templates(
            os.getenv(
                "SANDBOX_INPUT_TEMPLATES_FILE",
                str(ROOT / "sandbox_data" / "input_stub_templates.json"),
            )
        )
        if templates:
            stubs = [dict(random.choice(templates)) for _ in range(num)]

    if stubs is None:
        if target is not None:
            base = _stub_from_signature(target, smart=strat == "smart")
            stubs = [dict(base) for _ in range(num)]
        else:
            conf_env = os.getenv("SANDBOX_STUB_RANDOM_CONFIG", "")
            try:
                conf = json.loads(conf_env) if conf_env else {}
            except Exception:
                conf = {}
            stubs = _random_strategy(num, conf) or [{}]

    for prov in providers:
        try:
            new = prov(stubs, {"strategy": strat, "target": target})
            if new:
                stubs = [dict(s) for s in new if isinstance(s, dict)]
        except Exception:
            logger.exception("stub provider %s failed", getattr(prov, "__name__", "?"))

    return stubs


# ----------------------------------------------------------------------
def run_repo_section_simulations(
    repo_path: str,
    input_stubs: List[Dict[str, Any]] | None = None,
    env_presets: List[Dict[str, Any]] | None = None,
    *,
    return_details: bool = False,
) -> "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]:
    """Analyse sections and simulate execution environment per section."""
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager

    if input_stubs is None:
        input_stubs = generate_input_stubs()
    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import generate_presets

                env_presets = generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, Dict[str, list[Dict[str, Any]]]]]
    ):
        from sandbox_runner import scan_repo_sections

        logger.info("scanning repository sections in %s", repo_path)
        sections = scan_repo_sections(repo_path)
        tracker = ROITracker()
        scenario_names = []
        for i, preset in enumerate(env_presets):
            name = preset.get("SCENARIO_NAME", f"scenario_{i}")
            scenario_names.append(name)
        details: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
        synergy_data: Dict[str, Dict[str, list]] = {
            name: {"roi": [], "metrics": []} for name in scenario_names
        }
        scenario_synergy: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in scenario_names
        }

        tasks: list[
            tuple[int, asyncio.Task, str, str, str, Dict[str, Any], Dict[str, Any]]
        ] = []
        index = 0
        max_cpu = (
            max(float(p.get("CPU_LIMIT", 1)) for p in env_presets)
            if env_presets
            else 1.0
        )
        max_mem = (
            max(_parse_size(p.get("MEMORY_LIMIT", 0)) for p in env_presets)
            if env_presets
            else 0
        )
        max_gpu = (
            max(int(p.get("GPU_LIMIT", 0)) for p in env_presets) if env_presets else 0
        )
        total_cpu = multiprocessing.cpu_count() or 1
        if psutil:
            total_mem = psutil.virtual_memory().total
        else:
            total_mem = 0
        total_gpu = int(os.getenv("NUM_GPUS", "0"))
        workers_cpu = max(1, int(total_cpu / max(1.0, max_cpu)))
        workers_mem = (
            max(1, int(total_mem / max_mem)) if total_mem and max_mem else workers_cpu
        )
        workers_gpu = max(1, int(total_gpu / max_gpu)) if max_gpu else workers_cpu
        max_workers = min(workers_cpu, workers_mem, workers_gpu, len(sections)) or 1
        sem = asyncio.Semaphore(max_workers)

        all_diminished = True

        async def _gather_tasks() -> None:
            nonlocal index, all_diminished
            for module, sec_map in sections.items():
                tmp_dir = tempfile.mkdtemp(prefix="section_")
                shutil.copytree(repo_path, tmp_dir, dirs_exist_ok=True)
                debugger = SelfDebuggerSandbox(
                    object(), SelfCodingEngine(CodeDB(), MenaceMemoryManager())
                )
                try:
                    for sec_name, lines in sec_map.items():
                        code_str = "\n".join(lines)
                        for p_idx, preset in enumerate(env_presets):
                            scenario = scenario_names[p_idx]
                            logger.info(
                                "simulate %s:%s under scenario %s",
                                module,
                                sec_name,
                                scenario,
                            )
                            for stub in input_stubs:
                                env_input = dict(preset)
                                env_input.update(stub)

                                for _ in range(3):
                                    result = simulate_execution_environment(
                                        code_str, env_input
                                    )
                                    if not result.get("risk_flags_triggered"):
                                        break
                                    debugger.analyse_and_fix()

                                await sem.acquire()

                                async def _task() -> tuple[
                                    Dict[str, Any],
                                    list[tuple[float, float, Dict[str, float]]],
                                ]:
                                    try:
                                        return await _section_worker(
                                            code_str,
                                            env_input,
                                            tracker.diminishing(),
                                        )
                                    finally:
                                        sem.release()

                                fut = asyncio.create_task(_task())
                                tasks.append(
                                    (
                                        index,
                                        fut,
                                        module,
                                        sec_name,
                                        scenario,
                                        preset,
                                        stub,
                                    )
                                )
                                index += 1
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            sorted_tasks = sorted(tasks, key=lambda x: x[0])
            results = await asyncio.gather(*(t[1] for t in sorted_tasks))
            for (_, _fut, module, sec_name, scenario, preset, stub), (
                res,
                updates,
            ) in zip(sorted_tasks, results):
                logger.info(
                    "result %s:%s scenario %s exit=%s",
                    module,
                    sec_name,
                    scenario,
                    res.get("exit_code"),
                )
                for prev, actual, metrics in updates:
                    scenario_metrics = {
                        f"{k}:{scenario}": v for k, v in metrics.items()
                    }
                    pred_roi, _ = tracker.forecast()
                    tracker.record_metric_prediction("roi", pred_roi, actual)
                    tracker.update(
                        prev,
                        actual,
                        modules=[f"{module}:{sec_name}", scenario],
                        metrics={**metrics, **scenario_metrics},
                    )
                if updates:
                    synergy_data[scenario]["roi"].append(updates[-1][1])
                    synergy_data[scenario]["metrics"].append(updates[-1][2])
                if return_details:
                    details.setdefault(module, {}).setdefault(sec_name, []).append(
                        {"preset": preset, "stub": stub, "result": res}
                    )
                if res.get("exit_code") not in (0, None):
                    all_diminished = False

        await _gather_tasks()

        if all_diminished:
            combined: List[str] = []
            for sec_map in sections.values():
                for lines in sec_map.values():
                    combined.extend(lines)
            all_modules = list(sections)
            for p_idx, preset in enumerate(env_presets):
                scenario = scenario_names[p_idx]
                for stub in input_stubs:
                    env_input = dict(preset)
                    env_input.update(stub)
                    logger.info("combined run for scenario %s", scenario)
                    res, updates = await _section_worker(
                        "\n".join(combined),
                        env_input,
                        tracker.diminishing(),
                    )
                    for prev, actual, metrics in updates:
                        scenario_metrics = {
                            f"{k}:{scenario}": v for k, v in metrics.items()
                        }
                        pred_roi, _ = tracker.forecast()
                        tracker.record_metric_prediction("roi", pred_roi, actual)
                        tracker.update(
                            prev,
                            actual,
                            modules=all_modules + [scenario],
                            metrics={**metrics, **scenario_metrics},
                        )
                    if updates:
                        roi_sum = sum(float(r) for r in synergy_data[scenario]["roi"])
                        metric_totals: Dict[str, float] = {}
                        metric_counts: Dict[str, int] = {}
                        for m_dict in synergy_data[scenario]["metrics"]:
                            for m, val in m_dict.items():
                                metric_totals[m] = metric_totals.get(m, 0.0) + float(
                                    val
                                )
                                metric_counts[m] = metric_counts.get(m, 0) + 1
                        avg_metrics = {
                            m: metric_totals[m] / metric_counts[m]
                            for m in metric_totals
                            if metric_counts.get(m)
                        }
                        combined_metrics = updates[-1][2]
                        synergy_metrics = {
                            f"synergy_{k}": combined_metrics.get(k, 0.0)
                            - avg_metrics.get(k, 0.0)
                            for k in set(avg_metrics) | set(combined_metrics)
                        }
                        synergy_metrics["synergy_roi"] = updates[-1][1] - roi_sum
                        synergy_metrics.setdefault(
                            "synergy_profitability", synergy_metrics["synergy_roi"]
                        )
                        synergy_metrics.setdefault(
                            "synergy_revenue", synergy_metrics["synergy_roi"]
                        )
                        synergy_metrics.setdefault(
                            "synergy_projected_lucrativity",
                            combined_metrics.get("projected_lucrativity", 0.0)
                            - avg_metrics.get("projected_lucrativity", 0.0),
                        )
                        for m in (
                            "maintainability",
                            "code_quality",
                            "network_latency",
                            "throughput",
                        ):
                            synergy_metrics.setdefault(
                                f"synergy_{m}",
                                combined_metrics.get(m, 0.0) - avg_metrics.get(m, 0.0),
                            )
                        if hasattr(tracker, "register_metrics"):
                            tracker.register_metrics(*synergy_metrics.keys())
                        tracker.update(
                            roi_sum,
                            updates[-1][1],
                            modules=all_modules + [scenario],
                            metrics=synergy_metrics,
                        )
                        scenario_synergy.setdefault(scenario, []).append(
                            synergy_metrics
                        )
                        if hasattr(tracker, "scenario_synergy"):
                            tracker.scenario_synergy.setdefault(scenario, []).append(
                                synergy_metrics
                            )
                    if return_details:
                        details.setdefault("_combined", {}).setdefault(
                            "all", []
                        ).append({"preset": preset, "stub": stub, "result": res})

        if hasattr(tracker, "scenario_synergy"):
            tracker.scenario_synergy = scenario_synergy
        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


# ----------------------------------------------------------------------
def simulate_full_environment(preset: Dict[str, Any]) -> "ROITracker":
    """Execute an isolated sandbox run using ``preset`` environment vars."""

    tmp_dir = tempfile.mkdtemp(prefix="full_env_")
    try:
        repo_path = SANDBOX_REPO_PATH
        data_dir = Path(tmp_dir) / "data"
        env = os.environ.copy()
        env.update({k: str(v) for k, v in preset.items()})
        env.pop("SANDBOX_ENV_PRESETS", None)

        use_docker = str(os.getenv("SANDBOX_DOCKER", "0")).lower() not in {
            "0",
            "false",
            "no",
            "",
        }
        os_type = env.get("OS_TYPE", "").lower()
        vm_used = False
        if use_docker:
            container_repo = "/repo"
            sandbox_tmp = "/sandbox_tmp"
            env["SANDBOX_DATA_DIR"] = f"{sandbox_tmp}/data"

            code = (
                "import subprocess, os\n"
                "subprocess.run(['python', 'sandbox_runner.py'], cwd='"
                + container_repo
                + "')\n"
            )
            try:
                _execute_in_container(
                    code,
                    env,
                    mounts={str(repo_path): container_repo, tmp_dir: sandbox_tmp},
                    network_disabled=False,
                )
            except Exception:
                logger.exception("docker execution failed, falling back to local run")
                use_docker = False

        if not use_docker and os_type in {"windows", "macos"}:
            vm = shutil.which("qemu-system-x86_64")
            vm_settings = preset.get("VM_SETTINGS", {})
            image = vm_settings.get(f"{os_type}_image") or vm_settings.get("image")
            memory = str(vm_settings.get("memory", "2G"))
            if vm and image:
                vm_repo = "/repo"
                sandbox_tmp = "/sandbox_tmp"
                env["SANDBOX_DATA_DIR"] = f"{sandbox_tmp}/data"
                cmd = [
                    vm,
                    "-m",
                    memory,
                    "-snapshot",
                    "-drive",
                    f"file={image},if=virtio",
                    "-virtfs",
                    f"local,path={repo_path},mount_tag=repo,security_model=none",
                    "-virtfs",
                    f"local,path={tmp_dir},mount_tag=sandbox_tmp,security_model=none",
                    "-nographic",
                    "-serial",
                    "stdio",
                    "-append",
                    f"python {vm_repo}/sandbox_runner.py",
                ]
                try:
                    subprocess.run(
                        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    vm_used = True
                except Exception:
                    logger.exception("VM execution failed, falling back to local run")
                    vm_used = False
            else:
                logger.warning("qemu binary or VM image missing, running locally")

        if not use_docker and not vm_used:
            env["SANDBOX_DATA_DIR"] = str(data_dir)
            subprocess.run(
                ["python", "sandbox_runner.py"],
                cwd=repo_path,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        from menace.roi_tracker import ROITracker

        tracker = ROITracker()
        tracker.load_history(str(data_dir / "roi_history.json"))
        return tracker
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------
def run_workflow_simulations(
    workflows_db: str | Path = "workflows.db",
    env_presets: List[Dict[str, Any]] | None = None,
    *,
    return_details: bool = False,
    tracker: "ROITracker" | None = None,
) -> "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]:
    """Execute stored workflows under optional environment presets."""
    from menace.task_handoff_bot import WorkflowDB
    from menace.roi_tracker import ROITracker
    from menace.self_debugger_sandbox import SelfDebuggerSandbox
    from menace.self_coding_engine import SelfCodingEngine
    from menace.code_database import CodeDB
    from menace.menace_memory_manager import MenaceMemoryManager

    if env_presets is None:
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import generate_presets

                env_presets = generate_presets()
            except Exception:
                env_presets = [{}]
        else:
            env_presets = [{}]

    tracker = tracker or ROITracker()
    scenario_names = [
        p.get("SCENARIO_NAME", f"scenario_{i}") for i, p in enumerate(env_presets)
    ]

    wf_db = WorkflowDB(Path(workflows_db))
    workflows = wf_db.fetch()

    async def _run() -> (
        "ROITracker" | tuple["ROITracker", Dict[str, list[Dict[str, Any]]]]
    ):
        details: Dict[str, list[Dict[str, Any]]] = {}

        tasks: list[tuple[int, asyncio.Task, int, str, Dict[str, Any]]] = []
        index = 0
        synergy_data: Dict[str, Dict[str, list]] = {
            name: {"roi": [], "metrics": []} for name in scenario_names
        }
        combined_results: Dict[str, Dict[str, Any]] = {}

        def _wf_snippet(steps: list[str]) -> str:
            imports: list[str] = []
            calls: list[str] = []
            for idx, step in enumerate(steps):
                mod = ""
                func = ""
                if ":" in step:
                    mod, func = step.split(":", 1)
                elif "." in step:
                    mod, func = step.rsplit(".", 1)
                else:
                    mod, func = "simple_functions", step
                alias = f"_wf_{idx}"
                imports.append(f"from {mod} import {func} as {alias}")
                calls.append(f"{alias}()")
            if not calls:
                return "\n".join(f"# {s}" for s in steps) + "\npass\n"
            return "\n".join(imports + [""] + calls) + "\n"

        for wf in workflows:
            snippet = _wf_snippet(wf.workflow)
            debugger = SelfDebuggerSandbox(
                object(), SelfCodingEngine(CodeDB(), MenaceMemoryManager())
            )
            for p_idx, preset in enumerate(env_presets):
                scenario = scenario_names[p_idx]
                env_input = dict(preset)
                for _ in range(3):
                    result = simulate_execution_environment(snippet, env_input)
                    if not result.get("risk_flags_triggered"):
                        break
                    debugger.analyse_and_fix()
                fut = asyncio.create_task(
                    _section_worker(
                        snippet,
                        env_input,
                        tracker.diminishing(),
                    )
                )
                tasks.append((index, fut, wf.wid, scenario, preset))
                index += 1

        for _, fut, wid, scenario, preset in sorted(tasks, key=lambda x: x[0]):
            res, updates = await fut
            for prev, actual, metrics in updates:
                scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, actual)
                tracker.update(
                    prev,
                    actual,
                    modules=[f"workflow_{wid}", scenario],
                    metrics={**metrics, **scenario_metrics},
                )
            if updates:
                synergy_data[scenario]["roi"].append(updates[-1][1])
                synergy_data[scenario]["metrics"].append(updates[-1][2])
            if return_details:
                details.setdefault(str(wid), []).append(
                    {"preset": preset, "result": res}
                )

        combined_steps: list[str] = []
        for wf in workflows:
            combined_steps.extend(wf.workflow)
        combined_snippet = _wf_snippet(combined_steps)
        workflow_modules = [f"workflow_{wf.wid}" for wf in workflows]
        for p_idx, preset in enumerate(env_presets):
            scenario = scenario_names[p_idx]
            env_input = dict(preset)
            res, updates = await _section_worker(
                combined_snippet,
                env_input,
                tracker.diminishing(),
            )
            for prev, actual, metrics in updates:
                scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()}
                pred_roi, _ = tracker.forecast()
                tracker.record_metric_prediction("roi", pred_roi, actual)
                tracker.update(
                    prev,
                    actual,
                    modules=["all_workflows", scenario],
                    metrics={**metrics, **scenario_metrics},
                )
            if updates:
                combined_results[scenario] = {
                    "roi": updates[-1][1],
                    "metrics": updates[-1][2],
                }
                roi_sum = sum(float(r) for r in synergy_data[scenario]["roi"])
                metric_totals: Dict[str, float] = {}
                metric_counts: Dict[str, int] = {}
                for m_dict in synergy_data[scenario]["metrics"]:
                    for m, val in m_dict.items():
                        metric_totals[m] = metric_totals.get(m, 0.0) + float(val)
                        metric_counts[m] = metric_counts.get(m, 0) + 1
                avg_metrics = {
                    m: metric_totals[m] / metric_counts[m]
                    for m in metric_totals
                    if metric_counts.get(m)
                }
                combined_metrics = combined_results[scenario]["metrics"]
                synergy_metrics = {
                    f"synergy_{k}": combined_metrics.get(k, 0.0)
                    - avg_metrics.get(k, 0.0)
                    for k in set(avg_metrics) | set(combined_metrics)
                }
                synergy_metrics["synergy_roi"] = (
                    combined_results[scenario]["roi"] - roi_sum
                )
                if "synergy_profitability" not in synergy_metrics:
                    synergy_metrics["synergy_profitability"] = synergy_metrics[
                        "synergy_roi"
                    ]
                if "synergy_revenue" not in synergy_metrics:
                    synergy_metrics["synergy_revenue"] = synergy_metrics["synergy_roi"]
                if "synergy_projected_lucrativity" not in synergy_metrics:
                    synergy_metrics["synergy_projected_lucrativity"] = (
                        combined_metrics.get("projected_lucrativity", 0.0)
                        - avg_metrics.get("projected_lucrativity", 0.0)
                    )
                for m in (
                    "maintainability",
                    "code_quality",
                    "network_latency",
                    "throughput",
                ):
                    synergy_metrics.setdefault(
                        f"synergy_{m}",
                        combined_metrics.get(m, 0.0) - avg_metrics.get(m, 0.0),
                    )
                if hasattr(tracker, "register_metrics"):
                    tracker.register_metrics(*synergy_metrics.keys())
                tracker.update(
                    roi_sum,
                    combined_results[scenario]["roi"],
                    modules=workflow_modules + [scenario],
                    metrics=synergy_metrics,
                )
            if return_details:
                details.setdefault("_combined", []).append(
                    {"preset": preset, "result": res}
                )

        return (tracker, details) if return_details else tracker

    return asyncio.run(_run())


# ----------------------------------------------------------------------
def aggregate_synergy_metrics(
    paths: list[str], metric: str = "roi"
) -> list[tuple[str, float]]:
    """Return scenarios sorted by cumulative synergy ``metric``.

    Parameters
    ----------
    paths:
        Paths to run directories or ``roi_history.json`` files.
    metric:
        Metric name without the ``synergy_`` prefix. Defaults to ``"roi"``.
    """

    from menace.roi_tracker import ROITracker

    metric_name = metric if str(metric).startswith("synergy_") else f"synergy_{metric}"

    results: list[tuple[str, float]] = []
    for entry in paths:
        p = Path(entry)
        hist_path = p / "roi_history.json" if p.is_dir() else p
        name = p.name if p.is_dir() else p.stem
        tracker = ROITracker()
        try:
            tracker.load_history(str(hist_path))
        except Exception:
            logger.exception("failed to load history %s", hist_path)
            continue
        vals = tracker.metrics_history.get(metric_name)
        if vals is None:
            vals = tracker.synergy_metrics_history.get(metric_name, [])
        else:
            vals = list(vals)
        total = sum(float(v) for v in vals)
        results.append((name, total))

    return sorted(results, key=lambda x: x[1], reverse=True)
