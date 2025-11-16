from __future__ import annotations

"""Generate sandbox environment presets for scenario testing.

Adaptive preset agents persist their state to ``<path>.state.json`` and keep a
rolling ``.bak`` history.  On startup the loader attempts to recover from the
backup if the primary JSON file is corrupted.  Administrators can manually
restore by copying ``<path>.state.json.bak`` over the main file and restarting
the service.
"""

import random
import os
import ast
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, TYPE_CHECKING
import json
import logging
from logging_utils import log_record
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully when PyYAML missing
    try:
        from .yaml_fallback import get_yaml
    except Exception:  # pragma: no cover - allow flat execution
        from yaml_fallback import get_yaml  # type: ignore

    yaml = get_yaml(__name__)

try:  # pragma: no cover - support package and flat layouts
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

logger = logging.getLogger(__name__)
debug = os.getenv("PRESET_DEBUG") == "1"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .roi_tracker import ROITracker


# failure modes injected by the sandbox to mimic unstable environments
_FAILURE_MODES = [
    None,
    "disk",
    "network",
    "cpu",
    "memory",
    "timeout",
    "disk_corruption",
    "network_partition",
    "cpu_spike",
    "concurrency_spike",
    "hostile_input",
    "malicious_data",
    "user_misuse",
    "api_latency",
    "schema_drift",
    "flaky_upstream",
]

# chance that multiple failure modes will be combined in one preset
_MULTI_FAILURE_CHANCE = 0.2

_DISK_LIMITS = ["512Mi", "1Gi", "2Gi", "4Gi", "8Gi"]
_LATENCIES = [10, 50, 100, 200, 500]  # milliseconds
_API_LATENCIES = [50, 100, 200, 500, 1000]
_BANDWIDTHS = ["1Mbps", "5Mbps", "10Mbps", "50Mbps", "100Mbps"]
_CPU_LIMITS = ["0.5", "1", "2", "4", "8"]
_MEMORY_LIMITS = [f"{m}Mi" for m in [128, 256, 512, 1024, 2048, 4096]]
_PACKET_LOSS = [0.0, 0.01, 0.05, 0.1]
_JITTERS = [0, 5, 10, 20, 50]  # milliseconds
_PACKET_DUPLICATION = [0.0, 0.01, 0.05]
_SECURITY_LEVELS = [1, 2, 3, 4, 5]
_THREAT_INTENSITIES = [10, 30, 50, 70, 90]
_THREAD_BURSTS = [10, 20, 50]
_ASYNC_BURSTS = [20, 50, 100]
_CONCURRENCY_LEVELS = [1, 5, 10, 20, 50, 100]

# named profiles for deterministic scenarios with severity levels
_PROFILES: Dict[str, Dict[str, Any]] = {
    "high_latency_api": {
        "levels": {
            "low": {
                "NETWORK_LATENCY_MS": 200,
                "NETWORK_JITTER_MS": 20,
                "PACKET_LOSS": 0.05,
                "API_LATENCY_MS": 500,
                "FAILURE_MODES": "api_latency",
                "THREAT_INTENSITY": 30,
            },
            "high": {
                "NETWORK_LATENCY_MS": 500,
                "NETWORK_JITTER_MS": 50,
                "PACKET_LOSS": 0.1,
                "API_LATENCY_MS": 1000,
                "FAILURE_MODES": "api_latency",
                "THREAT_INTENSITY": 70,
            },
        }
    },
    "hostile_input": {
        "levels": {
            "low": {
                "FAILURE_MODES": "hostile_input",
                "SANDBOX_STUB_STRATEGY": "hostile",
                "PAYLOAD_INDICATOR": "corrupted_bytes",
                "MALICIOUS_DATA": True,
                "THREAT_INTENSITY": 30,
            },
            "high": {
                "FAILURE_MODES": "hostile_input",
                "SANDBOX_STUB_STRATEGY": "hostile",
                "PAYLOAD_INDICATOR": "corrupted_bytes",
                "MALICIOUS_DATA": True,
                "THREAT_INTENSITY": 50,
            },
        }
    },
    "user_misuse": {
        "levels": {
            "low": {
                "FAILURE_MODES": "user_misuse",
                "SANDBOX_STUB_STRATEGY": "misuse",
                "INVALID_CONFIG": True,
                "INVALID_PARAM_TYPES": True,
                "UNEXPECTED_API_CALLS": True,
                "THREAT_INTENSITY": 20,
            },
            "high": {
                "FAILURE_MODES": "user_misuse",
                "SANDBOX_STUB_STRATEGY": "misuse",
                "INVALID_CONFIG": True,
                "INVALID_PARAM_TYPES": True,
                "UNEXPECTED_API_CALLS": True,
                "THREAT_INTENSITY": 30,
            },
        }
    },
    "concurrency_spike": {
        "levels": {
            "low": {
                "FAILURE_MODES": ["concurrency_spike", "cpu_spike"],
                "THREAD_BURST": 10,
                "ASYNC_TASK_BURST": 20,
                "MAX_THREADS": 100,
                "CPU_SPIKE": True,
                "CONCURRENCY_LEVEL": 20,
                "THREAT_INTENSITY": 30,
            },
            "high": {
                "FAILURE_MODES": ["concurrency_spike", "cpu_spike"],
                "THREAD_BURST": 50,
                "ASYNC_TASK_BURST": 100,
                "MAX_THREADS": 200,
                "CPU_SPIKE": True,
                "CONCURRENCY_LEVEL": 100,
                "THREAT_INTENSITY": 50,
            },
        }
    },
    "schema_drift": {
        "levels": {
            "low": {
                "FAILURE_MODES": "schema_drift",
                "SCHEMA_MISMATCHES": 5,
                "SCHEMA_CHECKS": 100,
                "THREAT_INTENSITY": 20,
                "SANDBOX_STUB_STRATEGY": "legacy_schema",
            },
            "high": {
                "FAILURE_MODES": "schema_drift",
                "SCHEMA_MISMATCHES": 20,
                "SCHEMA_CHECKS": 100,
                "THREAT_INTENSITY": 40,
                "SANDBOX_STUB_STRATEGY": "legacy_schema",
            },
        }
    },
    "flaky_upstream": {
        "levels": {
            "low": {
                "FAILURE_MODES": "flaky_upstream",
                "UPSTREAM_FAILURES": 1,
                "UPSTREAM_REQUESTS": 20,
                "THREAT_INTENSITY": 20,
                "SANDBOX_STUB_STRATEGY": "flaky_upstream",
            },
            "high": {
                "FAILURE_MODES": "flaky_upstream",
                "UPSTREAM_FAILURES": 5,
                "UPSTREAM_REQUESTS": 20,
                "THREAT_INTENSITY": 40,
                "SANDBOX_STUB_STRATEGY": "flaky_upstream",
            },
        }
    },
}

# canonical profile names for core sandbox scenarios
CANONICAL_PROFILES: List[str] = [
    "high_latency_api",
    "hostile_input",
    "user_misuse",
    "concurrency_spike",
    "schema_drift",
    "flaky_upstream",
]

# legacy aliases mapping to canonical scenario names
_PROFILE_ALIASES = {
    "high_latency": "high_latency_api",
    "malicious_data": "hostile_input",
    "schema_mismatch": "schema_drift",
    "upstream_failure": "flaky_upstream",
}

# keyword based suggestions for module-specific profiles
_KEYWORD_PROFILE_MAP: Dict[str, List[str]] = {
    "api": ["high_latency_api"],
    "network": ["high_latency_api"],
    "parser": ["hostile_input", "user_misuse"],
    "input": ["hostile_input"],
    "concurrency": ["concurrency_spike"],
    "thread": ["concurrency_spike"],
    "database": ["high_latency_api", "concurrency_spike", "schema_drift"],
    "cache": ["high_latency_api"],
    "auth": ["user_misuse", "hostile_input"],
    "schema": ["schema_drift"],
    "legacy": ["schema_drift"],
    "flaky": ["flaky_upstream"],
    "upstream": ["flaky_upstream"],
    # ambiguous or generic modules should exercise all core scenarios
    "util": CANONICAL_PROFILES,
    "utils": CANONICAL_PROFILES,
    "common": CANONICAL_PROFILES,
    "misc": CANONICAL_PROFILES,
    "shared": CANONICAL_PROFILES,
}


def _load_keyword_overrides() -> Dict[str, List[str]]:
    """Load optional keyword profile overrides from YAML config."""

    path = os.getenv("SANDBOX_SETTINGS_YAML", "sandbox_settings.yaml")
    try:
        resolved = resolve_path(path)
        with resolved.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - log and ignore
        logger.debug("failed to load %s: %s", path, exc)
        return {}

    overrides = data.get("keyword_profiles", {})
    result: Dict[str, List[str]] = {}
    if isinstance(overrides, dict):
        for key, profs in overrides.items():
            if isinstance(profs, str):
                result[key] = [profs]
            else:
                try:
                    result[key] = list(profs)
                except TypeError:
                    continue
    return result


# merge optional overrides without modifying global if file missing
_KEYWORD_PROFILE_MAP.update(_load_keyword_overrides())


def infer_profiles_from_ast(module_path: str) -> List[str]:
    """Infer scenario profiles by inspecting a module's AST.

    The parser looks for import statements, decorator names and uppercase
    configuration flags. Any tokens that match :data:`_KEYWORD_PROFILE_MAP`
    keys will contribute their associated profiles. The resulting list
    preserves order and removes duplicates.
    """

    tokens: List[str] = []
    try:
        with open(module_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=module_path)
    except Exception:
        return []

    def _name(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        if isinstance(expr, ast.Call):
            return _name(expr.func)
        return ""

    class _Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover - trivial
            for alias in node.names:
                tokens.append(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pragma: no cover - trivial
            if node.module:
                tokens.append(node.module)
            for alias in node.names:
                tokens.append(alias.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - trivial
            for dec in node.decorator_list:
                tokens.append(_name(dec))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[attr-defined]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - trivial
            for dec in node.decorator_list:
                tokens.append(_name(dec))
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:  # pragma: no cover - trivial
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.isupper():
                    tokens.append(tgt.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # pragma: no cover - trivial
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id.isupper():
                tokens.append(tgt.id)
            self.generic_visit(node)

    _Visitor().visit(tree)

    tokens_l = [t.lower() for t in tokens]
    profiles: List[str] = []
    for key, profs in _KEYWORD_PROFILE_MAP.items():
        if any(key in t for t in tokens_l):
            profiles.extend(profs)

    seen: set[str] = set()
    result: List[str] = []
    for prof in profiles:
        if prof not in seen:
            result.append(prof)
            seen.add(prof)
    return result


def suggest_profiles_for_module(module_name: str) -> List[str]:
    """Return profile names relevant to ``module_name``.

    The helper performs a keyword lookup on the module path and supplements
    the results with :func:`infer_profiles_from_ast` which analyses the
    module's source code for imports, decorators and configuration flags.
    When multiple hints match the resulting list preserves order and removes
    duplicates. If no hints are found the full canonical profile list is
    returned to ensure broad coverage.
    """

    name = module_name.lower()
    profiles: List[str] = []
    for key, profs in _KEYWORD_PROFILE_MAP.items():
        if key and key in name:
            profiles.extend(profs)

    path = module_name
    if not os.path.isfile(path):
        mod_path = module_name.replace(".", "/")
        try:
            if os.path.isdir(mod_path):
                mod_path = resolve_path(os.path.join(mod_path, "__init__.py"))
            else:
                mod_path = resolve_path(f"{mod_path}.py")
            path = str(mod_path)
        except FileNotFoundError:
            path = ""
    else:
        path = str(resolve_path(path))

    if path:
        profiles.extend(infer_profiles_from_ast(path))

    if not profiles:
        return list(CANONICAL_PROFILES)

    seen: set[str] = set()
    unique: List[str] = []
    for prof in profiles:
        if prof not in seen:
            unique.append(prof)
            seen.add(prof)
    return unique


# probability of injecting a random profile when none specified
_PROFILE_PROB = 0.3

# minimum ROI history length for the adaptive RL agent
_ADAPTIVE_THRESHOLD = 5

# optional hardware and platform settings
_GPU_LIMITS = ["0", "1", "2", "4"]
_OS_TYPES = ["linux", "windows", "macos"]
# OS flavours that require a VM when Docker is unavailable
_ALT_OS_TYPES = ["windows", "macos"]
# probability that a preset targets a non-Linux OS
_ALT_OS_PROB = 0.3


def _select_failures() -> list[str]:
    """Return a list of failure modes to apply."""

    if random.random() < _MULTI_FAILURE_CHANCE:
        count = random.randint(2, min(3, len(_FAILURE_MODES) - 1))
        return random.sample([m for m in _FAILURE_MODES if m], count)
    mode = random.choice(_FAILURE_MODES)
    return [mode] if mode else []


def generate_canonical_presets() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return deterministic presets for common sandbox scenarios.

    Each canonical profile exposes a ``low`` and ``high`` severity level as
    defined in :data:`_PROFILES`. The returned mapping groups presets by
    scenario name and then by severity level so callers can iterate over the
    different intensities explicitly.
    """

    presets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in CANONICAL_PROFILES:
        prof = _PROFILES.get(name)
        if not prof:
            continue
        levels = prof.get("levels") if isinstance(prof, dict) else None
        if not levels:
            continue
        lvl_map: Dict[str, Dict[str, Any]] = {}
        for level_name, data in levels.items():
            preset = {
                "SCENARIO_NAME": name,
                "CPU_LIMIT": "1",
                "MEMORY_LIMIT": "512Mi",
            }
            preset.update(data)
            lvl_map[level_name] = preset
        presets[name] = lvl_map
    return presets


def generate_combined_presets(
    combinations: Sequence[Sequence[str]],
    *,
    severity: str | Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Return deterministic presets for explicit profile combinations.

    Each item in ``combinations`` is a sequence of profile names. The
    corresponding profile data is merged in order with duplicate failure
    modes removed while preserving their first occurrence. ``severity``
    behaves like in :func:`generate_presets` and may be a single string or a
    mapping of profile name to level.
    """

    def _sev(name: str) -> str:
        if isinstance(severity, dict):
            return (
                severity.get(name)
                or severity.get(_PROFILE_ALIASES.get(name, name))
                or "high"
            )
        return severity or "high"

    def _profile_data(name: str) -> Dict[str, Any] | None:
        data = _PROFILES.get(name)
        if not data:
            return None
        levels = data.get("levels") if isinstance(data, dict) else None
        if levels:
            lvl = _sev(name)
            return levels.get(lvl) or levels.get("high") or next(iter(levels.values()))
        return data

    presets: List[Dict[str, Any]] = []
    for combo in combinations:
        base: Dict[str, Any] = {"CPU_LIMIT": "1", "MEMORY_LIMIT": "512Mi"}
        failures: List[str] = []
        canonical = [_PROFILE_ALIASES.get(n, n) for n in combo]
        for name in canonical:
            data = _profile_data(name)
            if not data:
                continue
            fm = data.get("FAILURE_MODES")
            if fm:
                if isinstance(fm, list):
                    failures.extend(fm)
                else:
                    failures.append(fm)
            for k, v in data.items():
                if k != "FAILURE_MODES":
                    base[k] = v
        if failures:
            uniq: List[str] = []
            for fm in failures:
                if fm not in uniq:
                    uniq.append(fm)
            base["FAILURE_MODES"] = uniq if len(uniq) > 1 else uniq[0]
        base["SCENARIO_NAME"] = "+".join(canonical)
        presets.append(base)
    return presets


def _random_preset() -> Dict[str, Any]:
    """Build a single random preset."""

    cpu = random.choice(_CPU_LIMITS)
    memory = random.choice(_MEMORY_LIMITS)
    disk = random.choice(_DISK_LIMITS)
    jitter = random.choice(_JITTERS)
    bw_idx1 = random.randrange(len(_BANDWIDTHS))
    bw_idx2 = random.randrange(bw_idx1, len(_BANDWIDTHS))
    api_latency = random.choice(_API_LATENCIES)
    concurrency_level = random.choice(_CONCURRENCY_LEVELS)
    preset = {
        "CPU_LIMIT": cpu,
        "MEMORY_LIMIT": memory,
        "DISK_LIMIT": disk,
        "NETWORK_LATENCY_MS": random.choice(_LATENCIES),
        "NETWORK_JITTER_MS": jitter,
        "API_LATENCY_MS": api_latency,
        "MIN_BANDWIDTH": _BANDWIDTHS[bw_idx1],
        "MAX_BANDWIDTH": _BANDWIDTHS[bw_idx2],
        "BANDWIDTH_LIMIT": random.choice(_BANDWIDTHS),
        "PACKET_LOSS": random.choice(_PACKET_LOSS),
        "PACKET_DUPLICATION": random.choice(_PACKET_DUPLICATION),
        "SECURITY_LEVEL": random.choice(_SECURITY_LEVELS),
        "THREAT_INTENSITY": random.choice(_THREAT_INTENSITIES),
        "CONCURRENCY_LEVEL": concurrency_level,
    }
    if random.random() < 0.5:
        preset["GPU_LIMIT"] = random.choice(_GPU_LIMITS)
    if random.random() < _ALT_OS_PROB:
        os_type = random.choice(_ALT_OS_TYPES)
        preset["OS_TYPE"] = os_type
        preset["CONTAINER_IMAGE"] = f"python:3.11-{os_type}"
        preset["VM_SETTINGS"] = {
            f"{os_type}_image": f"{os_type}-base.qcow2",
            "memory": "4Gi",
        }
    # identify high-latency scenarios even without explicit failure modes
    if (
        preset["NETWORK_LATENCY_MS"] >= 200
        and (
            preset["NETWORK_JITTER_MS"] >= 20
            or preset["PACKET_LOSS"] >= 0.05
        )
    ) or preset["API_LATENCY_MS"] >= 500:
        preset.setdefault(
            "FAILURE_MODES",
            "api_latency" if preset["API_LATENCY_MS"] >= 500 else "network",
        )
        preset.setdefault("SCENARIO_NAME", "high_latency_api")

    failures = _select_failures()
    if failures:
        preset["FAILURE_MODES"] = failures[0] if len(failures) == 1 else failures
        if "api_latency" in failures:
            preset["API_LATENCY_MS"] = random.choice(_API_LATENCIES[3:])
            preset["SCENARIO_NAME"] = "high_latency_api"
        if "hostile_input" in failures or "malicious_data" in failures:
            preset["SCENARIO_NAME"] = (
                "hostile_input" if "hostile_input" in failures else "malicious_data"
            )
            preset["SANDBOX_STUB_STRATEGY"] = "hostile"
            preset.setdefault("PAYLOAD_INDICATOR", "corrupted_bytes")
            preset.setdefault("MALICIOUS_DATA", True)
        if "user_misuse" in failures:
            preset["SCENARIO_NAME"] = "user_misuse"
            preset["SANDBOX_STUB_STRATEGY"] = "misuse"
            preset.setdefault("INVALID_CONFIG", True)
            preset.setdefault("INVALID_PARAM_TYPES", True)
            preset.setdefault("UNEXPECTED_API_CALLS", True)
        if "concurrency_spike" in failures:
            preset.setdefault("THREAD_BURST", random.choice(_THREAD_BURSTS))
            preset.setdefault(
                "ASYNC_TASK_BURST", random.choice(_ASYNC_BURSTS)
            )
            preset.setdefault("MAX_THREADS", random.choice([100, 200, 500]))
            preset.setdefault("CPU_SPIKE", True)
            preset["SCENARIO_NAME"] = "concurrency_spike"
            preset["CONCURRENCY_LEVEL"] = max(
                preset.get("CONCURRENCY_LEVEL", 0),
                random.choice(_CONCURRENCY_LEVELS[3:]),
            )
    return preset


def generate_presets(
    count: int | None = None,
    *,
    profiles: Sequence[Union[str, Sequence[str]]] | None = None,
    severity: str | Dict[str, str] | None = None,
    agent: "AdaptivePresetAgent" | None = None,
    tracker: "ROITracker" | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of environment presets.

    ``profiles`` may contain individual profile names or sequences of profile
    names. When a sequence is supplied the corresponding profile data is
    merged in order, producing a combined scenario. ``severity`` selects a
    predefined tier for these profiles. It may be a single string applied to
    all profiles or a mapping of profile name to severity level. When
    ``profiles`` is omitted a few presets may still randomly pick one of the
    predefined profiles.
    """
    num = 3 if count is None else max(0, count)
    presets: List[Dict[str, Any]] = []

    def _sev(name: str) -> str:
        if isinstance(severity, dict):
            return severity.get(name) or severity.get(_PROFILE_ALIASES.get(name, name)) or "high"
        return severity or "high"

    def _profile_data(name: str) -> Dict[str, Any] | None:
        data = _PROFILES.get(name)
        if not data:
            return None
        levels = data.get("levels") if isinstance(data, dict) else None
        if levels:
            lvl = _sev(name)
            return levels.get(lvl) or levels.get("high") or next(iter(levels.values()))
        return data

    if profiles:
        for name in profiles:
            if isinstance(name, (list, tuple, set)):
                canonical_list = [_PROFILE_ALIASES.get(n, n) for n in name]
                base = _random_preset()
                failures: List[str] = []
                for cname in canonical_list:
                    data = _profile_data(cname)
                    if not data:
                        continue
                    fm = data.get("FAILURE_MODES")
                    if fm:
                        if isinstance(fm, list):
                            failures.extend(fm)
                        else:
                            failures.append(fm)
                    for k, v in data.items():
                        if k != "FAILURE_MODES":
                            base[k] = v
                if failures:
                    uniq: List[str] = []
                    for fm in failures:
                        if fm not in uniq:
                            uniq.append(fm)
                    base["FAILURE_MODES"] = uniq if len(uniq) > 1 else uniq[0]
                base["SCENARIO_NAME"] = "+".join(canonical_list)
                presets.append(base)
            else:
                canonical = _PROFILE_ALIASES.get(name, name)
                base = _random_preset()
                data = _profile_data(canonical)
                if data:
                    base.update(data)
                base["SCENARIO_NAME"] = canonical
                presets.append(base)

    remaining = num - len(presets)
    for _ in range(max(0, remaining)):
        preset = _random_preset()
        if (
            not profiles
            and "FAILURE_MODES" not in preset
            and _PROFILES
            and random.SystemRandom().random() < _PROFILE_PROB
        ):
            name = random.choice(list(_PROFILES))
            data = _profile_data(name)
            if data:
                preset.update(data)
            preset["SCENARIO_NAME"] = name
        presets.append(preset)

    if tracker:
        def _adj(seq: List[Any], cur: Any, up: bool) -> Any:
            lookup = [str(x) for x in seq]
            try:
                idx = lookup.index(str(cur))
            except ValueError:
                idx = 0
            idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
            return seq[idx]

        eff_vals = tracker.metrics_history.get("synergy_efficiency", [])
        vals = eff_vals[-3:]
        if vals:
            avg_eff = sum(vals) / len(vals)

            if avg_eff > 0.05:
                for p in presets:
                    lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[-1])
                    p["NETWORK_LATENCY_MS"] = _adj(_LATENCIES, lat, False)
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                    p["BANDWIDTH_LIMIT"] = _adj(_BANDWIDTHS, bw, True)
                    p["MAX_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                    )
                    p["MIN_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                    )
            elif avg_eff < -0.05:
                for p in presets:
                    lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
                    p["NETWORK_LATENCY_MS"] = _adj(_LATENCIES, lat, True)
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[-1]))
                    p["BANDWIDTH_LIMIT"] = _adj(_BANDWIDTHS, bw, False)
                    p["MAX_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                    )
                    p["MIN_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                    )

        lat_vals = tracker.metrics_history.get("synergy_network_latency", [])
        vals_lat = lat_vals[-3:]
        if vals_lat:
            avg_lat = sum(vals_lat) / len(vals_lat)
            for p in presets:
                cur = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
                if avg_lat > 1.0:
                    p["NETWORK_LATENCY_MS"] = _adj(_LATENCIES, cur, True)
                elif avg_lat < -1.0:
                    p["NETWORK_LATENCY_MS"] = _adj(_LATENCIES, cur, False)

        tp_vals = tracker.metrics_history.get("synergy_throughput", [])
        vals_tp = tp_vals[-3:]
        if vals_tp:
            avg_tp = sum(vals_tp) / len(vals_tp)
            for p in presets:
                bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[0]))
                if avg_tp > 5.0:
                    p["MAX_BANDWIDTH"] = _adj(_BANDWIDTHS, bw, True)
                    p["MIN_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                    )
                elif avg_tp < -5.0:
                    p["MAX_BANDWIDTH"] = _adj(_BANDWIDTHS, bw, False)
                    p["MIN_BANDWIDTH"] = _adj(
                        _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                    )

    if (
        agent
        and tracker
        and len(getattr(tracker, "roi_history", [])) >= _ADAPTIVE_THRESHOLD
    ):
        try:
            actions = agent.decide(tracker)
            if debug:
                logger.debug(
                    "RL actions %s", actions, extra=log_record(agent="rl", actions=actions)
                )

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            def _lvl(val: int, up: bool) -> int:
                levels = _THREAT_INTENSITIES
                if up:
                    for lvl in levels:
                        if lvl > val:
                            return lvl
                    return levels[-1]
                for lvl in reversed(levels):
                    if lvl < val:
                        return lvl
                return levels[0]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("bandwidth"):
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                    p["BANDWIDTH_LIMIT"] = _nxt(
                        _BANDWIDTHS, bw, actions["bandwidth"] > 0
                    )
                    p["MAX_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MAX_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                    p["MIN_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MIN_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            agent.save()
        except Exception as exc:
            logger.exception(
                "preset adaptation failed for policy %s with %d presets: %s",
                getattr(getattr(agent, "policy", None), "path", "n/a"),
                len(presets),
                exc,
                extra=log_record(
                    presets=presets,
                    tracker_state=getattr(tracker, "__dict__", {}),
                ),
            )

    return presets


__all__ = ["generate_presets", "generate_canonical_presets", "AdaptivePresetAgent"]


class AdaptivePresetAgent:
    """Simple RL agent using ROI and synergy history to adjust presets.

    When ``strategy`` is set to ``"deep_q"`` a small neural network is used to
    estimate Q-values via :class:`DeepQLearningStrategy`. ``"double_dqn"``
    selects the :class:`DoubleDQNStrategy` if PyTorch is available.
    """

    ACTIONS: tuple[dict[str, int], ...] = (
        {"cpu": 1, "memory": 0, "threat": 0},
        {"cpu": -1, "memory": 0, "threat": 0},
        {"cpu": 0, "memory": 1, "threat": 0},
        {"cpu": 0, "memory": -1, "threat": 0},
        {"cpu": 0, "memory": 0, "threat": 1},
        {"cpu": 0, "memory": 0, "threat": -1},
        {"cpu": 0, "memory": 0, "threat": 0},
    )

    def __init__(self, path: str | None = None, *, strategy: str | None = None) -> None:
        from .self_improvement_policy import (
            SelfImprovementPolicy,
            strategy_factory,
            DoubleDQNStrategy,
            torch as _torch,
            nn as _nn,
        )

        if strategy is None:
            strategy = (
                os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY")
                or os.getenv("SANDBOX_PRESET_RL_STRATEGY")
                or "q_learning"
            )

        name = str(strategy).replace("-", "_")
        if name == "double_dqn" and _torch is not None and _nn is not None:
            rl_strategy = DoubleDQNStrategy()
        else:
            env_var = (
                "SANDBOX_ADAPTIVE_AGENT_STRATEGY"
                if os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY") is not None
                else "SANDBOX_PRESET_RL_STRATEGY"
            )
            rl_strategy = strategy_factory(strategy, env_var=env_var)

        self.policy = SelfImprovementPolicy(path=path, strategy=rl_strategy)
        self.state_file = f"{path}.state.json" if path else None
        self.prev_state: tuple[int, ...] | None = None
        self.prev_action: int | None = None
        self._load_state()

    # --------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        data: dict[str, object] | None = None
        try:
            with open(self.state_file) as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "state" not in data or "action" not in data:
                raise ValueError("missing keys")
        except Exception as exc:
            logging.warning(
                "Failed to load RL state from %s: %s",
                self.state_file,
                exc,
            )
            bak = f"{self.state_file}.bak"
            if os.path.exists(bak):
                try:
                    with open(bak) as fh:
                        data = json.load(fh)
                    if not isinstance(data, dict) or "state" not in data or "action" not in data:
                        raise ValueError("missing keys")
                    os.replace(bak, self.state_file)
                except Exception as exc2:
                    logging.warning(
                        "Failed to load backup RL state from %s: %s",
                        bak,
                        exc2,
                    )
                    try:
                        os.remove(self.state_file)
                    except OSError:
                        pass
                    return
            else:
                try:
                    os.remove(self.state_file)
                except OSError:
                    pass
                return
        st = data.get("state") if data is not None else None
        self.prev_state = tuple(st) if isinstance(st, (list, tuple)) else None
        self.prev_action = data.get("action") if data is not None else None

    def _save_state(self) -> None:
        if not self.state_file:
            return
        tmp_file = f"{self.state_file}.tmp"
        bak_file = f"{self.state_file}.bak"
        old_bak = f"{bak_file}.1"
        try:
            with open(tmp_file, "w") as fh:
                json.dump({"state": self.prev_state, "action": self.prev_action}, fh)
            if os.path.exists(bak_file):
                os.replace(bak_file, old_bak)
            if os.path.exists(self.state_file):
                os.replace(self.state_file, bak_file)
            os.replace(tmp_file, self.state_file)
        except Exception as exc:
            logging.warning(
                "Failed to save RL state to %s: %s",
                self.state_file,
                exc,
            )
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass

    # --------------------------------------------------------------
    def _state(self, tracker: "ROITracker") -> tuple[int, ...]:
        """Return a compact representation of the current environment state.

        The state encodes ROI and synergy trends along with recent CPU usage,
        memory usage and threat intensity information obtained from
        :class:`ROITracker`.
        """

        roi_hist = tracker.roi_history
        syn_hist = tracker.metrics_history.get("synergy_roi", [])
        eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
        res_hist = tracker.metrics_history.get("synergy_resilience", [])
        cpu_hist = tracker.metrics_history.get("cpu_usage", [])
        mem_hist = tracker.metrics_history.get("memory_usage", [])
        threat_hist = tracker.metrics_history.get("threat_intensity", [])

        def _trend(vals: list[float]) -> float:
            if len(vals) >= 2:
                return float(vals[-1] - vals[-2])
            return float(vals[-1]) if vals else 0.0

        roi_trend = _trend(roi_hist)
        syn_trend = _trend(syn_hist)
        eff_trend = _trend(eff_hist)
        res_trend = _trend(res_hist)
        cpu_trend = _trend(cpu_hist)
        mem_trend = _trend(mem_hist)
        threat_trend = _trend(threat_hist)

        def _sign(val: float) -> int:
            return 1 if val > 0 else (-1 if val < 0 else 0)

        return (
            _sign(roi_trend),
            _sign(syn_trend),
            _sign(eff_trend),
            _sign(res_trend),
            _sign(cpu_trend),
            _sign(mem_trend),
            _sign(threat_trend),
        )

    def _reward(self, tracker: "ROITracker") -> float:
        """Return reward based on ROI, synergy and resource changes."""

        roi_hist = tracker.roi_history
        roi_delta = roi_hist[-1] - roi_hist[-2] if len(roi_hist) >= 2 else 0.0

        syn_hist = tracker.metrics_history.get("synergy_roi", [])
        eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
        res_hist = tracker.metrics_history.get("synergy_resilience", [])
        syn_delta = 0.0
        if len(syn_hist) >= 2:
            syn_delta = syn_hist[-1] - syn_hist[-2]
        elif syn_hist:
            syn_delta = syn_hist[-1]
        eff_delta = (
            eff_hist[-1] - eff_hist[-2]
            if len(eff_hist) >= 2
            else (eff_hist[-1] if eff_hist else 0.0)
        )
        res_delta = (
            res_hist[-1] - res_hist[-2]
            if len(res_hist) >= 2
            else (res_hist[-1] if res_hist else 0.0)
        )

        cpu_hist = tracker.metrics_history.get("cpu_usage", [])
        mem_hist = tracker.metrics_history.get("memory_usage", [])
        threat_hist = tracker.metrics_history.get("threat_intensity", [])

        cpu_delta = cpu_hist[-1] - cpu_hist[-2] if len(cpu_hist) >= 2 else 0.0
        mem_delta = mem_hist[-1] - mem_hist[-2] if len(mem_hist) >= 2 else 0.0
        threat_delta = (
            threat_hist[-1] - threat_hist[-2] if len(threat_hist) >= 2 else 0.0
        )

        # Penalise increased resource usage and threat intensity
        return float(
            roi_delta
            + syn_delta
            + eff_delta
            + res_delta
            - cpu_delta
            - mem_delta
            - threat_delta
        )

    # --------------------------------------------------------------
    def decide(self, tracker: "ROITracker") -> Dict[str, int]:
        state = self._state(tracker)
        if self.prev_state is not None and self.prev_action is not None:
            reward = self._reward(tracker)
            self.policy.update(self.prev_state, reward, state, action=self.prev_action)
        action_idx = self.policy.select_action(state)
        self.prev_state = state
        self.prev_action = action_idx
        return dict(self.ACTIONS[action_idx])

    def save(self) -> None:
        self.policy.save()
        self._save_state()

    # --------------------------------------------------------------
    def export_policy(self) -> Dict[tuple[int, ...], Dict[int, float]]:
        """Return policy values for external analysis."""

        return {k: dict(v) for k, v in self.policy.values.items()}

    def import_policy(self, data: Dict[tuple[int, ...], Dict[int, float]]) -> None:
        """Load policy values from ``data``."""

        new_table: Dict[tuple[int, ...], Dict[int, float]] = {}
        for key, val in data.items():
            tup = tuple(key) if not isinstance(key, tuple) else key
            new_table[tup] = {int(a): float(q) for a, q in val.items()}
        self.policy.values = new_table


def adapt_presets(
    tracker: "ROITracker", presets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return ``presets`` adjusted based on ``tracker`` metrics.

    When the recent average ``security_score`` is high, the next higher
    ``THREAT_INTENSITY`` level is selected to further stress security
    mechanisms. Low scores decrease the intensity accordingly.
    """

    actions: list[str] = []
    agent = None
    rl_path = os.getenv("SANDBOX_PRESET_RL_PATH")
    if not rl_path:
        rl_path = str(resolve_path("sandbox_data/preset_policy.json"))
    rl_strategy = os.getenv("SANDBOX_PRESET_RL_STRATEGY")
    if rl_path:
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)
        try:
            agent = getattr(adapt_presets, "_rl_agent", None)
            strat_name = (
                getattr(
                    getattr(agent, "policy", None), "strategy", None
                ).__class__.__name__.lower()
                if agent
                else None
            )
            if (
                agent is None
                or getattr(getattr(agent, "policy", None), "path", None) != rl_path
                or (rl_strategy and strat_name != rl_strategy.replace("-", "_").lower())
            ):
                agent = AdaptivePresetAgent(rl_path, strategy=rl_strategy)
                adapt_presets._rl_agent = agent
        except Exception:
            agent = None

    adapt_agent = getattr(adapt_presets, "_adaptive_agent", None)
    if adapt_agent is None:
        try:
            path = os.getenv("SANDBOX_ADAPTIVE_AGENT_PATH")
            adapt_strategy = os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY")
            adapt_agent = AdaptivePresetAgent(path, strategy=adapt_strategy)
            adapt_presets._adaptive_agent = adapt_agent
        except Exception:
            adapt_agent = None

    sec_vals = tracker.metrics_history.get("security_score", [])
    if not sec_vals:
        adapt_presets.last_actions = []
        return presets

    recent = sec_vals[-3:]
    avg_sec = sum(recent) / len(recent)

    def _next_level(current: int, up: bool) -> int:
        levels = _THREAT_INTENSITIES
        if up:
            for lvl in levels:
                if lvl > current:
                    return lvl
            return levels[-1]
        for lvl in reversed(levels):
            if lvl < current:
                return lvl
        return levels[0]

    sec_decision = "none"
    if avg_sec >= 80.0:
        sec_decision = "increase"
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            p["THREAT_INTENSITY"] = _next_level(cur, True)
    elif avg_sec < 50.0:
        sec_decision = "decrease"
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            p["THREAT_INTENSITY"] = _next_level(cur, False)
    if sec_decision != "none":
        actions.append(f"security_{sec_decision}")
    if debug:
        logger.debug(
            "security avg %.2f -> %s threat intensity",
            avg_sec,
            sec_decision,
            extra=log_record(avg_security=avg_sec, threat_decision=sec_decision),
        )

    # --------------------------------------------------------------
    # ROI-driven resource scaling
    def _next_val(seq: List[Any], current: Any, up: bool) -> Any:
        lookup = [str(x) for x in seq]
        try:
            idx = lookup.index(str(current))
        except ValueError:
            idx = 0
        idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
        return seq[idx]

    roi_hist = getattr(tracker, "roi_history", [])
    if len(roi_hist) >= 2:
        deltas = [roi_hist[i] - roi_hist[i - 1] for i in range(1, len(roi_hist))]
        recent_deltas = deltas[-3:]
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        tol = getattr(tracker, "diminishing", lambda: 0.01)()
        roi_avg = sum(roi_hist[-3:]) / min(3, len(roi_hist))
        roi_decision = "none"

        if abs(avg_delta) <= tol:
            # ROI stagnates -> scale up resources
            roi_decision = "scale_up"
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
        elif avg_delta > tol:
            # ROI improving -> slowly tighten limits
            roi_decision = "tighten"
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )
        if debug:
            logger.debug(
                "ROI avg %.2f delta %.3f -> %s resources",
                roi_avg,
                avg_delta,
                roi_decision,
                extra=log_record(
                    avg_roi=roi_avg,
                    roi_delta=avg_delta,
                    roi_decision=roi_decision,
                ),
            )
        if roi_decision != "none":
            actions.append(f"roi_{roi_decision}")

    # --------------------------------------------------------------
    # Use reinforcement learning when sufficient history is available
    if (
        agent
        and len(tracker.roi_history) >= 3
        and tracker.metrics_history.get("synergy_roi")
    ):
        try:
            actions = agent.decide(tracker)

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("bandwidth"):
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                    p["BANDWIDTH_LIMIT"] = _nxt(
                        _BANDWIDTHS, bw, actions["bandwidth"] > 0
                    )
                    p["MAX_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MAX_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                    p["MIN_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MIN_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))

                    def _lvl(val: int, up: bool) -> int:
                        levels = _THREAT_INTENSITIES
                        if up:
                            for lvl in levels:
                                if lvl > val:
                                    return lvl
                            return levels[-1]
                        for lvl in reversed(levels):
                            if lvl < val:
                                return lvl
                        return levels[0]

                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            agent.save()
            return presets
        except Exception as exc:
            logger.exception(
                "preset adaptation failed for policy %s with %d presets: %s",
                getattr(getattr(agent, "policy", None), "path", "n/a"),
                len(presets),
                exc,
                extra=log_record(
                    presets=presets,
                    tracker_state=getattr(tracker, "__dict__", {}),
                ),
            )

    elif adapt_agent and len(tracker.roi_history) >= _ADAPTIVE_THRESHOLD:
        try:
            actions = adapt_agent.decide(tracker)
            if debug:
                logger.debug(
                    "adaptive actions %s",
                    actions,
                    extra=log_record(agent="adaptive", actions=actions),
                )

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))

                    def _lvl(val: int, up: bool) -> int:
                        levels = _THREAT_INTENSITIES
                        if up:
                            for lvl in levels:
                                if lvl > val:
                                    return lvl
                            return levels[-1]
                        for lvl in reversed(levels):
                            if lvl < val:
                                return lvl
                        return levels[0]

                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            adapt_agent.save()
            return presets
        except Exception as exc:
            logger.exception(
                "preset adaptation failed for adaptive agent with %d presets: %s",
                len(presets),
                exc,
                extra=log_record(
                    presets=presets,
                    tracker_state=getattr(tracker, "__dict__", {}),
                ),
            )

    # --------------------------------------------------------------
    # Synergy-driven adjustments with prediction support
    syn_roi_vals = tracker.metrics_history.get("synergy_roi", [])
    try:
        if hasattr(tracker, "forecast_synergy"):
            pred_synergy_roi = float(tracker.forecast_synergy()[0])
        elif hasattr(tracker, "predict_synergy"):
            pred_synergy_roi = float(tracker.predict_synergy())
        elif hasattr(tracker, "forecast"):
            pred_synergy_roi = float(tracker.forecast()[0])
        else:
            pred_synergy_roi = 0.0
    except Exception:
        pred_synergy_roi = 0.0
    vals = syn_roi_vals[-3:]
    if pred_synergy_roi:
        vals.append(pred_synergy_roi)
    if vals:
        avg_syn = sum(vals) / len(vals)
        if avg_syn > 0.1:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, True)
                lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, False)
                bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[0]))
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, True)
        elif avg_syn < -0.1:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, False)
                lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[-1])
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, True)
                bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[-1]))
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, False)

    syn_sec_vals = tracker.metrics_history.get("synergy_security_score", [])
    try:
        pred_syn_sec = float(
            getattr(tracker, "predict_synergy_metric")("security_score")
        )
    except Exception:
        pred_syn_sec = 0.0
    vals_sec = syn_sec_vals[-3:]
    if pred_syn_sec:
        vals_sec.append(pred_syn_sec)
    if vals_sec:
        avg_syn_sec = sum(vals_sec) / len(vals_sec)

        def _next_sec_level(current: int, up: bool) -> int:
            levels = _SECURITY_LEVELS
            if up:
                for lvl in levels:
                    if lvl > current:
                        return lvl
                return levels[-1]
            for lvl in reversed(levels):
                if lvl < current:
                    return lvl
            return levels[0]

        if avg_syn_sec > 5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, True)
        elif avg_syn_sec < -5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, False)

    syn_safe_vals = tracker.metrics_history.get("synergy_safety_rating", [])
    try:
        pred_syn_safe = float(
            getattr(tracker, "predict_synergy_metric")("safety_rating")
        )
    except Exception:
        pred_syn_safe = 0.0
    vals_safe = syn_safe_vals[-3:]
    if pred_syn_safe:
        vals_safe.append(pred_syn_safe)
    if vals_safe:
        avg_syn_safe = sum(vals_safe) / len(vals_safe)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_safe > 5.0:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_safe < -5.0:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_risk_vals = tracker.metrics_history.get("synergy_risk_index", [])
    try:
        pred_syn_risk = float(getattr(tracker, "predict_synergy_metric")("risk_index"))
    except Exception:
        pred_syn_risk = 0.0
    vals_risk = syn_risk_vals[-3:]
    if pred_syn_risk:
        vals_risk.append(pred_syn_risk)
    if vals_risk:
        avg_syn_risk = sum(vals_risk) / len(vals_risk)

        def _next_sec_level(current: int, up: bool) -> int:
            levels = _SECURITY_LEVELS
            if up:
                for lvl in levels:
                    if lvl > current:
                        return lvl
                return levels[-1]
            for lvl in reversed(levels):
                if lvl < current:
                    return lvl
            return levels[0]

        if avg_syn_risk > 5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, True)
        elif avg_syn_risk < -5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, False)

    # ------------------------------------------------------------------
    # Synergy metrics for efficiency, antifragility and resilience
    syn_eff_vals = tracker.metrics_history.get("synergy_efficiency", [])
    try:
        pred_syn_eff = float(getattr(tracker, "predict_synergy_metric")("efficiency"))
    except Exception:
        pred_syn_eff = 0.0
    vals_eff = syn_eff_vals[-3:]
    if pred_syn_eff:
        vals_eff.append(pred_syn_eff)
    if vals_eff:
        avg_syn_eff = sum(vals_eff) / len(vals_eff)
        if avg_syn_eff > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_eff < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_adapt_vals = tracker.metrics_history.get("synergy_adaptability", [])
    try:
        pred_syn_adapt = float(
            getattr(tracker, "predict_synergy_metric")("adaptability")
        )
    except Exception:
        pred_syn_adapt = 0.0
    vals_adapt = syn_adapt_vals[-3:]
    if pred_syn_adapt:
        vals_adapt.append(pred_syn_adapt)
    if vals_adapt:
        avg_syn_adapt = sum(vals_adapt) / len(vals_adapt)
        if avg_syn_adapt > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_adapt < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_anti_vals = tracker.metrics_history.get("synergy_antifragility", [])
    try:
        pred_syn_af = float(getattr(tracker, "predict_synergy_metric")("antifragility"))
    except Exception:
        pred_syn_af = 0.0
    vals_af = syn_anti_vals[-3:]
    if pred_syn_af:
        vals_af.append(pred_syn_af)
    if vals_af:
        avg_syn_af = sum(vals_af) / len(vals_af)
        if avg_syn_af > 0.05:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, True)
        elif avg_syn_af < -0.05:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_res_vals = tracker.metrics_history.get("synergy_resilience", [])
    try:
        pred_syn_res = float(getattr(tracker, "predict_synergy_metric")("resilience"))
    except Exception:
        pred_syn_res = 0.0
    vals_res = syn_res_vals[-3:]
    if pred_syn_res:
        vals_res.append(pred_syn_res)
    if vals_res:
        avg_syn_res = sum(vals_res) / len(vals_res)
        if avg_syn_res > 0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
        elif avg_syn_res < -0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[-1]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )

    syn_rec_vals = tracker.metrics_history.get("synergy_recovery_time", [])
    try:
        pred_syn_rec = float(getattr(tracker, "predict_synergy_recovery_time")())
    except Exception:
        pred_syn_rec = 0.0
    vals_rec = syn_rec_vals[-3:]
    if pred_syn_rec:
        vals_rec.append(pred_syn_rec)
    if vals_rec:
        avg_syn_rec = sum(vals_rec) / len(vals_rec)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_rec > 0.1:
                p["THREAT_INTENSITY"] = _next_level(cur, False)
            elif avg_syn_rec < -0.1:
                p["THREAT_INTENSITY"] = _next_level(cur, True)

    # ------------------------------------------------------------------
    # Additional synergy metrics controlling resources
    syn_ent_vals = tracker.metrics_history.get("synergy_shannon_entropy", [])
    try:
        pred_syn_ent = float(
            getattr(tracker, "predict_synergy_metric")("shannon_entropy")
        )
    except Exception:
        pred_syn_ent = 0.0
    vals_ent = syn_ent_vals[-3:]
    if pred_syn_ent:
        vals_ent.append(pred_syn_ent)
    if vals_ent:
        avg_syn_ent = sum(vals_ent) / len(vals_ent)
        if avg_syn_ent > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
        elif avg_syn_ent < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )

    syn_flex_vals = tracker.metrics_history.get("synergy_flexibility", [])
    try:
        pred_syn_flex = float(getattr(tracker, "predict_synergy_metric")("flexibility"))
    except Exception:
        pred_syn_flex = 0.0
    vals_flex = syn_flex_vals[-3:]
    if pred_syn_flex:
        vals_flex.append(pred_syn_flex)
    if vals_flex:
        avg_syn_flex = sum(vals_flex) / len(vals_flex)
        if avg_syn_flex > 0.05:
            for p in presets:
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_flex < -0.05:
            for p in presets:
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_energy_vals = tracker.metrics_history.get("synergy_energy_consumption", [])
    try:
        pred_syn_energy = float(
            getattr(tracker, "predict_synergy_metric")("energy_consumption")
        )
    except Exception:
        pred_syn_energy = 0.0
    vals_energy = syn_energy_vals[-3:]
    if pred_syn_energy:
        vals_energy.append(pred_syn_energy)
    if vals_energy:
        avg_syn_energy = sum(vals_energy) / len(vals_energy)
        if avg_syn_energy > 0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[-1]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )
        elif avg_syn_energy < -0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )

    syn_profit_vals = tracker.metrics_history.get("synergy_profitability", [])
    try:
        pred_syn_profit = float(
            getattr(tracker, "predict_synergy_metric")("profitability")
        )
    except Exception:
        pred_syn_profit = 0.0
    vals_profit = syn_profit_vals[-3:]
    if pred_syn_profit:
        vals_profit.append(pred_syn_profit)
    if vals_profit:
        avg_syn_profit = sum(vals_profit) / len(vals_profit)
        if avg_syn_profit > 0.05:
            for p in presets:
                cur = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, cur, True)
        elif avg_syn_profit < -0.05:
            for p in presets:
                cur = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, cur, False)

    syn_rev_vals = tracker.metrics_history.get("synergy_revenue", [])
    try:
        pred_syn_rev = float(getattr(tracker, "predict_synergy_metric")("revenue"))
    except Exception:
        pred_syn_rev = 0.0
    vals_rev = syn_rev_vals[-3:]
    if pred_syn_rev:
        vals_rev.append(pred_syn_rev)
    if vals_rev:
        avg_syn_rev = sum(vals_rev) / len(vals_rev)
        for p in presets:
            mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
            if avg_syn_rev > 0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
            elif avg_syn_rev < -0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)

    syn_lucr_vals = tracker.metrics_history.get("synergy_projected_lucrativity", [])
    try:
        pred_syn_lucr = float(
            getattr(tracker, "predict_synergy_metric")("projected_lucrativity")
        )
    except Exception:
        pred_syn_lucr = 0.0
    vals_lucr = syn_lucr_vals[-3:]
    if pred_syn_lucr:
        vals_lucr.append(pred_syn_lucr)
    if vals_lucr:
        avg_syn_lucr = sum(vals_lucr) / len(vals_lucr)
        if avg_syn_lucr > 0.05:
            for p in presets:
                cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, True)
        elif avg_syn_lucr < -0.05:
            for p in presets:
                cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, False)

    # ----------------------------------------------
    # New synergy metrics for maintainability, code quality
    syn_maint_vals = tracker.metrics_history.get("synergy_maintainability", [])
    try:
        pred_syn_maint = float(
            getattr(tracker, "predict_synergy_metric")("maintainability")
        )
    except Exception:
        pred_syn_maint = 0.0
    vals_maint = syn_maint_vals[-3:]
    if pred_syn_maint:
        vals_maint.append(pred_syn_maint)
    if vals_maint:
        avg_syn_maint = sum(vals_maint) / len(vals_maint)
        for p in presets:
            cpu = str(p.get("CPU_LIMIT", "1"))
            if avg_syn_maint > 0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cpu, False)
            elif avg_syn_maint < -0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cpu, True)

    syn_cq_vals = tracker.metrics_history.get("synergy_code_quality", [])
    try:
        pred_syn_cq = float(getattr(tracker, "predict_synergy_metric")("code_quality"))
    except Exception:
        pred_syn_cq = 0.0
    vals_cq = syn_cq_vals[-3:]
    if pred_syn_cq:
        vals_cq.append(pred_syn_cq)
    if vals_cq:
        avg_syn_cq = sum(vals_cq) / len(vals_cq)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_cq > 0.05:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_cq < -0.05:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_lat_vals = tracker.metrics_history.get("synergy_network_latency", [])
    try:
        pred_syn_lat = float(
            getattr(tracker, "predict_synergy_metric")("network_latency")
        )
    except Exception:
        pred_syn_lat = 0.0
    vals_lat = syn_lat_vals[-3:]
    if pred_syn_lat:
        vals_lat.append(pred_syn_lat)
    if vals_lat:
        avg_syn_lat = sum(vals_lat) / len(vals_lat)
        for p in presets:
            lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
            if avg_syn_lat > 1.0:
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, True)
            elif avg_syn_lat < -1.0:
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, False)

    syn_tp_vals = tracker.metrics_history.get("synergy_throughput", [])
    try:
        pred_syn_tp = float(getattr(tracker, "predict_synergy_metric")("throughput"))
    except Exception:
        pred_syn_tp = 0.0
    vals_tp = syn_tp_vals[-3:]
    if pred_syn_tp:
        vals_tp.append(pred_syn_tp)
    if vals_tp:
        avg_syn_tp = sum(vals_tp) / len(vals_tp)
        for p in presets:
            bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[0]))
            if avg_syn_tp > 5.0:
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, True)
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
            elif avg_syn_tp < -5.0:
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, False)
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )

    # ------------------------------------------------------------------
    # Additional synergy metrics reacting to usage and discrepancies
    syn_disc_vals = tracker.metrics_history.get("synergy_discrepancy_count", [])
    try:
        pred_syn_disc = float(
            getattr(tracker, "predict_synergy_metric")("discrepancy_count")
        )
    except Exception:
        pred_syn_disc = 0.0
    vals_disc = syn_disc_vals[-3:]
    if pred_syn_disc:
        vals_disc.append(pred_syn_disc)
    if vals_disc:
        avg_syn_disc = sum(vals_disc) / len(vals_disc)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_disc > 1.0:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_disc < -1.0:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_gpu_vals = tracker.metrics_history.get("synergy_gpu_usage", [])
    try:
        pred_syn_gpu = float(getattr(tracker, "predict_synergy_metric")("gpu_usage"))
    except Exception:
        pred_syn_gpu = 0.0
    vals_gpu = syn_gpu_vals[-3:]
    if pred_syn_gpu:
        vals_gpu.append(pred_syn_gpu)
    if vals_gpu:
        avg_syn_gpu = sum(vals_gpu) / len(vals_gpu)
        for p in presets:
            cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
            if avg_syn_gpu > 0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, True)
            elif avg_syn_gpu < -0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, False)

    syn_cpu_vals = tracker.metrics_history.get("synergy_cpu_usage", [])
    try:
        pred_syn_cpu = float(getattr(tracker, "predict_synergy_metric")("cpu_usage"))
    except Exception:
        pred_syn_cpu = 0.0
    vals_cpu = syn_cpu_vals[-3:]
    if pred_syn_cpu:
        vals_cpu.append(pred_syn_cpu)
    if vals_cpu:
        avg_syn_cpu = sum(vals_cpu) / len(vals_cpu)
        for p in presets:
            cur = str(p.get("CPU_LIMIT", "1"))
            if avg_syn_cpu > 0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cur, True)
            elif avg_syn_cpu < -0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cur, False)

    syn_mem_vals = tracker.metrics_history.get("synergy_memory_usage", [])
    try:
        pred_syn_mem = float(getattr(tracker, "predict_synergy_metric")("memory_usage"))
    except Exception:
        pred_syn_mem = 0.0
    vals_mem = syn_mem_vals[-3:]
    if pred_syn_mem:
        vals_mem.append(pred_syn_mem)
    if vals_mem:
        avg_syn_mem = sum(vals_mem) / len(vals_mem)
        for p in presets:
            mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
            if avg_syn_mem > 0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
            elif avg_syn_mem < -0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)

    syn_long_vals = tracker.metrics_history.get("synergy_long_term_lucrativity", [])
    try:
        pred_syn_long = float(
            getattr(tracker, "predict_synergy_metric")("long_term_lucrativity")
        )
    except Exception:
        pred_syn_long = 0.0
    vals_long = syn_long_vals[-3:]
    if pred_syn_long:
        vals_long.append(pred_syn_long)
    if vals_long:
        avg_syn_long = sum(vals_long) / len(vals_long)
        for p in presets:
            gpu = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
            disk = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
            if avg_syn_long > 0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, gpu, True)
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, disk, True)
            elif avg_syn_long < -0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, gpu, False)
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, disk, False)

    adapt_presets.last_actions = actions

    return presets


def export_preset_policy() -> Dict[tuple[int, ...], Dict[int, float]]:
    """Return policy data from the active RL agent."""

    agent = getattr(adapt_presets, "_rl_agent", None)
    return agent.export_policy() if agent else {}


def import_preset_policy(data: Dict[tuple[int, ...], Dict[int, float]]) -> None:
    """Load policy ``data`` into the active RL agent if available."""

    agent = getattr(adapt_presets, "_rl_agent", None)
    if agent:
        agent.import_policy(data)


def generate_presets_from_history(
    data_dir: str = "sandbox_data", count: int | None = None
) -> List[Dict[str, Any]]:
    """Return presets adapted using ROI/security history from ``data_dir``."""
    try:
        data_dir = resolve_path(data_dir)
    except FileNotFoundError:
        data_dir = Path(data_dir)
    history_path = data_dir / "roi_history.json"
    try:
        history = str(resolve_path(history_path))
    except FileNotFoundError:
        history = history_path.as_posix()
    tracker = None
    if Path(history).exists():
        try:
            from .roi_tracker import ROITracker

            tracker = ROITracker()
            tracker.load_history(history)
        except Exception:
            tracker = None
    presets = generate_presets(count)
    source = "static generation"
    if tracker and tracker.metrics_history.get("security_score"):
        try:
            rl_agent = getattr(adapt_presets, "_rl_agent", None)
            source = "RL agent" if rl_agent else "history adaptation"
            logger.info(
                "adapting %d presets via %s (policy=%s)",
                len(presets),
                source,
                getattr(getattr(rl_agent, "policy", None), "path", "n/a"),
            )
            presets = adapt_presets(tracker, presets)
        except Exception as exc:
            logger.exception(
                "preset evolution failed for %s using %d presets: %s",
                history,
                len(presets),
                exc,
            )
    logger.debug(
        "generate_presets_from_history using %s from %s",
        source,
        history,
    )
    return presets


__all__.append("adapt_presets")
__all__.extend([
    "export_preset_policy",
    "import_preset_policy",
    "generate_presets_from_history",
])
