# flake8: noqa
from __future__ import annotations

"""Execute Menace patches in a controlled sandbox environment.

The module initialises :data:`GLOBAL_ROUTER` early via :func:`init_db_router`
so that subsequent imports can perform database operations safely. Repository
location and related settings are provided by :mod:`sandbox_runner.config`,
which reads ``SANDBOX_REPO_URL`` and ``SANDBOX_REPO_PATH`` from environment
variables or a ``SandboxSettings`` instance.
"""

import importlib.util
import logging
import os
import shutil
import sys
import signal
import uuid
from typing import TYPE_CHECKING

from db_router import init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope
from dynamic_path_router import resolve_path, repo_root, path_for_prompt

# Initialise a router for this process with a unique menace_id so
# ``GLOBAL_ROUTER`` becomes available to imported modules.  Import modules that
# require database access afterwards so they can rely on ``GLOBAL_ROUTER``. Local
# code uses the placeholder ``router`` which defaults to ``None`` to preserve
# legacy behaviour where sandbox utilities operate without a database connection
# when possible. All DB access must go through the router.
MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)
router = GLOBAL_ROUTER

from log_tags import INSIGHT, IMPROVEMENT_PATH, FEEDBACK, ERROR_FIX
from memory_aware_gpt_client import ask_with_memory
from shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE, LocalKnowledgeModule
from vector_service import FallbackResult, ContextBuilder
try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service lacks explicit class."""

        pass


def _verify_required_dependencies(settings: "SandboxSettings | None" = None) -> None:
    """Exit if required or production optional dependencies are missing.

    Dependency lists are drawn from :class:`SandboxSettings` so deployments can
    override them via configuration or environment variables.  Invalid entries
    are ignored with a warning.
    """

    def _have_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    if settings is None:
        try:
            from sandbox_settings import SandboxSettings, load_sandbox_settings

            path = os.getenv("SANDBOX_SETTINGS_PATH")
            settings = (
                load_sandbox_settings(path) if path else SandboxSettings()
            )
        except Exception:  # pragma: no cover - fallback when config unavailable
            settings = None

    def _clean_list(name: str, items: list[str] | None) -> list[str]:
        valid: list[str] = []
        invalid: list[str] = []
        for item in items or []:
            if isinstance(item, str) and item.strip():
                valid.append(item.strip())
            else:
                invalid.append(str(item))
        if invalid:
            logging.warning(
                "Ignoring unrecognised %s entries: %s", name, ", ".join(invalid)
            )
        return valid

    req_tools = _clean_list(
        "system tool",
        getattr(settings, "required_system_tools", ["ffmpeg", "tesseract", "qemu-system-x86_64"]),
    )
    req_pkgs = _clean_list(
        "python package",
        getattr(
            settings,
            "required_python_packages",
            ["pydantic", "dotenv", "foresight_tracker", "filelock"],
        ),
    )
    opt_pkgs = _clean_list(
        "optional python package",
        getattr(
            settings,
            "optional_python_packages",
            [
                "matplotlib",
                "statsmodels",
                "uvicorn",
                "fastapi",
                "sklearn",
                "stripe",
                "httpx",
            ],
        ),
    )

    missing_sys = [t for t in req_tools if shutil.which(t) is None]
    missing_req = [p for p in req_pkgs if not _have_spec(p)]
    missing_opt = [p for p in opt_pkgs if not _have_spec(p)]

    mode = os.getenv("MENACE_MODE", "test").lower()

    messages: list[str] = []
    if missing_sys:
        messages.append("Missing system packages: " + ", ".join(missing_sys))
        pkg_line = " ".join(missing_sys)
        messages.append("Install them on Debian/Ubuntu with:")
        messages.append(f"  sudo apt-get install {pkg_line}")
        messages.append("Or on macOS with:")
        messages.append(f"  brew install {pkg_line}")
    if missing_req:
        messages.append(
            "Missing Python packages: "
            + ", ".join(missing_req)
            + ". Install them with 'pip install <package>'."
        )
    if missing_opt and mode == "production":
        messages.append(
            "Missing optional Python packages: "
            + ", ".join(missing_opt)
            + ". Install them with 'pip install <package>'."
        )

    if messages:
        messages.append(
            f"Refer to {path_for_prompt('docs/autonomous_sandbox.md')} for manual setup instructions."
        )
        msg = "\n".join(messages)
        logging.error(msg)
        raise SystemExit(msg)

    if missing_opt:
        logging.warning("Missing optional Python packages: %s", ", ".join(missing_opt))


_verify_required_dependencies()

import json
import os
import re
import subprocess
import tempfile
import concurrent.futures
import multiprocessing
import asyncio
import time
import textwrap
import argparse
import logging
from logging_utils import log_record, get_logger, setup_logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING, Iterable
from menace.unified_event_bus import UnifiedEventBus
from menace.menace_orchestrator import MenaceOrchestrator
from menace.self_improvement_policy import SelfImprovementPolicy
from menace.self_improvement import SelfImprovementEngine
from menace.patch_score_backend import backend_from_url
from menace.self_test_service import SelfTestService
from menace.code_database import PatchHistoryDB, CodeDB
try:  # patch suggestion DB may reside at top level during tests
    from menace.patch_suggestion_db import PatchSuggestionDB
except Exception:  # pragma: no cover - fallback for test stubs
    from patch_suggestion_db import PatchSuggestionDB
from menace.audit_trail import AuditTrail
from menace.error_bot import ErrorBot, ErrorDB
from menace.data_bot import MetricsDB, DataBot
from menace.composite_workflow_scorer import CompositeWorkflowScorer
from menace.neuroplasticity import PathwayDB
from sandbox_runner.metrics_plugins import (
    discover_metrics_plugins,
    load_metrics_plugins,
)
from sandbox_runner.orphan_discovery import (
    discover_orphan_modules,
    discover_recursive_orphans,
)
from menace.discrepancy_detection_bot import DiscrepancyDetectionBot
from jinja2 import Template
from sandbox_settings import SandboxSettings
from menace.error_logger import ErrorLogger
from menace.knowledge_graph import KnowledgeGraph
from menace.error_forecaster import ErrorForecaster
from menace.quick_fix_engine import QuickFixEngine
try:  # optional metrics exporter
    from metrics_exporter import (
        sandbox_cpu_percent,
        sandbox_memory_mb,
        sandbox_crashes_total,
    )
except Exception:  # pragma: no cover - metrics exporter missing
    sandbox_cpu_percent = sandbox_memory_mb = sandbox_crashes_total = None

try:  # optional system metrics
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil not installed
    psutil = None

from foresight_tracker import ForesightTracker
from relevancy_metrics_db import RelevancyMetricsDB
from relevancy_radar import scan as relevancy_radar_scan, radar
from sandbox_runner.cycle import _async_track_usage
try:  # telemetry optional
    from sandbox_runner.meta_logger import _SandboxMetaLogger
except ImportError as exc:  # pragma: no cover - meta logger missing
    _SandboxMetaLogger = None  # type: ignore
    get_logger(__name__).warning(
        "sandbox meta logging unavailable: %s", exc
    )

try:
    from menace.pre_execution_roi_bot import PreExecutionROIBot
except Exception:  # pragma: no cover - optional dependency
    PreExecutionROIBot = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from menace.roi_tracker import ROITracker

__path__ = [resolve_path("sandbox_runner").as_posix()]
logger = get_logger(__name__)

def _get_local_knowledge() -> LocalKnowledgeModule:
    """Return the process-wide :class:`LocalKnowledgeModule` instance."""

    return LOCAL_KNOWLEDGE_MODULE
LOCAL_KNOWLEDGE_REFRESH_EVERY = int(
    os.getenv("LOCAL_KNOWLEDGE_REFRESH_EVERY", "10")
)
_local_knowledge_refresh_counter = 0

import sandbox_runner.config as sandbox_config

_TPL_PATH = Path(
    os.getenv(
        "GPT_SECTION_TEMPLATE",
        str(resolve_path("templates/gpt_section_prompt.j2")),
    )
)
_TPL: Template | None = None
_AUTO_PROMPTS_DIR = resolve_path("templates/auto_prompts")
# Loaded auto templates cached globally to avoid repeated disk reads
_AUTO_TEMPLATES: list[tuple[str, Template]] | None = None
_SUGGESTION_DB: PatchSuggestionDB | None = None
_prompt_len = os.getenv("GPT_SECTION_PROMPT_MAX_LENGTH")
try:
    GPT_SECTION_PROMPT_MAX_LENGTH: int | None = (
        int(_prompt_len) if _prompt_len else None
    )
except Exception:
    GPT_SECTION_PROMPT_MAX_LENGTH = None
_summ_depth = os.getenv("GPT_SECTION_SUMMARY_DEPTH")
try:
    GPT_SECTION_SUMMARY_DEPTH: int = int(_summ_depth) if _summ_depth else 3
except Exception:
    GPT_SECTION_SUMMARY_DEPTH = 3

from sandbox_runner.environment import (
    SANDBOX_EXTRA_METRICS,
    SANDBOX_ENV_PRESETS,
    simulate_execution_environment,
    simulate_full_environment,
    generate_sandbox_report,
    run_repo_section_simulations,
    run_workflow_simulations,
    run_scenarios as _env_run_scenarios,
    simulate_temporal_trajectory as _simulate_temporal_trajectory,
    _section_worker,
    validate_preset,
)
from sandbox_runner.cycle import _sandbox_cycle_runner, map_module_identifier
from sandbox_runner.cli import _run_sandbox, rank_scenarios, main
from meta_workflow_planner import simulate_meta_workflow as _simulate_meta_workflow


# ----------------------------------------------------------------------
def run_scenarios(workflow, tracker=None, foresight_tracker=None, presets=None):
    """Proxy to :func:`sandbox_runner.environment.run_scenarios`.

    Exposes the scenario runner at the top-level ``sandbox_runner`` module so
    callers can simply ``from sandbox_runner import run_scenarios``. All
    arguments are forwarded to
    :func:`sandbox_runner.environment.run_scenarios`.
    """

    return _env_run_scenarios(
        workflow, tracker=tracker, foresight_tracker=foresight_tracker, presets=presets
    )


# ----------------------------------------------------------------------
def simulate_temporal_trajectory(workflow_id, workflow, tracker=None, foresight_tracker=None):
    """Proxy to :func:`sandbox_runner.environment.simulate_temporal_trajectory`.

    Exposes the temporal trajectory simulator at the top-level ``sandbox_runner``
    module so callers can simply ``from sandbox_runner import
    simulate_temporal_trajectory``. All arguments are forwarded to
    :func:`sandbox_runner.environment.simulate_temporal_trajectory`.
    """

    return _simulate_temporal_trajectory(
        workflow_id,
        workflow,
        tracker=tracker,
        foresight_tracker=foresight_tracker,
    )


# ----------------------------------------------------------------------
def simulate_meta_workflow(meta_spec, workflows=None, runner=None):
    """Proxy to :func:`meta_workflow_planner.simulate_meta_workflow`."""

    return _simulate_meta_workflow(meta_spec, workflows=workflows, runner=runner)


# ----------------------------------------------------------------------
def load_modified_code(code_path: str) -> str:
    path = resolve_path(code_path)
    logger.debug("loading code from %s", path_for_prompt(path))
    with path.open("r", encoding="utf-8") as fh:
        content = fh.read()
    logger.debug("loaded %d bytes from %s", len(content), path_for_prompt(path))
    return content


# ----------------------------------------------------------------------
def scan_repo_sections(
    repo_path: str, modules: Iterable[str] | None = None
) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping of module -> section name -> lines.

    When ``modules`` is provided only those relative paths within ``repo_path``
    are traversed. Each entry may be a directory or a specific file path.
    """
    from menace.codebase_diff_checker import _extract_sections

    repo_path = resolve_path(repo_path)
    sections: Dict[str, Dict[str, List[str]]] = {}
    targets: List[str] = []

    if modules:
        for mod in modules:
            root = repo_path / mod
            if root.is_dir():
                for base, _, files in os.walk(root):
                    for name in files:
                        if name.endswith(".py"):
                            rel = os.path.relpath(Path(base) / name, repo_path)
                            targets.append(rel)
            elif root.suffix == ".py" and root.is_file():
                targets.append(os.path.relpath(root, repo_path))
    else:
        for base, _, files in os.walk(repo_path):
            for name in files:
                if name.endswith(".py"):
                    rel = os.path.relpath(Path(base) / name, repo_path)
                    targets.append(rel)

    for rel in targets:
        path = repo_path / rel
        try:
            sections[rel] = _extract_sections(path)
        except Exception:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    sections[rel] = {"__file__": fh.read().splitlines()}
            except Exception:
                sections[rel] = {}

    return sections


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
def run_relevancy_radar_scan(
    settings: SandboxSettings | None = None,
) -> Dict[str, str]:
    """Run an on-demand relevancy radar scan and return flagged modules."""

    settings = settings or SandboxSettings()
    if not getattr(settings, "enable_relevancy_radar", True):
        logger.info("relevancy radar disabled")
        return {}

    db_path = resolve_path(settings.sandbox_data_dir) / "relevancy_metrics.db"
    flags = relevancy_radar_scan(
        db_path=db_path,
        min_calls=settings.relevancy_radar_min_calls,
        compress_ratio=settings.relevancy_radar_compress_ratio,
        replace_ratio=settings.relevancy_radar_replace_ratio,
    )

    retention = settings.relevancy_metrics_retention_days
    if retention is not None:
        cutoff = time.time() - retention * 86400
        try:
            if db_path.exists() and db_path.stat().st_mtime < cutoff:
                db_path.unlink()
        except Exception:
            logger.exception("failed to apply relevancy radar retention policy")

    return flags


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
def prepare_snippet(
    snippet: str,
    *,
    max_length: int = 1000,
    summary_depth: int = GPT_SECTION_SUMMARY_DEPTH,
) -> tuple[str, str]:
    """Return a truncated or summarised snippet and extracted comment summary."""

    if not snippet:
        return "", ""

    comments = re.findall(r"(?m)^\s*#\s*(.*)", snippet)
    docstrings = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', snippet, re.S)
    docstrings = [d[0] or d[1] for d in docstrings]
    text = " ".join(comments + docstrings).strip()
    summary = text[: summary_depth * 50] if text else ""

    if len(snippet) > max_length:
        lines = snippet.splitlines()
        depth = max(1, summary_depth)
        head = lines[:depth]
        tail = lines[-depth:] if len(lines) > depth else []
        snippet = "\n".join(head + ["# ..."] + tail)
        if len(snippet) > max_length:
            snippet = snippet[:max_length]

    return snippet, summary


# ----------------------------------------------------------------------
def build_section_prompt(
    section: str,
    tracker: "ROITracker",
    snippet: str | None = None,
    prior: str | None = None,
    *,
    max_length: int = 1000,
    max_prompt_length: int | None = GPT_SECTION_PROMPT_MAX_LENGTH,
    summary_depth: int = GPT_SECTION_SUMMARY_DEPTH,
) -> str:
    global _TPL, _AUTO_TEMPLATES

    if _AUTO_TEMPLATES is None:
        templates: list[tuple[str, Template]] = []
        if _AUTO_PROMPTS_DIR.exists():
            for p in sorted(_AUTO_PROMPTS_DIR.glob("*.j2")):
                try:
                    templates.append((p.stem, Template(p.read_text())))
                except Exception:
                    continue
        _AUTO_TEMPLATES = templates

    if _TPL is None:
        try:
            _TPL = Template(_TPL_PATH.read_text())
        except Exception:
            _TPL = Template(
                "ROI for {{ section }} is declining. Recent ROI: [{{ history }}]. "
                "Metrics: {{ metrics }}{% if metrics_summary %} Metrics summary: {{ metrics_summary }}{% endif %}"
                "{% if synergy %} Synergy: {{ synergy }}{% endif %}{% if synergy_summary %} Synergy summary: {{ synergy_summary }}{% endif %}"
                "{% if summary %} Purpose: {{ summary }}{% endif %}\n"
                "Suggest a concise improvement:\n{{ snippet }}"
            )

    hist = tracker.module_deltas.get(section.split(":", 1)[0], [])
    hist_str = ", ".join(f"{v:.2f}" for v in hist[-5:]) if hist else ""

    metric_values = {
        m: vals[-1]
        for m, vals in tracker.metrics_history.items()
        if vals and not m.startswith("synergy_")
    }
    metric_values["ROI"] = tracker.roi_history[-1] if tracker.roi_history else 0.0
    metric_values["RAROI"] = (
        tracker.raroi_history[-1] if tracker.raroi_history else 0.0
    )
    top_metrics = sorted(metric_values.items(), key=lambda x: abs(x[1]), reverse=True)[
        :3
    ]
    metric_str = ", ".join(f"{k}={v:.2f}" for k, v in top_metrics)

    metric_summary = []
    for name in metric_values:
        vals = tracker.metrics_history.get(name, [])
        if len(vals) >= 2:
            delta = vals[-1] - vals[-2]
        elif vals:
            delta = vals[-1]
        else:
            delta = 0.0
        metric_summary.append(f"{name}:{delta:+.2f}")
    metric_summary_str = ", ".join(metric_summary[:3])

    synergy_values = {
        m: vals[-1]
        for m, vals in tracker.metrics_history.items()
        if vals and m.startswith("synergy_")
    }
    synergy_top = sorted(synergy_values.items(), key=lambda x: abs(x[1]), reverse=True)[
        :2
    ]
    synergy_str = ", ".join(f"{k}={v:.2f}" for k, v in synergy_top)

    synergy_summary = []
    for name in synergy_values:
        vals = tracker.metrics_history.get(name, [])
        if len(vals) >= 2:
            delta = vals[-1] - vals[-2]
        elif vals:
            delta = vals[-1]
        else:
            delta = 0.0
        synergy_summary.append(f"{name}:{delta:+.2f}")
    synergy_summary_str = ", ".join(synergy_summary[:2])

    snippet_part, comment_summary = prepare_snippet(
        snippet or "",
        max_length=max_length,
        summary_depth=summary_depth,
    )

    roi_deltas = [
        round(tracker.roi_history[i] - tracker.roi_history[i - 1], 2)
        for i in range(1, len(tracker.roi_history))
    ]
    roi_deltas = roi_deltas[-3:]
    raroi_deltas = [
        round(tracker.raroi_history[i] - tracker.raroi_history[i - 1], 2)
        for i in range(1, len(tracker.raroi_history))
    ]
    raroi_deltas = raroi_deltas[-3:]
    discrepancy_hist = tracker.metrics_history.get("discrepancy_count", [])[-3:]

    tpl = _TPL

    def _delta(vals):
        if len(vals) < 2:
            return 0.0
        return vals[-1] - vals[-2]

    sec_hist = tracker.metrics_history.get("security_score", [])
    eff_hist = tracker.metrics_history.get("efficiency", [])

    if _AUTO_TEMPLATES:
        sec_drop = max(0.0, -_delta(sec_hist))
        eff_drop = max(0.0, -_delta(eff_hist))
        raroi_drop = max(0.0, -_delta(tracker.raroi_history))
        syn_deltas = [
            _delta(v) for k, v in tracker.metrics_history.items() if k.startswith("synergy_") and v
        ]
        synergy_drop = max(0.0, -sum(syn_deltas) / len(syn_deltas)) if syn_deltas else 0.0

        weights = {
            "security": 1.0,
            "efficiency": 1.0,
            "raroi": 1.0,
            "synergy": 0.5,
        }

        best_score = float("-inf")
        chosen = tpl
        for name, t in _AUTO_TEMPLATES:
            score = weights["synergy"] * synergy_drop
            if "security" in name:
                score += weights["security"] * sec_drop
            if "efficiency" in name:
                score += weights["efficiency"] * eff_drop
            if "roi" in name:
                score += weights["raroi"] * raroi_drop
            if score > best_score:
                best_score = score
                chosen = t
        tpl = chosen

    prompt = tpl.render(
        section=section,
        history=hist_str,
        metrics=metric_str,
        metrics_summary=metric_summary_str,
        synergy=synergy_str,
        synergy_summary=synergy_summary_str,
        summary=comment_summary.strip(),
        snippet=snippet_part,
        deltas=roi_deltas,
        discrepancy_history=discrepancy_hist,
        prior=prior,
    )

    if max_prompt_length and len(prompt) > max_prompt_length:
        while True:
            excess = len(prompt) - max_prompt_length
            if excess <= 0:
                break
            if snippet_part:
                trim = min(excess, len(snippet_part))
                snippet_part = snippet_part[:-trim]
                excess -= trim
            elif metric_str:
                trim = min(excess, len(metric_str))
                metric_str = metric_str[:-trim]
                excess -= trim
            elif metric_summary_str:
                trim = min(excess, len(metric_summary_str))
                metric_summary_str = metric_summary_str[:-trim]
                excess -= trim
            elif synergy_str:
                trim = min(excess, len(synergy_str))
                synergy_str = synergy_str[:-trim]
                excess -= trim
            elif synergy_summary_str:
                # keep at least the prefix; if cannot trim further break
                if len(synergy_summary_str) > 10:
                    trim = min(excess, len(synergy_summary_str) - 10)
                    synergy_summary_str = synergy_summary_str[:-trim]
                    excess -= trim
                else:
                    break
            else:
                break
            prompt = tpl.render(
                section=section,
                history=hist_str,
                metrics=metric_str,
                metrics_summary=metric_summary_str,
                synergy=synergy_str,
                synergy_summary=synergy_summary_str,
                summary=comment_summary.strip(),
                snippet=snippet_part,
                prior=prior,
            )
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length]

    return prompt



@dataclass
class SandboxContext:
    tmp: str
    repo: Path
    orig_cwd: Path
    data_dir: Path
    event_bus: UnifiedEventBus
    policy: SelfImprovementPolicy
    patch_db: PatchHistoryDB
    patch_db_path: Path
    orchestrator: MenaceOrchestrator
    improver: SelfImprovementEngine
    sandbox: Any
    tester: SelfTestService
    tracker: "ROITracker"
    foresight_tracker: ForesightTracker
    meta_log: _SandboxMetaLogger
    backups: Dict[str, Any]
    env: Dict[str, str]
    settings: SandboxSettings
    models: List[str]
    module_counts: Dict[str, int]
    changed_modules: Any
    res_db: Any
    pre_roi_bot: Any
    va_client: Any
    gpt_client: Any
    engine: Any
    dd_bot: DiscrepancyDetectionBot
    data_bot: DataBot
    pathway_db: PathwayDB
    telem_db: ErrorDB
    context_builder: ContextBuilder
    plugins: list
    extra_metrics: Dict[str, float]
    cycles: int
    base_roi_tolerance: float
    roi_tolerance: float
    prev_roi: float
    predicted_roi: float | None
    predicted_lucrativity: float | None
    brainstorm_interval: int
    brainstorm_retries: int
    patch_retries: int
    sections: Dict[str, Dict[str, List[str]]]
    all_section_names: set[str]
    roi_history_file: Path
    foresight_history_file: Path
    brainstorm_history: List[str]
    conversations: Dict[str, List[Dict[str, str]]]
    offline_suggestions: bool = False
    adapt_presets: bool = True
    suggestion_cache: Dict[str, str] = field(default_factory=dict)
    suggestion_db: PatchSuggestionDB | None = None
    synergy_needed: bool = False
    best_roi: float = 0.0
    best_synergy_metrics: Dict[str, float] = field(default_factory=dict)
    module_map: set[str] = field(default_factory=set)
    orphan_traces: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _sandbox_init(
    preset: Dict[str, Any],
    args: argparse.Namespace,
    context_builder: ContextBuilder,
) -> SandboxContext:
    import sandbox_runner.environment as env

    if not isinstance(context_builder, ContextBuilder):
        raise ValueError("context_builder must be a ContextBuilder")

    env._cleanup_pools()

    logger.info(
        "initialising sandbox",
        extra=log_record(env=dict(os.environ), preset=preset),
    )

    def _handle_signal(signum, frame) -> None:  # pragma: no cover - signal path
        env._cleanup_pools()
        env._await_cleanup_task()
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            logger.exception("signal handler setup failed")

    tmp = tempfile.mkdtemp(prefix="menace_sandbox_")
    logger.info("sandbox temporary directory", extra=log_record(path=tmp))
    repo = sandbox_config.get_sandbox_repo_path()
    orig_cwd = Path.cwd()

    data_dir = Path(
        getattr(args, "sandbox_data_dir", None)
        or os.getenv("SANDBOX_DATA_DIR", str(resolve_path("sandbox_data")))
    )
    if not data_dir.is_absolute():
        data_dir = repo_root() / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("using data directory", extra=log_record(path=str(data_dir)))
    policy_file = data_dir / "improvement_policy.pkl"
    patch_file = data_dir / "patch_history.db"
    module_map_file = data_dir / "module_map.json"

    refresh_map = bool(
        getattr(args, "refresh_module_map", False)
        or os.getenv("SANDBOX_REFRESH_MODULE_MAP") == "1"
    )
    auto_map = bool(
        getattr(args, "autodiscover_modules", False)
        or os.getenv("SANDBOX_AUTO_MAP")
        or os.getenv("SANDBOX_AUTODISCOVER_MODULES")
    )
    if not os.getenv("SANDBOX_AUTO_MAP") and os.getenv("SANDBOX_AUTODISCOVER_MODULES"):
        logger.warning(
            "SANDBOX_AUTODISCOVER_MODULES is deprecated; use SANDBOX_AUTO_MAP",
        )

    try:
        from menace.module_index_db import ModuleIndexDB  # type: ignore
    except Exception:
        ModuleIndexDB = None  # type: ignore
    module_index = ModuleIndexDB(module_map_file, auto_map=False) if ModuleIndexDB else None
    if module_index and (refresh_map or auto_map or not module_map_file.exists()):
        try:
            module_index.refresh(force=True)
            logger.info(
                "module map generated",
                extra=log_record(path=str(module_map_file)),
            )
        except Exception:
            logger.exception("module map generation failed")

    _env_vars = {
        "DATABASE_URL": "sqlite:///:memory:",
        "BOT_DB_PATH": str(Path(tmp) / "bots.db"),
        "BOT_PERFORMANCE_DB": str(Path(tmp) / "perf.db"),
        "MAINTENANCE_DB": str(Path(tmp) / "maint.db"),
    }
    preset = {k: v for k, v in preset.items() if k != "SANDBOX_ENV_PRESETS"}
    backups = {k: os.environ.get(k) for k in {**_env_vars, **preset}}
    os.environ.update(_env_vars)
    os.environ.update({k: str(v) for k, v in preset.items()})
    logger.debug("temporary env vars set: %s", _env_vars)

    event_bus = UnifiedEventBus(persist_path=str(Path(tmp) / "events.db"))
    logger.info("event bus initialised")

    relevancy_db = RelevancyMetricsDB(data_dir / "relevancy_metrics.db")
    meta_log = _SandboxMetaLogger(data_dir / "sandbox_meta.log")
    meta_log.module_index = module_index
    meta_log.metrics_db = relevancy_db
    metrics_db = MetricsDB(data_dir / "metrics.db")
    pathway_db = PathwayDB(data_dir / "pathways.db")
    data_bot = DataBot(metrics_db)
    dd_bot = DiscrepancyDetectionBot()
    module_counts: Dict[str, int] = {}
    patch_db_path = patch_file

    def _changed_modules(last_id: int) -> tuple[list[str], int]:
        if not patch_db_path.exists():
            return [], last_id

        try:
            with router.get_connection("patch_history") as conn:
                rows = conn.execute(
                    "SELECT id, filename FROM patch_history WHERE id>?",
                    (last_id,),
                ).fetchall()
        except Exception as exc:
            logger.exception("patch history read failed: %s", exc)
            return [], last_id
        modules = [str(r[1]) for r in rows]
        new_last = max([last_id, *[r[0] for r in rows]]) if rows else last_id
        return modules, new_last

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    models = env.get("MODELS", "demo").split(",")
    os.chdir(repo)
    orchestrator = MenaceOrchestrator()
    orchestrator.create_oversight("root", "L1")
    policy = SelfImprovementPolicy(path=str(policy_file))
    patch_db = PatchHistoryDB(patch_db_path)
    score_backend = None
    backend_url = os.getenv("PATCH_SCORE_BACKEND_URL")
    if backend_url:
        try:
            score_backend = backend_from_url(backend_url)
        except Exception:
            logger.exception("patch score backend init failed")
    gpt_memory = _get_local_knowledge().memory
    improver = SelfImprovementEngine(
        meta_logger=meta_log,
        module_index=module_index,
        patch_db=patch_db,
        policy=policy,
        score_backend=score_backend,
        auto_refresh_map=bool(getattr(args, "refresh_module_map", False)),
        gpt_memory=gpt_memory,
    )
    meta_log.module_index = improver.module_index

    graph = KnowledgeGraph()
    telem_db = ErrorDB(Path(tmp) / "errors.db", graph=graph)
    error_logger = ErrorLogger(telem_db, knowledge_graph=graph)
    forecaster = ErrorForecaster(metrics_db, graph=graph)
    improver.error_bot = ErrorBot(
        telem_db,
        metrics_db,
        graph=graph,
        forecaster=forecaster,
        improvement_engine=improver,
    )
    improver.error_bot.error_logger = error_logger
    settings = SandboxSettings()

    class _TelemProxy:
        def __init__(self, db: ErrorDB) -> None:
            self.db = db

        def recent_errors(
            self,
            limit: int = 5,
            *,
            scope: Scope | str = "local",
            source_menace_id: str | None = None,
        ) -> list[str]:
            menace_id = self.db._menace_id(source_menace_id)
            clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
            query = apply_scope(
                "SELECT stack_trace FROM telemetry",
                clause,
            ) + " ORDER BY id DESC LIMIT ?"
            cur = self.db.conn.execute(query, [*params, limit])
            return [str(r[0]) for r in cur.fetchall()]

    from menace.self_coding_engine import SelfCodingEngine
    from menace.menace_memory_manager import MenaceMemoryManager
    from menace.self_debugger_sandbox import SelfDebuggerSandbox

    # ``suggestion_db`` is assigned later once paths are available but it is
    # passed into ``SelfCodingEngine`` below, so initialise it here to avoid an
    # UnboundLocalError during argument evaluation.
    suggestion_db: PatchSuggestionDB | None = None

    try:
        from menace.visual_agent_client import VisualAgentClient
    except Exception as exc:
        logger.warning("VisualAgentClient import failed: %s", exc)
        try:
            from menace.visual_agent_client import (
                VisualAgentClientStub as VisualAgentClient,
            )
        except Exception:
            VisualAgentClient = None  # type: ignore

    va_client = None
    if VisualAgentClient:
        try:
            va_client = VisualAgentClient()
        except Exception as exc:
            logger.warning("VisualAgentClient init failed: %s", exc)
            try:
                from menace.visual_agent_client import VisualAgentClientStub

                va_client = VisualAgentClientStub()
                logger.info("using VisualAgentClientStub due to failure")
            except Exception:
                va_client = None
    engine = SelfCodingEngine(
        CodeDB(),
        MenaceMemoryManager(),
        llm_client=va_client,
        patch_suggestion_db=suggestion_db,
        gpt_memory=gpt_memory,
        context_builder=context_builder,
    )
    from menace.self_coding_manager import SelfCodingManager
    from menace.model_automation_pipeline import ModelAutomationPipeline

    quick_manager = SelfCodingManager(
        engine, ModelAutomationPipeline(context_builder=context_builder), bot_name="menace"
    )
    quick_fix_engine = QuickFixEngine(
        telem_db, quick_manager, graph=graph, context_builder=context_builder
    )

    gpt_client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from menace.chatgpt_idea_bot import ChatGPTClient

            gpt_client = ChatGPTClient(
                model="gpt-4",
                gpt_memory=_get_local_knowledge().memory,
                context_builder=context_builder,
            )
        except Exception:
            logger.exception("GPT client init failed")
            gpt_client = None
    sandbox = SelfDebuggerSandbox(
        _TelemProxy(telem_db),
        engine,
        policy=policy,
        state_getter=improver._policy_state,
    )
    sandbox.context_builder = context_builder
    sandbox.error_logger = error_logger
    sandbox.error_forecaster = forecaster
    sandbox.quick_fix_engine = quick_fix_engine
    sandbox.graph = graph
    tester = SelfTestService(
        telem_db,
        include_orphans=include_orphans,
        discover_orphans=discover_orphans,
        discover_isolated=discover_isolated,
        recursive_orphans=recursive_orphans,
        recursive_isolated=recursive_isolated,
        auto_include_isolated=settings.auto_include_isolated,
        integration_callback=lambda mods: SelfImprovementEngine._refresh_module_map(
            improver, mods
        ),
    )
    from menace.roi_tracker import ROITracker

    try:
        from menace.resources_bot import ROIHistoryDB
    except Exception:  # pragma: no cover
        ROIHistoryDB = None  # type: ignore

    res_db = None
    res_path = env.get("SANDBOX_RESOURCE_DB")
    if res_path and ROIHistoryDB:
        try:
            res_db = ROIHistoryDB(res_path)
        except Exception:
            logger.exception(
                "failed to load resource db: %s", path_for_prompt(res_path)
            )

    pre_roi_bot = None
    if PreExecutionROIBot:
        try:
            pre_roi_bot = PreExecutionROIBot(data_bot=data_bot)
            manager = getattr(pre_roi_bot, "prediction_manager", None)
            if manager is None:
                from menace.prediction_manager_bot import PredictionManager

                metric_names = PredictionManager.DEFAULT_METRIC_BOTS
                manager = PredictionManager(
                    data_bot=data_bot,
                    default_metric_bots=metric_names,
                )
                pre_roi_bot.prediction_manager = manager
                if hasattr(pre_roi_bot, "assigned_prediction_bots"):
                    try:
                        pre_roi_bot.assigned_prediction_bots = (
                            manager.assign_prediction_bots(pre_roi_bot)
                        )
                    except Exception as exc:
                        logger.exception("failed to assign prediction bots: %s", exc)
        except Exception as exc:
            logger.exception("failed to init PreExecutionROIBot")
            try:
                from menace.pre_execution_roi_bot import PreExecutionROIBotStub

                pre_roi_bot = PreExecutionROIBotStub()
                logger.info("using PreExecutionROIBotStub due to error: %s", exc)
            except Exception:
                pre_roi_bot = None
    else:
        try:
            from menace.pre_execution_roi_bot import PreExecutionROIBotStub

            pre_roi_bot = PreExecutionROIBotStub()
            logger.info("using PreExecutionROIBotStub due to missing implementation")
        except Exception:
            pre_roi_bot = None

    roi_tolerance = float(env.get("SANDBOX_ROI_TOLERANCE", "0.01"))
    entropy_threshold = float(
        env.get("SANDBOX_ENTROPY_THRESHOLD", str(roi_tolerance))
    )
    volatility_threshold = float(env.get("SANDBOX_VOLATILITY_THRESHOLD", "1.0"))
    tracker = ROITracker(
        resource_db=res_db,
        cluster_map=improver.module_clusters,
        entropy_threshold=entropy_threshold,
    )
    foresight_tracker = getattr(args, "foresight_tracker", None)
    if foresight_tracker is None:
        foresight_tracker = ForesightTracker(
            max_cycles=10,
            volatility_threshold=volatility_threshold,
        )
    roi_history_file = data_dir / "roi_history.json"
    foresight_history_file = data_dir / "foresight_history.json"
    try:
        tracker.load_history(str(roi_history_file))
    except Exception:
        logger.exception("failed to load roi history")
    if foresight_history_file.exists():
        try:
            with foresight_history_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            foresight_tracker = ForesightTracker.from_dict(
                data,
                volatility_threshold=volatility_threshold,
            )
        except Exception:
            logger.exception("failed to load foresight history")
    meta_log.module_deltas.update(tracker.module_deltas)
    prev_roi = 0.0
    predicted_roi = None
    predicted_lucrativity = None
    base_roi_tolerance = roi_tolerance
    cycles = int(env.get("SANDBOX_CYCLES", "5"))
    brainstorm_interval = int(env.get("SANDBOX_BRAINSTORM_INTERVAL", "0"))
    brainstorm_retries = int(env.get("SANDBOX_BRAINSTORM_RETRIES", "3"))
    patch_retries = int(env.get("SANDBOX_PATCH_RETRIES", "3"))

    sections = scan_repo_sections(str(repo))
    all_section_names: set[str] = set()
    for mod, sec_map in sections.items():
        for name in sec_map:
            all_section_names.add(f"{mod}:{name}")

    try:
        plugins = discover_metrics_plugins(env)
    except Exception:
        logger.exception("failed to load metrics plugins")
        plugins = []

    metrics_cfg_path = os.getenv(
        "SANDBOX_METRICS_FILE", str(resolve_path("sandbox_metrics.yaml"))
    )
    extra_metrics = SANDBOX_EXTRA_METRICS

    brainstorm_history: list[str] = []
    conversations: Dict[str, List[Dict[str, str]]] = {}
    offline_suggestions = bool(
        getattr(args, "offline_suggestions", False)
        or os.getenv("SANDBOX_OFFLINE_SUGGESTIONS", "0") == "1"
    )
    os.environ.setdefault("SANDBOX_DISCOVER_ISOLATED", "1")
    include_orphans = True
    if os.getenv("SANDBOX_DISABLE_ORPHANS") == "1":
        include_orphans = False
    env_val = os.getenv("SANDBOX_INCLUDE_ORPHANS")
    if env_val is not None:
        include_orphans = env_val.lower() in {"1", "true", "yes"}
    if getattr(args, "include_orphans") is False:
        include_orphans = False
    args.include_orphans = include_orphans
    os.environ["SANDBOX_INCLUDE_ORPHANS"] = "1" if include_orphans else "0"
    os.environ["SELF_TEST_INCLUDE_ORPHANS"] = (
        "1" if include_orphans else "0"
    )
    if not include_orphans:
        os.environ["SANDBOX_DISABLE_ORPHANS"] = "1"

    discover_orphans = True
    if os.getenv("SANDBOX_DISABLE_ORPHAN_SCAN") == "1":
        discover_orphans = False
    if getattr(args, "discover_orphans") is False:
        discover_orphans = False
    args.discover_orphans = discover_orphans
    os.environ["SELF_TEST_DISCOVER_ORPHANS"] = (
        "1" if discover_orphans else "0"
    )
    if getattr(settings, "auto_include_isolated", True):
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1"
        os.environ["SANDBOX_RECURSIVE_ISOLATED"] = "1"
    discover_isolated = True
    arg_iso = getattr(args, "discover_isolated", None)
    if arg_iso is not None:
        discover_isolated = arg_iso
    recursive_isolated = getattr(settings, "recursive_isolated", True)
    arg_rec_iso = getattr(args, "recursive_isolated", None)
    if arg_rec_iso is not None:
        recursive_isolated = arg_rec_iso
    if getattr(settings, "auto_include_isolated", True):
        recursive_isolated = True
        discover_isolated = True
    recursive_orphans = getattr(settings, "recursive_orphan_scan", True)
    arg_rec = getattr(args, "recursive_orphans", None)
    if arg_rec is not None:
        recursive_orphans = arg_rec
    args.recursive_orphans = recursive_orphans
    val = "1" if recursive_orphans else "0"
    os.environ["SANDBOX_DISABLE_ORPHAN_SCAN"] = "1" if not discover_orphans else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    os.environ["SANDBOX_RECURSIVE_ISOLATED"] = "1" if recursive_isolated else "0"
    os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1" if discover_isolated else "0"
    adapt_env = os.getenv("SANDBOX_ADAPT_PRESETS")
    if adapt_env is not None:
        adapt_presets_flag = adapt_env not in {"0", "false", "False"}
    else:
        adapt_presets_flag = not getattr(args, "no_preset_adapt", False)
    suggestion_cache: Dict[str, str] = {}
    cache_path = getattr(args, "suggestion_cache", None) or os.getenv(
        "SANDBOX_SUGGESTION_CACHE"
    )
    if cache_path:
        try:
            cache_data = json.loads(Path(cache_path).read_text())
            if isinstance(cache_data, dict):
                suggestion_cache = cache_data
        except Exception:
            logger.exception(
                "failed to load suggestion cache: %s", path_for_prompt(cache_path)
            )
            suggestion_cache = {}
    global _SUGGESTION_DB
    suggestion_db_path = suggestion_cache.get("db", str(data_dir / "module_suggestions.db"))
    if _SUGGESTION_DB is None or _SUGGESTION_DB.path != Path(suggestion_db_path):
        _SUGGESTION_DB = PatchSuggestionDB(suggestion_db_path)
    suggestion_db = _SUGGESTION_DB

    module_map_set: set[str] = set()
    try:
        if getattr(improver, "module_index", None):
            module_map_set = {
                Path(str(k)).as_posix() for k in getattr(improver.module_index, "_map", {})
            }
    except Exception:
        module_map_set = set()

    return SandboxContext(
        tmp=tmp,
        repo=repo,
        orig_cwd=orig_cwd,
        data_dir=data_dir,
        event_bus=event_bus,
        policy=policy,
        patch_db=patch_db,
        patch_db_path=patch_db_path,
        orchestrator=orchestrator,
        improver=improver,
        sandbox=sandbox,
        tester=tester,
        tracker=tracker,
        foresight_tracker=foresight_tracker,
        meta_log=meta_log,
        backups=backups,
        env=env,
        settings=settings,
        models=models,
        module_counts=module_counts,
        changed_modules=_changed_modules,
        res_db=res_db,
        pre_roi_bot=pre_roi_bot,
        va_client=va_client,
        gpt_client=gpt_client,
        engine=engine,
        dd_bot=dd_bot,
        data_bot=data_bot,
        pathway_db=pathway_db,
        telem_db=telem_db,
        context_builder=context_builder,
        plugins=plugins,
        extra_metrics=extra_metrics,
        cycles=cycles,
        base_roi_tolerance=base_roi_tolerance,
        roi_tolerance=roi_tolerance,
        prev_roi=prev_roi,
        predicted_roi=predicted_roi,
        predicted_lucrativity=predicted_lucrativity,
        brainstorm_interval=brainstorm_interval,
        brainstorm_retries=brainstorm_retries,
        patch_retries=patch_retries,
        sections=sections,
        all_section_names=all_section_names,
        roi_history_file=roi_history_file,
        foresight_history_file=foresight_history_file,
        brainstorm_history=brainstorm_history,
        conversations=conversations,
        offline_suggestions=offline_suggestions,
        adapt_presets=adapt_presets_flag,
        suggestion_cache=suggestion_cache,
        suggestion_db=suggestion_db,
        module_map=module_map_set,
        orphan_traces={},
    )


def _sandbox_cleanup(ctx: SandboxContext) -> None:
    logger.debug(
        "sandbox cleanup starting",
        extra=log_record(path=str(ctx.tmp)),
    )
    os.chdir(ctx.orig_cwd)
    ctx.policy.save()
    for k, v in ctx.backups.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    err_logger = getattr(ctx.sandbox, "error_logger", None)
    if err_logger:
        try:
            if getattr(err_logger, "replicator", None):
                err_logger.replicator.flush()
        except Exception:
            logger.exception("telemetry flush failed")
        try:
            db_conn = getattr(getattr(err_logger, "db", None), "conn", None)
            if db_conn:
                db_conn.commit()
        except Exception:
            logger.exception("telemetry db commit failed")
    try:
        ctx.telem_db.conn.commit()
    except Exception:
        logger.exception("telemetry db commit failed")
    builder = getattr(ctx, "context_builder", None)
    if builder is not None:
        try:
            close = getattr(builder, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.exception("context builder close failed")
    ctx.event_bus.close()
    shutil.rmtree(ctx.tmp)
    logger.info(
        "sandbox cleanup complete",
        extra=log_record(
            tmp_dir=str(ctx.tmp),
            data_dir=str(ctx.data_dir),
            policy_saved=bool(ctx.policy),
        ),
    )
    module = _get_local_knowledge()
    try:
        module.refresh()
        module.memory.conn.commit()
    except Exception:
        logger.exception("failed to refresh local knowledge module")


@radar.track
def _sandbox_main(preset: Dict[str, Any], args: argparse.Namespace) -> "ROITracker":
    from menace.roi_tracker import ROITracker

    global SANDBOX_ENV_PRESETS, _local_knowledge_refresh_counter
    logger.info("starting sandbox run", extra=log_record(preset=preset))
    context_builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    context_builder.refresh_db_weights()
    ctx = _sandbox_init(preset, args, context_builder)
    graph = getattr(ctx.sandbox, "graph", KnowledgeGraph())
    err_logger = getattr(
        ctx.sandbox, "error_logger", ErrorLogger(knowledge_graph=graph)
    )

    @radar.track
    def _cycle(
        section: str | None,
        snippet: str | None,
        tracker: "ROITracker",
        scenario: str | None = None,
    ) -> None:
        workflow_id = section or "workflow"
        def _record_metrics() -> None:
            if sandbox_cpu_percent is None or sandbox_memory_mb is None:
                return
            cpu = 0.0
            mem = 0.0
            if psutil:
                try:
                    cpu = float(psutil.cpu_percent())
                except Exception:
                    cpu = 0.0
                try:
                    mem = float(psutil.Process().memory_info().rss) / (1024 * 1024)
                except Exception:
                    mem = 0.0
            else:
                try:
                    if hasattr(os, "getloadavg") and os.cpu_count():
                        load = os.getloadavg()[0]
                        cpu = min(100.0, 100.0 * load / (os.cpu_count() or 1))
                except Exception:
                    cpu = 0.0
                try:
                    import resource

                    mem = float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ) / 1024
                except Exception:
                    mem = 0.0
            try:
                sandbox_cpu_percent.set(cpu)
                sandbox_memory_mb.set(mem)
            except Exception:
                pass

        def _run() -> bool:
            try:
                _sandbox_cycle_runner(ctx, section, snippet, tracker, scenario)
                return True
            except Exception as exc:
                err_logger.log(exc, section, "sandbox_runner")
                if sandbox_crashes_total is not None:
                    try:
                        sandbox_crashes_total.inc()
                    except Exception:
                        pass
                return False
            finally:
                _record_metrics()
                forecaster = getattr(ctx.sandbox, "error_forecaster", None)
                qfix = getattr(ctx.sandbox, "quick_fix_engine", None)
                if forecaster:
                    try:
                        forecaster.train()
                        bots: list[str] = []
                        try:
                            df = forecaster.metrics_db.fetch(None)
                            if hasattr(df, "empty"):
                                if not getattr(df, "empty", True):
                                    bots = list(dict.fromkeys(df["bot"].tolist()))
                            elif isinstance(df, list):
                                bots = list(
                                    dict.fromkeys(r.get("bot") for r in df if r.get("bot"))
                                )
                        except Exception:
                            bots = []
                        for b in bots:
                            try:
                                probs = forecaster.predict_error_prob(b, steps=1)
                            except Exception:
                                continue
                            if probs and probs[0] > 0.8:
                                modules: list[str] = []
                                if getattr(ctx.sandbox, "graph", None):
                                    try:
                                        chain_nodes = forecaster.predict_failure_chain(
                                            b, ctx.sandbox.graph, steps=3
                                        )
                                        modules = [
                                            n.split(":", 1)[1]
                                            for n in chain_nodes
                                            if n.startswith("module:")
                                        ]
                                    except Exception:
                                        modules = []
                                if modules:
                                    logger.info(
                                        "predicted high risk",
                                        extra=log_record(bot=b, modules=modules),
                                    )
                                if qfix:
                                    try:
                                        qfix.run(b)
                                    except Exception:
                                        logger.exception(
                                            "quick fix engine failed for %s", b
                                        )
                    except Exception:
                        logger.exception("error forecasting failed")

        scorer = CompositeWorkflowScorer(
            ctx.data_bot.db, ctx.pathway_db, tracker=tracker
        )
        scorer.score_workflow(workflow_id, {workflow_id: _run})

    switched = False
    section_results: dict[str, dict[str, list]] = {}
    for mod, sec_map in ctx.sections.items():
        for name, lines in sec_map.items():
            section_name = f"{mod}:{name}"
            if section_name in ctx.meta_log.flagged_sections:
                continue
            logger.info("processing section", extra=log_record(section=section_name))
            snippet = "\n".join(lines[:5])
            section_trackers: list[ROITracker] = []
            for p_idx, env_preset in enumerate(SANDBOX_ENV_PRESETS):
                if not validate_preset(env_preset):
                    continue
                env_updates = {
                    k: v for k, v in env_preset.items() if k != "SANDBOX_ENV_PRESETS"
                }
                scenario = env_preset.get("SCENARIO_NAME", f"scenario_{p_idx}")
                logger.info(
                    "running scenario",
                    extra=log_record(section=section_name, scenario=scenario),
                )
                backups_p = {k: os.environ.get(k) for k in env_updates}
                os.environ.update({k: str(v) for k, v in env_updates.items()})
                ctx.prev_roi = 0.0
                ctx.predicted_roi = None
                sec_tracker = ROITracker(
                    resource_db=ctx.res_db,
                    cluster_map=ctx.improver.module_clusters,
                    entropy_threshold=entropy_threshold,
                )
                _cycle(section_name, snippet, sec_tracker, scenario)
                if LOCAL_KNOWLEDGE_REFRESH_EVERY > 0:
                    _local_knowledge_refresh_counter += 1
                    if (
                        _local_knowledge_refresh_counter
                        % LOCAL_KNOWLEDGE_REFRESH_EVERY
                        == 0
                    ):
                        module = _get_local_knowledge()
                        try:
                            module.refresh()
                            module.memory.conn.commit()
                        except Exception:
                            logger.exception(
                                "failed to refresh local knowledge module"
                            )
                section_trackers.append(sec_tracker)
                sec_res = section_results.setdefault(
                    section_name, {"roi": [], "metrics": []}
                )
                sec_res["roi"].append(
                    sec_tracker.roi_history[-1] if sec_tracker.roi_history else 0.0
                )
                sec_res["metrics"].append(
                    {k: v[-1] for k, v in sec_tracker.metrics_history.items()}
                )
                for k, v in backups_p.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                if ctx.meta_log.flagged_sections >= ctx.all_section_names:
                    switched = True
                    break
            if section_trackers and ctx.adapt_presets:
                try:
                    from menace.environment_generator import adapt_presets

                    agg = ROITracker(entropy_threshold=entropy_threshold)
                    for t in section_trackers:
                        agg.roi_history.extend(t.roi_history)
                        for m, vals in t.metrics_history.items():
                            agg.metrics_history.setdefault(m, []).extend(vals)
                    thr = agg.diminishing()
                    e_thr = ctx.settings.entropy_plateau_threshold or thr
                    e_consec = ctx.settings.entropy_plateau_consecutive or 3
                    flagged = ctx.meta_log.diminishing(
                        thr, consecutive=e_consec, entropy_threshold=e_thr
                    )
                    if flagged:
                        SANDBOX_ENV_PRESETS = adapt_presets(agg, SANDBOX_ENV_PRESETS)
                        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(SANDBOX_ENV_PRESETS)
                except Exception:
                    logger.exception("preset adaptation failed")
            if switched:
                break
        if switched:
            break

    if ctx.meta_log.flagged_sections:
        from statistics import mean

        roi_sum = 0.0
        metric_totals: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for name in ctx.meta_log.flagged_sections:
            res = section_results.get(name)
            if not res:
                continue
            if res["roi"]:
                roi_sum += mean(res["roi"])
            for metric_dict in res["metrics"]:
                for m, val in metric_dict.items():
                    metric_totals[m] = metric_totals.get(m, 0.0) + float(val)
                    metric_counts[m] = metric_counts.get(m, 0) + 1

        avg_metrics = {
            m: metric_totals[m] / metric_counts[m]
            for m in metric_totals
            if metric_counts.get(m)
        }
        synergy_history: list[dict[str, float]] = ctx.tracker.synergy_history
        while True:
            synergy_tracker = ROITracker(
                resource_db=ctx.res_db,
                cluster_map=ctx.improver.module_clusters,
                entropy_threshold=entropy_threshold,
            )
            ctx.prev_roi = 0.0
            ctx.predicted_roi = None
            _cycle(None, None, synergy_tracker)
            combined_roi = (
                synergy_tracker.roi_history[-1] if synergy_tracker.roi_history else 0.0
            )
            combined_metrics = {
                k: v[-1] for k, v in synergy_tracker.metrics_history.items()
            }
            synergy_metrics = {
                f"synergy_{k}": combined_metrics.get(k, 0.0) - avg_metrics.get(k, 0.0)
                for k in set(avg_metrics) | set(combined_metrics)
            }
            synergy_metrics["synergy_recovery_time"] = combined_metrics.get(
                "recovery_time", 0.0
            ) - avg_metrics.get("recovery_time", 0.0)
            synergy_metrics["synergy_roi"] = combined_roi - roi_sum
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
            synergy_metrics.setdefault(
                "synergy_risk_index",
                combined_metrics.get("risk_index", 0.0)
                - avg_metrics.get("risk_index", 0.0),
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
            for m in (
                "shannon_entropy",
                "flexibility",
                "energy_consumption",
                "efficiency",
                "antifragility",
                "resilience",
                "safety_rating",
                "security_score",
            ):
                synergy_metrics.setdefault(
                    f"synergy_{m}",
                    combined_metrics.get(m, 0.0) - avg_metrics.get(m, 0.0),
                )
            try:
                predicted = ctx.tracker.predict_synergy()
                logger.info(
                    "synergy prediction",
                    extra=log_record(
                        roi=predicted,
                        module=section_name,
                        actual_synergy_roi=synergy_metrics.get("synergy_roi", 0.0),
                    ),
                )
            except Exception:
                logger.exception("synergy prediction failed")
            ctx.tracker.register_metrics(*synergy_metrics.keys())
            for name, val in synergy_metrics.items():
                try:
                    pred, _ = ctx.tracker.forecast_metric(name)
                    ctx.tracker.record_metric_prediction(name, pred, float(val))
                except Exception:
                    pass
            ctx.tracker.update(
                roi_sum,
                combined_roi,
                [
                    map_module_identifier(m, ctx.repo, combined_roi)
                    for m in ctx.meta_log.flagged_sections
                ],
                metrics=synergy_metrics,
            )
            synergy_history.append(synergy_metrics)
            ctx.best_roi = max(ctx.best_roi, combined_roi)
            for k, v in synergy_metrics.items():
                prev = ctx.best_synergy_metrics.get(k)
                if prev is None or v > prev:
                    ctx.best_synergy_metrics[k] = v
            stall = False
            reliability = ctx.tracker.reliability(metric="synergy_roi")
            synergy_mae = ctx.tracker.synergy_reliability()
            thr_scale = min(1.0, synergy_mae)
            threshold = max(
                ctx.tracker.diminishing() * thr_scale,
                ctx.tracker.diminishing() * 0.1,
            )
            if len(synergy_history) >= 2:
                prev = synergy_history[-2]
                roi_delta = synergy_metrics.get("synergy_roi", 0.0) - prev.get(
                    "synergy_roi", 0.0
                )
                metric_delta = max(
                    synergy_metrics.get(k, 0.0) - prev.get(k, 0.0)
                    for k in synergy_metrics
                )
                if reliability >= 0.8 and roi_delta <= threshold and metric_delta <= threshold:
                    break
                stall = (
                    roi_delta <= threshold
                    and metric_delta <= threshold
                )
                drop = synergy_metrics.get("synergy_roi", 0.0) < prev.get(
                    "synergy_roi", 0.0
                )
                if not drop:
                    for k, v in synergy_metrics.items():
                        if v < prev.get(k, v):
                            drop = True
                            break
                stall = stall or drop
            if stall and ctx.gpt_client:
                try:
                    summary = "; ".join(
                        f"{k}:{combined_metrics.get(k, 0.0)}"
                        for k in sorted(combined_metrics)
                    )
                    prior = "; ".join(ctx.brainstorm_history[-3:])
                    prompt = build_section_prompt(
                        "overall",
                        ctx.tracker,
                        f"Brainstorm improvements. Current metrics: {summary}",
                        prior=prior if prior else None,
                        max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                    )
                    hist = ctx.conversations.get("brainstorm", [])
                    module = _get_local_knowledge()
                    builder = getattr(ctx, "context_builder", None)
                    mem_ctx = ""
                    if builder is not None:
                        cb_session = uuid.uuid4().hex
                        try:
                            mem_ctx = builder.build("brainstorm", session_id=cb_session)
                            if isinstance(mem_ctx, (FallbackResult, ErrorResult)):
                                mem_ctx = ""
                        except Exception:
                            mem_ctx = ""
                    if mem_ctx:
                        prompt = mem_ctx + "\n\n" + prompt
                    history_text = "\n".join(
                        f"{m.get('role')}: {m.get('content')}" for m in hist
                    )
                    prompt_text = (
                        f"{history_text}\nuser: {prompt}" if history_text else prompt
                    )
                    resp = ask_with_memory(
                        ctx.gpt_client,
                        "sandbox_runner.brainstorm",
                        prompt_text,
                        memory=getattr(ctx.gpt_client, "gpt_memory", None),
                        tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                    )
                    idea = (
                        resp.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    hist = hist + [{"role": "user", "content": prompt}]
                    if idea:
                        ctx.brainstorm_history.append(idea)
                        hist.append({"role": "assistant", "content": idea})
                        logger.info("brainstorm", extra=log_record(idea=idea))
                    module = _get_local_knowledge()
                    try:
                        module.log(
                            prompt,
                            idea,
                            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                        )
                    except Exception:
                        logger.exception("local knowledge logging failed")
                    if len(hist) > 6:
                        hist = hist[-6:]
                    ctx.conversations["brainstorm"] = hist
                except Exception:
                    logger.exception("synergy brainstorming failed")
            if stall or ctx.synergy_needed:
                ctx.synergy_needed = False
                continue
            break

    ctx.prev_roi = 0.0
    ctx.predicted_roi = None
    _cycle(None, None, ctx.tracker)

    flagged = []
    if ctx.adapt_presets:
        try:
            thr = ctx.tracker.diminishing()
            e_thr = ctx.settings.entropy_plateau_threshold or thr
            e_consec = ctx.settings.entropy_plateau_consecutive or 3
            flagged = ctx.meta_log.diminishing(
                thr, consecutive=e_consec, entropy_threshold=e_thr
            )
        except Exception:
            flagged = []
    if ctx.adapt_presets and flagged:
        try:
            from menace.environment_generator import adapt_presets

            SANDBOX_ENV_PRESETS = adapt_presets(ctx.tracker, SANDBOX_ENV_PRESETS)
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(SANDBOX_ENV_PRESETS)
        except Exception:
            logger.exception("preset adaptation failed")

    if not getattr(args, "no_workflow_run", False):
        try:
            run_workflow_simulations(
                getattr(args, "workflow_db", "workflows.db"),
                SANDBOX_ENV_PRESETS,
                tracker=ctx.tracker,
                foresight_tracker=ctx.foresight_tracker,
                dynamic_workflows=getattr(args, "dynamic_workflows", False),
                module_algorithm=getattr(args, "module_algorithm", "greedy"),
                module_threshold=float(getattr(args, "module_threshold", 0.1)),
                module_semantic=getattr(args, "module_semantic", False),
            )
        except Exception:
            logger.exception("workflow simulations failed")

    ranking = ctx.tracker.rankings()
    e_thr = ctx.settings.entropy_plateau_threshold or ctx.tracker.diminishing()
    e_consec = ctx.settings.entropy_plateau_consecutive or 3
    flags = ctx.meta_log.diminishing(
        consecutive=e_consec, entropy_threshold=e_thr
    )
    if ranking:
        logger.info("sandbox roi ranking", extra=log_record(ranking=ranking))
    if flags:
        logger.info("sandbox diminishing", extra=log_record(modules=flags))
    try:
        ctx.tracker.save_history(str(ctx.roi_history_file))
    except Exception:
        logger.exception("failed to save roi history")
    try:
        if getattr(ctx, "foresight_tracker", None) and getattr(ctx, "foresight_history_file", None):
            with ctx.foresight_history_file.open("w", encoding="utf-8") as fh:
                json.dump(ctx.foresight_tracker.to_dict(), fh, indent=2)
    except Exception:
        logger.exception("failed to save foresight history")
    logger.info("sandbox run complete")
    _sandbox_cleanup(ctx)
    return ctx.tracker


__all__ = [
    "load_modified_code",
    "scan_repo_sections",
    "discover_orphan_modules",
    "discover_recursive_orphans",
    "build_section_prompt",
    "simulate_execution_environment",
    "simulate_full_environment",
    "generate_sandbox_report",
    "run_repo_section_simulations",
    "run_scenarios",
    "run_workflow_simulations",
    "simulate_meta_workflow",
    "simulate_temporal_trajectory",
    "_section_worker",
    "_sandbox_cycle_runner",
    "_SandboxMetaLogger",
    "_sandbox_init",
    "_sandbox_cleanup",
    "_sandbox_main",
    "_run_sandbox",
    "run_relevancy_radar_scan",
    "rank_scenarios",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    setup_logging()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--radar-scan",
        action="store_true",
        help="run a relevancy radar scan and print summary",
    )
    args, remaining = parser.parse_known_args()
    if args.radar_scan:
        flags = run_relevancy_radar_scan()
        if not flags:
            print("No modules flagged by relevancy radar")
        else:
            for mod, status in sorted(flags.items()):
                print(f"{mod}: {status}")
        raise SystemExit(0)
    main(remaining)
