from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import signal


REQUIRED_SYSTEM_TOOLS = ["ffmpeg", "tesseract", "qemu-system-x86_64"]
REQUIRED_PYTHON_PKGS = ["pydantic", "dotenv"]
# Optional packages used by sandbox utilities
OPTIONAL_PYTHON_PKGS = [
    "matplotlib",
    "statsmodels",
    "uvicorn",
    "fastapi",
    "sklearn",
    "stripe",
    "httpx",
]


def _verify_required_dependencies() -> None:
    """Exit if required or production optional dependencies are missing."""

    def _have_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    missing_sys = [t for t in REQUIRED_SYSTEM_TOOLS if shutil.which(t) is None]
    missing_req = [p for p in REQUIRED_PYTHON_PKGS if not _have_spec(p)]
    missing_opt = [p for p in OPTIONAL_PYTHON_PKGS if not _have_spec(p)]

    mode = os.getenv("MENACE_MODE", "test").lower()

    messages: list[str] = []
    if missing_sys:
        messages.append(
            "Missing system packages: " + ", ".join(missing_sys)
        )
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
            "Refer to docs/autonomous_sandbox.md for manual setup instructions."
        )
        msg = "\n".join(messages)
        logging.error(msg)
        raise SystemExit(msg)

    if missing_opt:
        logging.warning("Missing optional Python packages: %s", ", ".join(missing_opt))


_verify_required_dependencies()

import ast
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
from typing import Any, Dict, List, TYPE_CHECKING

from menace.unified_event_bus import UnifiedEventBus
from menace.menace_orchestrator import MenaceOrchestrator
from menace.self_improvement_policy import SelfImprovementPolicy
from menace.self_improvement_engine import SelfImprovementEngine
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
from sandbox_runner.metrics_plugins import (
    discover_metrics_plugins,
    load_metrics_plugins,
)
from menace.discrepancy_detection_bot import DiscrepancyDetectionBot
from jinja2 import Template

try:
    from menace.pre_execution_roi_bot import PreExecutionROIBot
except Exception:  # pragma: no cover - optional dependency
    PreExecutionROIBot = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from menace.roi_tracker import ROITracker

__path__ = [os.path.join(os.path.dirname(__file__), "sandbox_runner")]
logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent

from sandbox_runner.config import SANDBOX_REPO_URL, SANDBOX_REPO_PATH

_TPL_PATH = Path(
    os.getenv("GPT_SECTION_TEMPLATE", ROOT / "templates" / "gpt_section_prompt.j2")
)
_TPL: Template | None = None
_AUTO_PROMPTS_DIR = ROOT / "templates" / "auto_prompts"
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
    _section_worker,
    validate_preset,
)
from sandbox_runner.cycle import _sandbox_cycle_runner, map_module_identifier
from sandbox_runner.cli import _run_sandbox, rank_scenarios, main


# ----------------------------------------------------------------------
def load_modified_code(code_path: str) -> str:
    logger.debug("loading code from %s", code_path)
    with open(code_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    logger.debug("loaded %d bytes from %s", len(content), code_path)
    return content


# ----------------------------------------------------------------------
def scan_repo_sections(repo_path: str) -> Dict[str, Dict[str, List[str]]]:
    from menace.codebase_diff_checker import _extract_sections

    sections: Dict[str, Dict[str, List[str]]] = {}
    for base, _, files in os.walk(repo_path):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, repo_path)
            try:
                sections[rel] = _extract_sections(path)
            except Exception:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        sections[rel] = {"__file__": fh.read().splitlines()}
                except Exception:
                    sections[rel] = {}
    return sections


# ----------------------------------------------------------------------
def discover_orphan_modules(repo_path: str) -> List[str]:
    """Return module names that are never imported by other modules."""

    repo_path = os.path.abspath(repo_path)
    imported: set[str] = set()
    modules: dict[str, str] = {}

    for base, _, files in os.walk(repo_path):
        rel_base = os.path.relpath(base, repo_path)
        if rel_base.split(os.sep)[0] == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if name == "__init__.py":
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, repo_path)
            if rel.split(os.sep)[0] == "tests":
                continue
            try:
                text = open(path, "r", encoding="utf-8").read()
            except Exception:
                continue
            if "if __name__ == '__main__'" in text or 'if __name__ == "__main__"' in text:
                continue

            module = os.path.splitext(rel)[0].replace(os.sep, ".")
            modules[module] = path

            try:
                tree = ast.parse(text)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    pkg_parts = module.split(".")[:-1]
                    if node.level:
                        if node.level - 1 <= len(pkg_parts):
                            base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                        else:
                            base_prefix = []
                    else:
                        base_prefix = pkg_parts

                    if node.module:
                        imported.add(".".join(base_prefix + node.module.split(".")))
                    elif node.names:
                        for alias in node.names:
                            imported.add(".".join(base_prefix + alias.name.split(".")))

    orphans = [m for m in modules if m not in imported]
    orphans.sort()
    return orphans


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
        roi_drop = max(0.0, -_delta(tracker.roi_history))
        syn_deltas = [
            _delta(v) for k, v in tracker.metrics_history.items() if k.startswith("synergy_") and v
        ]
        synergy_drop = max(0.0, -sum(syn_deltas) / len(syn_deltas)) if syn_deltas else 0.0

        weights = {
            "security": 1.0,
            "efficiency": 1.0,
            "roi": 1.0,
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
                score += weights["roi"] * roi_drop
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
class _CycleMeta:
    cycle: int
    roi: float
    delta: float
    modules: list[str]
    reason: str


class _SandboxMetaLogger:
    def __init__(self, path: Path, module_index: "ModuleIndexDB" | None = None) -> None:
        self.path = path
        self.audit = AuditTrail(str(path))
        self.records: list[_CycleMeta] = []
        self.module_deltas: dict[str, list[float]] = {}
        self.flagged_sections: set[str] = set()
        self.last_patch_id = 0
        self.module_index = module_index
        logger.debug("SandboxMetaLogger initialised at %s", path)

    def log_cycle(
        self, cycle: int, roi: float, modules: list[str], reason: str
    ) -> None:
        prev = self.records[-1].roi if self.records else 0.0
        delta = roi - prev
        self.records.append(_CycleMeta(cycle, roi, delta, modules, reason))
        for m in modules:
            if self.module_index:
                try:
                    from pathlib import Path

                    gid = str(self.module_index.get(Path(m).name))
                except Exception:
                    gid = m
            else:
                gid = m
            self.module_deltas.setdefault(gid, []).append(delta)
        try:
            self.audit.record(
                {
                    "cycle": cycle,
                    "roi": roi,
                    "delta": delta,
                    "modules": modules,
                    "reason": reason,
                }
            )
        except Exception:
            logger.exception("meta log record failed")
        logger.debug(
            "cycle %d logged roi=%s delta=%s modules=%s", cycle, roi, delta, modules
        )

    def rankings(self) -> list[tuple[str, float]]:
        totals = {m: sum(v) for m, v in self.module_deltas.items()}
        logger.debug("rankings computed: %s", totals)
        return sorted(totals.items(), key=lambda x: x[1], reverse=True)

    def diminishing(
        self, threshold: float | None = None, consecutive: int = 3
    ) -> list[str]:
        flags: list[str] = []
        thr = 0.0 if threshold is None else float(threshold)
        eps = 1e-3
        for m, vals in self.module_deltas.items():
            if m in self.flagged_sections:
                continue
            if len(vals) < consecutive:
                continue
            window = vals[-consecutive:]
            mean = sum(window) / consecutive
            if len(window) > 1:
                var = sum((v - mean) ** 2 for v in window) / len(window)
                std = var ** 0.5
            else:
                std = 0.0
            if abs(mean) <= thr and std < eps:
                flags.append(m)
                self.flagged_sections.add(m)
        if flags:
            logger.debug("modules with diminishing returns: %s", flags)
        return flags


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
    meta_log: _SandboxMetaLogger
    backups: Dict[str, Any]
    env: Dict[str, str]
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
    brainstorm_history: List[str]
    conversations: Dict[str, List[Dict[str, str]]]
    offline_suggestions: bool = False
    adapt_presets: bool = True
    suggestion_cache: Dict[str, str] = field(default_factory=dict)
    suggestion_db: PatchSuggestionDB | None = None
    synergy_needed: bool = False
    best_roi: float = 0.0
    best_synergy_metrics: Dict[str, float] = field(default_factory=dict)


def _sandbox_init(preset: Dict[str, Any], args: argparse.Namespace) -> SandboxContext:
    import sandbox_runner.environment as env

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
    repo = SANDBOX_REPO_PATH
    orig_cwd = Path.cwd()

    data_dir = Path(
        getattr(args, "sandbox_data_dir", None)
        or os.getenv("SANDBOX_DATA_DIR", str(ROOT / "sandbox_data"))
    )
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("using data directory", extra=log_record(path=str(data_dir)))
    policy_file = data_dir / "improvement_policy.pkl"
    patch_file = data_dir / "patch_history.db"
    module_map_file = data_dir / "module_map.json"

    refresh_map = os.getenv("SANDBOX_REFRESH_MODULE_MAP") == "1"
    auto_map = bool(
        getattr(args, "autodiscover_modules", False)
        or os.getenv("SANDBOX_AUTO_MAP")
        or os.getenv("SANDBOX_AUTODISCOVER_MODULES")
    )
    if not os.getenv("SANDBOX_AUTO_MAP") and os.getenv("SANDBOX_AUTODISCOVER_MODULES"):
        logger.warning(
            "SANDBOX_AUTODISCOVER_MODULES is deprecated; use SANDBOX_AUTO_MAP",
        )

    if refresh_map or auto_map or not module_map_file.exists():
        try:
            from scripts.generate_module_map import generate_module_map

            algo = (
                getattr(args, "module_algorithm", None)
                or os.getenv("SANDBOX_MODULE_ALGO", "greedy")
            )
            thr_arg = getattr(args, "module_threshold", None)
            if thr_arg is not None:
                threshold = thr_arg
            else:
                try:
                    threshold = float(os.getenv("SANDBOX_MODULE_THRESHOLD", "0.1"))
                except Exception:
                    threshold = 0.1
            sem_arg = getattr(args, "module_semantic", None)
            if sem_arg is None:
                sem_env = os.getenv("SANDBOX_SEMANTIC_MODULES")
                if sem_env is None:
                    sem_env = os.getenv("SANDBOX_MODULE_SEMANTIC")  # legacy
                use_semantic = auto_map if sem_env is None else sem_env == "1"
            else:
                use_semantic = bool(sem_arg)

            mapping = generate_module_map(
                module_map_file,
                root=Path(repo),
                algorithm=algo,
                threshold=threshold,
                semantic=use_semantic,
            )
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

    meta_log = _SandboxMetaLogger(data_dir / "sandbox_meta.log")
    metrics_db = MetricsDB(data_dir / "metrics.db")
    data_bot = DataBot(metrics_db)
    dd_bot = DiscrepancyDetectionBot()
    module_counts: Dict[str, int] = {}
    patch_db_path = patch_file

    def _changed_modules(last_id: int) -> tuple[list[str], int]:
        if not patch_db_path.exists():
            return [], last_id
        import sqlite3

        try:
            with sqlite3.connect(patch_db_path, check_same_thread=False) as conn:
                rows = conn.execute(
                    "SELECT id, filename FROM patch_history WHERE id>?",
                    (last_id,),
                ).fetchall()
        except sqlite3.Error as exc:
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
    try:
        from menace.module_index_db import ModuleIndexDB  # type: ignore
    except Exception:
        ModuleIndexDB = None  # type: ignore
    policy = SelfImprovementPolicy(path=str(policy_file))
    patch_db = PatchHistoryDB(patch_db_path)
    score_backend = None
    backend_url = os.getenv("PATCH_SCORE_BACKEND_URL")
    if backend_url:
        try:
            score_backend = backend_from_url(backend_url)
        except Exception:
            logger.exception("patch score backend init failed")
    improver = SelfImprovementEngine(
        meta_logger=meta_log,
        module_index=ModuleIndexDB(module_map_file) if ModuleIndexDB else None,
        patch_db=patch_db,
        policy=policy,
        score_backend=score_backend,
    )
    meta_log.module_index = improver.module_index

    telem_db = ErrorDB(Path(tmp) / "errors.db")
    improver.error_bot = ErrorBot(telem_db, MetricsDB())

    class _TelemProxy:
        def __init__(self, db: ErrorDB) -> None:
            self.db = db

        def recent_errors(self, limit: int = 5) -> list[str]:
            cur = self.db.conn.execute(
                "SELECT stack_trace FROM telemetry ORDER BY id DESC LIMIT ?",
                (limit,),
            )
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
    )
    gpt_client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from menace.chatgpt_idea_bot import ChatGPTClient

            gpt_client = ChatGPTClient(model="gpt-4")
        except Exception:
            logger.exception("GPT client init failed")
            gpt_client = None
    sandbox = SelfDebuggerSandbox(
        _TelemProxy(telem_db),
        engine,
        policy=policy,
        state_getter=improver._policy_state,
    )
    tester = SelfTestService(telem_db)
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
            logger.exception("failed to load resource db: %s", res_path)

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

    tracker = ROITracker(resource_db=res_db, cluster_map=improver.module_clusters)
    roi_history_file = data_dir / "roi_history.json"
    try:
        tracker.load_history(str(roi_history_file))
    except Exception:
        logger.exception("failed to load roi history")
    meta_log.module_deltas.update(tracker.module_deltas)
    prev_roi = 0.0
    predicted_roi = None
    predicted_lucrativity = None
    roi_tolerance = float(env.get("SANDBOX_ROI_TOLERANCE", "0.01"))
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
        "SANDBOX_METRICS_FILE", str(ROOT / "sandbox_metrics.yaml")
    )
    extra_metrics = SANDBOX_EXTRA_METRICS

    brainstorm_history: list[str] = []
    conversations: Dict[str, List[Dict[str, str]]] = {}
    offline_suggestions = bool(
        getattr(args, "offline_suggestions", False)
        or os.getenv("SANDBOX_OFFLINE_SUGGESTIONS", "0") == "1"
    )
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
            logger.exception("failed to load suggestion cache: %s", cache_path)
            suggestion_cache = {}
    global _SUGGESTION_DB
    suggestion_db_path = suggestion_cache.get("db", str(data_dir / "module_suggestions.db"))
    if _SUGGESTION_DB is None or _SUGGESTION_DB.path != Path(suggestion_db_path):
        _SUGGESTION_DB = PatchSuggestionDB(suggestion_db_path)
    suggestion_db = _SUGGESTION_DB

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
        meta_log=meta_log,
        backups=backups,
        env=env,
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
        brainstorm_history=brainstorm_history,
        conversations=conversations,
        offline_suggestions=offline_suggestions,
        adapt_presets=adapt_presets_flag,
        suggestion_cache=suggestion_cache,
        suggestion_db=suggestion_db,
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


def _sandbox_main(preset: Dict[str, Any], args: argparse.Namespace) -> "ROITracker":
    from menace.roi_tracker import ROITracker

    global SANDBOX_ENV_PRESETS
    logger.info("starting sandbox run", extra=log_record(preset=preset))
    ctx = _sandbox_init(preset, args)

    def _cycle(
        section: str | None,
        snippet: str | None,
        tracker: "ROITracker",
        scenario: str | None = None,
    ) -> None:
        _sandbox_cycle_runner(ctx, section, snippet, tracker, scenario)

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
                sec_tracker = ROITracker(resource_db=ctx.res_db, cluster_map=ctx.improver.module_clusters)
                _cycle(section_name, snippet, sec_tracker, scenario)
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

                    agg = ROITracker()
                    for t in section_trackers:
                        agg.roi_history.extend(t.roi_history)
                        for m, vals in t.metrics_history.items():
                            agg.metrics_history.setdefault(m, []).extend(vals)
                    flagged = ctx.meta_log.diminishing(agg.diminishing())
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
            synergy_tracker = ROITracker(resource_db=ctx.res_db, cluster_map=ctx.improver.module_clusters)
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
                [map_module_identifier(m, ctx.repo) for m in ctx.meta_log.flagged_sections],
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
                    resp = ctx.gpt_client.ask(
                        hist + [{"role": "user", "content": prompt}]
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
            flagged = ctx.meta_log.diminishing(ctx.tracker.diminishing())
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
            )
        except Exception:
            logger.exception("workflow simulations failed")

    ranking = ctx.meta_log.rankings()
    flags = ctx.meta_log.diminishing()
    if ranking:
        logger.info("sandbox roi ranking", extra=log_record(ranking=ranking))
    if flags:
        logger.info("sandbox diminishing", extra=log_record(modules=flags))
    try:
        ctx.tracker.save_history(str(ctx.roi_history_file))
    except Exception:
        logger.exception("failed to save roi history")
    logger.info("sandbox run complete")
    _sandbox_cleanup(ctx)
    return ctx.tracker


__all__ = [
    "load_modified_code",
    "scan_repo_sections",
    "discover_orphan_modules",
    "build_section_prompt",
    "simulate_execution_environment",
    "simulate_full_environment",
    "generate_sandbox_report",
    "run_repo_section_simulations",
    "run_workflow_simulations",
    "_section_worker",
    "_sandbox_cycle_runner",
    "_SandboxMetaLogger",
    "_sandbox_init",
    "_sandbox_cleanup",
    "_sandbox_main",
    "_run_sandbox",
    "rank_scenarios",
    "main",
]
