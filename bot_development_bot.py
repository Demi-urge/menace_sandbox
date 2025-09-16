"""Bot Development Bot for building bots from handoff specs."""

from __future__ import annotations

import json
import os
import time
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Iterable, Callable, Type
import importlib.util
import keyword

import logging
import subprocess
import shutil
import sys
import uuid
from dynamic_path_router import resolve_path
from context_builder_util import ensure_fresh_weights, create_context_builder
from secret_redactor import redact_secrets

try:
    from packaging.requirements import Requirement  # type: ignore
    from importlib.metadata import version as pkg_version  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Requirement = None  # type: ignore
    pkg_version = None  # type: ignore

from typing import TYPE_CHECKING

from .db_router import DBRouter
from . import RAISE_ERRORS
from .bot_dev_config import BotDevConfig
from .models_repo import (
    MODELS_REPO_PATH,
    ACTIVE_MODEL_FILE,
    ensure_models_repo,
)
from .coding_bot_interface import self_coding_managed
from vector_service.context_builder import ContextBuilder
from prompt_types import Prompt
from .codex_output_analyzer import (
    validate_stripe_usage,
)
from .self_coding_manager import (
    SelfCodingManager,
    internalize_coding_bot,
    _manager_generate_helper_with_builder as manager_generate_helper,
)
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .data_bot import DataBot, persist_sc_thresholds
from .code_database import CodeDB
from .menace_memory_manager import MenaceMemoryManager
from .bot_registry import BotRegistry
from .threshold_service import ThresholdService
from .self_coding_thresholds import get_thresholds
from .shared_evolution_orchestrator import get_orchestrator

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator = get_orchestrator("BotDevelopmentBot", data_bot, engine)
_th = get_thresholds("BotDevelopmentBot")
persist_sc_thresholds(
    "BotDevelopmentBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)
manager = internalize_coding_bot(
    "BotDevelopmentBot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=evolution_orchestrator,
    roi_threshold=_th.roi_drop,
    error_threshold=_th.error_increase,
    test_failure_threshold=_th.test_failure_increase,
    threshold_service=ThresholdService(),
)

if TYPE_CHECKING:  # pragma: no cover - heavy dependency
    from .watchdog import Watchdog

try:
    import yaml  # type: ignore
    if not hasattr(yaml, "safe_dump"):
        yaml.safe_dump = lambda data, *a, **k: json.dumps(data)  # type: ignore
    if not hasattr(yaml, "safe_load"):
        yaml.safe_load = lambda s, *a, **k: json.loads(s)  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

try:
    from elasticsearch import Elasticsearch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Elasticsearch = None  # type: ignore

try:
    from git import Repo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Repo = None  # type: ignore


@dataclass
class RetryStrategy:
    """Simple configurable retry strategy."""

    attempts: int = 3
    delay: float = 1.0
    factor: float = 2.0

    def run(
        self,
        func: Callable[[], Any],
        *,
        exc: Type[BaseException] | tuple[Type[BaseException], ...] = Exception,
        logger: logging.Logger | None = None,
    ) -> Any:
        if isinstance(exc, type):
            exc_types = (exc,)
        else:
            exc_types = exc
        backoff = self.delay
        for i in range(self.attempts):
            try:
                return func()
            except exc_types as e:
                if i == self.attempts - 1:
                    raise
                log = logger or logging
                log.warning(
                    "retry %s/%s after error: %s",
                    i + 1,
                    self.attempts,
                    e,
                )
                time.sleep(backoff)
                backoff *= self.factor
        raise RuntimeError("retry strategy exhausted")


DEFAULT_TEMPLATE = resolve_path("config/prompt_templates.v2.json")
PROMPT_TEMPLATES_PATH = resolve_path(
    os.getenv("PROMPT_TEMPLATES_PATH") or DEFAULT_TEMPLATE
)

TEMPLATE_SECTION_KEY = "templates"
TEMPLATE_VERSION_KEY = "version"

SECTION_CODING_STANDARDS = "coding_standards"
SECTION_REPOSITORY_LAYOUT = "repository_layout"
SECTION_METADATA = "metadata"
SECTION_VERSION_CONTROL = "version_control"
SECTION_TESTING = "testing"
SECTION_CONTEXT = "context"
SECTION_BILLING = "billing"

INSTRUCTION_SECTIONS = [
    SECTION_CODING_STANDARDS,
    SECTION_REPOSITORY_LAYOUT,
    SECTION_METADATA,
    SECTION_VERSION_CONTROL,
    SECTION_TESTING,
    SECTION_BILLING,
    SECTION_CONTEXT,
]

PREVIOUS_FAILURE_TEMPLATE = "Previous failure: {error}"

RESPONSE_FORMAT_HINT = (
    "Return JSON like {'status': 'completed', 'message': <optional string>}"
)

# Mapping of common helper functions to their expected logic.
FUNCTION_IMPLEMENTATION_MAP: Dict[str, str] = {
    "click_target": (
        "Use an automation library to click the given coordinates or selector "
        "and return None"
    ),
    "type_text": (
        "Send keyboard input to the active field using an automation library"
    )
}


@dataclass
class BotSpec:
    """Specification for a bot to be developed."""

    name: str
    purpose: str
    functions: List[str] = field(default_factory=list)
    io_format: str = ""
    language: str = "python"
    dependencies: List[str] = field(default_factory=list)
    level: str = ""
    description: str = ""
    function_docs: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the spec."""

        return asdict(self)


@dataclass
class EngineResult:
    """Result from :meth:`_call_codex_api`."""

    success: bool
    code: str | None = None
    error: str | None = None


@dataclass
class InstructionTemplate:
    """Representation of a single instruction template."""

    text: str
    weight: float = 1.0


class PromptTemplateEngine:
    """Render instruction templates with optional weighting."""

    def __init__(self, templates: Dict[str, List[Any]] | None = None) -> None:
        templates = templates or {}
        self.templates: Dict[str, List[InstructionTemplate]] = {}
        for section, items in templates.items():
            parsed: List[InstructionTemplate] = []
            for it in items:
                if isinstance(it, dict):
                    parsed.append(
                        InstructionTemplate(
                            text=str(it.get("text", "")),
                            weight=float(it.get("weight", 1.0)),
                        )
                    )
                else:
                    parsed.append(InstructionTemplate(text=str(it)))
            parsed.sort(key=lambda t: t.weight, reverse=True)
            self.templates[section] = parsed

    def render(
        self,
        section: str,
        context: Dict[str, Any],
        *,
        limit: int | None = None,
    ) -> List[str]:
        items = self.templates.get(section, [])
        rendered = [tmpl.text.format(**context) for tmpl in items]
        if limit is not None:
            rendered = rendered[:limit]
        return rendered


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class BotDevelopmentBot:
    """Receive bot specs and generate starter code repositories."""

    def __init__(
        self,
        repo_base: Path | str | None = None,
        es_url: str | None = None,
        db_steward: "DBRouter" | None = None,
        watchdog: "Watchdog" | None = None,
        *,
        config: BotDevConfig | None = None,
        context_builder: ContextBuilder,
        manager: SelfCodingManager | None = None,
        engine: SelfCodingEngine | None = None,
    ) -> None:
        self.config = config or BotDevConfig()
        if repo_base is not None:
            self.config.repo_base = Path(repo_base)
        else:
            self.config.repo_base = Path(self.config.repo_base)
        if es_url is not None:
            self.config.es_url = es_url
        self.config.repo_base.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BotDev")
        self.watchdog = watchdog
        self.repo_base = self.config.repo_base
        self.es = (
            Elasticsearch(self.config.es_url)
            if self.config.es_url and Elasticsearch
            else None
        )
        if self.es:
            try:
                self.es.info()
            except Exception as exc:  # pragma: no cover - external service
                self.logger.warning("Elasticsearch unavailable: %s", exc)
                self.es = None
        self.errors: List[str] = []
        self.db_steward = db_steward
        self.error_sinks = list(self.config.error_sinks)
        self.concurrency = self.config.concurrency_workers
        self.config.validate()
        self.file_write_retry = RetryStrategy(
            attempts=self.config.file_write_attempts,
            delay=self.config.file_write_retry_delay,
        )
        self.engine_retry = RetryStrategy(
            attempts=self.config.engine_attempts,
            delay=self.config.engine_retry_delay,
        )
        self.prompt_templates_version = 1
        try:
            with PROMPT_TEMPLATES_PATH.open() as fh:
                data = json.load(fh)
            self.prompt_templates = data.get(TEMPLATE_SECTION_KEY, data)
            self.prompt_templates_version = int(data.get(TEMPLATE_VERSION_KEY, 1))
        except Exception as exc:
            self.logger.warning("failed to load prompt templates: %s", exc)
            self.prompt_templates = {}
        self.template_engine = PromptTemplateEngine(self.prompt_templates)
        if not isinstance(context_builder, ContextBuilder):
            msg = "ContextBuilder instance is required"
            try:
                self._escalate(msg)
            except Exception:
                self.logger.error(msg)
            raise ValueError(msg)
        self.context_builder = context_builder
        try:
            ensure_fresh_weights(self.context_builder)
        except Exception as exc:
            self.logger.error("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc
        self.engine = getattr(manager, "engine", engine)
        # warn about missing optional dependencies
        for dep_name, mod in {
            "yaml": yaml,
            "Elasticsearch": Elasticsearch,
            "git": Repo,
        }.items():
            if mod is None:
                self.logger.warning("optional dependency %s unavailable", dep_name)
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = DataBot()

    @property
    def coding_engine(self) -> SelfCodingEngine:  # pragma: no cover - backward compat
        return self.engine

    @property
    def self_coding_engine(self) -> SelfCodingEngine:  # pragma: no cover - backward compat
        return self.engine

    def _escalate(self, message: str, level: str = "error") -> None:
        """Send an escalation message to configured sinks."""
        if self.watchdog:
            try:
                self.watchdog.escalate(message)
            except Exception:
                self.logger.exception("watchdog escalate failed")
        for sink in self.error_sinks:
            try:
                sink(level, message)
            except Exception:
                self.logger.exception("error sink failed")

    def _sanitize_functions(self, funcs: Iterable[str]) -> List[str]:
        seen = set()
        valid = []
        for func in funcs:
            if not func or not func.isidentifier() or keyword.iskeyword(func):
                self.logger.warning("invalid function name: %s", func)
                continue
            if func in seen:
                self.logger.warning("duplicate function name: %s", func)
                continue
            seen.add(func)
            valid.append(func)
        return valid

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Return a safe repository name."""
        safe = re.sub(r"\W+", "_", name).strip("_")
        return safe or "bot"

    def _sanitize_capabilities(self, caps: Iterable[str]) -> List[str]:
        seen = set()
        valid: List[str] = []
        for cap in caps:
            if not cap:
                continue
            cap = re.sub(r"\s+", "_", str(cap)).lower()
            cap = re.sub(r"[^a-z0-9_]+", "", cap)
            if not cap:
                self.logger.warning("invalid capability: %s", cap)
                continue
            if cap in seen:
                continue
            seen.add(cap)
            valid.append(cap)
        return valid

    def _function_guidance(self, funcs: Iterable[str]) -> List[str]:
        lines: List[str] = []
        for func in funcs:
            info = FUNCTION_IMPLEMENTATION_MAP.get(func)
            if info:
                lines.append(f"{func}: {info}")
        return lines

    def decide_instructions(self, spec: BotSpec) -> Dict[str, List[str]]:
        """Return rendered instruction templates for ``spec``."""
        context = {"name": spec.name}
        instructions: Dict[str, List[str]] = {}
        complexity = len(spec.capabilities) + len(spec.dependencies)
        limit = 5 if complexity > 5 or spec.level.lower() == "advanced" else 3
        for section in INSTRUCTION_SECTIONS:
            items = self.template_engine.render(section, context, limit=limit)
            instructions[section] = items
        if self.errors:
            instructions.setdefault(SECTION_CONTEXT, []).append(
                PREVIOUS_FAILURE_TEMPLATE.format(error=self.errors[-1])
            )
        return instructions

    def parse_plan(self, data: str) -> List[BotSpec]:
        """Parse JSON or YAML plan text into specs."""
        try:
            obj = json.loads(data)
        except Exception:
            if yaml:
                obj = yaml.safe_load(data)
            else:
                raise
        if isinstance(obj, dict):
            obj = [obj]
        specs = []
        for entry in obj:
            funcs = self._sanitize_functions(entry.get("functions", []))
            caps = self._sanitize_capabilities(entry.get("capabilities", []))
            func_docs_src = entry.get("function_docs", {})
            func_docs: Dict[str, str] = {}
            if isinstance(func_docs_src, dict):
                for f in funcs:
                    val = func_docs_src.get(f)
                    if isinstance(val, str):
                        func_docs[f] = val
            specs.append(
                BotSpec(
                    name=self._sanitize_name(entry.get("name", "bot")),
                    purpose=entry.get("purpose", ""),
                    functions=funcs,
                    io_format=entry.get("io", ""),
                    language=entry.get("language", "python"),
                    dependencies=list(entry.get("dependencies", [])),
                    level=entry.get("level", ""),
                    description=entry.get("description", ""),
                    function_docs=func_docs,
                    capabilities=caps,
                )
            )
        return specs

    def create_env(self, spec: BotSpec, model_id: int | None = None) -> Path:
        """Create or reuse a repository directory."""
        if self.repo_base == MODELS_REPO_PATH:
            ensure_models_repo()

        base = self._sanitize_name(spec.name)
        path = self.repo_base / base
        editing = path.exists()
        if editing:
            if model_id is not None:
                while ACTIVE_MODEL_FILE.exists():
                    time.sleep(0.1)
                src = self.repo_base.parent / str(model_id)
                if src.exists():
                    if (path / ".git").exists():
                        try:
                            subprocess.run(
                                ["git", "fetch", str(src)], cwd=path, check=True
                            )
                            subprocess.run(
                                ["git", "reset", "--hard", "FETCH_HEAD"],
                                cwd=path,
                                check=True,
                            )
                        except Exception:
                            shutil.rmtree(path)
                            subprocess.run(
                                ["git", "clone", str(src), str(path)], check=True
                            )
                    else:
                        if path.exists():
                            shutil.rmtree(path)
                        subprocess.run(
                            ["git", "clone", str(src), str(path)], check=True
                        )
            return path

        counter = 1
        while path.exists():
            path = self.repo_base / f"{base}_{counter}"
            counter += 1
        if path.name != spec.name:
            spec.name = path.name
        path.mkdir(parents=True, exist_ok=True)
        if Repo and not (path / ".git").exists():
            Repo.init(path)
        return path

    def _write_with_retry(self, path: Path, data: str) -> None:
        """Write ``data`` to ``path`` atomically with retries."""

        def writer() -> None:
            if self.manager is not None:
                desc = f"update {path.name} content\n\n{data}"
                self.manager.auto_run_patch(path, desc)
                registry = getattr(self.manager, "bot_registry", None)
                if registry is not None:
                    try:
                        name = getattr(
                            self,
                            "name",
                            getattr(self, "bot_name", self.__class__.__name__),
                        )
                        registry.update_bot(name, str(path))
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception("bot registry update failed")
                return
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as fh:
                fh.write(data)
            tmp.replace(path)

        self.file_write_retry.run(writer, logger=self.logger)

    def lint_code(self, path: Path) -> None:
        """Run style and type checks for Python code."""
        if path.suffix != ".py":
            return

        tools = [
            ("black", ["black", "--check", str(path)]),
            ("flake8", ["flake8", str(path)]),
            ("mypy", ["mypy", str(path)]),
        ]
        for name, cmd in tools:
            if shutil.which(cmd[0]) is None:
                self.logger.warning("%s not installed", name)
                continue
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                msg = (proc.stdout + proc.stderr).strip()
                self.logger.warning(
                    "%s failed for %s: %s", name, path, msg
                )
                self._escalate(f"{name} failed for {path}: {msg}")

        stripe_check = Path(resolve_path("scripts/check_stripe_imports.py"))
        proc = subprocess.run(
            [sys.executable, str(stripe_check), str(path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = (proc.stdout + proc.stderr).strip()
            self.logger.warning(
                "stripe import check failed for %s: %s", path, msg
            )
            self._escalate(
                f"stripe import check failed for {path}: {msg}"
            )
            raise RuntimeError(
                f"stripe import check failed for {path}: {msg}"
            )

    def version_control(
        self, repo_dir: Path, paths: List[Path], message: str = "Auto-generated bot"
    ) -> None:
        """Commit generated files using GitPython if available."""
        repo_dir = Path(resolve_path(repo_dir))
        if Repo:
            try:
                repo = Repo(repo_dir)
                repo.index.add([str(p) for p in paths])
                repo.index.commit(message)
                return
            except Exception as exc:  # pragma: no cover - git errors
                self.logger.warning("GitPython commit failed: %s", exc)
        try:
            subprocess.run(
                ["git", "add", *[str(p) for p in paths]], cwd=repo_dir, check=True
            )
            subprocess.run(["git", "commit", "-m", message], cwd=repo_dir, check=True)
        except Exception as exc:
            self.logger.exception("git commit failed: %s", exc)
            self._escalate(f"git commit failed: {exc}")
            if RAISE_ERRORS:
                raise

    def _create_requirements(self, repo_dir: Path, spec: BotSpec) -> Path | None:
        repo_dir = Path(resolve_path(repo_dir))
        if not spec.dependencies:
            return None
        req = repo_dir / "requirements.txt"
        self._write_with_retry(req, "\n".join(spec.dependencies))
        self._validate_dependencies(spec.dependencies)
        return Path(resolve_path(req))

    def _validate_dependencies(self, deps: Iterable[str]) -> None:
        for dep in deps:
            if Requirement and pkg_version:
                try:
                    req = Requirement(dep)
                    importlib.import_module(req.name)
                    installed = pkg_version(req.name)
                    if req.specifier and installed not in req.specifier:
                        self.logger.warning(
                            "dependency '%s' version %s does not satisfy %s",
                            dep,
                            installed,
                            req.specifier,
                        )
                except Exception as exc:
                    self.logger.warning("dependency check failed for %s: %s", dep, exc)
            else:
                try:
                    importlib.import_module(dep)
                except Exception:
                    self.logger.warning("dependency '%s' not importable", dep)

    def _write_meta(self, repo_dir: Path, spec: BotSpec) -> Path:
        repo_dir = Path(resolve_path(repo_dir))
        meta = repo_dir / "meta.yaml"
        data = {
            "name": spec.name,
            "purpose": spec.purpose,
            "functions": spec.functions,
            "capabilities": spec.capabilities,
            "io_format": spec.io_format,
            "language": spec.language,
            "dependencies": spec.dependencies,
            "level": spec.level,
            "timestamp": time.time(),
        }
        if yaml and hasattr(yaml, "safe_dump"):
            self._write_with_retry(meta, yaml.safe_dump(data))
        else:
            self._write_with_retry(meta, json.dumps(data, indent=2))
        return Path(resolve_path(meta))

    def _create_tests(self, repo_dir: Path, spec: BotSpec) -> list[Path]:
        repo_dir = Path(resolve_path(repo_dir))
        tests_dir = repo_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        tests_dir = Path(resolve_path(tests_dir))
        (tests_dir / "__init__.py").touch()
        files: list[Path] = []
        for func in spec.functions:
            if not func.isidentifier():
                continue
            arg = ""
            if func.startswith("fetch") or func.startswith("get"):
                arg = "'http://example.com'"
            elif func.startswith("save") or func.startswith("write"):
                arg = "'out.txt', 'data'"
            tf = tests_dir / f"test_{func}.py"
            lines = [
                f"from {spec.name} import {func}",
                "from pathlib import Path",
                "import pytest",
                "",
                f"def test_{func}():",
            ]
            call = f"{func}({arg})" if arg else f"{func}()"
            lines.append("    try:")
            lines.append(f"        result = {call}")
            lines.append("    except Exception as exc:")
            lines.append("        pytest.fail(f'function raised {exc}')")
            if func.startswith("save") or func.startswith("write"):
                lines.append("    assert Path('out.txt').exists()")
                lines.append("    Path('out.txt').unlink()")
            elif func.startswith("fetch") or func.startswith("get"):
                lines.append("    assert result is not None")
            else:
                lines.append("    assert result is None")
            self._write_with_retry(tf, "\n".join(lines) + "\n")
            files.append(Path(resolve_path(tf)))
        return files

    def _validate_repo(self, repo_dir: Path, spec: BotSpec) -> bool:
        repo_dir = Path(resolve_path(repo_dir))
        try:
            module_path = Path(resolve_path(repo_dir / f"{spec.name}.py"))
            with module_path.open("r", encoding="utf-8") as fh:
                validate_stripe_usage(fh.read())
            spec_obj = importlib.util.spec_from_file_location(spec.name, module_path)
            if not spec_obj or not spec_obj.loader:
                raise ImportError("cannot load module")
            mod = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(mod)
            for func in spec.functions:
                if not hasattr(mod, func):
                    raise AttributeError(f"missing function {func}")
            return True
        except Exception as exc:
            self.logger.exception(
                "runtime validation failed for %s: %s", spec.name, exc
            )
            self._escalate(f"runtime validation failed for {spec.name}: {exc}")
            return False

    # ------------------------------------------------------------------
    def _call_codex_api(
        self, prompt: Prompt | list[dict[str, str]], path: Path | None = None
    ) -> EngineResult:
        """Produce helper code via :class:`SelfCodingManager`."""

        if isinstance(prompt, Prompt):
            content = prompt.user
            if prompt.examples:
                content += "\n\n" + "\n".join(prompt.examples)
            messages: list[dict[str, str]] = [
                {"role": "user", "content": content}
            ]
            if prompt.system:
                messages.insert(0, {"role": "system", "content": prompt.system})
        else:
            messages = prompt

        prompt_parts: list[str] = []
        user_found = False
        for message in messages:
            role = message.get("role")
            if not role:
                continue
            prompt_parts.append(f"{role}: {message.get('content', '')}")
            if role == "user":
                user_found = True

        if not user_found:
            msg = "no user prompt provided"
            self.logger.warning(msg)
            self._escalate(msg, level="warning")
            self.errors.append(msg)
            if self.config.raise_errors:
                raise ValueError(msg)
            return EngineResult(False, None, msg)

        prompt = "\n".join(prompt_parts)

        if not prompt.strip():
            msg = "empty prompt"
            self.logger.error(msg)
            self._escalate(msg)
            self.errors.append(msg)
            if self.config.raise_errors:
                raise ValueError(msg)
            return EngineResult(False, None, msg)

        prompt_snippet = prompt[: self.config.max_prompt_log_chars]
        prompt_snippet = redact_secrets(prompt_snippet)
        self.logger.info("generate_helper prompt: %s", prompt_snippet)

        manager = getattr(self, "manager", None)
        if manager is None:
            msg = "SelfCodingManager is required"
            self.logger.error(msg)
            self._escalate(msg)
            self.errors.append(msg)
            if self.config.raise_errors:
                raise RuntimeError(msg)
            return EngineResult(False, None, msg)
        try:
            if path is not None:
                self.engine_retry.run(
                    lambda: manager.auto_run_patch(path, prompt),
                    logger=self.logger,
                )
                code = path.read_text()
            else:
                code = self.engine_retry.run(
                    lambda: manager_generate_helper(
                        manager,
                        prompt,
                        context_builder=self.context_builder,
                    ),
                    logger=self.logger,
                )
            return EngineResult(True, code, None)
        except Exception as exc:
            msg = f"engine request failed: {exc}"
            self.logger.exception(msg)
            self._escalate(msg, level="error")
            self.errors.append(msg)
            if self.config.raise_errors:
                raise
            return EngineResult(False, None, msg)

    def build_training_prompt(
        self,
        spec: BotSpec,
        *,
        context_builder: ContextBuilder,
        sample_limit: int = 5,
        _sample_sort_by: str = "confidence",
        _sample_with_vectors: bool = True,
    ) -> Prompt:
        """Return an enriched :class:`Prompt` for code generation.

        Parameters
        ----------
        spec:
            Bot specification describing the desired bot.
        sample_limit:
            Maximum number of context snippets retrieved by the
            :class:`ContextBuilder`.
        _sample_sort_by:
            Deprecated, retained for backward compatibility.
        _sample_with_vectors:
            Deprecated, retained for backward compatibility.
        """

        if context_builder is None:
            raise ValueError("context_builder is required")
        if self.engine is None:
            raise RuntimeError("SelfCodingEngine is required for prompt generation")

        problem_lines: list[str] = [
            f"# Bot specification: {spec.name}",
            f"Template version: v{self.prompt_templates_version}",
        ]
        problem_lines.extend([
            "## Overview",
            f"Language: {spec.language}",
        ])
        if spec.level:
            problem_lines.append(f"Level: {spec.level}")
        if spec.io_format:
            problem_lines.append(f"IO Format: {spec.io_format}")
        purpose_line = spec.description or spec.purpose
        if purpose_line:
            problem_lines.append(f"Purpose: {purpose_line}")

        if spec.function_docs or spec.functions:
            problem_lines.append("\n## Functions")
            for func in spec.functions:
                doc = spec.function_docs.get(func, "").strip()
                doc_line = doc.splitlines()[0].strip() if doc else ""
                bullet = f"- {func}: {doc_line}" if doc_line else f"- {func}"
                problem_lines.append(bullet)

        if spec.dependencies:
            deps = ", ".join(spec.dependencies)
            problem_lines.append("\n## Dependencies")
            problem_lines.append(deps)

        if spec.capabilities:
            problem_lines.append("\n## Required Capabilities")
            for cap in spec.capabilities:
                problem_lines.append(f"- Implement capability: {cap}")

        templates = self.decide_instructions(spec)
        instruction_lines: list[str] = []
        for section in INSTRUCTION_SECTIONS:
            items = templates.get(section, [])
            if not items:
                continue
            instruction_lines.append(section.replace("_", " ").title() + ":")
            instruction_lines.extend(items)

        func_lines = self._function_guidance(spec.functions)
        if func_lines:
            instruction_lines.append("Function Guidance:")
            instruction_lines.extend(func_lines)

        constraint_lines = [RESPONSE_FORMAT_HINT]

        problem_context = "\n".join(problem_lines)
        implementation_instructions = "\n".join(instruction_lines)
        developer_constraints = "\n".join(constraint_lines)

        intent_text = (
            "INSTRUCTION MODE: Generate Python code based on the inputs below.\n"
            "Problem Context:\n"
            f"{problem_context}\n\n"
            "Implementation Instructions:\n"
            f"{implementation_instructions}\n\n"
            "Developer Constraints:\n"
            f"{developer_constraints}\n\n"
            "Expected Output:\n"
            "Return only the complete Python code without explanations or markdown."
        )
        session_id = uuid.uuid4().hex
        spec_meta = (
            spec.to_dict() if hasattr(spec, "to_dict") else asdict(spec)
        )
        intent_payload: Dict[str, Any] = {
            "instruction": intent_text,
            "sample_limit": sample_limit,
            "top_k": sample_limit,
            "session_id": session_id,
            "session_ids": [session_id],
        }
        if isinstance(spec_meta, dict):
            intent_payload.update(spec_meta)

        build_fn = getattr(self.engine, "build_enriched_prompt", None)
        if callable(build_fn):
            prompt_obj = build_fn(
                intent_text,
                intent=intent_payload,
                context_builder=context_builder,
            )
        else:
            prompt_obj = context_builder.build_prompt(
                intent_text,
                intent=intent_payload,
                top_k=sample_limit,
                session_id=session_id,
            )

        meta = dict(getattr(prompt_obj, "metadata", {}) or {})
        meta.setdefault("instruction", intent_text)
        meta.setdefault("sample_limit", sample_limit)
        existing_sessions = meta.setdefault("session_ids", [])
        for sid in intent_payload["session_ids"]:
            if sid not in existing_sessions:
                existing_sessions.append(sid)
        meta.setdefault("sample_context", list(prompt_obj.examples))
        prompt_obj.metadata = meta
        return prompt_obj

    def _build_prompt(
        self,
        spec: BotSpec,
        *,
        context_builder: ContextBuilder,
        sample_limit: int = 5,
        _sample_sort_by: str = "confidence",
        _sample_with_vectors: bool = True,
    ) -> Prompt:
        """Backward compatible alias for :meth:`build_training_prompt`."""

        return self.build_training_prompt(
            spec,
            context_builder=context_builder,
            sample_limit=sample_limit,
            _sample_sort_by=_sample_sort_by,
            _sample_with_vectors=_sample_with_vectors,
        )

    def build_bot(
        self,
        spec: BotSpec,
        *,
        context_builder: ContextBuilder,
        model_id: int | None = None,
        sample_limit: int = 5,
        sample_sort_by: str = "outcome_score",
        sample_with_vectors: bool = True,
    ) -> Path:
        """Create code from a spec and save it to a repo.

        Parameters
        ----------
        spec:
            Description of the bot to generate.
        model_id:
            Optional model selector for code generation.
        sample_limit:
            Maximum number of training samples fetched per source for prompt
            generation.
        sample_sort_by:
            Field used when ranking training samples.
        sample_with_vectors:
            Whether to request embeddings for training samples.
        """
        start_time = time.time()
        errors = 0
        try:
            # check for existing code templates first
            if self.db_steward:
                try:
                    existing = self.db_steward.existing_code(spec.name)
                except Exception as exc:
                    self.logger.warning("existing code lookup failed: %s", exc)
                    self._escalate(f"existing code lookup failed: {exc}")
                    if RAISE_ERRORS:
                        errors = 1
                        raise
                    existing = None
            else:
                existing = None

            repo_dir = Path(resolve_path(self.create_env(spec, model_id=model_id)))
            meta = self._write_meta(repo_dir, spec)

            if existing:
                file_path = Path(resolve_path(repo_dir)) / f"{spec.name}.py"
                self._write_with_retry(file_path, existing)
                file_path = Path(resolve_path(file_path))
                self.lint_code(file_path)
                req = self._create_requirements(repo_dir, spec)
                tests = self._create_tests(repo_dir, spec)
                self._validate_repo(repo_dir, spec)
                paths = [file_path, meta] + ([req] if req else []) + tests
                self.version_control(
                    repo_dir, paths, message=f"Initial version of {spec.name}"
                )
                return file_path

            prompt = self._build_prompt(
                spec,
                context_builder=context_builder,
                sample_limit=sample_limit,
                sample_sort_by=sample_sort_by,
                sample_with_vectors=sample_with_vectors,
            )
            file_path = Path(resolve_path(repo_dir)) / f"{spec.name}.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=True)
            result = self._call_codex_api(prompt, path=file_path)
            if not result.success or not result.code:
                errors = 1
                raise RuntimeError(result.error or "engine request failed")
            file_path = Path(resolve_path(file_path))
            self.lint_code(file_path)
            req = self._create_requirements(repo_dir, spec)
            tests = self._create_tests(repo_dir, spec)
            self._validate_repo(repo_dir, spec)
            paths = [file_path, meta] + ([req] if req else []) + tests
            self.version_control(repo_dir, paths, message=f"Initial version of {spec.name}")

            # Explicit manual registration for auditability
            try:
                registry = getattr(self.manager, "bot_registry", None)
                d_bot = getattr(self.manager, "data_bot", None)
                orchestrator = getattr(self.manager, "evolution_orchestrator", None)
                engine = getattr(self.manager, "engine", None)
                pipeline = getattr(self.manager, "pipeline", None)
                if registry and d_bot and engine and pipeline:
                    bot_name = spec.name
                    module_path = str(file_path)
                    _th = get_thresholds(bot_name)
                    registry.register_bot(
                        bot_name,
                        manager=self.manager,
                        data_bot=d_bot,
                        is_coding_bot=True,
                        roi_threshold=_th.roi_drop,
                        error_threshold=_th.error_increase,
                        test_failure_threshold=_th.test_failure_increase,
                    )
                    registry.update_bot(bot_name, module_path)
                    d_bot.reload_thresholds(bot_name)
                    persist_sc_thresholds(
                        bot_name,
                        roi_drop=_th.roi_drop,
                        error_increase=_th.error_increase,
                        test_failure_increase=_th.test_failure_increase,
                    )
                    internalize_coding_bot(
                        bot_name,
                        engine,
                        pipeline,
                        data_bot=d_bot,
                        bot_registry=registry,
                        evolution_orchestrator=orchestrator,
                        roi_threshold=_th.roi_drop,
                        error_threshold=_th.error_increase,
                        test_failure_threshold=_th.test_failure_increase,
                    )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("dynamic bot registration failed for %s", spec.name)

            return file_path
        except Exception:
            errors = 1 if not errors else errors
            raise
        finally:
            self.data_bot.collect(
                bot=self.name,
                response_time=time.time() - start_time,
                errors=errors,
                tests_failed=0,
                tests_run=0,
                revenue=0.0,
                expense=0.0,
            )

    def build_from_plan(
        self,
        data: str,
        *,
        model_id: int | None = None,
    ) -> List[Path]:
        start_time = time.time()
        errors = 0
        specs = self.parse_plan(data)
        try:
            if self.concurrency > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                    files = list(
                        ex.map(
                            lambda s: self.build_bot(
                                s,
                                context_builder=self.context_builder,
                                model_id=model_id,
                            ),
                            specs,
                        )
                    )
            else:
                files = [
                    self.build_bot(
                        s,
                        context_builder=self.context_builder,
                        model_id=model_id,
                    )
                    for s in specs
                ]
        except Exception as exc:
            errors = 1
            msg = f"build_from_plan failed: {exc}"
            self.logger.exception(msg)
            self._escalate(msg)
            raise
        finally:
            self.data_bot.collect(
                bot=self.name,
                response_time=time.time() - start_time,
                errors=errors,
                tests_failed=0,
                tests_run=0,
                revenue=0.0,
                expense=0.0,
            )
        return files


__all__ = ["BotSpec", "BotDevelopmentBot", "RetryStrategy"]
