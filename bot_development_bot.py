"""Bot Development Bot for building bots from handoff specs."""

from __future__ import annotations

import json
import os
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Iterable, Callable, Type
import importlib.util
import keyword

import logging
import subprocess
import shutil
import sys
from dynamic_path_router import resolve_path
import uuid
from snippet_compressor import compress_snippets
from context_builder_util import ensure_fresh_weights
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
from vector_service.context_builder import ContextBuilder, FallbackResult, ErrorResult
from .codex_output_analyzer import (
    validate_stripe_usage,
)
from .self_coding_engine import SelfCodingEngine
from .code_database import CodeDB
from .menace_memory_manager import MenaceMemoryManager

try:  # pragma: no cover - optional dependency
    from . import codex_db_helpers as cdh
except Exception:  # pragma: no cover - optional dependency
    cdh = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - heavy dependency
    from .watchdog import Watchdog

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


try:  # pragma: no cover - optional dependency
    import mss  # type: ignore
    import numpy as np  # type: ignore
    import pytesseract  # type: ignore
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mss = np = pytesseract = cv2 = None  # type: ignore

from . import vision_utils

try:  # pragma: no cover - optional dependency
    from .micro_models.tool_predictor import predict_tools  # type: ignore
    from .micro_models.prefix_injector import inject_prefix  # type: ignore
except Exception:  # pragma: no cover - allow running without predictor
    def predict_tools(spec):  # type: ignore
        return []

    def inject_prefix(prompt, prefix, conf, role="system"):  # type: ignore
        return prompt

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
    "ocr_image": (
        "Call vision_utils.detect_text on the image path or bytes and return "
        "the extracted text"
    ),
    "type_text": (
        "Send keyboard input to the active field using an automation library"
    ),
    "capture_screenshot": (
        "Save a screenshot to the provided path and return the path"
    ),
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
        self.visual_token = self.config.visual_agent_token
        self.token_refresh_cmd = self.config.visual_token_refresh_cmd
        self.desktop_url = self.config.desktop_url
        self.laptop_url = self.config.laptop_url
        self.visual_agents = list(filter(None, self.config.visual_agent_urls)) or [
            self.desktop_url,
            self.laptop_url,
        ]
        self.code_generators = {"python": self._generate_python_code}
        self.denial_phrases = self.config.denial_phrases
        self.headless = self.config.headless
        self.errors: List[str] = []
        self.db_steward = db_steward
        self.default_templates = self.config.default_templates
        self.ocr_region = self.config.ocr_region
        self.error_sinks = list(self.config.error_sinks)
        self.concurrency = self.config.concurrency_workers
        self.config.validate()
        self.file_write_retry = RetryStrategy(
            attempts=self.config.file_write_attempts,
            delay=self.config.file_write_retry_delay,
        )
        self.send_prompt_retry = RetryStrategy(
            attempts=self.config.send_prompt_attempts,
            delay=self.config.send_prompt_retry_delay,
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
        if engine is None:
            try:
                code_db = CodeDB()
            except Exception as exc:  # pragma: no cover - allow running without DB
                self.logger.debug("CodeDB init failed: %s", exc)
                code_db = None  # type: ignore[arg-type]
            try:
                memory_mgr = MenaceMemoryManager()
            except Exception as exc:  # pragma: no cover - allow running without memory
                self.logger.debug("Memory manager init failed: %s", exc)
                memory_mgr = None  # type: ignore[arg-type]
            engine = SelfCodingEngine(
                code_db, memory_mgr, context_builder=self.context_builder
            )
        self.engine = engine
        # warn about missing optional dependencies
        for dep_name, mod in {
            "requests": requests,
            "mss": mss,
            "numpy": np,
            "pytesseract": pytesseract,
            "cv2": cv2,
            "yaml": yaml,
            "Elasticsearch": Elasticsearch,
            "git": Repo,
        }.items():
            if mod is None:
                self.logger.warning("optional dependency %s unavailable", dep_name)

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

    def fetch_patterns(self, spec: BotSpec) -> Dict[str, Any]:
        """Retrieve patterns from Elasticsearch if available."""
        if not self.es:
            return {}
        try:
            query: Dict[str, Any] = {
                "bool": {"must": [{"match": {"language": spec.language}}]}
            }
            if spec.level:
                query["bool"].setdefault("should", []).append(
                    {"match": {"level": spec.level}}
                )
            res = self.es.search(index=self.config.es_index, query=query)
            hits = sorted(
                res.get("hits", {}).get("hits", []),
                key=lambda h: h.get("_score", 0),
                reverse=True,
            )
            patterns: Dict[str, list[tuple[float, str]]] = {}
            for hit in hits:
                src = hit.get("_source", {})
                score = float(hit.get("_score", 0))
                for key, val in src.items():
                    lst = patterns.setdefault(key, [])
                    if str(val) not in [v for _, v in lst]:
                        lst.append((score, str(val)))
            for key in patterns:
                patterns[key].sort(key=lambda t: t[0], reverse=True)
                patterns[key] = patterns[key][:5]
            return patterns
        except Exception as exc:  # pragma: no cover - external service
            self.logger.exception("pattern fetch failed for %s: %s", spec.name, exc)
            self._escalate(f"pattern fetch failed for {spec.name}: {exc}")
            if RAISE_ERRORS:
                raise
            return {}

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

    def generate_code(self, spec: BotSpec, patterns: Dict[str, Any]) -> str:
        generator = self.code_generators.get(spec.language.lower())
        if not generator:
            self.logger.error("Unsupported language: %s", spec.language)
            return "# Unsupported language"
        return generator(spec, patterns)

    def _generate_python_code(self, spec: BotSpec, patterns: Dict[str, Any]) -> str:
        lines = [
            f'"""{spec.purpose}"""',
            "",
            "import logging",
            "",
            f'logger = logging.getLogger("{spec.name}")',
            "",
            "",
        ]
        for func in spec.functions:
            lines.extend(self._python_function_block(func, patterns.get(func, [])))
        return "\n".join(lines)

    def _python_function_block(
        self, func: str, pattern_list: Iterable[tuple[float, str]]
    ) -> List[str]:
        lines = []
        lines.append(f"def {func}(*args, **kwargs) -> None:")
        lines.append(
            f'    """Autogenerated function \'{func}\'.\n\n'
            '    Args:\n        *args: positional arguments.\n'
            '        **kwargs: keyword arguments.\n'
            '    Returns:\n        None."""'
        )
        body = ""
        if pattern_list:
            best = sorted(pattern_list, key=lambda t: t[0], reverse=True)[0]
            body = best[1].strip()
        if body:
            for ln in body.splitlines():
                lines.append(f"    {ln}")
        else:
            template = self._default_template(func)
            for ln in template:
                lines.append(f"    {ln}")
        lines.append("")
        return lines

    def _default_template(self, func: str) -> List[str]:
        for prefix, tmpl in self.default_templates.items():
            if func.startswith(prefix):
                return list(tmpl)
        if func.startswith("fetch") or func.startswith("get"):
            return [
                "import requests",
                "url = args[0] if args else ''",
                "if not isinstance(url, str):",
                "    raise ValueError('url must be a string')",
                "try:",
                "    resp = requests.get(url)",
                "    resp.raise_for_status()",
                "except Exception as exc:",
                "    logger.error('request failed: %s', exc)",
                "    return None",
                "return resp",
            ]
        if func.startswith("save") or func.startswith("write"):
            return [
                "path = args[0] if args else ''",
                "data = args[1] if len(args) > 1 else ''",
                "if not path:",
                "    raise ValueError('missing output path')",
                "try:",
                "    with open(path, 'w') as fh:",
                "        fh.write(str(data))",
                "except Exception as exc:",
                "    logger.error('write failed: %s', exc)",
                "    return False",
                "return True",
            ]
        return [f"logger.info('{func} called')", "return None"]

    def _write_with_retry(self, path: Path, data: str) -> None:
        """Write ``data`` to ``path`` atomically with retries."""

        def writer() -> None:
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
    # Visual agent helpers
    def _screen_denied(self) -> bool | None:
        """Return True if OCR detects refusal text.

        Returns ``None`` when OCR fails so callers can retry.
        """
        if self.headless or not (mss and cv2 and np):
            return False
        attempts = 0
        while attempts < 2:
            try:
                with mss.mss() as sct:
                    left, top, width, height = self.ocr_region
                    img = np.array(
                        sct.grab(
                            {"top": top, "left": left, "width": width, "height": height}
                        )
                    )
                _, png = cv2.imencode(".png", img)
                text = vision_utils.detect_text(png.tobytes())
                if not text and pytesseract:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray)
                if text:
                    text = text.lower()
                    return any(p in text for p in self.denial_phrases)
                attempts += 1
                time.sleep(0.5)
            except Exception as exc:  # pragma: no cover - ocr errors
                logging.error("screenshot OCR failed: %s", exc)
                return None
        return False

    def _poll_agent(self, base: str, name: str) -> tuple[bool, str]:
        """Poll /status until inactive; returns (ok, reason)."""
        if not requests:
            return False, "requests unavailable"
        while True:
            denied = self._screen_denied()
            if denied is True:
                return False, "denied by visual agent"
            try:
                resp = requests.get(f"{base}/status", timeout=10)
            except Exception as exc:
                self.logger.exception("status poll failed: %s", exc)
                self._escalate(f"visual agent status poll failed: {exc}")
                if RAISE_ERRORS:
                    raise
                return False, f"status poll failed: {exc}"
            if resp.status_code != 200:
                return False, f"unexpected status {resp.status_code}"
            data = resp.json()
            status = str(data.get("status", "")).lower()
            if status in {"failed", "error"}:
                snippet = json.dumps(data)[:200]
                self.logger.error("visual agent error for %s: %s", name, snippet)
                return False, f"error status {status}"
            if not data.get("active", False):
                return (
                    status in {"completed", "success", "done", ""},
                    status,
                )
            time.sleep(self.config.visual_agent_poll_interval)

    def _refresh_token(self) -> bool:
        """Refresh visual agent token using configured command."""
        if not self.token_refresh_cmd:
            return False
        for attempt in range(3):
            proc = subprocess.run(
                self.token_refresh_cmd,
                shell=True,
                text=True,
                capture_output=True,
            )
            output = (proc.stdout + proc.stderr).strip()
            if proc.returncode == 0 and proc.stdout.strip():
                self.visual_token = proc.stdout.strip()
                return True
            self.logger.warning(
                "token refresh attempt %s failed: %s",
                attempt + 1,
                output,
            )
            if attempt < 2:
                time.sleep(1.0)
            else:
                self._escalate(f"visual token refresh failed: {output}")
        return False

    def _call_codex_api(self, messages: list[dict[str, str]]) -> EngineResult:
        """Produce helper code via :class:`SelfCodingEngine`.

        All messages are concatenated with their role tags to form a single
        prompt for
        :meth:`SelfCodingEngine.generate_helper`.  If no user prompt is
        provided, the error is escalated and the method returns an
        :class:`EngineResult` describing the failure or raises
        :class:`ValueError` when :attr:`BotDevConfig.raise_errors` is true.
        """

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

        try:
            code = self.engine_retry.run(
                lambda: self.engine.generate_helper(prompt),
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

    def _send_prompt(self, base: str, prompt: str, name: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        def sender() -> tuple[bool, str]:
            resp = requests.post(
                f"{base}/run",
                headers={"x-token": self.visual_token},
                json={"prompt": prompt, "branch": None},
                timeout=10,
            )
            if resp.status_code == 401:
                if self._refresh_token():
                    return sender()
                return False, "unauthorized"
            if resp.status_code == 202:
                return self._poll_agent(base, name)
            snippet = resp.text[:200]
            self.logger.error(
                "visual agent returned %s for %s: %s", resp.status_code, name, snippet
            )
            return False, f"status {resp.status_code}"

        try:
            return self.send_prompt_retry.run(sender, logger=self.logger)
        except Exception as exc:
            self.logger.exception("prompt send failed: %s", exc)
            self._escalate(f"visual agent request failed: {exc}")
            if RAISE_ERRORS:
                raise
            return False, f"exception {exc}"

    def _visual_build(self, prompt: str, name: str) -> bool:
        """Attempt building via visual agents."""
        last_reason = ""
        for url in self.visual_agents:
            ok, reason = self._send_prompt(url, prompt, name)
            if ok:
                return True
            last_reason = reason
            self.logger.error("visual agent %s failed for %s: %s", url, name, reason)
        if last_reason:
            self.errors.append(last_reason)
        return False

    def _build_prompt(
        self,
        spec: BotSpec,
        *,
        context_builder: ContextBuilder,
        sample_limit: int = 5,
        sample_sort_by: str = "confidence",
        sample_with_vectors: bool = True,
    ) -> str:
        """Return the final prompt for the visual agent or Codex.

        Parameters
        ----------
        spec:
            Bot specification describing the desired bot.
        sample_limit:
            Maximum number of training examples fetched via
            :func:`codex_db_helpers.aggregate_samples`.
        sample_sort_by:
            Field used when ranking training examples.
        sample_with_vectors:
            Whether to request embedding vectors for the training examples.
        """
        if context_builder is None:
            raise ValueError("context_builder is required")
        query = spec.description or spec.purpose or spec.name
        session_id = uuid.uuid4().hex
        ctx_result = context_builder.build(
            query, session_id=session_id, include_vectors=True
        )
        context_session_id = session_id
        vector_metadata: list[tuple[str, str, float]] = []
        retrieval_context: str | Dict[str, Any] = ""
        if isinstance(ctx_result, (ErrorResult, FallbackResult)):
            retrieval_context = ""
        else:
            if isinstance(ctx_result, tuple):
                retrieval_context, context_session_id, vector_metadata = ctx_result
            else:
                retrieval_context = ctx_result
            retrieval_context = compress_snippets({"snippet": retrieval_context}).get(
                "snippet", retrieval_context
            )

        predicted_tool = ""
        pred_conf = 0.0
        try:
            preds = predict_tools(spec)
            if preds:
                predicted_tool, pred_conf = preds[0]
        except Exception:
            predicted_tool = ""
            pred_conf = 0.0

        # Gather historical examples to provide additional prompt context
        samples = []
        if cdh is not None:
            try:
                samples = cdh.aggregate_samples(
                    sort_by=sample_sort_by,
                    limit=sample_limit,
                    include_embeddings=sample_with_vectors,
                    scope=cdh.Scope.ALL,
                )
            except Exception:
                samples = []
        sample_context = "\n".join(
            s.content for s in samples if getattr(s, "content", "")
        )

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

        prompt = (
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
        if predicted_tool:
            prompt = inject_prefix(
                prompt,
                f"Suggested Tool: {predicted_tool}",
                pred_conf,
                role="system",
            )
        if retrieval_context:
            if not isinstance(retrieval_context, str):
                retrieval_context = json.dumps(retrieval_context, indent=2)
            prompt += "\n\nContext:\n" + retrieval_context
            meta_block = json.dumps(
                {"context_session_id": context_session_id, "vectors": vector_metadata},
                indent=2,
            )
            prompt += "\n\nContext Metadata:\n" + meta_block
        if sample_context:
            prompt += "\n\n### Training Examples\n" + sample_context

        return prompt

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
            Optional model selector for the visual agent.
        sample_limit:
            Maximum number of training samples fetched per source for prompt
            generation.
        sample_sort_by:
            Field used when ranking training samples.
        sample_with_vectors:
            Whether to request embeddings for training samples.
        """
        # check for existing code templates first
        if self.db_steward:
            try:
                existing = self.db_steward.existing_code(spec.name)
            except Exception as exc:
                self.logger.warning("existing code lookup failed: %s", exc)
                self._escalate(f"existing code lookup failed: {exc}")
                if RAISE_ERRORS:
                    raise
                existing = None
        else:
            existing = None

        patterns = self.fetch_patterns(spec)
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
        built = self._visual_build(prompt, spec.name)

        if built:
            # read repo contents back in
            for p in repo_dir.rglob("*.py"):
                try:
                    _ = p.read_text()
                    self.lint_code(p)
                except Exception as exc:
                    self.logger.exception("file read failed for %s: %s", spec.name, exc)
                    self._escalate(f"bot file read failed for {spec.name}: {exc}")
                    if RAISE_ERRORS:
                        raise
            files = [Path(resolve_path(p)) for p in repo_dir.glob(f"{spec.name}.py")]
            if files:
                req = self._create_requirements(repo_dir, spec)
                tests = self._create_tests(repo_dir, spec)
                self._validate_repo(repo_dir, spec)
                paths = files + [meta] + ([req] if req else []) + tests
                self.version_control(
                    repo_dir, paths, message=f"Initial version of {spec.name}"
                )
                return files[0]

        # visual agent failed -> fallback strategies
        self.logger.warning(
            "Visual build failed for %s, switching to fallback", spec.name
        )
        if not self.errors or not self.errors[-1].startswith("visual"):
            self.errors.append("visual build failed")
        self.logger.info("Attempting engine fallback for %s", spec.name)
        try:
            code = self.engine_retry.run(
                lambda: self.engine.generate_helper(prompt),
                logger=self.logger,
            )
            if not code:
                raise RuntimeError("empty response")
        except Exception as exc:
            msg = f"engine request failed: {exc}"
            self.logger.exception(msg)
            self._escalate(msg)
            self.errors.append(msg)
            if RAISE_ERRORS:
                raise RuntimeError("engine request failed") from exc
            code = self.generate_code(spec, patterns)

        file_path = Path(resolve_path(repo_dir)) / f"{spec.name}.py"
        self._write_with_retry(file_path, code)
        file_path = Path(resolve_path(file_path))
        self.lint_code(file_path)
        req = self._create_requirements(repo_dir, spec)
        tests = self._create_tests(repo_dir, spec)
        self._validate_repo(repo_dir, spec)
        paths = [file_path, meta] + ([req] if req else []) + tests
        self.version_control(repo_dir, paths, message=f"Initial version of {spec.name}")
        return file_path

    def build_from_plan(
        self,
        data: str,
        *,
        model_id: int | None = None,
    ) -> List[Path]:
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
            msg = f"build_from_plan failed: {exc}"
            self.logger.exception(msg)
            self._escalate(msg)
            raise
        return files


__all__ = ["BotSpec", "BotDevelopmentBot", "RetryStrategy"]
