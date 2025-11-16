"""Utility to analyse and flag AI generated Python code snippets."""

from __future__ import annotations

import ast
import json
import logging
import logging.config
import os
import argparse
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from importlib import import_module
import sys
import tokenize
import io
from pathlib import Path
from stripe_detection import (
    PAYMENT_KEYWORDS,
    HTTP_LIBRARIES,
    contains_payment_keyword,
)

from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# default location for bundled pattern configuration
_BASE_PATH = resolve_path(".")
try:
    DEFAULT_PATTERN_CONFIG_PATH = resolve_path(
        "codex_output_analyzer/config/default_pattern_config.json"
    )
except FileNotFoundError:
    DEFAULT_PATTERN_CONFIG_PATH = _BASE_PATH / "codex_output_analyzer/config/default_pattern_config.json"

# default severity map configuration
try:
    DEFAULT_SEVERITY_MAP_PATH = resolve_path(
        "codex_output_analyzer/config/default_severity_map.json"
    )
except FileNotFoundError:
    DEFAULT_SEVERITY_MAP_PATH = _BASE_PATH / "codex_output_analyzer/config/default_severity_map.json"

# default suspicious words for comment/docstring extraction
DEFAULT_SUSPICIOUS_WORDS: Set[str] = {
    "todo",
    "fixme",
    "hack",
    "xxx",
    "nosec",
    "noqa",
    "insecure",
}


class CriticalGenerationFailure(RuntimeError):
    """Raised when forbidden payment integrations are detected."""


_RAW_STRIPE_PATTERN = re.compile(
    r"api\.stripe\.com|"  # direct endpoint
    r"(?:sk|pk)_[A-Za-z0-9*xX]+|"  # raw or masked keys
    r"(?:c2tf|cGtf)[A-Za-z0-9+/=]{8,}|"  # base64-encoded keys
    r"(?:YXBpLnN0cmlwZS5jb20=?|aHR0cHM6Ly9hcGkuc3RyaXBlLmNvbQ==)"  # base64 api URL
)


def validate_stripe_usage_generic(text: str) -> None:
    """Raise if *text* contains Stripe endpoints, keys or keywords without router."""

    lowered = text.lower()
    if _RAW_STRIPE_PATTERN.search(text):
        if "stripe_billing_router" not in lowered:
            raise CriticalGenerationFailure(
                "critical generation failure: raw Stripe usage detected",
            )
    if any(keyword in lowered for keyword in PAYMENT_KEYWORDS):
        if "stripe_billing_router" not in lowered:
            raise CriticalGenerationFailure(
                "critical generation failure: payment keywords without stripe_billing_router",
            )


def validate_stripe_usage(code: str) -> None:
    """Scan *code* for disallowed Stripe usage.

    This checker looks for direct ``stripe`` imports, calls to the Stripe API
    via common HTTP libraries (``requests``, ``httpx``, ``aiohttp``, ``urllib``
    and ``urllib3``) and payment related identifiers used without the
    ``stripe_billing_router`` safety wrapper.

    Parameters
    ----------
    code:
        Source code to inspect.

    Raises
    ------
    CriticalGenerationFailure
        If Stripe is used directly or payment keywords appear without
        importing ``stripe_billing_router``.
    """
    validate_stripe_usage_generic(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:  # pragma: no cover - invalid code
        raise CriticalGenerationFailure(
            "critical generation failure: could not parse code"
        ) from exc

    class _StripeVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stripe_import = False
            self.has_router_import = False
            self.router_used = False
            self.has_payment_keyword = False
            self.raw_api_call = False
            self.http_names: dict[str, str] = {}

        def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover - simple
            for alias in node.names:
                module = alias.name
                asname = alias.asname or module
                if module == "stripe" or module.startswith("stripe."):
                    self.stripe_import = True
                if module in HTTP_LIBRARIES:
                    self.http_names[asname] = module
                if module == "stripe_billing_router":
                    self.has_router_import = True
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pragma: no cover - simple
            module = node.module or ""
            if module == "stripe" or module.startswith("stripe."):
                self.stripe_import = True
            if module in HTTP_LIBRARIES:
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.http_names[name] = module
            for alias in node.names:
                if alias.name == "stripe_billing_router":
                    self.has_router_import = True
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # pragma: no cover - simple
            root = _root_name(node.func)
            if root == "stripe_billing_router":
                self.router_used = True
            elif root in self.http_names:
                if any(
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and "api." "stripe.com" in arg.value
                    for arg in node.args
                ):
                    self.raw_api_call = True
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:  # pragma: no cover - simple
            if node.id == "stripe_billing_router" and isinstance(node.ctx, ast.Load):
                self.router_used = True
            elif isinstance(node.ctx, ast.Store):
                name = node.id
                if not name.isupper() and name != "stripe_billing_router":
                    if contains_payment_keyword(name):
                        self.has_payment_keyword = True
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:  # pragma: no cover - simple
            if isinstance(node.ctx, ast.Store):
                name = node.attr
                if not name.isupper() and name != "stripe_billing_router":
                    if contains_payment_keyword(name):
                        self.has_payment_keyword = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - simple
            if contains_payment_keyword(node.name):
                self.has_payment_keyword = True
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover
            if contains_payment_keyword(node.name):
                self.has_payment_keyword = True
            self.generic_visit(node)

    def _root_name(node: ast.AST) -> str | None:
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
        return None

    visitor = _StripeVisitor()
    visitor.visit(tree)

    if visitor.stripe_import:
        raise CriticalGenerationFailure(
            "critical generation failure: direct Stripe import detected"
        )
    if visitor.raw_api_call:
        raise CriticalGenerationFailure(
            "critical generation failure: direct Stripe API call detected"
        )
    if visitor.has_payment_keyword:
        if not visitor.has_router_import:
            raise CriticalGenerationFailure(
                "critical generation failure: payment keywords without stripe_billing_router import"
            )
        if not visitor.router_used:
            raise CriticalGenerationFailure(
                "critical generation failure: stripe_billing_router imported but unused"
            )


def configure_logging(
    logger_obj: Optional[logging.Logger] = None,
    *,
    level: str = "INFO",
    log_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    handlers: Optional[List[logging.Handler]] = None,
) -> logging.Logger:
    """Configure logging for :mod:`codex_output_analyzer`.

    Parameters
    ----------
    level:
        Logging level name. Ignored if *config* supplied.
    log_file:
        Optional file path for an additional :class:`~logging.FileHandler`.
    config:
        Optional `dictConfig` style dictionary.
    handlers:
        Optional list of pre-created handlers to attach when *config* is None.
    """

    if config:
        logging.config.dictConfig(config)
        return logging.getLogger(logger_obj.name if logger_obj else __name__)

    target = logger_obj or logger
    target.setLevel(getattr(logging, level.upper(), logging.INFO))
    target.handlers.clear()

    handlers = list(handlers or [])
    if not handlers:
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    for h in handlers:
        target.addHandler(h)

    return target


class AnalysisError(Enum):
    """Standardised error codes returned by this module."""

    SYNTAX_ERROR = "syntax_error"
    PLUGIN_ERROR = "plugin_error"
    CONFIG_ERROR = "config_error"


class Severity(Enum):
    """Severity ranking for unsafe patterns."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class PatternConfig:
    """Configuration for unsafe pattern detection."""

    dangerous_imports: Set[str]
    reward_keywords: Set[str]
    severity_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class UnsafePattern:
    """Structured information about a flagged pattern."""

    message: str
    severity: Severity
    line: Optional[int]
    node: Optional[str]
    count: int = 1

    def to_dict(
        self,
        *,
        include_context: bool = False,
        include_count: bool = False,
    ) -> Dict[str, Any]:
        data = {"message": self.message, "severity": self.severity.value}
        if include_context:
            data.update({"line": self.line, "node": self.node})
        if include_count:
            data["count"] = self.count
        return data


def _load_default_pattern_config(path: Optional[Path | str] = None) -> "PatternConfig":
    """Return pattern config loaded from JSON if available."""

    if path is None:
        env = os.getenv("CODEX_DEFAULT_PATTERN_CONFIG")
        if env:
            try:
                path = resolve_path(env)
            except FileNotFoundError:
                path = _BASE_PATH / env
        else:
            path = DEFAULT_PATTERN_CONFIG_PATH
    else:
        try:
            path = resolve_path(path)
        except FileNotFoundError:
            path = _BASE_PATH / str(path)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("loaded default pattern config from %s", path)
            return PatternConfig(
                dangerous_imports=set(data.get("dangerous_imports", [])),
                reward_keywords=set(data.get("reward_keywords", [])),
                severity_overrides=data.get("severity_overrides", {}),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to load default pattern config %s: %s", path, exc)

    logger.warning("using builtin pattern configuration")
    return PatternConfig(
        dangerous_imports={
            "os",
            "subprocess",
            "socket",
            "requests",
            "urllib",
            "http",
            "shutil",
        },
        reward_keywords={"reward", "kpi", "dispatch"},
        severity_overrides={},
    )


DEFAULT_PATTERN_CONFIG = _load_default_pattern_config()


def _load_default_severity_map(path: Optional[Path | str] = None) -> Dict[str, Severity]:
    """Return severity mapping loaded from JSON if available."""

    if path is None:
        env = os.getenv("CODEX_DEFAULT_SEVERITY_MAP")
        if env:
            try:
                path = resolve_path(env)
            except FileNotFoundError:
                path = _BASE_PATH / env
        else:
            path = DEFAULT_SEVERITY_MAP_PATH
    else:
        try:
            path = resolve_path(path)
        except FileNotFoundError:
            path = _BASE_PATH / str(path)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            logger.debug("loaded default severity map from %s", path)
            return {k: Severity(v) for k, v in raw.items()}
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to load default severity map %s: %s", path, exc)

    logger.warning("using builtin severity mapping")
    return {
        "dangerous_import": Severity.MEDIUM,
        "import_from_dangerous": Severity.MEDIUM,
        "wildcard_import": Severity.LOW,
        "filesystem_process_call": Severity.MEDIUM,
        "network_call": Severity.MEDIUM,
        "dangerous_builtin_open": Severity.MEDIUM,
        "dangerous_builtin_eval": Severity.HIGH,
        "dangerous_builtin_exec": Severity.HIGH,
        "dangerous_builtin_compile": Severity.MEDIUM,
        "subprocess_shell_true": Severity.HIGH,
        "eval_on_input": Severity.HIGH,
        "bare_except": Severity.LOW,
        "infinite_while": Severity.MEDIUM,
        "file_open_in_with": Severity.LOW,
        "suspicious_identifier": Severity.LOW,
        "suspicious_attribute": Severity.LOW,
    }


# version tag for generated analysis schemas
ANALYSIS_SCHEMA_VERSION = 1

PLUGIN_INTERFACE_VERSION = 1


def _config_from_dict(data: Dict[str, Any]) -> PatternConfig:
    """Validate and convert a raw dictionary to :class:`PatternConfig`."""

    if not isinstance(data, dict):
        raise TypeError("pattern config must be a dict")
    allowed = {"dangerous_imports", "reward_keywords", "severity_overrides"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys in pattern config: {sorted(unknown)}")
    dangerous = data.get("dangerous_imports", [])
    rewards = data.get("reward_keywords", [])
    overrides = data.get("severity_overrides", {})
    if not isinstance(dangerous, (list, set)):
        raise TypeError("dangerous_imports must be list or set")
    if not isinstance(rewards, (list, set)):
        raise TypeError("reward_keywords must be list or set")
    if not isinstance(overrides, dict):
        raise TypeError("severity_overrides must be a dict")
    for k, v in overrides.items():
        if str(v) not in {s.value for s in Severity}:
            raise ValueError(f"invalid severity level {v} for {k}")
    return PatternConfig(
        dangerous_imports=set(dangerous),
        reward_keywords=set(rewards),
        severity_overrides={k: str(v) for k, v in overrides.items()},
    )


def load_pattern_config(
    source: Optional[str] = None,
    *,
    default_path: Optional[str] = None,
) -> PatternConfig:
    """Load pattern configuration from JSON file or plugin module."""

    if source is None:
        source = os.environ.get("CODEX_PATTERN_SOURCE")

    if not source:
        if default_path is not None or os.getenv("CODEX_DEFAULT_PATTERN_CONFIG"):
            return _load_default_pattern_config(default_path)
        return DEFAULT_PATTERN_CONFIG

    if source.endswith(".json"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"pattern config file {source} not found")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _config_from_dict(data)

    try:
        plugin = import_module(source)
    except Exception as exc:  # pragma: no cover - plugin loading errors
        logger.exception("failed to load pattern plugin %s", source)
        raise ImportError(f"could not import pattern plugin {source}") from exc

    if not hasattr(plugin, "get_pattern_config"):
        raise AttributeError(f"plugin {source} missing get_pattern_config")

    version = getattr(plugin, "PATTERN_PLUGIN_VERSION", 1)
    try:
        major = int(str(version).split(".")[0])
    except Exception as exc:
        raise ValueError(f"invalid plugin version {version}") from exc
    if major != PLUGIN_INTERFACE_VERSION:
        raise ValueError(
            f"plugin {source} version {version} incompatible with {PLUGIN_INTERFACE_VERSION}.x"
        )

    cfg = plugin.get_pattern_config()
    if isinstance(cfg, PatternConfig):
        return cfg
    if isinstance(cfg, dict):
        return _config_from_dict(cfg)
    raise TypeError("get_pattern_config must return PatternConfig or dict")


def load_pattern_source(
    source: Optional[str] = None,
    *,
    default_path: Optional[str] = None,
) -> tuple[PatternConfig, List[ast.NodeVisitor]]:
    """Load pattern config and any custom AST visitors from a plugin."""

    config = load_pattern_config(source, default_path=default_path)
    visitors: List[ast.NodeVisitor] = []

    if source and not source.endswith(".json"):
        try:
            plugin = import_module(source)
            if hasattr(plugin, "get_custom_visitors"):
                for vis in plugin.get_custom_visitors() or []:
                    inst = vis() if isinstance(vis, type) else vis
                    if isinstance(inst, ast.NodeVisitor):
                        visitors.append(inst)
                    else:
                        logger.warning("custom visitor %s is not a NodeVisitor", vis)
        except Exception:  # pragma: no cover - plugin loading errors
            logger.exception("failed loading custom visitors from %s", source)
            raise

    return config, visitors


def _get_call_name(node: ast.AST) -> Optional[str]:
    """Return dotted name for a Call node if possible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _get_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


class _CodeAnalyzer(ast.NodeVisitor):
    """Collect high level information from AST."""

    def __init__(self) -> None:
        self.functions: List[str] = []
        self.function_details: List[Dict[str, Any]] = []
        self.classes: List[str] = []
        self.imports: List[str] = []
        self.calls: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        params = [a.arg for a in node.args.args]
        complexity = len(list(ast.walk(node)))
        self.function_details.append(
            {"name": node.name, "parameters": params, "complexity": complexity}
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        params = [a.arg for a in node.args.args]
        complexity = len(list(ast.walk(node)))
        self.function_details.append(
            {"name": node.name, "parameters": params, "complexity": complexity}
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            name = f"{module}.{alias.name}" if module else alias.name
            self.imports.append(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)


class _UnsafePatternFinder(ast.NodeVisitor):
    """Detect potentially risky patterns in the AST."""

    def __init__(
        self,
        config: Optional[PatternConfig] = None,
        severity_map: Optional[Dict[str, Severity]] = None,
    ) -> None:
        self.config = config or DEFAULT_PATTERN_CONFIG
        self.severity_map: Dict[str, Severity] = dict(
            severity_map or _load_default_severity_map()
        )
        for key, val in self.config.severity_overrides.items():
            try:
                self.severity_map[key] = Severity(val)
            except ValueError:
                logger.warning("invalid severity %s for %s", val, key)
        self.flags: List[UnsafePattern] = []

    def _add_flag(self, message: str, severity: Severity, node: ast.AST) -> None:
        self.flags.append(
            UnsafePattern(
                message,
                severity,
                getattr(node, "lineno", None),
                node.__class__.__name__,
            )
        )

    def _sev(self, key: str) -> Severity:
        if key not in self.severity_map:
            logger.info("unknown severity pattern %s; defaulting to MEDIUM", key)
            return Severity.MEDIUM
        return self.severity_map[key]

    # imports
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod in self.config.dangerous_imports:
                self._add_flag(
                    f"import of {alias.name}",
                    self._sev("dangerous_import"),
                    node,
                )
                logger.debug("unsafe import detected: %s", alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            mod = node.module.split(".")[0]
            if mod in self.config.dangerous_imports:
                self._add_flag(
                    f"import from {node.module}",
                    self._sev("import_from_dangerous"),
                    node,
                )
                logger.debug("unsafe import-from detected: %s", node.module)
        for alias in node.names:
            if alias.name == "*":
                self._add_flag(
                    f"wildcard import from {node.module or ''}",
                    self._sev("wildcard_import"),
                    node,
                )
                logger.debug("wildcard import detected: %s", node.module or "")
        self.generic_visit(node)

    # calls
    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node.func)
        if not name:
            self.generic_visit(node)
            return
        lowered = name.lower()
        if lowered.startswith(("os.", "subprocess.", "shutil.")):
            self._add_flag(
                f"filesystem or process call {name}",
                self._sev("filesystem_process_call"),
                node,
            )
            logger.debug("unsafe filesystem/process call: %s", name)
        if lowered.startswith(("requests", "urllib", "http", "socket")):
            self._add_flag(
                f"network call {name}",
                self._sev("network_call"),
                node,
            )
            logger.debug("unsafe network call: %s", name)
        if lowered in {"open", "eval", "exec", "compile"}:
            if lowered == "open":
                sev = self._sev("dangerous_builtin_open")
            elif lowered == "eval":
                sev = self._sev("dangerous_builtin_eval")
            elif lowered == "exec":
                sev = self._sev("dangerous_builtin_exec")
            else:
                sev = self._sev("dangerous_builtin_compile")
            self._add_flag(f"dangerous call {name}", sev, node)
            logger.debug("dangerous built-in call: %s", name)
        if lowered.startswith("subprocess"):
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    self._add_flag(
                        "subprocess with shell=True",
                        self._sev("subprocess_shell_true"),
                        node,
                    )
                    logger.debug("subprocess shell=True detected")
        if lowered in {"eval", "exec"}:
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    inner = _get_call_name(arg.func)
                    if inner and inner.lower() == "input":
                        self._add_flag(
                            f"{name} on input()",
                            self._sev("eval_on_input"),
                            node,
                        )
                        logger.debug("%s on input() detected", name)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if handler.type is None:
                self._add_flag(
                    "bare except",
                    self._sev("bare_except"),
                    node,
                )
                logger.debug("bare except detected")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self._add_flag(
                "infinite while loop",
                self._sev("infinite_while"),
                node,
            )
            logger.debug("infinite while loop detected")
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                cname = _get_call_name(ctx.func)
                if cname and cname.lower() == "open":
                    self._add_flag(
                        "file open in with",
                        self._sev("file_open_in_with"),
                        node,
                    )
                    logger.debug("file open in with detected")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.lower() in self.config.reward_keywords:
            self._add_flag(
                f"suspicious identifier {node.id}",
                self._sev("suspicious_identifier"),
                node,
            )
            logger.debug("suspicious identifier used: %s", node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr.lower()
        if attr in self.config.reward_keywords:
            self._add_flag(
                f"suspicious attribute {node.attr}",
                self._sev("suspicious_attribute"),
                node,
            )
            logger.debug("suspicious attribute used: %s", node.attr)
        self.generic_visit(node)


def analyze_generated_code(code_str: str) -> Dict[str, Any]:
    """Parse Python code and return structured information about its contents."""

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:  # pragma: no cover - invalid code
        return {
            "error": {
                "type": AnalysisError.SYNTAX_ERROR.value,
                "details": str(exc),
            }
        }

    analyzer = _CodeAnalyzer()
    analyzer.visit(tree)

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "functions": analyzer.functions,
        "function_details": analyzer.function_details,
        "classes": analyzer.classes,
        "imports": analyzer.imports,
        "calls": analyzer.calls,
    }


def flag_unsafe_patterns(
    parsed_ast: ast.AST,
    config: Optional[PatternConfig] = None,
    *,
    include_context: bool = False,
    extra_visitors: Optional[List[ast.NodeVisitor]] = None,
    include_all: bool = False,
    include_counts: bool = False,
    return_objects: bool = False,
) -> List[Any]:
    """Return descriptions of risky patterns detected in the AST."""

    finder = _UnsafePatternFinder(config)
    visitors = [finder] + list(extra_visitors or [])
    for visitor in visitors:
        visitor.visit(parsed_ast)

    all_flags: List[UnsafePattern] = []
    for visitor in visitors:
        all_flags.extend(getattr(visitor, "flags", []))

    if include_all:
        patterns = all_flags
    else:
        dedup: Dict[str, UnsafePattern] = {}
        for flag in all_flags:
            if flag.message not in dedup:
                dedup[flag.message] = UnsafePattern(
                    flag.message,
                    flag.severity,
                    flag.line,
                    flag.node,
                )
            else:
                dedup[flag.message].count += 1
        patterns = list(dedup.values())

    if return_objects:
        return patterns

    return [
        p.to_dict(include_context=include_context or include_all, include_count=include_counts)
        for p in patterns
    ]


def log_analysis_result(result: Dict[str, Any], output_path: str) -> None:
    """Persist analysis results as JSON for later auditing."""

    if result.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        logger.warning(
            "analysis result schema %s does not match expected %s",
            result.get("schema_version"),
            ANALYSIS_SCHEMA_VERSION,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_with_ts = dict(result)
    result_with_ts["timestamp"] = datetime.utcnow().isoformat()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_with_ts, f, indent=2, sort_keys=True)
    logger.info("analysis saved to %s", output_path)


def load_analysis_result(path: str) -> Dict[str, Any]:
    """Load previously saved analysis results and validate schema."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema_version")
    if version != ANALYSIS_SCHEMA_VERSION:
        logger.warning(
            "loaded analysis schema %s does not match expected %s",
            version,
            ANALYSIS_SCHEMA_VERSION,
        )
    return data


def run_pattern_tests(
    path: str, config: PatternConfig, visitors: Optional[List[ast.NodeVisitor]] = None
) -> bool:
    """Run pattern matching tests from a JSON file."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception as exc:
        logger.error("failed to load test cases: %s", exc)
        return False

    all_passed = True
    for idx, case in enumerate(cases):
        code = case.get("code", "")
        expected = set(case.get("flags", []))
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            logger.error("case %s syntax error: %s", idx, exc)
            all_passed = False
            continue
        found = {
            f["message"]
            for f in flag_unsafe_patterns(tree, config, extra_visitors=visitors or [])
        }
        if found != expected:
            logger.error(
                "case %s failed: expected %s got %s",
                idx,
                sorted(expected),
                sorted(found),
            )
            all_passed = False
    if all_passed:
        logger.info("all pattern tests passed")
    return all_passed


def extract_comments_and_docstrings(
    code_str: str, *, suspicious_words: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """Return list of comments and docstrings found in *code_str* with metadata."""

    suspicious_words = set(suspicious_words or DEFAULT_SUSPICIOUS_WORDS)
    items: List[Dict[str, Any]] = []
    tokens = tokenize.generate_tokens(io.StringIO(code_str).readline)
    for toknum, tokval, start, _, _ in tokens:
        if toknum == tokenize.COMMENT:
            content = tokval.lstrip("# ").strip()
            comment_type = "inline" if start[1] > 0 else "standalone"
            item = {
                "type": "comment",
                "content": content,
                "line": start[0],
                "comment_type": comment_type,
            }
            lowered = content.lower()
            flagged = [w for w in suspicious_words if w in lowered]
            if flagged:
                item["suspicious_keywords"] = flagged
            items.append(item)

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:  # pragma: no cover - invalid code
        logger.error("%s while extracting docstrings", exc)
        return items

    for node in [tree, *ast.walk(tree)]:
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            doc = ast.get_docstring(node)
            if doc:
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    lineno = node.body[0].value.lineno
                else:
                    lineno = getattr(node, "lineno", None)
                item = {
                    "type": "docstring",
                    "content": doc,
                    "line": lineno,
                    "node": node.__class__.__name__,
                    "docstring_type": node.__class__.__name__.lower(),
                }
                if hasattr(node, "name"):
                    item["name"] = node.name
                lowered = doc.lower()
                flagged = [w for w in suspicious_words if w in lowered]
                if flagged:
                    item["suspicious_keywords"] = flagged
                items.append(item)
    return items


def analyze_source(
    code: str,
    pattern_source: Optional[str] = None,
    *,
    include_flag_context: bool = False,
    default_pattern_config: Optional[str] = None,
    include_counts: bool = False,
) -> Dict[str, Any]:
    """Analyse *code* and return a structured report."""

    try:
        config, visitors = load_pattern_source(
            pattern_source, default_path=default_pattern_config
        )
    except Exception as exc:  # pragma: no cover - plugin errors
        logger.error("failed to load pattern configuration: %s", exc)
        config = DEFAULT_PATTERN_CONFIG
        visitors = []
        config_error = str(exc)
    else:
        config_error = None

    result = analyze_generated_code(code)
    if "error" not in result:
        flags = flag_unsafe_patterns(
            ast.parse(code),
            config=config,
            include_context=include_flag_context,
            extra_visitors=visitors,
            include_counts=include_counts,
        )
        result["flags"] = flags
        if config_error:
            result["config_error"] = {
                "type": AnalysisError.CONFIG_ERROR.value,
                "details": config_error,
            }
    return result


def analyze_file(
    path: str,
    pattern_source: Optional[str] = None,
    *,
    include_flag_context: bool = False,
    max_size: Optional[int] = None,
    default_pattern_config: Optional[str] = None,
    include_counts: bool = False,
) -> Dict[str, Any]:
    if max_size is not None and os.path.getsize(path) > max_size:
        raise ValueError(f"file {path} exceeds maximum allowed size {max_size} bytes")
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    return analyze_source(
        code,
        pattern_source,
        include_flag_context=include_flag_context,
        default_pattern_config=default_pattern_config,
        include_counts=include_counts,
    )


def cli(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for analysing Python code."""

    parser = argparse.ArgumentParser(description="Analyse generated Python code")
    parser.add_argument("path", help="Path to the Python file to analyse")
    parser.add_argument("--output", help="File to write JSON results to")
    parser.add_argument(
        "--pattern-source",
        help="JSON file or plugin module providing pattern configuration",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument("--log-file", help="Optional file to write logs to")
    parser.add_argument(
        "--test-cases",
        help="Run pattern matching tests from the specified JSON file and exit",
    )
    parser.add_argument(
        "--flag-context",
        action="store_true",
        help="Include full flag context in results",
    )
    parser.add_argument(
        "--flag-counts",
        action="store_true",
        help="Include count of deduplicated flags",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2 * 1024 * 1024,
        help="Maximum file size in bytes",
    )
    parser.add_argument(
        "--cpu-time",
        type=int,
        help="Optional CPU time limit in seconds",
    )
    parser.add_argument(
        "--default-pattern-config",
        help="Path to default pattern configuration JSON file",
    )
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, log_file=args.log_file)

    if args.test_cases:
        cfg, visitors = load_pattern_source(
            args.pattern_source, default_path=args.default_pattern_config
        )
        ok = run_pattern_tests(args.test_cases, cfg, visitors)
        return 0 if ok else 1

    if args.cpu_time:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_time, args.cpu_time))
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("resource module cpu limit failed: %s", exc)
            if os.name == "nt":
                import threading

                def _kill() -> None:
                    logger.error("CPU time limit exceeded")
                    os._exit(1)

                timer = threading.Timer(args.cpu_time, _kill)
                timer.daemon = True
                timer.start()
            else:
                logger.warning("failed to apply cpu limit: %s", exc)

    result = analyze_file(
        args.path,
        args.pattern_source,
        include_flag_context=args.flag_context,
        max_size=args.max_size,
        default_pattern_config=args.default_pattern_config,
        include_counts=args.flag_counts,
    )

    if args.output:
        log_analysis_result(result, args.output)
    else:
        print(json.dumps(result, indent=2))

    return 0


def main(argv: Optional[List[str]] = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
