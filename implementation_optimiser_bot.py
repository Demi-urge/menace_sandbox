"""Implementation Optimiser Bot for refining task packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
import ast
import logging
import time
from vector_service.context_builder import ContextBuilder
from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds
from .coding_bot_interface import self_coding_managed
from .task_handoff_bot import TaskPackage, TaskInfo
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator = get_orchestrator(
    "ImplementationOptimiserBot", data_bot, engine
)
_th = get_thresholds("ImplementationOptimiserBot")
persist_sc_thresholds(
    "ImplementationOptimiserBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)
manager = internalize_coding_bot(
    "ImplementationOptimiserBot",
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


@dataclass
class ImplementationAdvice:
    """Advice or optimised snippet for a task."""

    name: str
    optimised_code: str


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class ImplementationOptimiserBot:
    """Receive ``TaskPackage`` objects and refine them.

    When generating missing implementations this bot can produce simple Python
    or shell templates.  The Python template has two flavours: a minimal style
    that simply defines the required functions, and a logging style that wraps
    the body in basic ``try``/``except`` blocks and emits log messages.
    """

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        manager: SelfCodingManager | None = None,
    ) -> None:
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.history: List[TaskPackage] = []
        self.context_builder = context_builder
        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:
            logger.error("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc
        eng = getattr(manager, "engine", None)
        if eng is not None:
            try:
                existing_cb = eng.context_builder  # type: ignore[attr-defined]
            except AttributeError as exc:  # pragma: no cover - defensive check
                raise AttributeError(
                    "manager.engine must provide a context_builder"
                ) from exc
            if existing_cb is None:
                raise ValueError("manager.engine.context_builder cannot be None")
            eng.context_builder = context_builder  # type: ignore[attr-defined]
            cb = eng.context_builder  # type: ignore[attr-defined]
            if hasattr(cb, "refresh_db_weights"):
                try:
                    cb.refresh_db_weights()  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - best effort
                    pass
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot

    # ------------------------------------------------------------------
    @staticmethod
    def _optimise_python(code: str) -> str:
        """Apply trivial AST based clean up to Python code."""
        try:
            tree = ast.parse(code)

            class Cleaner(ast.NodeTransformer):
                def __init__(self) -> None:
                    super().__init__()
                    self._stack: list[ast.AST] = []

                def generic_visit(self, node: ast.AST):  # type: ignore[override]
                    self._stack.append(node)
                    new_node = super().generic_visit(node)
                    self._stack.pop()
                    return new_node

                def visit_Pass(self, node):  # type: ignore[override]
                    parent = self._stack[-1] if self._stack else None
                    body = getattr(parent, "body", None) if parent else None
                    if isinstance(body, list) and len(body) > 1:
                        return None
                    return node

                @staticmethod
                def _strip_blank_docstring(node: ast.AST) -> None:
                    body = getattr(node, "body", None)
                    if (
                        isinstance(body, list)
                        and body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                        and not body[0].value.value.strip()
                    ):
                        body.pop(0)

                def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
                    self.generic_visit(node)
                    self._strip_blank_docstring(node)
                    return node

                def visit_AsyncFunctionDef(
                    self, node: ast.AsyncFunctionDef
                ):  # type: ignore[override]
                    self.generic_visit(node)
                    self._strip_blank_docstring(node)
                    return node

                def visit_ClassDef(self, node: ast.ClassDef):  # type: ignore[override]
                    self.generic_visit(node)
                    self._strip_blank_docstring(node)
                    return node

                def visit_Module(self, node: ast.Module):  # type: ignore[override]
                    self.generic_visit(node)
                    self._strip_blank_docstring(node)
                    return node

            tree = Cleaner().visit(tree)
            ast.fix_missing_locations(tree)

            class NameCollector(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.names: set[str] = set()

                def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
                    self.names.add(node.id)
                    self.generic_visit(node)

            collector = NameCollector()
            collector.visit(tree)

            class ImportCleaner(ast.NodeTransformer):
                def __init__(self, used: set[str]) -> None:
                    self.used = used

                def visit_Import(self, node: ast.Import):  # type: ignore[override]
                    node.names = [
                        n for n in node.names if n.name == "*" or (n.asname or n.name) in self.used
                    ]
                    return None if not node.names else node

                def visit_ImportFrom(self, node: ast.ImportFrom):  # type: ignore[override]
                    node.names = [
                        n for n in node.names if n.name == "*" or (n.asname or n.name) in self.used
                    ]
                    return None if not node.names else node

            tree = ImportCleaner(collector.names).visit(tree)
            ast.fix_missing_locations(tree)

            return ast.unparse(tree)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to optimise python code")
            return code.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _python_template(t: TaskInfo, desc: str, *, style: str) -> str:
        """Return a Python implementation template for *t*."""

        functions = (
            t.metadata.get("functions", ["run"])
            if isinstance(t.metadata, dict)
            else ["run"]
        )

        imports = [f"import {d}" for d in t.dependencies]
        imports += ["import logging", "", "logger = logging.getLogger(__name__)"]

        if t.dependencies:
            dep_lines_log = ["results = []"]
            dep_lines_log += [f"results.append({d}())" for d in t.dependencies]
            dep_lines_log.append("return results")
            dep_lines_min = [f"{d}()" for d in t.dependencies]
        else:
            dep_lines_log = [f"logger.info({desc!r})", f"return {desc!r}"]
            dep_lines_min = [f"logger.info({desc!r})"]
        dep_block_log = "\n        ".join(dep_lines_log)
        dep_block_min = "\n        ".join(dep_lines_min)

        parts = list(imports) + [""]

        for fn in functions:
            if style == "logging":
                parts.append(f"def {fn}():")
                ret_type = "list" if t.dependencies else "str"
                dep_block = dep_block_log
            else:
                parts.append(f"def {fn}(*args, **kwargs):")
                ret_type = "bool"
                dep_block = dep_block_min
            parts.extend(
                [
                    "    \"\"\"",
                    f"    {desc} for task '{t.name}'.",
                    "",
                    "    Returns:",
                    f"        {ret_type}",
                    "    \"\"\"",
                ]
            )
            if style == "logging":
                parts.append(f"    logger.info('Task {t.name} starting')")
                parts.append("    try:")
                parts.append(f"        {dep_block}")
                parts.append("    except Exception as exc:")
                parts.append(f"        logger.error('Task {t.name} failed: %s', exc)")
                parts.append("        raise")
                parts.append(f"    logger.info('Task {t.name} completed')")
            else:
                parts.append("    try:")
                parts.append(f"        {dep_block}")
                parts.append("        return True")
                parts.append("    except Exception as exc:")
                parts.append(f"        logger.error('Task {t.name} failed: %s', exc)")
                parts.append("        return False")
            parts.append("")

        return "\n".join(parts).rstrip()

    @staticmethod
    def _shell_template(t: TaskInfo, desc: str) -> str:
        """Return a shell implementation template for *t*."""

        lines = [
            "#!/bin/sh",
            "set -e",
            f"# {desc} for task '{t.name}'",
            "",
            "main() {",
        ]
        if t.dependencies:
            lines.extend(f"    {d} \"$@\"" for d in t.dependencies)
        else:
            lines.append(f"    echo {desc!r}")
        lines.extend(
            [
                "}",
                "",
                "if main \"$@\"; then",
                "    exit 0",
                "else",
                "    exit 1",
                "fi",
            ]
        )
        return "\n".join(lines).rstrip()

    def process(self, package: TaskPackage) -> List[ImplementationAdvice]:
        """Record the package and return basic optimisation advice."""
        start_time = time.time()
        self.history.append(package)
        advice: List[ImplementationAdvice] = []
        for t in package.tasks:
            code = t.code or ""
            optimised = self._optimise_python(code)
            advice.append(ImplementationAdvice(name=t.name, optimised_code=optimised))
        self.data_bot.collect(
            bot=self.name,
            response_time=time.time() - start_time,
            errors=0,
            tests_failed=0,
            tests_run=0,
            revenue=0.0,
            expense=0.0,
        )
        return advice

    def fill_missing(
        self,
        package: TaskPackage,
        *,
        language: str = "python",
        style: str = "logging",
    ) -> TaskPackage:
        """Return a copy of *package* with missing code populated.

        Parameters
        ----------
        package:
            The :class:`TaskPackage` with potential gaps.
        language:
            Default language to generate (``"python"`` or ``"shell"``).
        style:
            Template style to use for Python code (``"logging"`` or ``"minimal"``).
        """

        start_time = time.time()
        tasks: List[TaskInfo] = []
        for t in package.tasks:
            if t.code:
                code = t.code
            else:
                desc = "Auto generated implementation"
                if isinstance(t.metadata, dict) and t.metadata.get("purpose"):
                    desc = str(t.metadata["purpose"])

                lang = (
                    t.metadata.get("language", language)
                    if isinstance(t.metadata, dict)
                    else language
                )
                tmpl_style = (
                    t.metadata.get("template", style)
                    if isinstance(t.metadata, dict)
                    else style
                )

                generated = ""
                if self.manager is not None and isinstance(t.metadata, dict):
                    path_str = t.metadata.get("path")
                    if path_str:
                        try:
                            path = Path(path_str)
                            outcome = self.manager.auto_run_patch(path, desc)
                            summary = outcome.get("summary") if outcome else None
                            failed_tests = int(summary.get("self_tests", {}).get("failed", 0)) if summary else 0
                            patch_id = outcome.get("patch_id") if outcome else None
                            if summary is None or failed_tests:
                                if summary is None:
                                    logger.warning("implementation validation summary unavailable")
                                else:
                                    logger.warning(
                                        "implementation validation failed: %s", failed_tests
                                    )
                                engine = getattr(self.manager, "engine", None)
                                if patch_id is not None and hasattr(engine, "rollback_patch"):
                                    try:
                                        engine.rollback_patch(str(patch_id))
                                    except Exception:
                                        logger.exception("implementation rollback failed")
                                generated = ""
                            else:
                                if getattr(self.manager, "bot_registry", None):
                                    try:
                                        name = getattr(
                                            self,
                                            "name",
                                            getattr(self, "bot_name", self.__class__.__name__),
                                        )
                                        self.manager.bot_registry.update_bot(name, str(path))
                                    except Exception:  # pragma: no cover - best effort
                                        logger.exception("bot registry update failed")
                                generated = path.read_text().rstrip()
                        except Exception:
                            generated = ""

                if generated:
                    code = generated
                else:
                    if lang.startswith("py"):
                        code = self._python_template(t, desc, style=tmpl_style)
                    elif lang in {"shell", "bash", "sh"}:
                        code = self._shell_template(t, desc)
                    else:
                        code = ""

            metadata = t.metadata or {"info": "added by optimiser"}
            tasks.append(
                TaskInfo(
                    name=t.name,
                    dependencies=t.dependencies,
                    resources=t.resources,
                    schedule=t.schedule,
                    code=code,
                    metadata=metadata,
                )
            )

        result = TaskPackage(tasks=tasks, version=package.version)
        self.data_bot.collect(
            bot=self.name,
            response_time=time.time() - start_time,
            errors=0,
            tests_failed=0,
            tests_run=0,
            revenue=0.0,
            expense=0.0,
        )
        return result


__all__ = ["ImplementationAdvice", "ImplementationOptimiserBot"]
