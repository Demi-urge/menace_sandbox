"""Implementation Optimiser Bot for refining task packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .self_coding_engine import SelfCodingEngine
import ast
import logging
from vector_service import ContextBuilder

logger = logging.getLogger(__name__)

from .task_handoff_bot import TaskPackage, TaskInfo


@dataclass
class ImplementationAdvice:
    """Advice or optimised snippet for a task."""

    name: str
    optimised_code: str


class ImplementationOptimiserBot:
    """Receive ``TaskPackage`` objects and refine them.

    When generating missing implementations this bot can produce simple Python
    or shell templates.  The Python template has two flavours: a minimal style
    that simply defines the required functions, and a logging style that wraps
    the body in basic ``try``/``except`` blocks and emits log messages.
    """

    def __init__(
        self,
        engine: SelfCodingEngine | None = None,
        *,
        context_builder: ContextBuilder,
    ) -> None:
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.history: List[TaskPackage] = []
        self.engine = engine
        self.context_builder = context_builder
        if self.engine is not None:
            try:
                self.engine.context_builder = context_builder  # type: ignore[attr-defined]
                if hasattr(self.engine.context_builder, "refresh_db_weights"):
                    self.engine.context_builder.refresh_db_weights()  # type: ignore[attr-defined]
            except Exception:
                pass

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

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # type: ignore[override]
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
        self.history.append(package)
        advice: List[ImplementationAdvice] = []
        for t in package.tasks:
            code = t.code or ""
            optimised = self._optimise_python(code)
            advice.append(ImplementationAdvice(name=t.name, optimised_code=optimised))
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
                if self.engine is not None:
                    try:
                        generated = self.engine.generate_helper(desc).rstrip()
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

        return TaskPackage(tasks=tasks, version=package.version)


__all__ = ["ImplementationAdvice", "ImplementationOptimiserBot"]
