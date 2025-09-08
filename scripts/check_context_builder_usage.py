#!/usr/bin/env python3
"""Static check for ``ContextBuilder`` usage.

This script scans the repository for calls to ``_build_prompt``,
``PromptEngine`` and patch helpers such as ``generate_patch`` to ensure that a
``context_builder`` keyword argument is supplied.  It now also checks *any*
function call named ``build_prompt`` or ``build_prompt_with_memory`` regardless
of whether it is accessed as an attribute.  Additionally, direct calls to
``openai.Completion.create`` and ``openai.ChatCompletion.create`` as well as the
``chat_completion_create`` wrapper are inspected.  Any invocation missing a
``context_builder`` keyword or an inline ``# nocb`` comment will be flagged.  The
check ignores files located in directories named ``tests`` or ``unit_tests``.

The linter also detects calls to ``LLMClient.generate`` or similar ``.generate``
wrappers whose class name ends with ``Client``, ``Provider`` or ``Wrapper`` and
requires a ``context_builder`` keyword.  These calls are reported when the
argument is absent and no ``# nocb`` marker is present on the line or the one
directly above.  Assignments or ``functools.partial`` wrappers of such
``.generate`` methods are tracked so that subsequent calls through variables or
partials are likewise validated.

Furthermore, the linter searches for imports or calls to
``get_default_context_builder`` outside of test directories.  Such usage is
reported unless the offending line (or the one immediately above it) contains a
``# nocb`` marker.  Calls to ``create_context_builder`` are allowed and
considered valid ``ContextBuilder`` instantiations.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOCB_MARK = "# nocb"

DEFAULT_BUILDER_NAME = "get_default_context_builder"
HELPER_NAMES = {
    "create_context_builder",
    "config.create_context_builder.create_context_builder",
    "menace_sandbox.config.create_context_builder.create_context_builder",
    "context_builder_util.create_context_builder",
}
REQUIRED_NAMES = {
    "PromptEngine",
    "_build_prompt",
    "generate_patch",
    "build_prompt",
    "build_prompt_with_memory",
    "chat_completion_create",
}

GENERATE_WRAPPER_SUFFIXES = ("client", "provider", "wrapper")
PARTIAL_NAMES = {"partial", "functools.partial"}


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in {"tests", "unit_tests"} for part in path.parts):
            continue
        yield path


def check_file(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text()
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - syntax errors
        print(f"Skipping {path}: {exc}", file=sys.stderr)
        return []

    lines = text.splitlines()
    errors: list[tuple[int, str]] = []

    def full_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None

    OPENAI_NAMES = {
        "openai.ChatCompletion.create",
        "openai.Completion.create",
    }

    def is_generate_wrapper(name: str) -> bool:
        """Return True if *name* looks like an unbound ``.generate`` wrapper."""

        if not name.endswith(".generate"):
            return False
        parts = name.split(".")
        if len(parts) != 2:
            return False
        cls = parts[0]
        return cls[0].isupper() and any(
            cls.lower().endswith(suf) for suf in GENERATE_WRAPPER_SUFFIXES
        )

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.generate_aliases: set[str] = set()

        @staticmethod
        def _has_none_default(arg: ast.arg, default: ast.AST | None) -> bool:
            return bool(
                default
                and isinstance(default, ast.Constant)
                and default.value is None
                and arg.arg == "context_builder"
            )

        def _check_args(self, node: ast.AST) -> None:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            args = node.args

            pos_args = args.posonlyargs + args.args
            defaults: list[ast.AST | None] = [None] * (
                len(pos_args) - len(args.defaults)
            ) + list(args.defaults)
            for arg, default in zip(pos_args, defaults):
                if self._has_none_default(arg, default):
                    line_no = arg.lineno
                    line = (
                        lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    )
                    if NOCB_MARK not in line:
                        errors.append(
                            (
                                line_no,
                                "context_builder default None disallowed or missing context_builder",
                            )
                        )

            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if self._has_none_default(arg, default):
                    line_no = arg.lineno
                    line = (
                        lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    )
                    if NOCB_MARK not in line:
                        errors.append(
                            (
                                line_no,
                                "context_builder default None disallowed or missing context_builder",
                            )
                        )

        def _record_alias(self, targets: list[ast.expr], value: ast.AST) -> None:
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names:
                return

            if isinstance(value, ast.Attribute):
                name = full_name(value)
                if name and is_generate_wrapper(name):
                    self.generate_aliases.update(names)
            elif isinstance(value, ast.Call):
                call_name = full_name(value.func)
                if call_name in PARTIAL_NAMES and value.args:
                    first = value.args[0]
                    gen_name = full_name(first)
                    if gen_name and is_generate_wrapper(gen_name):
                        if not any(kw.arg == "context_builder" for kw in value.keywords):
                            self.generate_aliases.update(names)

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: D401
            self._record_alias(node.targets, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: D401
            target = node.target
            value = node.value
            if isinstance(target, ast.Name) and value is not None:
                self._record_alias([target], value)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            name_full = full_name(node.func)
            name_simple = name_full.split(".")[-1] if name_full else None

            if name_simple == "getattr":
                second: ast.AST | None = None
                third: ast.AST | None = None
                if len(node.args) >= 2:
                    second = node.args[1]
                else:
                    for kw in node.keywords:
                        if kw.arg == "name":
                            second = kw.value
                if len(node.args) >= 3:
                    third = node.args[2]
                else:
                    for kw in node.keywords:
                        if kw.arg == "default":
                            third = kw.value
                if (
                    isinstance(second, ast.Constant)
                    and second.value == "context_builder"
                    and isinstance(third, ast.Constant)
                    and third.value is None
                ):
                    line_no = node.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                "getattr context_builder default None disallowed or missing context_builder",
                            )
                        )
                self.generic_visit(node)
                return

            if isinstance(node.func, ast.Name) and node.func.id in self.generate_aliases:
                has_kw = any(kw.arg == "context_builder" for kw in node.keywords)
                target = node.func.id
            elif name_simple == DEFAULT_BUILDER_NAME:
                line_no = node.lineno
                line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                prev = lines[line_no - 2] if line_no >= 2 else ""
                if NOCB_MARK not in line and NOCB_MARK not in prev:
                    errors.append(
                        (
                            line_no,
                            f"{DEFAULT_BUILDER_NAME} disallowed or missing context_builder",
                        )
                    )
                self.generic_visit(node)
                return
            else:
                has_kw = any(kw.arg == "context_builder" for kw in node.keywords)
                target = None
                if name_simple in REQUIRED_NAMES:
                    target = name_simple
                elif name_full in OPENAI_NAMES:
                    target = name_full
                elif name_full and is_generate_wrapper(name_full):
                    target = name_full
                elif isinstance(node.func, ast.Call):
                    inner = node.func
                    inner_name = full_name(inner.func)
                    if inner_name in PARTIAL_NAMES and inner.args:
                        first = inner.args[0]
                        gen_name = full_name(first)
                        if gen_name and (
                            is_generate_wrapper(gen_name)
                            or gen_name in self.generate_aliases
                        ):
                            has_kw = has_kw or any(
                                kw.arg == "context_builder" for kw in inner.keywords
                            )
                            target = gen_name

            if (
                isinstance(node.func, ast.Name)
                and node.func.id in self.generate_aliases
            ) or (target and not has_kw):
                line_no = node.lineno
                line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                prev = lines[line_no - 2] if line_no >= 2 else ""
                if NOCB_MARK not in line and NOCB_MARK not in prev:
                    errors.append(
                        (
                            line_no,
                            f"{target or node.func.id} disallowed or missing context_builder",
                        )
                    )

            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:  # noqa: D401
            for alias in node.names:
                name = alias.name.split(".")[-1]
                if name == DEFAULT_BUILDER_NAME or alias.asname == DEFAULT_BUILDER_NAME:
                    line_no = node.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                f"{DEFAULT_BUILDER_NAME} disallowed or missing context_builder",
                            )
                        )
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: D401
            for alias in node.names:
                if alias.name == DEFAULT_BUILDER_NAME or alias.asname == DEFAULT_BUILDER_NAME:
                    line_no = node.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                f"{DEFAULT_BUILDER_NAME} disallowed or missing context_builder",
                            )
                        )
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: D401
            self._check_args(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: D401
            self._check_args(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    REQUIRED_DB_STRINGS = {"bots.db", "code.db", "errors.db", "workflows.db"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = full_name(node.func)
        if not name or name in HELPER_NAMES:
            continue
        if name != "ContextBuilder" and not name.endswith(".ContextBuilder"):
            continue

        strings: list[str] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                strings.append(arg.value)
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                strings.append(kw.value.value)

        missing = sorted(db for db in REQUIRED_DB_STRINGS if db not in strings)
        if missing:
            line_no = node.lineno
            line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
            prev = lines[line_no - 2] if line_no >= 2 else ""
            if NOCB_MARK not in line and NOCB_MARK not in prev:
                errors.append(
                    (
                        line_no,
                        "ContextBuilder() missing " + ", ".join(missing),
                    )
                )

    return errors


def main() -> int:
    failures: list[str] = []
    for path in iter_python_files(ROOT):
        for lineno, message in check_file(path):
            failures.append(f"{path}:{lineno} -> {message}")
    if failures:
        for line in failures:
            print(line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
