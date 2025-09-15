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


Any ``# nocb`` comments found in production modules are likewise reported
unless the file path has been explicitly whitelisted.  Test directories are
still excluded from the scan.


The linter also detects calls to ``LLMClient.generate``/``async_generate`` or
similar ``.generate`` wrappers whose class name ends with ``Client``,
``Provider`` or ``Wrapper`` and requires a ``context_builder`` keyword.  These
calls are reported when the argument is absent and no ``# nocb`` marker is
present on the line or the one directly above.  Assignments or
``functools.partial`` wrappers of such generate methods are tracked so that
subsequent calls through variables or partials are likewise validated.

Variables assigned from ``LLMClient``-like classes are now remembered so that
instance method calls such as ``client.generate()`` also require a
``context_builder`` keyword.  Common aliases like ``llm`` or ``model`` are
heuristically treated as potential ``LLMClient`` instances and their
``.generate()`` or ``.async_generate()`` calls are checked even without a prior
assignment.

Furthermore, the linter searches for imports or calls to
``get_default_context_builder`` outside of test directories.  Such usage is
reported unless the offending line (or the one immediately above it) contains a
``# nocb`` marker.  Calls to ``create_context_builder`` are allowed and
considered valid ``ContextBuilder`` instantiations.

Functions invoking ``ContextBuilder.build`` must accept a non-optional builder
parameter.  Parameters defaulting to ``None`` or falling back to a new builder
within the function body are flagged to ensure that callers explicitly inject a
``ContextBuilder`` instance.

Manual string prompts sent directly to LLM clients are also detected.  Literal
strings, f-strings, or concatenated strings passed as ``prompt`` arguments will
be reported unless they originate from ``ContextBuilder.build_prompt`` or
``SelfCodingEngine.build_enriched_prompt``.  Additionally, direct invocations of
``PromptEngine.build_prompt`` are flagged to discourage bypassing the builder
infrastructure.  ``Prompt(...)`` calls must likewise receive output from
``context_builder.build_prompt`` or ``SelfCodingEngine.build_enriched_prompt``;
outside ``vector_service/context_builder.py`` direct ``Prompt`` construction is
rejected.  Lists or dicts supplied directly as ``messages`` to ``.generate``
methods are also disallowed; use the canonical builders instead.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOCB_MARK = "# nocb"

# Paths relative to repo root where ``# nocb`` markers are permitted.
NOCB_WHITELIST: set[str] = {"scripts/check_context_builder_usage.py"}

# Files allowed to instantiate ``Prompt`` directly.  These modules expose the
# canonical builders and may construct prompts internally without going through
# ``ContextBuilder.build_prompt``.
PROMPT_WHITELIST: set[str] = {"vector_service/context_builder.py"}

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
GENERATE_METHODS = {"generate", "async_generate"}
PARTIAL_NAMES = {"partial", "functools.partial"}
ALIAS_NAMES = {"llm", "model"}
PROMPT_HELPER_PREFIXES = ("generate_", "build_", "create_")
PROMPT_HELPER_KEYWORDS = ("prompt", "candidate")


def _is_string_expr(node: ast.AST) -> bool:
    """Return True if *node* represents a string literal or concatenation."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):  # f"{expr}"
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_string_expr(node.left) or _is_string_expr(node.right)
    return False


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

    try:
        rel = path.relative_to(ROOT).as_posix()
        outside_repo = False
    except ValueError:  # tmp files outside repo
        rel = path.as_posix()
        outside_repo = True
    if not outside_repo and rel not in NOCB_WHITELIST:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and NOCB_MARK in tok.string:
                errors.append((tok.start[0], "# nocb marker disallowed"))

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
        """Return True if *name* looks like an unbound generate wrapper."""

        if not any(name.endswith(f".{m}") for m in GENERATE_METHODS):
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
            self.llm_instances: set[str] = set()
            self.prompt_vars: set[str] = set()

        @staticmethod
        def _has_default(arg: ast.arg, default: ast.AST | None) -> bool:
            return bool(default is not None and arg.arg == "context_builder")

        def _check_args(self, node: ast.AST) -> None:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            args = node.args

            pos_args = args.posonlyargs + args.args
            defaults: list[ast.AST | None] = [None] * (
                len(pos_args) - len(args.defaults)
            ) + list(args.defaults)
            for arg, default in zip(pos_args, defaults):
                if self._has_default(arg, default):
                    line_no = arg.lineno
                    line = (
                        lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    )
                    if NOCB_MARK not in line:
                        errors.append(
                            (
                                line_no,
                                "context_builder default disallowed or missing "
                                "context_builder",
                            )
                        )

            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if self._has_default(arg, default):
                    line_no = arg.lineno
                    line = (
                        lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    )
                    if NOCB_MARK not in line:
                        errors.append(
                            (
                                line_no,
                                "context_builder default disallowed or missing "
                                "context_builder",
                            )
                        )

        def _record_alias(self, targets: list[ast.expr], value: ast.AST) -> None:
            names = [
                t.id if isinstance(t, ast.Name) else t.attr
                for t in targets
                if isinstance(t, (ast.Name, ast.Attribute))
            ]
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

        def _record_instance(self, targets: list[ast.expr], value: ast.AST) -> None:
            names = [
                t.id if isinstance(t, ast.Name) else t.attr
                for t in targets
                if isinstance(t, (ast.Name, ast.Attribute))
            ]
            if not names:
                return

            if isinstance(value, ast.Call):
                call_name = full_name(value.func)
                if call_name:
                    cls = call_name.split(".")[-1]
                    if cls and cls[0].isupper() and any(
                        cls.lower().endswith(suf) for suf in GENERATE_WRAPPER_SUFFIXES
                    ):
                        self.llm_instances.update(names)

        def _record_prompt_var(
            self, targets: list[ast.expr], value: ast.AST
        ) -> None:
            if not isinstance(value, ast.Call):
                return
            call_name = full_name(value.func)
            if call_name not in {
                "context_builder.build_prompt",
                "SelfCodingEngine.build_enriched_prompt",
            }:
                return
            for target in targets:
                if isinstance(target, ast.Name):
                    self.prompt_vars.add(target.id)

        def _has_builder_fallback(self, node: ast.AST, name: str) -> bool:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign):
                    if (
                        len(inner.targets) == 1
                        and isinstance(inner.targets[0], ast.Name)
                        and inner.targets[0].id == name
                    ):
                        val = inner.value
                        if (
                            isinstance(val, ast.BoolOp)
                            and isinstance(val.op, ast.Or)
                            and val.values
                            and isinstance(val.values[0], ast.Name)
                            and val.values[0].id == name
                        ):
                            return True
                elif isinstance(inner, ast.If):
                    test = inner.test
                    if (
                        isinstance(test, ast.Compare)
                        and len(test.ops) == 1
                        and isinstance(test.left, ast.Name)
                        and test.left.id == name
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None
                    ):
                        for stmt in inner.body:
                            if (
                                isinstance(stmt, ast.Assign)
                                and len(stmt.targets) == 1
                                and isinstance(stmt.targets[0], ast.Name)
                                and stmt.targets[0].id == name
                            ):
                                return True
            return False

        def _check_prompt_strings(self, node: ast.Call, is_llm_call: bool) -> None:
            if not is_llm_call:
                return

            candidates: list[ast.AST] = []
            if node.args:
                candidates.append(node.args[0])
            for kw in node.keywords:
                if kw.arg == "prompt":
                    candidates.append(kw.value)

            for expr in candidates:
                if _is_string_expr(expr):
                    line_no = expr.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                "manual string prompt disallowed; use "
                                "context_builder.build_prompt or "
                                "SelfCodingEngine.build_enriched_prompt",
                            )
                        )

        def _check_message_literals(self, node: ast.Call, is_llm_call: bool) -> None:
            if not is_llm_call:
                return

            candidates: list[ast.AST] = []
            if node.args:
                candidates.append(node.args[0])
            for kw in node.keywords:
                if kw.arg == "messages":
                    candidates.append(kw.value)

            for expr in candidates:
                if isinstance(expr, (ast.List, ast.Dict)):
                    line_no = expr.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                "direct message list/dict disallowed; use "
                                "ContextBuilder.build_prompt or "
                                "SelfCodingEngine.build_enriched_prompt",
                            )
                        )

        def _check_build_calls(self, node: ast.AST) -> None:
            params: dict[str, ast.AST | None] = {}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                pos_args = args.posonlyargs + args.args
                defaults = [None] * (len(pos_args) - len(args.defaults)) + list(
                    args.defaults
                )
                for arg, default in zip(pos_args, defaults):
                    if "builder" in arg.arg:
                        params[arg.arg] = default
                for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                    if "builder" in arg.arg:
                        params[arg.arg] = default
            else:
                return

            optional = {
                name
                for name, default in params.items()
                if isinstance(default, ast.Constant) and default.value is None
            }
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "build"
                    and isinstance(inner.func.value, ast.Name)
                ):
                    base = inner.func.value.id
                    if base in params and (
                        base in optional or self._has_builder_fallback(node, base)
                    ):
                        line_no = inner.lineno
                        line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                        prev = lines[line_no - 2] if line_no >= 2 else ""
                        if NOCB_MARK not in line and NOCB_MARK not in prev:
                            errors.append(
                                (
                                    line_no,
                                    f"{base}.build disallowed or missing context_builder",
                                )
                            )

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: D401
            self._record_alias(node.targets, node.value)
            self._record_instance(node.targets, node.value)
            self._record_prompt_var(node.targets, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: D401
            target = node.target
            value = node.value
            if value is not None:
                self._record_alias([target], value)
                self._record_instance([target], value)
                self._record_prompt_var([target], value)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            name_full = full_name(node.func)
            name_simple = name_full.split(".")[-1] if name_full else None

            if name_simple == "Prompt" and rel not in PROMPT_WHITELIST:
                arg = node.args[0] if node.args else None
                valid = False
                if isinstance(arg, ast.Call):
                    arg_name = full_name(arg.func)
                    if arg_name in {
                        "context_builder.build_prompt",
                        "SelfCodingEngine.build_enriched_prompt",
                    }:
                        valid = True
                elif isinstance(arg, ast.Name) and arg.id in self.prompt_vars:
                    valid = True
                if not valid:
                    line_no = node.lineno
                    line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                    prev = lines[line_no - 2] if line_no >= 2 else ""
                    if NOCB_MARK not in line and NOCB_MARK not in prev:
                        errors.append(
                            (
                                line_no,
                                "Prompt call requires context_builder.build_prompt "
                                "or SelfCodingEngine.build_enriched_prompt",
                            )
                        )

            if name_full == "PromptEngine.build_prompt":
                line_no = node.lineno
                line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                prev = lines[line_no - 2] if line_no >= 2 else ""
                if NOCB_MARK not in line and NOCB_MARK not in prev:
                    errors.append(
                        (
                            line_no,
                            "PromptEngine.build_prompt disallowed; use ContextBuilder.build_prompt",
                        )
                    )
                self.generic_visit(node)
                return

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
                                "getattr context_builder default None disallowed or missing "
                                "context_builder",
                            )
                        )
                self.generic_visit(node)
                return

            is_llm_call = False

            if isinstance(node.func, ast.Name) and node.func.id in self.generate_aliases:
                has_kw = any(kw.arg == "context_builder" for kw in node.keywords)
                target = node.func.id
                is_llm_call = True
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
                    if name_simple == "chat_completion_create":
                        is_llm_call = True
                elif name_full in OPENAI_NAMES:
                    target = name_full
                    is_llm_call = True
                elif name_full and is_generate_wrapper(name_full):
                    target = name_full
                    is_llm_call = True
                elif name_simple and any(
                    name_simple.startswith(prefix)
                    and any(key in name_simple for key in PROMPT_HELPER_KEYWORDS)
                    for prefix in PROMPT_HELPER_PREFIXES
                ):
                    target = name_simple
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
                            is_llm_call = True
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in GENERATE_METHODS
                ):
                    base = node.func.value
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    else:
                        base_name = None
                    if base_name and (
                        base_name in self.llm_instances or base_name in ALIAS_NAMES
                    ):
                        target = f"{base_name}.{node.func.attr}"
                        is_llm_call = True

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

            self._check_prompt_strings(node, is_llm_call)
            self._check_message_literals(node, is_llm_call)
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
            self._check_build_calls(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: D401
            self._check_args(node)
            self._check_build_calls(node)
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
