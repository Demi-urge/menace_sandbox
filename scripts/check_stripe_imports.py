"""This script guards against direct usage of the Stripe SDK and accidental
exposure of live Stripe credentials. Run it in two modes:

* Default mode checks Python files for ``stripe`` imports, raw ``api.stripe.com``
  calls via common HTTP libraries (``requests``, ``httpx``, ``aiohttp``,
  ``urllib`` and ``urllib3``) and payment keywords without
  ``stripe_billing_router``.
* ``--keys`` scans any files for strings such as ``sk_live``, ``pk_live``,
  ``STRIPE_SECRET_KEY`` or ``STRIPE_PUBLIC_KEY`` outside
  ``stripe_billing_router.py``.

Examples
--------
Check imports and keywords::

    python scripts/check_stripe_imports.py path/to/file.py

Scan for potential live keys::

    python scripts/check_stripe_imports.py --keys path/to/file.py
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

try:  # pragma: no cover - prefer package relative imports
    from menace_sandbox.dynamic_path_router import get_project_root, resolve_path
    from menace_sandbox.stripe_detection import (
        HTTP_LIBRARIES,
        contains_payment_keyword,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dynamic_path_router import get_project_root, resolve_path  # type: ignore
    from stripe_detection import (  # type: ignore
        HTTP_LIBRARIES,
        contains_payment_keyword,
    )

REPO_ROOT = get_project_root()

ALLOWED = {
    resolve_path("stripe_billing_router.py").resolve(),
    resolve_path("scripts/check_stripe_imports.py").resolve(),
    resolve_path("startup_checks.py").resolve(),
    resolve_path("bot_development_bot.py").resolve(),
    resolve_path("config_loader.py").resolve(),
    resolve_path("codex_output_analyzer.py").resolve(),
    resolve_path("stripe_detection.py").resolve(),
    resolve_path("billing/openai_wrapper.py").resolve(),
    resolve_path("stripe_watchdog.py").resolve(),
}
# Detect exposures of Stripe keys, including partially redacted ones with ``*``.
KEY_PATTERN = re.compile(
    (
        r"(?:sk|pk)_(?:live|test)?_[A-Za-z0-9]*\*+[A-Za-z0-9*]*"
        r"|sk_live|pk_live|STRIPE_SECRET_KEY|STRIPE_PUBLIC_KEY"
    )
)


class StripeAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stripe_lines: list[int] = []
        self.raw_api_lines: list[int] = []
        self.keyword_lines: set[int] = set()
        self.has_router_import = False
        self.http_names: dict[str, str] = {}
        self.openai_lines: list[int] = []

    def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover - simple
        for alias in node.names:
            module = alias.name
            asname = alias.asname or module
            if module == "stripe" or module.startswith("stripe."):
                self.stripe_lines.append(node.lineno)
            if module in HTTP_LIBRARIES:
                self.http_names[asname] = module
            if module == "stripe_billing_router":
                self.has_router_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pragma: no cover - simple
        module = node.module or ""
        if module == "stripe" or module.startswith("stripe."):
            self.stripe_lines.append(node.lineno)
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
        if root in self.http_names:
            if any(
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and "api." "stripe.com" in arg.value
                for arg in node.args
            ):
                self.raw_api_lines.append(node.lineno)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "openai"
            and node.func.value.attr == "ChatCompletion"
            and node.func.attr == "create"
        ):
            self.openai_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - simple
        if contains_payment_keyword(node.name):
            self.keyword_lines.add(node.lineno)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover
        if contains_payment_keyword(node.name):
            self.keyword_lines.add(node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # pragma: no cover - simple
        if isinstance(node.ctx, ast.Store):
            name = node.id
            if not name.isupper() and name != "stripe_billing_router":
                if contains_payment_keyword(name):
                    self.keyword_lines.add(node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # pragma: no cover - simple
        if isinstance(node.ctx, ast.Store):
            name = node.attr
            if not name.isupper() and name != "stripe_billing_router":
                if contains_payment_keyword(name):
                    self.keyword_lines.add(node.lineno)
        self.generic_visit(node)


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _check_ast(paths: list[Path]) -> list[str]:
    imports: list[str] = []
    raw: list[str] = []
    keywords: list[str] = []
    openai_calls: list[str] = []
    for path in paths:
        if path.suffix != ".py" or path in ALLOWED:
            continue
        if any(part in {"tests", "unit_tests", "fixtures", "finance_logs"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        analyzer = StripeAnalyzer()
        analyzer.visit(tree)
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        for lineno in analyzer.stripe_lines:
            imports.append(f"{rel}:{lineno}:import stripe")
        for lineno in analyzer.raw_api_lines:
            raw.append(f"{rel}:{lineno}:raw api.stripe.com call")
        for lineno in analyzer.openai_lines:
            openai_calls.append(
                f"{rel}:{lineno}:openai.ChatCompletion.create usage"
            )
        if analyzer.keyword_lines and not analyzer.has_router_import:
            lineno = min(analyzer.keyword_lines)
            keywords.append(f"{rel}:{lineno}:missing stripe_billing_router import")
    if imports:
        print("Direct Stripe imports detected (use stripe_billing_router):")
    if raw:
        print("Raw Stripe API usage detected (use stripe_billing_router):")
    if openai_calls:
        print(
            "Direct openai.ChatCompletion.create usage detected "
            "(use billing.openai_wrapper):"
        )
    if keywords:
        print("Payment/Stripe keywords without stripe_billing_router import detected:")
    return imports + raw + openai_calls + keywords


def _check_keys(paths: list[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        if path in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if KEY_PATTERN.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print(
            "Potential Stripe live keys or environment variables detected "
            "(use stripe_billing_router):",
        )
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keys",
        action="store_true",
        help="scan for live Stripe keys and environment variables",
    )
    args, files = parser.parse_known_args(argv)

    paths: list[Path] = []
    for filename in files:
        paths.append(resolve_path(filename))

    if args.keys:
        offenders = _check_keys(paths)
    else:
        offenders = _check_ast(paths)
    if offenders:
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
