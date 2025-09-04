#!/usr/bin/env python3
"""Detect direct Stripe SDK imports or Stripe live keys.

This script guards against direct usage of the Stripe SDK and accidental
exposure of live Stripe credentials. Run it in two modes:

* Default mode checks Python files for ``stripe`` imports, raw ``api.stripe.com``
  calls via common HTTP libraries and payment keywords without
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


def resolve_path(path: str) -> str:
    return path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from stripe_detection import PAYMENT_KEYWORDS, HTTP_LIBRARIES, contains_payment_keyword  # noqa: E402

ALLOWED = {
    (REPO_ROOT / "stripe_billing_router.py").resolve(),  # path-ignore
    (REPO_ROOT / "scripts/check_stripe_imports.py").resolve(),  # path-ignore
    (REPO_ROOT / "startup_checks.py").resolve(),  # path-ignore
    (REPO_ROOT / "bot_development_bot.py").resolve(),  # path-ignore
    (REPO_ROOT / "config_loader.py").resolve(),  # path-ignore
    (REPO_ROOT / "codex_output_analyzer.py").resolve(),  # path-ignore
}
KEY_PATTERN = re.compile(
    r"sk_live|pk_live|STRIPE_SECRET_KEY|STRIPE_PUBLIC_KEY"
)


class StripeAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stripe_lines: list[int] = []
        self.raw_api_lines: list[int] = []
        self.keyword_lines: set[int] = set()
        self.has_router_import = False
        self.http_names: dict[str, str] = {}

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
                and "api.stripe.com" in arg.value
                for arg in node.args
            ):
                self.raw_api_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - simple
        if contains_payment_keyword(node.name):
            self.keyword_lines.add(node.lineno)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover - simple
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
        if analyzer.keyword_lines and not analyzer.has_router_import:
            lineno = min(analyzer.keyword_lines)
            keywords.append(f"{rel}:{lineno}:missing stripe_billing_router import")
    if imports:
        print("Direct Stripe imports detected (use stripe_billing_router):")
    if raw:
        print("Raw Stripe API usage detected (use stripe_billing_router):")
    if keywords:
        print("Payment/Stripe keywords without stripe_billing_router import detected:")
    return imports + raw + keywords


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
        p = Path(resolve_path(filename))
        paths.append(p if p.is_absolute() else (REPO_ROOT / p).resolve())

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
