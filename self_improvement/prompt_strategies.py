"""Prompt strategy definitions and template rendering helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Mapping


class PromptStrategy(str, Enum):
    """Enumeration of supported prompt tuning strategies."""

    STRICT_FIX = "strict_fix"
    DELETE_REBUILD = "delete_rebuild"
    COMMENT_REFACTOR = "comment_refactor"
    UNIT_TEST_REWRITE = "unit_test_rewrite"


TemplateBuilder = Callable[[Mapping[str, Any]], str]


def _module_name(ctx: Mapping[str, Any]) -> str:
    """Extract a human readable module name from ``ctx``."""

    return str(ctx.get("module") or ctx.get("module_name") or ctx.get("name") or "the module")


def _strict_fix(ctx: Mapping[str, Any]) -> str:
    mod = _module_name(ctx)
    return (
        f"Apply the smallest change necessary in {mod} to address the issue without "
        "altering unrelated code."
    )


def _delete_rebuild(ctx: Mapping[str, Any]) -> str:
    mod = _module_name(ctx)
    return f"Remove the existing implementation of {mod} entirely and recreate it from scratch."


def _comment_refactor(ctx: Mapping[str, Any]) -> str:
    mod = _module_name(ctx)
    return (
        f"Restructure or improve comments in {mod} without modifying the runtime behaviour "
        "of the code."
    )


def _unit_test_rewrite(ctx: Mapping[str, Any]) -> str:
    mod = _module_name(ctx)
    return (
        f"Rewrite or expand unit tests for {mod} to clarify intent while avoiding changes "
        "to production code."
    )


TEMPLATE_BUILDERS: Dict[PromptStrategy, TemplateBuilder] = {
    PromptStrategy.STRICT_FIX: _strict_fix,
    PromptStrategy.DELETE_REBUILD: _delete_rebuild,
    PromptStrategy.COMMENT_REFACTOR: _comment_refactor,
    PromptStrategy.UNIT_TEST_REWRITE: _unit_test_rewrite,
}


def render_prompt(strategy: PromptStrategy, module_ctx: Mapping[str, Any]) -> str:
    """Render the template for ``strategy`` using ``module_ctx``.

    Parameters
    ----------
    strategy:
        The :class:`PromptStrategy` to render a prompt for.
    module_ctx:
        Mapping providing details about the target module. Keys such as
        ``module``, ``module_name`` or ``name`` are inspected for a human
        friendly description.
    """

    builder = TEMPLATE_BUILDERS[strategy]
    return builder(module_ctx)


__all__ = ["PromptStrategy", "TEMPLATE_BUILDERS", "render_prompt"]
