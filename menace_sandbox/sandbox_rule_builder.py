"""Build deterministic sandbox patch rules from error context."""

from __future__ import annotations

from typing import Any, Mapping
import re

from error_ontology import classify_error


_ANCHOR_PLACEHOLDER = "__MENACE_SANDBOX_NOOP_ANCHOR__"

_RULE_TEMPLATES: dict[str, dict[str, object]] = {
    "syntaxerror": {
        "rule_id": "syntax-error-anchor",
        "description": "Flag the nearest code declaration for syntax repair.",
        "tokens": ("def ", "class ", "lambda ", "return "),
        "suffix": "  # TODO: fix syntax error",
    },
    "importerror": {
        "rule_id": "import-error-anchor",
        "description": "Mark the closest import statement for dependency fixes.",
        "tokens": ("import ", "from "),
        "suffix": "  # TODO: verify import",
    },
    "typeerror-mismatch": {
        "rule_id": "type-mismatch-anchor",
        "description": "Annotate the likely type mismatch location.",
        "tokens": ("return ", "def ", "="),
        "suffix": "  # TODO: review type expectations",
    },
    "contractviolation": {
        "rule_id": "contract-violation-anchor",
        "description": "Highlight contract or assertion checks to revise.",
        "tokens": ("assert ", "contract", "expect", "require"),
        "suffix": "  # TODO: revisit contract",
    },
    "edgecasefailure": {
        "rule_id": "edge-case-anchor",
        "description": "Signal a control-flow branch for edge case handling.",
        "tokens": ("if ", "elif ", "for ", "while "),
        "suffix": "  # TODO: handle edge case",
    },
    "unhandledexception": {
        "rule_id": "unhandled-exception-anchor",
        "description": "Point to error handling for unhandled exceptions.",
        "tokens": ("try:", "except ", "raise "),
        "suffix": "  # TODO: add exception handling",
    },
    "invalidinput": {
        "rule_id": "invalid-input-anchor",
        "description": "Mark input validation logic for adjustments.",
        "tokens": ("input", "validate", "parse", "if "),
        "suffix": "  # TODO: validate input",
    },
    "missingreturn": {
        "rule_id": "missing-return-anchor",
        "description": "Target function definitions that may miss returns.",
        "tokens": ("def ", "return "),
        "suffix": "  # TODO: ensure return",
    },
    "configerror": {
        "rule_id": "config-error-anchor",
        "description": "Flag configuration reads for correction.",
        "tokens": ("config", "settings", "env", "os.getenv"),
        "suffix": "  # TODO: verify config",
    },
    "other": {
        "rule_id": "general-error-anchor",
        "description": "Insert a general error note near the top of the file.",
        "tokens": (),
        "suffix": "  # TODO: investigate error",
    },
}


def _normalize_rule_id(value: str | None) -> str:
    if not value:
        return "noop"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-_")
    return slug or "noop"


def _first_non_empty_line(lines: list[str]) -> str:
    for line in lines:
        if line.strip():
            return line
    return ""


def _find_anchor_line(source: str, tokens: tuple[str, ...]) -> str:
    lines = source.splitlines()
    if tokens:
        lowered_tokens = tuple(token.lower() for token in tokens)
        for line in lines:
            lowered_line = line.lower()
            if any(token in lowered_line for token in lowered_tokens):
                return line
    fallback = _first_non_empty_line(lines)
    if fallback:
        return fallback
    return source[:1]


def _build_replacement(anchor: str, suffix: str) -> str:
    if not anchor:
        return anchor
    if suffix and suffix not in anchor:
        return f"{anchor}{suffix}"
    return anchor


def build_rules(
    error_text: str | None,
    *,
    source: str | None = None,
    rule_source: str = "sandbox_runner",
) -> list[dict[str, Any]]:
    """Create a minimal deterministic ruleset for sandbox patch generation."""

    classification: Mapping[str, Any] | None = None
    if error_text:
        try:
            classification = classify_error(error_text)
        except Exception:
            classification = None

    category: str | None = None
    matched_rule_id: str | None = None
    if isinstance(classification, Mapping):
        status = classification.get("status")
        if isinstance(status, str):
            category = status
        data = classification.get("data")
        if isinstance(data, Mapping):
            matched = data.get("matched_rule_id")
            if isinstance(matched, str):
                matched_rule_id = matched

    normalized_category = _normalize_rule_id(category)
    source_text = source if isinstance(source, str) else ""
    source_length = len(source_text)

    template = _RULE_TEMPLATES.get(normalized_category, _RULE_TEMPLATES["other"])
    tokens = template["tokens"]
    if not isinstance(tokens, tuple):
        tokens = ()

    anchor = _find_anchor_line(source_text, tokens)

    suffix = template["suffix"]
    if not isinstance(suffix, str):
        suffix = ""
    replacement = _build_replacement(anchor, suffix)
    allow_zero_matches = not anchor

    rule_id = template["rule_id"] if isinstance(template["rule_id"], str) else "generic"
    description = (
        template["description"]
        if isinstance(template["description"], str)
        else "Deterministic sandbox rule"
    )

    return [
        {
            "type": "replace",
            "id": f"{rule_id}-{normalized_category}",
            "description": description,
            "anchor": anchor or _ANCHOR_PLACEHOLDER,
            "anchor_kind": "literal",
            "replacement": replacement,
            "count": 1,
            "allow_zero_matches": allow_zero_matches,
            "meta": {
                "source": rule_source,
                "intent": "anchor-replace",
                "error_category": category or "unknown",
                "error_rule_id": matched_rule_id or "unknown",
                "source_length": source_length,
                "anchor_tokens": list(tokens),
            },
        }
    ]


__all__ = ["build_rules"]
