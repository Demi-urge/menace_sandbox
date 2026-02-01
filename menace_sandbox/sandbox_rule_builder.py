"""Build deterministic sandbox patch rules from error context."""

from __future__ import annotations

from typing import Any, Mapping
import re

from error_ontology import classify_error


_ANCHOR_PLACEHOLDER = "__MENACE_SANDBOX_NOOP_ANCHOR__"


def _normalize_rule_id(value: str | None) -> str:
    if not value:
        return "noop"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-_")
    return slug or "noop"


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
    source_length = len(source) if isinstance(source, str) else 0

    return [
        {
            "type": "replace",
            "id": f"noop-{normalized_category}",
            "description": f"Placeholder rule for {category or 'unknown'} classification.",
            "anchor": _ANCHOR_PLACEHOLDER,
            "anchor_kind": "literal",
            "replacement": _ANCHOR_PLACEHOLDER,
            "count": 1,
            "allow_zero_matches": True,
            "meta": {
                "source": rule_source,
                "intent": "noop",
                "error_category": category or "unknown",
                "error_rule_id": matched_rule_id or "unknown",
                "source_length": source_length,
            },
        }
    ]


__all__ = ["build_rules"]
