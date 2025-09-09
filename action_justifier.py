"""Generate plain-English explanations for Security AI flagged actions."""

from __future__ import annotations

import json
import os
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from string import Template
from pydantic import BaseModel, ValidationError

from dynamic_path_router import resolve_dir, resolve_path
from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder, FallbackResult
except Exception:  # pragma: no cover - fall back to minimal stubs
    ContextBuilder = Any  # type: ignore

    class FallbackResult(list):  # type: ignore[misc]
        pass

try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    class ErrorResult(Exception):  # type: ignore[override]
        pass


logger = logging.getLogger(__name__)

try:  # transformers is optional and only used for offline models
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
_SETTINGS_PATH = os.getenv(
    "JUSTIFIER_SETTINGS_PATH",
    str(resolve_path("config/justifier_settings.json")),
)
_LOG_DIR = Path(os.getenv("JUSTIFIER_LOG_DIR") or resolve_dir("logs"))
_JUSTIFICATION_LOG = _LOG_DIR / os.getenv("JUSTIFICATION_LOG_FILE", "justifications.jsonl")
_CACHE_DIR = Path(os.getenv("JUSTIFIER_CACHE_DIR") or (_LOG_DIR / "cache"))


class ActionLogModel(BaseModel):
    action_type: str | None = None
    action_description: str | None = None
    generated_code: bool | None = False


class InputPayload(BaseModel):
    action_log: ActionLogModel
    violation_flags: List[str] = []
    risk_score: float
    domain: str


def _load_settings() -> Dict[str, Any]:
    """Return configuration for the justifier module."""
    if not os.path.exists(_SETTINGS_PATH):
        logger.warning("settings not found at %s", _SETTINGS_PATH)
        return {"llm_mode": False}
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "llm_mode" not in data:
            raise ValueError("invalid settings structure")
        return data
    except Exception as exc:  # pragma: no cover - robust fallback
        logger.warning("Failed to load settings: %s", exc)
        return {"llm_mode": False}


# ---------------------------------------------------------------------------
# Helper: Template based explanation
_TEMPLATES = {
    "urgent": Template(
        "URGENT: $reasons. Risk score $score for domain '$domain'."
    ),
    "concerning": Template(
        "This action was flagged because $reasons. Final risk score: $score for domain '$domain'."
    ),
    "cautionary": Template(
        "Note: $reasons. Risk score $score for domain '$domain'."
    ),
}


def _severity_tone(score: float) -> str:
    if score > 0.9:
        return "urgent"
    if score > 0.6:
        return "concerning"
    return "cautionary"


def _sanitize(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def _template_justification(
    action_log: Dict[str, Any],
    violation_flags: List[str],
    risk_score: float,
    domain: str,
) -> str:
    """Return a justification string using a static template."""
    reasons: List[str] = []
    if violation_flags:
        reasons.append("it triggered rules: " + ", ".join(map(_sanitize, violation_flags)))
    action_type = _sanitize(str(action_log.get("action_type", "unspecified")))
    desc = _sanitize(str(action_log.get("action_description", "")))
    if desc:
        reasons.append(f"it attempted '{desc}' ({action_type})")
    else:
        reasons.append(f"action type '{action_type}'")
    if action_log.get("generated_code"):
        reasons.append("the generated code could modify system behaviour")
    if not reasons:
        reasons.append("risk scoring heuristics identified suspicious behaviour")
    joined = " and ".join(reasons)
    tone = _severity_tone(risk_score)
    tmpl = _TEMPLATES[tone]
    return tmpl.substitute(reasons=joined, score=f"{risk_score:.2f}", domain=_sanitize(domain))


# ---------------------------------------------------------------------------
# Helper: LLM-based explanation (offline only)

def _llm_justification(
    action_log: Dict[str, Any],
    violation_flags: List[str],
    risk_score: float,
    domain: str,
    settings: Dict[str, Any],
    *,
    context_builder: ContextBuilder,
) -> str | None:
    """Return a justification using a local language model, if available."""
    if not _TRANSFORMERS_AVAILABLE:
        return None
    model_path = settings.get("model_path")
    if not model_path:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    except Exception:
        return None
    base_prompt = (
        f"Action type: {action_log.get('action_type', 'unknown')}\n"
        f"Description: {action_log.get('action_description', '')}\n"
        f"Violations: {', '.join(violation_flags) if violation_flags else 'none'}\n"
        f"Domain: {domain}\n"
        f"Risk score: {risk_score:.2f}\n"
        "Explain briefly why this action was flagged:"
    )
    vec_ctx = ""
    try:
        payload = json.dumps(
            {
                "action_log": action_log,
                "violation_flags": violation_flags,
                "risk_score": risk_score,
                "domain": domain,
            }
        )
        ctx_res = context_builder.build(payload)
        vec_ctx = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
        if isinstance(vec_ctx, (FallbackResult, ErrorResult)):
            vec_ctx = ""
        elif vec_ctx:
            vec_ctx = compress_snippets({"snippet": vec_ctx}).get("snippet", vec_ctx)
    except Exception:
        vec_ctx = ""
    prompt = f"{vec_ctx}\n\n{base_prompt}" if vec_ctx else base_prompt
    cache_key = hashlib.sha256(
        json.dumps(
            {"log": action_log, "v": violation_flags, "r": risk_score, "d": domain},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_file = _CACHE_DIR / f"{cache_key}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text().strip()
        except Exception:
            logger.exception("failed reading justification cache")
    try:
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(tokens, max_new_tokens=60, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = text[len(prompt):].strip()
    except Exception:
        return None
    if explanation:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(explanation)
        except Exception:
            logger.exception("failed writing justification cache")
    return explanation or None


# ---------------------------------------------------------------------------
# Public API

def generate_justification(
    action_log: Dict[str, Any],
    violation_flags: List[str],
    risk_score: float,
    domain: str,
    *,
    context_builder: ContextBuilder,
) -> str:
    """Return a concise explanation of why an action was flagged."""
    try:
        _ = InputPayload(
            action_log=action_log,
            violation_flags=violation_flags,
            risk_score=risk_score,
            domain=domain,
        )
    except ValidationError as exc:
        raise ValueError(f"invalid input: {exc}") from exc

    settings = _load_settings()
    if settings.get("llm_mode"):
        llm_text = _llm_justification(
            action_log,
            violation_flags,
            risk_score,
            domain,
            settings,
            context_builder=context_builder,
        )
        if llm_text:
            return llm_text
    # Fallback to deterministic template approach
    return _template_justification(action_log, violation_flags, risk_score, domain)


def save_justification_to_log(justification: str, action_id: str) -> None:
    """Append the justification to the log file with timestamp."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action_id": action_id,
        "justification": justification,
    }
    try:
        with open(_JUSTIFICATION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover - disk issues
        logger.error("failed to write justification log: %s", exc)


__all__ = ["generate_justification", "save_justification_to_log"]
