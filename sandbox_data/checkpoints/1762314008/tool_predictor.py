"""Predict the next tool a bot may require.

The predictor attempts to load a fine-tuned language model from disk. When a
model is unavailable a couple of lightweight heuristic fallbacks are used so
callers can depend on the function without heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - allow heuristic fallback
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore

# ``resolve_path`` ensures model files are located relative to the project root
# while falling back to a simple Path when the model directory is absent.
try:
    DEFAULT_PATH = resolve_path("micro_models/tool_predictor_model")
except FileNotFoundError:  # pragma: no cover - model may be missing in tests
    DEFAULT_PATH = Path("micro_models/tool_predictor_model")
_MODEL_CACHE: tuple | None = None


def _load_model(path: Path) -> tuple | None:
    """Return ``(tokenizer, model)`` if a fine-tuned model exists at ``path``."""

    global _MODEL_CACHE
    if _MODEL_CACHE is None and AutoModelForCausalLM is not None:
        try:
            tok = AutoTokenizer.from_pretrained(path)  # type: ignore[union-attr]
            mdl = AutoModelForCausalLM.from_pretrained(path)  # type: ignore[union-attr]
            _MODEL_CACHE = (tok, mdl)
        except Exception:  # pragma: no cover - model not present
            _MODEL_CACHE = (None, None)
    return _MODEL_CACHE


def _heuristic_predictions(spec) -> List[Tuple[str, float]]:
    """Fallback suggestions when no language model is available."""

    tools: List[Tuple[str, float]] = []
    caps = getattr(spec, "capabilities", [])
    if caps:
        tools.extend((cap, 0.5) for cap in caps[:3])
    else:
        funcs = getattr(spec, "functions", [])
        tools.extend((func, 0.1) for func in funcs[:3])
    return tools


def predict_tools(spec, *, model_path: Path | None = None) -> List[Tuple[str, float]]:
    """Return a ranked list of ``(tool, score)`` for ``spec``.

    The function gracefully degrades when optional dependencies or models are not
    available and therefore may return heuristic suggestions with low
    confidence scores.
    """

    path = model_path or DEFAULT_PATH
    if AutoModelForCausalLM is not None:
        tok, mdl = _load_model(path) or (None, None)
        if tok is not None and mdl is not None:
            try:
                desc = spec.description or spec.purpose or spec.name
                fields = [f"Name: {spec.name}", f"Description: {desc}"]
                if getattr(spec, "capabilities", None):
                    fields.append(
                        "Capabilities: " + ", ".join(getattr(spec, "capabilities"))
                    )
                if getattr(spec, "functions", None):
                    fields.append("Functions: " + ", ".join(getattr(spec, "functions")))
                prompt = (
                    "Suggest the next tool for this bot based on the details.\n"
                    + "\n".join(fields)
                    + "\nTool:"
                )
                inputs = tok(prompt, return_tensors="pt")  # type: ignore[union-attr]
                out_ids = mdl.generate(**inputs, max_new_tokens=32)  # type: ignore[union-attr]
                text = tok.decode(out_ids[0], skip_special_tokens=True)
                tools: List[Tuple[str, float]] = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    tools.append((line, 0.9))
                if tools:
                    return tools
            except Exception:  # pragma: no cover - inference failure
                pass
    return _heuristic_predictions(spec)


__all__ = ["predict_tools"]

