from __future__ import annotations

import random
from typing import Dict, List
import uuid

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional heavy dep
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

try:
    from vector_service import ContextBuilder, FallbackResult, ErrorResult
except Exception as exc:  # pragma: no cover - explicit failure
    raise ImportError(
        "vector_service is required for response generation; "
        "install via `pip install vector_service`"
    ) from exc

from snippet_compressor import compress_snippets


# ---------------------------------------------------------------------------

def redundancy_filter(candidates: List[str], threshold: float = 0.7) -> List[str]:
    """Remove semantically redundant responses using Jaccard metrics."""

    def ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
        tokens = text.lower().split()
        return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()

    filtered: List[str] = []
    for cand in candidates:
        keep = True
        cand_set = set(cand.lower().split())
        cand_ng = ngrams(cand)
        for prev in filtered:
            prev_set = set(prev.lower().split())
            inter = cand_set & prev_set
            union = cand_set | prev_set
            jaccard = len(inter) / len(union) if union else 0.0
            prev_ng = ngrams(prev)
            ng_union = cand_ng | prev_ng
            ng_inter = cand_ng & prev_ng
            ng_jaccard = len(ng_inter) / len(ng_union) if ng_union else 0.0
            if max(jaccard, ng_jaccard) >= threshold:
                keep = False
                break
        if keep:
            filtered.append(cand)
    return filtered


# ---------------------------------------------------------------------------

class ResponseCandidateGenerator:
    """Generate response candidates from scripts, language model, and history."""

    def __init__(self, *, context_builder: ContextBuilder) -> None:
        self.context_builder = context_builder
        # Ensure ranking weights are loaded before generating any context
        self.context_builder.refresh_db_weights()
        self.static_scripts: Dict[str, List[str]] = {
            "curiosity": [
                "Did you know our latest offer?",
                "Ever wondered how pros succeed so quickly?",
            ],
            "fear": [
                "Imagine the cost of missing out.",
                "Don't let this opportunity slip away.",
            ],
            "tribal": [
                "Join others just like you today!",
                "Everyone in the community loves this approach.",
            ],
        }
        scripts = [s for lst in self.static_scripts.values() for s in lst]
        self.script_vectorizer = TfidfVectorizer(stop_words="english") if TfidfVectorizer else None
        self.past_vectorizer = TfidfVectorizer(stop_words="english") if TfidfVectorizer else None
        self.script_matrix = (
            self.script_vectorizer.fit_transform(scripts) if self.script_vectorizer else None
        )
        self.script_texts = scripts
        self.past_responses: List[str] = []
        self.past_matrix = None
        self.tokenizer = None
        self.model = None
        if AutoTokenizer and AutoModelForCausalLM and torch is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
            except Exception:
                self.tokenizer = None
                self.model = None

    # ------------------------------------------------------------------
    def add_past_response(self, response: str) -> None:
        self.past_responses.append(response)
        if self.past_vectorizer is not None:
            self.past_matrix = self.past_vectorizer.fit_transform(self.past_responses)

    # ------------------------------------------------------------------
    def _static_candidates(self, message: str, top_k: int = 3) -> List[str]:
        if not self.script_vectorizer or self.script_matrix is None:
            return random.sample(self.script_texts, min(top_k, len(self.script_texts)))
        query = self.script_vectorizer.transform([message])
        sims = cosine_similarity(query, self.script_matrix)[0]
        idx = sims.argsort()[-top_k:][::-1]
        return [self.script_texts[i] for i in idx]

    # ------------------------------------------------------------------
    def _past_candidates(self, message: str, top_k: int = 3) -> List[str]:
        if not self.past_responses:
            return []
        if not self.past_vectorizer or self.past_matrix is None:
            return self.past_responses[:top_k]
        query = self.past_vectorizer.transform([message])
        sims = cosine_similarity(query, self.past_matrix)[0]
        idx = sims.argsort()[-top_k:][::-1]
        return [self.past_responses[i] for i in idx]

    # ------------------------------------------------------------------
    def _dynamic_candidates(
        self,
        message: str,
        history: List[str],
        archetype: str,
        n: int = 3,
        *,
        context_builder: ContextBuilder,
    ) -> List[str]:
        if self.tokenizer and self.model and torch is not None:
            try:
                prompt = " ".join(history + [message, archetype])
                session_id = uuid.uuid4().hex
                ctx_res = context_builder.build(message, session_id=session_id)
                ctx = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
                if isinstance(ctx, (FallbackResult, ErrorResult)):
                    ctx = ""
                if ctx:
                    if isinstance(ctx, dict):
                        ctx = compress_snippets(ctx).get("snippet", "")
                    else:
                        ctx = compress_snippets({"snippet": ctx}).get("snippet", ctx)
                    if ctx:
                        prompt = f"{ctx}\n\n{prompt}"
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                outputs = self.model.generate(  # nocb
                    input_ids,
                    max_length=input_ids.shape[1] + 20,
                    num_return_sequences=n,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
                decoded = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                return decoded
            except Exception:
                pass
        base = f"{archetype}: " if archetype else ""
        return [f"{base}{message} ..." for _ in range(n)]

    # ------------------------------------------------------------------
    def generate_candidates(
        self,
        message: str,
        history: List[str],
        archetype: str = "",
        *,
        context_builder: ContextBuilder,
    ) -> List[str]:
        candidates: List[str] = []
        candidates.extend(self._static_candidates(message))
        candidates.extend(
            self._dynamic_candidates(
                message,
                history,
                archetype,
                context_builder=context_builder,
            )
        )
        candidates.extend(self._past_candidates(message))
        return redundancy_filter(candidates)
