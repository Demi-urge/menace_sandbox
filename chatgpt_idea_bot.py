"""ChatGPT Idea Bot for generating and validating business models."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Any, TYPE_CHECKING, Callable
from pathlib import Path
from context_builder import handle_failure, PromptBuildError
from billing.prompt_notice import prepend_payment_notice
from prompt_types import Prompt
from llm_interface import LLMClient, LLMResult, VALID_PROMPT_ORIGINS
try:  # pragma: no cover - optional billing dependency
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - best effort
    stripe_billing_router = None  # type: ignore
from dynamic_path_router import resolve_path

OPENAI_CHAT_URL = os.environ.get(
    "OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions"
)
TWITTER_SEARCH_URL = os.environ.get(
    "TWITTER_SEARCH_URL", "https://api.twitter.com/2/tweets/search/recent"
)
REDDIT_SEARCH_URL = os.environ.get(
    "REDDIT_SEARCH_URL", "https://www.reddit.com/search.json"
)
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "30"))
TWITTER_TIMEOUT = int(os.environ.get("TWITTER_TIMEOUT", "10"))
REDDIT_TIMEOUT = int(os.environ.get("REDDIT_TIMEOUT", "10"))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from . import database_manager, RAISE_ERRORS  # noqa: E402
from .database_management_bot import DatabaseManagementBot  # noqa: E402
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore
try:  # helper for GPT memory tagging
    from .memory_logging import log_with_tags
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_logging import log_with_tags  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .local_knowledge_module import LocalKnowledgeModule
except Exception:  # pragma: no cover - fallback for flat layout
    from local_knowledge_module import LocalKnowledgeModule  # type: ignore
try:  # contextual history retrieval
    from .knowledge_retriever import (
        get_feedback,
        get_improvement_paths,
        get_error_fixes,
    )
except Exception:  # pragma: no cover - fallback for flat layout
    from knowledge_retriever import (  # type: ignore
        get_feedback,
        get_improvement_paths,
        get_error_fixes,
    )
try:
    from .run_autonomous import LOCAL_KNOWLEDGE_MODULE as _LOCAL_KNOWLEDGE
except Exception:
    try:
        from .sandbox_runner import LOCAL_KNOWLEDGE_MODULE as _LOCAL_KNOWLEDGE
    except Exception:  # pragma: no cover - fallback
        _LOCAL_KNOWLEDGE = None
if _LOCAL_KNOWLEDGE is None:
    _LOCAL_KNOWLEDGE = LocalKnowledgeModule(manager=GPT_MEMORY_MANAGER)
LOCAL_KNOWLEDGE_MODULE = _LOCAL_KNOWLEDGE

from governed_retrieval import govern_retrieval, redact  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from gpt_memory_interface import GPTMemoryInterface

# Optional dependency for advanced retrieval
try:  # pragma: no cover - optional
    from vector_service import Retriever, FallbackResult
except Exception:  # pragma: no cover - missing dependency
    Retriever = None  # type: ignore

    class FallbackResult(list):  # type: ignore
        pass

try:  # pragma: no cover - optional
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback
    class ErrorResult(Exception):  # type: ignore
        pass

try:  # pragma: no cover - optional
    from vector_service.context_builder import (
        ContextBuilder,
        build_prompt as _context_build_prompt,
    )
except Exception:  # pragma: no cover - fallback when service missing
    ContextBuilder = Any  # type: ignore
    _context_build_prompt = None  # type: ignore

DEFAULT_IDEA_DB = database_manager.DB_PATH
IDEA_DB_PATH = Path(resolve_path(os.environ.get("IDEA_DB_PATH", str(DEFAULT_IDEA_DB))))


def _build_contextual_prompt(
    goal: str,
    *,
    context_builder: ContextBuilder,
    intent_metadata: Dict[str, Any] | None = None,
    fallback_goal: Any | None = None,
    fallback_kwargs: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Prompt:
    """Build a :class:`Prompt` using the active context engine."""

    if context_builder is None:
        raise ValueError("context_builder is required")

    if _context_build_prompt is not None and isinstance(goal, str) and goal.strip():
        try:
            return _context_build_prompt(
                goal,
                intent=intent_metadata,
                context_builder=context_builder,
                **kwargs,
            )
        except PromptBuildError:
            raise
        except (AttributeError, TypeError, ValueError):
            # Fallback to legacy builder signature below.
            pass

    builder_fn = getattr(context_builder, "build_prompt", None)
    if not callable(builder_fn):
        raise TypeError("context_builder missing build_prompt")

    target = fallback_goal if fallback_goal is not None else goal
    call_kwargs: Dict[str, Any] = {}
    if fallback_kwargs:
        call_kwargs.update(fallback_kwargs)
    if intent_metadata is not None:
        call_kwargs.setdefault("intent_metadata", intent_metadata)
    call_kwargs.update(kwargs)

    try:
        return builder_fn(target, **call_kwargs)
    except TypeError:
        if "intent_metadata" in call_kwargs and "intent" not in call_kwargs:
            meta = call_kwargs.pop("intent_metadata")
            call_kwargs["intent"] = meta
        return builder_fn(target, **call_kwargs)


@dataclass
class ChatGPTClient(LLMClient):
    """Wrapper for SelfCodingEngine-driven chat completion with offline fallback.

    The former OpenAI-based approach is retained for legacy compatibility.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"))
    session: requests.Session | None = None
    offline_cache_path: str | None = field(
        default_factory=lambda: os.environ.get("CHATGPT_CACHE_FILE")
    )
    timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_RETRIES", "1")))
    gpt_memory: "GPTMemoryInterface | None" = field(
        default_factory=lambda: GPT_MEMORY_MANAGER
    )
    context_builder: ContextBuilder = field(kw_only=True)
    log_prompts: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        LLMClient.__init__(self, model=self.model, log_prompts=self.log_prompts)
        if not self.session:
            if requests is None:
                raise ImportError("requests library required for ChatGPTClient")
            self.session = requests.Session()
        self._cache: dict[str, object] = {}
        if self.offline_cache_path:
            try:
                path = Path(self.offline_cache_path)
                self._cache = json.loads(path.read_text())
                logger.info("loaded offline cache from %s", path)
            except FileNotFoundError:
                logger.warning("offline cache %s not found", self.offline_cache_path)
            except Exception as exc:  # pragma: no cover - best effort
                logger.exception("failed to load offline cache: %s", exc)
                if RAISE_ERRORS:
                    raise
        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:  # pragma: no cover - validation
            logger.exception("failed to refresh context builder")
            raise RuntimeError(
                "provided ContextBuilder cannot query local databases"
            ) from exc

    def generate(
        self,
        prompt: Prompt,
        *,
        parse_fn: Callable[[str], Any] | None = None,
        backend: str | None = None,
        context_builder: ContextBuilder,
        tags: Iterable[str] | None = None,
    ) -> LLMResult:  # type: ignore[override]
        """Extend :meth:`LLMClient.generate` to accept optional ``tags``."""

        raw_tags: List[Any]
        if tags is not None:
            raw_tags = list(tags)
        else:
            raw_tags = []

        canonical_tags = [str(tag) for tag in raw_tags if str(tag)]
        unique_tags = list(dict.fromkeys(canonical_tags)) if canonical_tags else []

        metadata_attr = getattr(prompt, "metadata", None)
        if isinstance(metadata_attr, dict):
            metadata_snapshot = dict(metadata_attr)
        else:
            metadata_snapshot = dict(metadata_attr or {})

        origin = getattr(prompt, "origin", "") or metadata_snapshot.get("origin")
        if origin not in VALID_PROMPT_ORIGINS:
            raise ValueError("ChatGPTClient requires prompts built by context_builder")
        if getattr(prompt, "origin", None) != origin and origin:
            try:
                setattr(prompt, "origin", origin)
            except Exception:
                pass

        prompt_tags = list(getattr(prompt, "tags", []) or [])
        metadata_tags = list(metadata_snapshot.get("tags", []) or [])
        metadata_intent_tags = list(metadata_snapshot.get("intent_tags", []) or [])

        merged_tags = (
            list(dict.fromkeys([*prompt_tags, *metadata_tags, *unique_tags]))
            if (prompt_tags or metadata_tags or unique_tags)
            else []
        )
        merged_intent_tags = list(
            dict.fromkeys([*metadata_intent_tags, *unique_tags])
        )
        if not merged_intent_tags and merged_tags:
            merged_intent_tags = list(merged_tags)

        metadata_payload: Dict[str, Any] = dict(metadata_snapshot)
        if merged_tags:
            metadata_payload["tags"] = list(merged_tags)
        if merged_intent_tags:
            metadata_payload["intent_tags"] = list(merged_intent_tags)
        if origin:
            metadata_payload["origin"] = origin

        final_prompt = prompt
        enrich_fn = getattr(context_builder, "enrich_prompt", None)
        if callable(enrich_fn):
            try:
                result = enrich_fn(
                    prompt,
                    tags=list(merged_tags) or None,
                    metadata=dict(metadata_payload),
                    origin=origin,
                )
                if result is not None:
                    final_prompt = result
            except PromptBuildError:
                raise
            except Exception:
                logger.exception("context builder enrichment failed")
                if RAISE_ERRORS:
                    raise

        def _ensure_prompt_fields(target: Prompt) -> Prompt:
            meta_attr = getattr(target, "metadata", None)
            if isinstance(meta_attr, dict):
                meta_dict = meta_attr
            else:
                meta_dict = dict(meta_attr or {})

            if merged_tags:
                existing_meta_tags = list(meta_dict.get("tags", []) or [])
                combined_meta_tags = list(
                    dict.fromkeys([*existing_meta_tags, *merged_tags])
                )
                if combined_meta_tags:
                    meta_dict["tags"] = combined_meta_tags

            if merged_intent_tags:
                existing_intent = list(meta_dict.get("intent_tags", []) or [])
                combined_intent = list(
                    dict.fromkeys([*existing_intent, *merged_intent_tags])
                )
                if combined_intent:
                    meta_dict["intent_tags"] = combined_intent

            if origin and meta_dict.get("origin") != origin:
                meta_dict["origin"] = origin

            if not isinstance(meta_attr, dict):
                try:
                    setattr(target, "metadata", meta_dict)
                except Exception:
                    pass

            if merged_tags:
                current_tags = list(getattr(target, "tags", []) or [])
                combined_tags = list(
                    dict.fromkeys([*current_tags, *merged_tags])
                )
                if combined_tags and combined_tags != current_tags:
                    try:
                        setattr(target, "tags", combined_tags)
                    except Exception:
                        pass

            if origin and getattr(target, "origin", None) != origin:
                try:
                    setattr(target, "origin", origin)
                except Exception:
                    pass

            return target

        final_prompt = _ensure_prompt_fields(final_prompt)

        final_meta_attr = getattr(final_prompt, "metadata", None)
        if isinstance(final_meta_attr, dict):
            final_meta = final_meta_attr
        else:
            final_meta = dict(final_meta_attr or {})
        final_origin = getattr(final_prompt, "origin", "") or final_meta.get("origin")
        if final_origin not in VALID_PROMPT_ORIGINS:
            raise ValueError("context builder returned unexpected prompt origin")

        return super().generate(
            final_prompt,
            parse_fn=parse_fn,
            backend=backend,
            context_builder=context_builder,
        )

    def ask(
        self,
        messages: Prompt | List[Dict[str, str]],
        *,
        timeout: int | None = None,
        max_retries: int | None = None,
        validate: bool = True,
        knowledge: Any | None = None,
        retriever: "Retriever | None" = None,
        tags: Iterable[str] | None = None,
        memory_manager: "GPTMemoryInterface | None" = None,
        use_memory: bool | None = None,
        relevance_threshold: float = 0.0,
        max_summary_length: int = 500,
    ) -> Dict[str, object]:
        # Normalize messages to OpenAI chat format, accepting Prompt objects.
        prompt_obj: Prompt | None
        if isinstance(messages, Prompt) or (
            hasattr(messages, "user") and hasattr(messages, "metadata")
        ):
            prompt_obj = messages
            prompt_tags = list(getattr(prompt_obj, "tags", []) or [])
            if tags is None and prompt_tags:
                tags = prompt_tags
            norm_messages: List[Dict[str, str]] = []
            if prompt_obj.system:
                norm_messages.append({"role": "system", "content": prompt_obj.system})
            for ex in getattr(prompt_obj, "examples", []) or []:
                norm_messages.append({"role": "user", "content": ex})
            user_msg: Dict[str, Any] = {"role": "user", "content": prompt_obj.user}
            if getattr(prompt_obj, "metadata", None):
                user_msg["metadata"] = dict(prompt_obj.metadata)
            norm_messages.append(user_msg)
            messages = norm_messages
        else:
            prompt_obj = None

        memory: Any | None = memory_manager or knowledge or self.gpt_memory
        use_mem = use_memory if use_memory is not None else memory is not None

        def _log(request: List[Dict[str, str]], response: str) -> None:
            prompt_str = request[-1].get("content", "") if request else ""
            if tags is not None:
                mem_tags = list(tags)
                global_tags = list(tags)
            elif memory is knowledge:
                mem_tags = []
                global_tags = [IMPROVEMENT_PATH, INSIGHT]
            else:
                mem_tags = [IMPROVEMENT_PATH, INSIGHT]
                global_tags = [IMPROVEMENT_PATH, INSIGHT]
            try:
                if memory and mem_tags:
                    log_with_tags(
                        memory,
                        prompt_str,
                        response,
                        tags=["chatgpt_idea_bot.generate", *mem_tags],
                    )
                if self.gpt_memory and self.gpt_memory is not memory:
                    log_with_tags(
                        self.gpt_memory,
                        prompt_str,
                        response,
                        tags=["chatgpt_idea_bot.generate", *global_tags],
                    )
            except Exception:
                logger.exception("failed to log interaction")

        user_prompt = messages[-1].get("content", "") if messages else ""
        messages_for_api = prepend_payment_notice(list(messages))
        if use_mem:
            try:
                ctx_parts: List[str] = []
                if memory is not None:
                    def _fmt(entries: Iterable[Any], title: str) -> str:
                        parts: List[str] = []
                        for e in entries:
                            prompt = getattr(e, "prompt", "")
                            response = getattr(e, "response", "")
                            text = response or prompt
                            if text:
                                parts.append(f"- {text}")
                        if not parts:
                            return ""
                        body = "\n".join(parts)
                        return f"### {title}\n{body}"

                    try:
                        fb = get_feedback(memory, user_prompt, limit=5)
                        ctx = _fmt(fb, "Feedback")
                        if ctx:
                            ctx_parts.append(ctx)
                    except Exception:
                        pass
                    try:
                        fixes = get_error_fixes(memory, user_prompt, limit=3)
                        ctx = _fmt(fixes, "Error fixes")
                        if ctx:
                            ctx_parts.append(ctx)
                    except Exception:
                        pass
                    try:
                        improv = get_improvement_paths(memory, user_prompt, limit=3)
                        ctx = _fmt(improv, "Improvement paths")
                        if ctx:
                            ctx_parts.append(ctx)
                    except Exception:
                        pass

                if retriever is not None:
                    try:
                        hits = retriever.search(user_prompt, top_k=5)
                        if isinstance(hits, (FallbackResult, ErrorResult)):
                            if isinstance(hits, FallbackResult):
                                logger.debug(
                                    "retriever returned fallback for prompt: %s",
                                    getattr(hits, "reason", ""),
                                )
                            hits = []
                    except Exception:
                        hits = []
                    for h in hits:
                        meta = h.get("metadata", {})
                        snippet = meta.get("summary") or meta.get("message") or str(meta)
                        ctx_parts.append(snippet)

                if memory is not None:
                    entries: List[Any] = []
                    if hasattr(memory, "get_similar_entries"):
                        try:
                            vec_matches = memory.get_similar_entries(
                                user_prompt, limit=5, use_embeddings=True
                            )
                        except Exception:
                            vec_matches = []
                        try:
                            kw_matches = memory.get_similar_entries(
                                user_prompt, limit=5, use_embeddings=False
                            )
                        except Exception:
                            kw_matches = []
                        seen: set[str] = set()
                        for score, entry in vec_matches + kw_matches:
                            if score < relevance_threshold:
                                continue
                            key = f"{getattr(entry, 'prompt', '')}|{getattr(entry, 'response', '')}"
                            if key in seen:
                                continue
                            seen.add(key)
                            entries.append(entry)
                    else:
                        try:
                            if hasattr(memory, "search_context"):
                                entries = memory.search_context(user_prompt, limit=5)
                            elif hasattr(memory, "retrieve"):
                                entries = memory.retrieve(user_prompt, limit=5)
                        except Exception:
                            entries = []
                    if entries:
                        for e in entries:
                            text = (
                                f"Prompt: {getattr(e, 'prompt', '')}\n"
                                f"Response: {getattr(e, 'response', '')}"
                            )
                            if govern_retrieval(text) is None:
                                continue
                            ctx_parts.append(redact(text))
                    context_text = ""
                    if ctx_parts:
                        context_text = "\n\n".join(ctx_parts)
                        if len(context_text) > max_summary_length:
                            context_text = context_text[:max_summary_length]
                    if context_text:
                        messages_for_api[0]["content"] += "\n" + context_text
            except Exception:
                logger.exception("context retrieval failed")

        if not self.session or requests is None:
            logger.error("HTTP session unavailable, using offline response")
            result = self._offline_response(messages_for_api)
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            _log(messages_for_api, text)
            return result
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages_for_api}

        attempts = max_retries if max_retries is not None else self.max_retries
        t_out = timeout if timeout is not None else self.timeout

        for attempt in range(max(attempts, 1)):
            try:
                resp = self.session.post(
                    OPENAI_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=t_out,
                )
            except requests.Timeout:
                logger.warning("OpenAI API request timed out")
                if attempt >= attempts - 1:
                    if RAISE_ERRORS:
                        raise
                    result = self._offline_response(messages_for_api)
                    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    _log(messages_for_api, text)
                    return result
                continue
            except requests.RequestException as exc:
                logger.error("chat completion request error: %s", exc)
                if attempt >= attempts - 1:
                    if RAISE_ERRORS:
                        raise
                    result = self._offline_response(messages_for_api)
                    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    _log(messages_for_api, text)
                    return result
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except json.JSONDecodeError as exc:
                    logger.exception("invalid JSON from API: %s", exc)
                    if attempt >= attempts - 1:
                        if RAISE_ERRORS:
                            raise
                        result = self._offline_response(messages_for_api)
                        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        _log(messages_for_api, text)
                        return result
                    continue
                if validate and not self._valid_schema(data):
                    logger.error("invalid response schema from API")
                    if attempt >= attempts - 1:
                        result = self._offline_response(messages_for_api)
                        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        _log(messages_for_api, text)
                        return result
                    continue
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                _log(messages_for_api, text)
                return data
            elif resp.status_code in (401, 403):
                logger.error("authorization error with OpenAI API (status %s)", resp.status_code)
                if RAISE_ERRORS:
                    raise RuntimeError("unauthorized")
                result = self._offline_response(messages_for_api)
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                _log(messages_for_api, text)
                return result
            elif resp.status_code == 429:
                logger.warning("rate limited by OpenAI (attempt %s)", attempt + 1)
                if attempt >= attempts - 1:
                    break
                continue
            else:
                logger.warning("unexpected status %s from API", resp.status_code)
                if attempt >= attempts - 1:
                    break
        result = self._offline_response(messages_for_api)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        _log(messages_for_api, text)
        return result

    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:  # type: ignore[override]
        """Return an :class:`LLMResult` for ``prompt`` using :meth:`ask`."""

        data = self.ask(prompt)
        if isinstance(data, LLMResult):
            return data

        text = ""
        prompt_tokens = completion_tokens = None
        input_tokens = output_tokens = None
        cost = None
        latency = None

        if isinstance(data, dict):
            raw: Dict[str, Any] = dict(data)
            raw.setdefault("backend", "chatgpt_client")
            raw.setdefault("model", self.model)
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict):
                    text = message.get("content", "") or ""
            usage = raw.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                input_tokens = usage.get("input_tokens") or prompt_tokens
                output_tokens = usage.get("output_tokens") or completion_tokens
                cost = usage.get("cost") or usage.get("total_cost")
                latency = usage.get("latency_ms")
            return LLMResult(
                raw=raw,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency,
            )

        if isinstance(data, str):
            text = data
            raw = {
                "response": data,
                "backend": "chatgpt_client",
                "model": self.model,
            }
        else:
            text = str(data)
            raw = {
                "response": data,
                "backend": "chatgpt_client",
                "model": self.model,
            }
        return LLMResult(raw=raw, text=text)

    def _valid_schema(self, data: Dict[str, object]) -> bool:
        try:
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return False
            message = choices[0].get("message")
            if not isinstance(message, dict):
                return False
            if "content" not in message:
                return False
        except Exception:
            return False
        return True

    def _offline_response(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        if not self._cache:
            logger.info("using empty offline response")
            return {"choices": [{"message": {"content": "offline"}}]}
        key = messages[-1].get("content", "")[:100]
        cached = self._cache.get(key)
        if cached:
            logger.debug("returning cached response for %s", key)
            return cached
        logger.info("no cached response for %s", key)
        return {"choices": [{"message": {"content": "offline"}}]}

    def build_prompt_with_memory(
        self,
        tags: list[str],
        *,
        prior: str | None = None,
        context_builder: ContextBuilder,
        intent_metadata: Dict[str, Any] | None = None,
    ) -> Prompt:
        """Build a :class:`Prompt` via ``context_builder`` including memory."""

        if context_builder is None:
            raise ValueError("context_builder is required")

        tag_list = list(tags)
        canonical_tags = [str(tag) for tag in tag_list if str(tag)]
        intent_meta: Dict[str, Any] = {"tags": list(canonical_tags)}
        if prior:
            intent_meta["prior_ideas"] = prior
        if intent_metadata:
            intent_meta.update(intent_metadata)

        query = " ".join(canonical_tags).strip() or (prior or "").strip()
        if not query:
            raise ValueError("at least one tag or prior is required to build a prompt")

        base_prompt: Prompt | None = None

        engine = getattr(self, "engine", None)
        build_enriched = getattr(engine, "build_enriched_prompt", None)
        if callable(build_enriched):
            try:
                base_prompt = build_enriched(
                    query,
                    intent=dict(intent_meta),
                    context_builder=context_builder,
                )
            except PromptBuildError:
                raise
            except Exception:
                logger.exception("engine-based prompt enrichment failed")
                if RAISE_ERRORS:
                    raise

        if base_prompt is None:
            try:
                base_prompt = _build_contextual_prompt(
                    query,
                    context_builder=context_builder,
                    intent_metadata=intent_meta,
                    fallback_goal=list(tag_list),
                    fallback_kwargs={"prior": prior} if prior is not None else None,
                )
            except Exception as exc:
                if isinstance(exc, PromptBuildError):
                    raise
                handle_failure(
                    "failed to build prompt from context builder",
                    exc,
                    logger=logger,
                )

        if base_prompt is None:
            raise RuntimeError("context builder returned no prompt")

        metadata_payload = dict(intent_meta)
        metadata_payload.setdefault("intent_tags", list(tag_list))

        origin = (
            getattr(base_prompt, "origin", "")
            or metadata_payload.get("origin")
            or "context_builder"
        )

        enrich_fn = getattr(context_builder, "enrich_prompt", None)
        if not callable(enrich_fn):
            enrich_method = getattr(ContextBuilder, "enrich_prompt", None)

            if callable(enrich_method):

                def enrich_fn(
                    prompt: Prompt,
                    *,
                    tags: Iterable[str] | None = None,
                    metadata: Dict[str, Any] | None = None,
                    origin: str | None = None,
                ) -> Prompt:
                    return enrich_method(
                        context_builder,
                        prompt,
                        tags=tags,
                        metadata=metadata,
                        origin=origin,
                    )

        if callable(enrich_fn):
            try:
                result = enrich_fn(
                    base_prompt,
                    tags=canonical_tags or None,
                    metadata=metadata_payload,
                    origin=origin,
                )
                if result is not None:
                    base_prompt = result
            except PromptBuildError:
                raise
            except Exception:
                logger.exception("context builder enrichment failed")
                if RAISE_ERRORS:
                    raise
        else:
            metadata_attr = getattr(base_prompt, "metadata", None)
            if isinstance(metadata_attr, dict):
                metadata = metadata_attr
            else:
                metadata = dict(metadata_attr or {})

            for key, value in metadata_payload.items():
                metadata.setdefault(key, value)

            if canonical_tags:
                existing_tags = list(getattr(base_prompt, "tags", []) or [])
                merged_tags = list(dict.fromkeys([*existing_tags, *canonical_tags]))
            else:
                merged_tags = list(getattr(base_prompt, "tags", []) or [])

            if merged_tags:
                metadata.setdefault("tags", list(merged_tags))
                if list(getattr(base_prompt, "tags", []) or []) != list(merged_tags):
                    try:
                        setattr(base_prompt, "tags", list(merged_tags))
                    except Exception:
                        pass

            metadata.setdefault("origin", origin)

            if metadata_attr is not metadata:
                try:
                    setattr(base_prompt, "metadata", metadata)
                except Exception:
                    pass

        if getattr(base_prompt, "origin", None) not in VALID_PROMPT_ORIGINS:
            try:
                setattr(base_prompt, "origin", origin)
            except Exception:
                pass

        return base_prompt


def build_prompt(
    client: ChatGPTClient,
    context_builder: ContextBuilder,
    tags: Iterable[str],
    prior: str | None = None,
) -> Prompt:
    """Construct a prompt via ``client`` and ``context_builder``."""
    if context_builder is None:
        raise ValueError("context_builder is required")
    base_tags = [IMPROVEMENT_PATH, *tags]
    try:
        return client.build_prompt_with_memory(
            base_tags,
            prior=prior,
            context_builder=context_builder,
        )
    except PromptBuildError:
        raise
    except Exception as exc:
        handle_failure("prompt construction failed", exc, logger=logger)


@dataclass
class Idea:
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    unique: bool = True
    insight: str | None = None


@dataclass
class SocialValidator:
    twitter_token: str | None = field(
        default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN")
    )
    reddit_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "REDDIT_USER_AGENT", "menace-bot/0.1"
        )
    )

    def _twitter_search(self, query: str) -> bool:
        if not self.twitter_token:
            return True
        headers = {"Authorization": f"Bearer {self.twitter_token}"}
        params = {"query": query, "max_results": 10}
        try:
            resp = requests.get(
                TWITTER_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=TWITTER_TIMEOUT,
            )
        except Exception as exc:
            logger.exception("twitter search failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return True
        if resp.status_code != 200:
            logger.warning("twitter api status %s", resp.status_code)
            return True
        data = resp.json()
        return not data.get("data")

    def _reddit_search(self, query: str) -> bool:
        params = {"q": query, "limit": 5, "sort": "new"}
        headers = {"User-Agent": self.reddit_user_agent}
        try:
            resp = requests.get(
                REDDIT_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=REDDIT_TIMEOUT,
            )
        except Exception as exc:
            logger.exception("reddit search failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return True
        if resp.status_code != 200:
            logger.warning("reddit api status %s", resp.status_code)
            return True
        data = resp.json()
        return not data.get("data", {}).get("children")

    def is_unique_online(self, idea_name: str) -> bool:
        return self._twitter_search(idea_name) and self._reddit_search(idea_name)


def parse_ideas(data: Dict[str, object] | str | LLMResult) -> List[Idea]:
    """Parse JSON ideas from ChatGPT response."""
    if isinstance(data, LLMResult):
        if data.raw is not None:
            return parse_ideas(data.raw)
        return parse_ideas(data.text)
    if isinstance(data, str):
        text = data
    else:
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    try:
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("expected list of ideas")
    except Exception as exc:
        logger.exception("failed to parse ideas: %s", exc)
        if RAISE_ERRORS:
            raise
        return []
    ideas: List[Idea] = []
    for item in items:
        ideas.append(
            Idea(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                tags=list(item.get("tags", [])),
            )
        )
    return ideas


def follow_up(
    client: ChatGPTClient,
    idea: Idea,
    context_builder: ContextBuilder,
) -> str:
    """Request additional insight for a single idea."""
    prompt = "Provide deeper insight or variations for this business model."
    try:
        prompt_obj = _build_contextual_prompt(
            prompt,
            context_builder=context_builder,
            intent_metadata={
                "idea_name": idea.name,
                "description": idea.description,
            },
        )
    except Exception as exc:
        if isinstance(exc, PromptBuildError):
            raise
        handle_failure(
            "failed to build follow-up prompt",
            exc,
            logger=logger,
        )

    try:
        result = client.generate(
            prompt_obj,
            context_builder=context_builder,
            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
        )
    except PromptBuildError:
        raise
    except Exception as exc:  # pragma: no cover - network/parse failures
        handle_failure(
            "follow-up inference failed",
            exc,
            logger=logger,
        )

    text = result.text
    if not text and isinstance(result.raw, dict):
        text = (
            result.raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    idea.insight = text or ""
    if not idea.insight:
        logger.info("no follow-up insight returned for %s", idea.name)
    return idea.insight


def generate_and_filter(
    tags: Iterable[str],
    client: ChatGPTClient,
    validator: SocialValidator,
    context_builder: ContextBuilder,
) -> List[Idea]:
    """Generate ideas and filter them for novelty."""
    logger.info("requesting ideas for tags: %s", ", ".join(tags))
    prompt = (
        "Suggest five new online business models. "
        "Respond in JSON list format with fields name, description and tags."
    )
    try:
        prompt_obj = _build_contextual_prompt(
            prompt,
            context_builder=context_builder,
            intent_metadata={"tags": list(tags)},
        )
    except Exception as exc:
        if isinstance(exc, PromptBuildError):
            raise
        handle_failure("failed to build idea generation prompt", exc, logger=logger)

    try:
        result = client.generate(
            prompt_obj,
            context_builder=context_builder,
            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
        )
    except PromptBuildError:
        raise
    except Exception as exc:
        handle_failure("idea generation request failed", exc, logger=logger)
    ideas = parse_ideas(result)
    novel: List[Idea] = []
    for idea in ideas:
        if not validator.is_unique_online(idea.name):
            logger.debug("idea %s not unique online", idea.name)
            continue
        existing = database_manager.search_models(idea.name)
        if existing:
            logger.debug("idea %s already in database", idea.name)
            continue
        try:
            follow_up(client, idea, context_builder=context_builder)
        except PromptBuildError:
            raise
        except Exception as exc:
            handle_failure(
                f"failed to enrich idea {idea.name}",
                exc,
                logger=logger,
            )
        novel.append(idea)
    logger.info("generated %d novel ideas", len(novel))
    return novel


def handoff_to_database(
    idea: Idea,
    source: str = "idea_bot",
    *,
    db_path: Path = IDEA_DB_PATH,
) -> None:
    """Send idea details to the database management bot only."""
    try:
        bot = DatabaseManagementBot(db_path=db_path)
        bot.ingest_idea(idea.name, tags=idea.tags, source=source, urls=[])
    except Exception as exc:
        logger.exception("database handoff failed: %s", exc)
        if RAISE_ERRORS:
            raise
    else:
        logger.info("idea %s stored in database", idea.name)
