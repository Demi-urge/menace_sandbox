"""ChatGPT Idea Bot for generating and validating business models."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Any, TYPE_CHECKING
from pathlib import Path
from billing.prompt_notice import prepend_payment_notice
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
try:  # memory-aware wrapper
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
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
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback when service missing
    ContextBuilder = Any  # type: ignore

from snippet_compressor import compress_snippets  # noqa: E402
DEFAULT_IDEA_DB = database_manager.DB_PATH
IDEA_DB_PATH = Path(resolve_path(os.environ.get("IDEA_DB_PATH", str(DEFAULT_IDEA_DB))))


@dataclass
class ChatGPTClient:
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

    def __post_init__(self) -> None:
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

    def ask(
        self,
        messages: List[Dict[str, str]],
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
                    if ctx_parts:
                        context_text = "\n\n".join(ctx_parts)
                    if len(context_text) > max_summary_length:
                        context_text = context_text[:max_summary_length]
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
        prompt: str,
        *,
        prior: str | None = None,
        context_builder: ContextBuilder,
    ) -> List[Dict[str, Any]]:
        """Prepend builder and memory-derived context to ``prompt``."""

        if context_builder is None:
            raise ValueError("context_builder is required")

        builder = context_builder
        builder_ctx = ""
        session_id = uuid.uuid4().hex
        try:
            query_parts = [*tags]
            if prior:
                query_parts.append(prior)
            query = " ".join(query_parts)
            ctx_res = builder.build(query, session_id=session_id)
            builder_ctx = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
            if isinstance(builder_ctx, (FallbackResult, ErrorResult)):
                builder_ctx = ""
            elif builder_ctx:
                builder_ctx = compress_snippets({"snippet": builder_ctx}).get(
                    "snippet", builder_ctx
                )
        except Exception:
            logger.exception("failed to build vector context")

        mem_ctx = ""
        if self.gpt_memory is not None:
            try:
                if hasattr(self.gpt_memory, "search_context"):
                    entries = self.gpt_memory.search_context("", tags=tags)
                    if entries:
                        mem_ctx = "\n".join(
                            f"{getattr(e, 'prompt', '')} {getattr(e, 'response', '')}"
                            for e in entries
                        )
                elif hasattr(self.gpt_memory, "fetch_context"):
                    mem_ctx = self.gpt_memory.fetch_context(tags)
            except Exception:
                logger.exception("failed to fetch memory context")

        combined_ctx = "\n".join(part for part in [builder_ctx, mem_ctx] if part)
        if combined_ctx:
            prompt = f"{prompt}\n{combined_ctx}"

        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        messages[0].setdefault("metadata", {})["retrieval_session_id"] = session_id
        return messages


def build_prompt(
    client: ChatGPTClient,
    context_builder: ContextBuilder,
    tags: Iterable[str],
    prior: str | None = None,
) -> List[Dict[str, Any]]:
    """Construct a prompt and fetch memory-aware messages via ``client``."""
    if context_builder is None:
        raise ValueError("context_builder is required")
    parts = ["Suggest five new online business models"]
    if prior:
        parts.append(f"building on {prior}")
    if tags:
        parts.append("with a focus on " + ", ".join(tags))
    prompt = (
        " ".join(parts)
        + ". Respond in JSON list format with fields name, description and tags."
    )
    base_tags = [IMPROVEMENT_PATH, *tags]
    return client.build_prompt_with_memory(
        base_tags,
        prompt,
        prior=prior,
        context_builder=context_builder,
    )


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


def parse_ideas(data: Dict[str, object]) -> List[Idea]:
    """Parse JSON ideas from ChatGPT response."""
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
    prompt = (
        f"Provide deeper insight or variations for this business model: {idea.name} - "
        f"{idea.description}"
    )
    try:
        data = ask_with_memory(
            client,
            "chatgpt_idea_bot.follow_up",
            prompt,
            memory=LOCAL_KNOWLEDGE_MODULE,
            context_builder=context_builder,
            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
        )
        idea.insight = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception as exc:  # pragma: no cover - network/parse failures
        logger.exception("follow-up request failed: %s", exc)
        if RAISE_ERRORS:
            raise
        idea.insight = None
    if not idea.insight:
        logger.info("no follow-up insight returned for %s", idea.name)
    return idea.insight or ""


def generate_and_filter(
    tags: Iterable[str],
    client: ChatGPTClient,
    validator: SocialValidator,
    context_builder: ContextBuilder,
) -> List[Idea]:
    """Generate ideas and filter them for novelty."""
    logger.info("requesting ideas for tags: %s", ", ".join(tags))
    parts = ["Suggest five new online business models"]
    if tags:
        parts.append("with a focus on " + ", ".join(tags))
    prompt = (
        " ".join(parts)
        + ". Respond in JSON list format with fields name, description and tags."
    )
    response = ask_with_memory(
        client,
        "chatgpt_idea_bot.generate_and_filter",
        prompt,
        memory=LOCAL_KNOWLEDGE_MODULE,
        context_builder=context_builder,
        tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
    )
    ideas = parse_ideas(response)
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
        except Exception:
            logger.exception("failed to enrich idea %s", idea.name)
            if RAISE_ERRORS:
                raise
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
