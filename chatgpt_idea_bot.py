"""ChatGPT Idea Bot for generating and validating business models."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional, Any, TYPE_CHECKING
from pathlib import Path

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

from . import database_manager, RAISE_ERRORS
from .database_management_bot import DatabaseManagementBot

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .gpt_memory import GPTMemoryManager

DEFAULT_IDEA_DB = database_manager.DB_PATH
IDEA_DB_PATH = Path(os.environ.get("IDEA_DB_PATH", str(DEFAULT_IDEA_DB)))


@dataclass
class ChatGPTClient:
    """Simple wrapper for the OpenAI chat completion API with offline fallback."""

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"))
    session: requests.Session | None = None
    offline_cache_path: str | None = field(default_factory=lambda: os.environ.get("CHATGPT_CACHE_FILE"))
    timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_RETRIES", "1")))

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

    def ask(
        self,
        messages: List[Dict[str, str]],
        *,
        timeout: int | None = None,
        max_retries: int | None = None,
        validate: bool = True,
        knowledge: Any | None = None,
        tags: Iterable[str] | None = None,
        memory_manager: "GPTMemoryManager" | None = None,
        use_memory: bool = False,
        relevance_threshold: float = 0.0,
        max_summary_length: int = 500,
    ) -> Dict[str, object]:
        memory = memory_manager
        if memory is None:
            if knowledge is not None:
                if (
                    hasattr(knowledge, "log_interaction")
                    and (
                        hasattr(knowledge, "search_context")
                        or hasattr(knowledge, "get_similar_entries")
                    )
                ):
                    memory = knowledge
                elif hasattr(knowledge, "GPTMemory"):
                    try:
                        memory = knowledge.GPTMemory()
                    except Exception:
                        memory = None
            else:
                try:
                    from .gpt_memory import GPTMemoryManager as _GPTMemory  # type: ignore

                    memory = _GPTMemory()
                except Exception:
                    memory = None

        def _log(prompt: str, response: str) -> None:
            if memory and hasattr(memory, "log_interaction"):
                try:
                    memory.log_interaction(prompt, response, tags or [])
                except Exception:
                    logger.exception("failed to log interaction")

        user_prompt = messages[-1].get("content", "") if messages else ""
        messages_for_api = messages
        if use_memory and memory is not None:
            try:
                if hasattr(memory, "get_similar_entries"):
                    matches = memory.get_similar_entries(user_prompt, limit=5)
                    relevant = [e for s, e in matches if s >= relevance_threshold]
                elif hasattr(memory, "search_context"):
                    relevant = memory.search_context(user_prompt)
                else:
                    relevant = []
                if relevant:
                    ctx_parts = [
                        f"Prompt: {getattr(e, 'prompt', '')}\nResponse: {getattr(e, 'response', '')}"
                        for e in relevant
                    ]
                    context_text = "\n\n".join(ctx_parts)
                    if len(context_text) > max_summary_length:
                        context_text = context_text[:max_summary_length]
                    messages_for_api = [{"role": "system", "content": context_text}] + messages
            except Exception:
                logger.exception("context retrieval failed")

        if not self.session or requests is None:
            logger.error("HTTP session unavailable, using offline response")
            result = self._offline_response(messages_for_api)
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            _log(user_prompt, text)
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
                    _log(user_prompt, text)
                    return result
                continue
            except requests.RequestException as exc:
                logger.error("chat completion request error: %s", exc)
                if attempt >= attempts - 1:
                    if RAISE_ERRORS:
                        raise
                    result = self._offline_response(messages_for_api)
                    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    _log(user_prompt, text)
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
                        _log(user_prompt, text)
                        return result
                    continue
                if validate and not self._valid_schema(data):
                    logger.error("invalid response schema from API")
                    if attempt >= attempts - 1:
                        result = self._offline_response(messages_for_api)
                        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        _log(user_prompt, text)
                        return result
                    continue
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                _log(user_prompt, text)
                return data
            elif resp.status_code in (401, 403):
                logger.error("authorization error with OpenAI API (status %s)", resp.status_code)
                if RAISE_ERRORS:
                    raise RuntimeError("unauthorized")
                result = self._offline_response(messages_for_api)
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                _log(user_prompt, text)
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
        _log(user_prompt, text)
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


def build_prompt(tags: Iterable[str], prior: str | None = None) -> List[Dict[str, str]]:
    """Construct a prompt using tags and optional prior context."""
    parts = ["Suggest five new online business models"]
    if prior:
        parts.append(f"building on {prior}")
    if tags:
        parts.append("with a focus on " + ", ".join(tags))
    prompt = " ".join(parts) + ". Respond in JSON list format with fields name, description and tags."
    return [{"role": "user", "content": prompt}]


@dataclass
class Idea:
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    unique: bool = True
    insight: str | None = None


@dataclass
class SocialValidator:
    twitter_token: str | None = field(default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN"))
    reddit_user_agent: str = field(default_factory=lambda: os.environ.get("REDDIT_USER_AGENT", "menace-bot/0.1"))

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


def follow_up(client: ChatGPTClient, idea: Idea) -> str:
    """Request additional insight for a single idea."""
    messages = [
        {
            "role": "user",
            "content": f"Provide deeper insight or variations for this business model: {idea.name} - {idea.description}",
        }
    ]
    try:
        data = client.ask(messages)
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
) -> List[Idea]:
    """Generate ideas and filter them for novelty."""
    logger.info("requesting ideas for tags: %s", ", ".join(tags))
    messages = build_prompt(tags)
    response = client.ask(messages)
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
            follow_up(client, idea)
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
