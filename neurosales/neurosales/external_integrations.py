from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, TYPE_CHECKING
import time
from . import config
import os
import logging
import requests
from billing.openai_wrapper import chat_completion_create

try:  # optional dependency
    import pinecone  # type: ignore
except Exception:  # pragma: no cover - optional dep
    pinecone = None  # type: ignore

try:  # optional dependency
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dep
    GraphDatabase = None  # type: ignore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from vector_service.context_builder import ContextBuilder


class RedditHarvester:
    """Harvest dopamine-heavy comment trees from specific subreddits."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    def harvest(
        self, subreddits: List[str], keywords: List[str], limit: int = 100
    ) -> List[Dict[str, Any]]:
        url = "https://api.pushshift.io/reddit/comment/search/"
        params = {
            "subreddit": ",".join(subreddits),
            "q": " | ".join(keywords),
            "size": limit,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        resp = self.session.get(url, params=params, timeout=10)
        data = resp.json().get("data", [])
        return [
            {
                "id": c.get("id"),
                "parent_id": c.get("parent_id"),
                "body": c.get("body"),
                "link_id": c.get("link_id"),
            }
            for c in data
        ]

    # ------------------------------------------------------------------
    def comment_tree(self, link_id: str) -> List[Dict[str, Any]]:
        url = "https://api.pushshift.io/reddit/comment/search/"
        params = {
            "link_id": link_id,
            "sort": "asc",
            "sort_type": "created_utc",
            "size": 500,
        }
        resp = self.session.get(url, params=params, timeout=10)
        return resp.json().get("data", [])


class TwitterTracker:
    """Track hashtag surges with auto-refreshing OAuth token."""

    def __init__(self, token_hook: Callable[[], str]) -> None:
        self.token_hook = token_hook
        self.token = token_hook()
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _refresh(self) -> None:
        self.token = self.token_hook()

    # ------------------------------------------------------------------
    def search_hashtag(self, hashtag: str, max_results: int = 10) -> Dict[str, Any]:
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {"query": hashtag, "max_results": max_results}
        resp = self.session.get(url, headers=self._headers(), params=params, timeout=10)
        if resp.status_code == 401:
            self._refresh()
            resp = self.session.get(
                url, headers=self._headers(), params=params, timeout=10
            )
        return resp.json()


class GPT4Client:
    """Stream GPT-4 chat completions for adaptive ranking."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        context_builder: "ContextBuilder",
    ) -> None:
        if api_key is None:
            api_key = config.load_config().openai_key or os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        self.context_builder = context_builder
        self.enabled = api_key is not None
        if self.enabled:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        else:  # pragma: no cover - warning path
            logger.warning("GPT4Client disabled: backend unavailable")

    # ------------------------------------------------------------------
    def stream_chat(
        self,
        archetype: str,
        emotion_tensor: List[float],
        objective: str,
        text: str,
    ) -> Iterator[str]:
        if not self.enabled:
            logger.warning("GPT4Client disabled: backend unavailable")
            yield ""
            return
        prompt = self.context_builder.build_prompt(
            text,
            intent={
                "archetype": archetype,
                "emotion_tensor": emotion_tensor,
                "objective": objective,
            },
        )
        try:
            resp = chat_completion_create(prompt, model="gpt-4")
            content = ""
            try:
                content = resp["choices"][0]["message"]["content"]
            except Exception:
                pass
            yield content
        except Exception:  # pragma: no cover - best effort
            logger.warning("GPT4Client disabled: backend unavailable")
            yield ""


class PineconeLogger:
    """Persist and query utterance embeddings in Pinecone."""

    def __init__(
        self,
        index_name: str | None = None,
        *,
        api_key: str | None = None,
        environment: str | None = None,
    ) -> None:
        self.enabled = pinecone is not None
        if not self.enabled:
            self.index = None
            return

        cfg = config.load_config()
        index_name = index_name or cfg.pinecone_index
        api_key = api_key or cfg.pinecone_key
        environment = environment or cfg.pinecone_env

        if not (index_name and api_key and environment):
            logger.warning("PineconeLogger disabled: missing configuration")
            self.enabled = False
            self.index = None
            return

        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536)
        self.index = pinecone.Index(index_name)

    # ------------------------------------------------------------------
    def log(self, user_id: str, embedding: List[float], text: str) -> None:
        if not self.enabled or self.index is None:
            logger.warning("PineconeLogger disabled: library unavailable")
            return
        uid = f"{user_id}-{int(time.time()*1000)}"
        self.index.upsert([(uid, embedding, {"text": text})])

    # ------------------------------------------------------------------
    def query(self, embedding: List[float], top_k: int = 3) -> Dict[str, Any]:
        if not self.enabled or self.index is None:
            logger.warning("PineconeLogger disabled: library unavailable")
            return {"matches": []}
        return self.index.query(embedding, top_k=top_k, include_metadata=True)


class InfluenceGraphUpdater:
    """Batch update archetype nodes and edges in Neo4j."""

    def __init__(
        self, uri: Optional[str] = None, auth: Optional[tuple[str, str]] = None
    ) -> None:
        cfg = config.load_config()
        uri = uri or cfg.neo4j_uri
        if auth is None:
            if cfg.neo4j_user and cfg.neo4j_pass:
                auth = (cfg.neo4j_user, cfg.neo4j_pass)

        self.enabled = GraphDatabase is not None and uri and auth
        if self.enabled:
            self.driver = GraphDatabase.driver(uri, auth=auth)
        else:  # pragma: no cover - optional dep missing
            self.driver = None

    # ------------------------------------------------------------------
    def batch_update(
        self, archetypes: Iterable[str], edges: Iterable[tuple[str, str, str]]
    ) -> None:
        if not self.enabled or self.driver is None:
            logger.warning("InfluenceGraphUpdater disabled: library unavailable")
            return
        query = """
        UNWIND $nodes AS n
        MERGE (:Archetype {name:n})
        WITH $edges AS es
        UNWIND es AS e
        MATCH (a:Archetype {name:e[0]}), (b:Archetype {name:e[1]})
        MERGE (a)-[:REL {type:e[2]}]->(b)
        """
        with self.driver.session() as session:
            session.run(query, nodes=list(archetypes), edges=list(edges))

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.enabled and self.driver is not None:
            self.driver.close()
