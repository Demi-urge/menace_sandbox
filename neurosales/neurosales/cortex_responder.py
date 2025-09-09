from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .memory import ConversationMemory
from .response_generation import ResponseCandidateGenerator
from .scoring import CandidateResponseScorer, ResponsePriorityQueue
from .external_integrations import GPT4Client, PineconeLogger
from typing import TYPE_CHECKING
from context_builder_util import create_context_builder

try:  # pragma: no cover - optional dependency
    from vector_service import ContextBuilder
except Exception:  # pragma: no cover - fallback when vector service missing
    class ContextBuilder:  # type: ignore[misc]
        pass

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .user_preferences import PreferenceProfile


@dataclass
class ResponseRecord:
    """Record stored in Postgres for top ranked responses."""

    session_id: str
    response: str
    score: float
    timestamp: float = field(default_factory=time.time)


class InMemoryResponseDB:
    """Very small stand-in for a Postgres table of responses."""

    def __init__(self) -> None:
        self.rows: List[ResponseRecord] = []

    def insert(self, record: ResponseRecord) -> None:
        self.rows.append(record)


class CortexAwareResponder:
    """Pipeline performing GPT-4 generation, embedding storage, and ranking."""

    def __init__(
        self,
        openai_key: str | None = None,
        *,
        pinecone_index: str,
        pinecone_key: str,
        pinecone_env: str,
        pg: Optional[InMemoryResponseDB] = None,
        context_builder: Optional[ContextBuilder] = None,  # nocb
    ) -> None:
        builder = context_builder or create_context_builder()
        self.client = GPT4Client(openai_key, context_builder=builder)
        self.pinecone = PineconeLogger(
            pinecone_index, api_key=pinecone_key, environment=pinecone_env
        )
        self.pg = pg or InMemoryResponseDB()
        # allow access to the builder for overrides
        self.context_builder = self.client.context_builder
        self.generator = ResponseCandidateGenerator(
            context_builder=self.client.context_builder
        )
        self.scorer = CandidateResponseScorer()
        self.queue = ResponsePriorityQueue()

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        from .embedding import embed_text

        return embed_text(text)

    # ------------------------------------------------------------------
    def generate_response(
        self,
        session_id: str,
        user_id: str,
        text: str,
        memory: ConversationMemory,
        profile: "PreferenceProfile",
    ) -> str:
        # GPT-4 first pass
        first_pass = "".join(
            self.client.stream_chat(user_id, [], profile.archetype, text)
        )

        history_texts = [m.content for m in memory.get_recent_messages()]

        # log embeddings to Pinecone
        embed_text = " ".join([text, first_pass] + history_texts)
        vec = self._embed(embed_text)
        self.pinecone.log(session_id, vec, embed_text)

        # candidate generation
        self.generator.add_past_response(first_pass)
        candidates = self.generator.generate_candidates(  # nocb
            text,
            history_texts,
            profile.archetype,
        )
        if first_pass not in candidates:
            candidates.append(first_pass)

        scores = self.scorer.score_candidates(text, candidates, profile, history_texts)
        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best, best_score = ranked[0]
            for resp, sc in ranked[1:]:
                self.queue.add_response(resp, {"score": sc})
        else:
            best = first_pass
            best_score = 0.0

        self.pg.insert(ResponseRecord(session_id, best, best_score))
        return best
