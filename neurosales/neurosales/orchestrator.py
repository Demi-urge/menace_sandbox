from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .memory import DatabaseConversationMemory
from .embedding_memory import EmbeddingConversationMemory
from .user_preferences import PreferenceEngine, PreferenceProfile
try:  # pragma: no cover - optional dependency
    from .response_generation import ResponseCandidateGenerator
    from vector_service import ContextBuilder
except Exception as exc:  # pragma: no cover - explicit failure
    raise ImportError(
        "vector_service is required for SandboxOrchestrator; install via `pip install vector_service`"
    ) from exc
from .scoring import CandidateResponseScorer
from .rl_integration import DatabaseRLResponseRanker
from .self_learning import SelfLearningEngine
from .reactions import ReactionHistory
from .dynamic_harvest import DynamicHarvester


class SandboxOrchestrator:
    """Coordinate memory, scoring, and ranking for chat sessions."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        persistent: bool = False,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.persistent = persistent
        self.session_factory = session_factory
        self.db_url = db_url
        self.context_builder = context_builder
        self.memories: Dict[str, DatabaseConversationMemory | EmbeddingConversationMemory] = {}
        self.reactions: Dict[str, ReactionHistory] = {}
        self.preferences = PreferenceEngine()
        self.generator = ResponseCandidateGenerator(
            context_builder=self.context_builder
        )
        self.scorer = CandidateResponseScorer()
        self.ranker = DatabaseRLResponseRanker(
            session_factory=session_factory, db_url=db_url
        )
        self.learner = SelfLearningEngine(
            session_factory=session_factory, db_url=db_url
        )
        self.harvester = DynamicHarvester()
        self._pending_state: Dict[str, Tuple[int, ...]] = {}
        self._pending_reply: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def _get_memory(self, user_id: str):
        mem = self.memories.get(user_id)
        if mem is None:
            if self.persistent:
                mem = DatabaseConversationMemory(
                    user_id=user_id,
                    session_factory=self.session_factory,
                    db_url=self.db_url,
                )
            else:
                mem = EmbeddingConversationMemory()
            self.memories[user_id] = mem
        return mem

    # ------------------------------------------------------------------
    def _get_reactions(self, user_id: str) -> ReactionHistory:
        hist = self.reactions.get(user_id)
        if hist is None:
            hist = ReactionHistory(
                session_factory=self.session_factory,
                db_url=self.db_url,
            )
            self.reactions[user_id] = hist
        return hist

    # ------------------------------------------------------------------
    def _capture_feedback(
        self,
        user_id: str,
        followup: str,
        next_actions: List[str],
        correction: Optional[str] = None,
    ) -> None:
        if user_id not in self._pending_reply:
            return

        features = [len(followup), followup.count("!"), followup.count("?")]
        engagement = self.scorer._predict_engagement(features)

        last_reply = self._pending_reply.pop(user_id)
        state = self._pending_state.pop(user_id)

        self._get_reactions(user_id).add_pair(last_reply, followup)
        self.learner.log_interaction(
            last_reply, followup, correction=correction, engagement=engagement
        )

        mem = self._get_memory(user_id)
        next_state = (len(mem.get_recent_messages()),)
        self.ranker.log_outcome(user_id, state, last_reply, engagement, next_state, next_actions)

    # ------------------------------------------------------------------
    def handle_chat(self, user_id: str, text: str) -> Tuple[str, float]:
        mem = self._get_memory(user_id)
        mem.add_message("user", text)
        self.preferences.add_message(user_id, text)
        profile: PreferenceProfile = self.preferences.get_profile(user_id)
        history = [m.content for m in mem.get_recent_messages()]
        cands = self.generator.generate_candidates(text, history, profile.archetype)  # nocb
        scores = self.scorer.score_candidates(text, cands, profile, history)

        self._capture_feedback(user_id, text, list(scores))

        ranked = self.ranker.rank(user_id, scores, history)
        if ranked:
            reply = ranked[0]
            confidence = scores.get(reply, 0.0)
        else:
            reply = ""
            confidence = 0.0
        mem.add_message("assistant", reply)
        self.generator.add_past_response(reply)
        self._get_reactions(user_id).add_pair(text, reply)
        self.learner.log_interaction(text, reply)
        self._pending_state[user_id] = (len(history),)
        self._pending_reply[user_id] = reply
        return reply, confidence

    # ------------------------------------------------------------------
    def harvest_content(
        self,
        url: str,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        selector: str = "article",
    ) -> List[str]:
        if username and password:
            return self.harvester.harvest_dashboard(url, username, password)
        return self.harvester.harvest_infinite_scroll(url, selector)


__all__ = ["SandboxOrchestrator"]
