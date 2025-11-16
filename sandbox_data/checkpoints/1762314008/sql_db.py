from __future__ import annotations

import time
import os
from typing import Optional
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class UserProfile(Base):
    """User profile with basic metadata and interaction timestamp."""

    __tablename__ = "user_profiles"

    id = sa.Column(sa.String, primary_key=True)
    username = sa.Column(sa.String, nullable=False)
    elo = sa.Column(sa.Float, default=1000.0)
    archetype = sa.Column(sa.String, default="")
    last_interaction = sa.Column(sa.Float, default=lambda: time.time())

    preferences = relationship(
        "UserPreference", back_populates="user", cascade="all, delete-orphan"
    )
    messages = relationship(
        "ConversationMessage", back_populates="user", cascade="all, delete-orphan"
    )
    matches = relationship(
        "MatchHistory", back_populates="user", cascade="all, delete-orphan"
    )
    preference_messages = relationship(
        "PreferenceMessage", back_populates="user", cascade="all, delete-orphan"
    )


class UserPreference(Base):
    """Store key/value preferences and confidence per user."""

    __tablename__ = "user_preferences"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, sa.ForeignKey("user_profiles.id"))
    key = sa.Column(sa.String)
    value = sa.Column(sa.String)
    confidence = sa.Column(sa.Float, default=0.0)

    user = relationship("UserProfile", back_populates="preferences")


class ConversationMessage(Base):
    """History of messages exchanged with a user."""

    __tablename__ = "conversation_history"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, sa.ForeignKey("user_profiles.id"))
    role = sa.Column(sa.String)
    message = sa.Column(sa.Text)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())

    user = relationship("UserProfile", back_populates="messages")


class ArchetypeStats(Base):
    """Track archetype difficulty via elo and interactions."""

    __tablename__ = "archetypes"

    name = sa.Column(sa.String, primary_key=True)
    elo = sa.Column(sa.Float, default=1000.0)
    interactions = sa.Column(sa.Integer, default=0)


class MatchHistory(Base):
    """Outcome of conversations for elo adjustment."""

    __tablename__ = "match_history"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, sa.ForeignKey("user_profiles.id"))
    archetype = sa.Column(sa.String, sa.ForeignKey("archetypes.name"))
    outcome = sa.Column(sa.String)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())

    user = relationship("UserProfile", back_populates="matches")


class SelfLearningEvent(Base):
    """Feedback events for the self learning system."""

    __tablename__ = "self_learning_events"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    text = sa.Column(sa.Text)
    issue = sa.Column(sa.String)
    correction = sa.Column(sa.Text, nullable=True)
    engagement = sa.Column(sa.Float, default=1.0)


class SelfLearningState(Base):
    """Serialized engine state for persistence across restarts."""

    __tablename__ = "self_learning_state"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    data = sa.Column(sa.JSON)


class ReactionPair(Base):
    """Persisted phrase/reaction pair for per-user history."""

    __tablename__ = "reaction_pairs"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String)
    phrase = sa.Column(sa.Text)
    reaction = sa.Column(sa.Text)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    archived = sa.Column(sa.Boolean, default=False)


class PreferenceMessage(Base):
    """Persisted preference analysis per message."""

    __tablename__ = "preference_messages"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, sa.ForeignKey("user_profiles.id"))
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    tokens = sa.Column(sa.JSON)
    embedding = sa.Column(sa.JSON)
    intensity = sa.Column(sa.Float)
    contradiction = sa.Column(sa.Boolean)

    user = relationship("UserProfile", back_populates="preference_messages")


class EmbeddingMessage(Base):
    """Persisted embedding or vector message."""

    __tablename__ = "embedding_messages"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    role = sa.Column(sa.String)
    content = sa.Column(sa.Text)
    embedding = sa.Column(sa.JSON)
    priority = sa.Column(sa.Float, default=1.0)
    synced = sa.Column(sa.Boolean, default=False)


class EmotionEntry(Base):
    """Logged emotional state per persona."""

    __tablename__ = "emotion_entries"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    persona = sa.Column(sa.String)
    label = sa.Column(sa.String)
    intensity = sa.Column(sa.Float)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())


class RewardEntry(Base):
    """Persisted reward ledger line for a user."""

    __tablename__ = "reward_entries"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    green = sa.Column(sa.Float, default=0.0)
    violet = sa.Column(sa.Float, default=0.0)
    gold = sa.Column(sa.Float, default=0.0)
    iron = sa.Column(sa.Float, default=0.0)
    sentiment_before = sa.Column(sa.Float, default=0.0)
    sentiment_after = sa.Column(sa.Float, default=0.0)
    followups = sa.Column(sa.Integer, default=0)
    session_delta = sa.Column(sa.Float, default=0.0)
    fact_error = sa.Column(sa.Boolean, default=False)
    lost_user = sa.Column(sa.Boolean, default=False)
    confidence = sa.Column(sa.Float, default=1.0)


class RLFeedback(Base):
    """General feedback entry for RL modules."""

    __tablename__ = "rl_feedback"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    timestamp = sa.Column(sa.Float, default=lambda: time.time())
    text = sa.Column(sa.Text)
    feedback = sa.Column(sa.Text)
    score = sa.Column(sa.Float, default=1.0)
    processed = sa.Column(sa.Boolean, default=False)


class RLPolicyWeight(Base):
    """Policy weight and usage count for a response."""

    __tablename__ = "rl_policy_weights"

    response = sa.Column(sa.Text, primary_key=True)
    weight = sa.Column(sa.Float, default=0.0)
    count = sa.Column(sa.Integer, default=0)


class ReplayExperience(Base):
    """Recorded transition for Q-learning replay."""

    __tablename__ = "replay_experiences"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String)
    state = sa.Column(sa.JSON)
    action = sa.Column(sa.String)
    reward = sa.Column(sa.Float)
    next_state = sa.Column(sa.JSON)


def create_session(db_url: Optional[str] = None, **engine_kwargs):
    """Return a session factory for the schema.

    If ``db_url`` is omitted it will be taken from the ``NEURO_DB_URL``
    environment variable. When neither is provided an in-memory SQLite
    database is used.

    Extra keyword arguments are forwarded to ``sqlalchemy.create_engine``.
    This allows tweaking connection pooling with options such as
    ``pool_size`` or ``max_overflow``.
    """

    db_url = db_url or os.environ.get("NEURO_DB_URL", "sqlite://")

    if "pool_size" not in engine_kwargs:
        env_pool = os.environ.get("NEURO_DB_POOL_SIZE")
        if env_pool:
            try:
                engine_kwargs["pool_size"] = int(env_pool)
            except ValueError:
                pass
    if "max_overflow" not in engine_kwargs:
        env_overflow = os.environ.get("NEURO_DB_MAX_OVERFLOW")
        if env_overflow:
            try:
                engine_kwargs["max_overflow"] = int(env_overflow)
            except ValueError:
                pass

    if db_url.startswith("sqlite"):
        engine = sa.create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=sa.pool.StaticPool,
            **engine_kwargs,
        )
        Base.metadata.create_all(engine, checkfirst=True)
    else:
        engine = sa.create_engine(db_url, **engine_kwargs)
        try:
            Base.metadata.create_all(engine, checkfirst=True)
        except Exception:
            # database might be unreachable during initialization
            pass
    return sessionmaker(bind=engine)


def run_migrations(db_url: str, revision: str = "head") -> None:
    """Apply Alembic migrations to the given database URL."""

    import importlib
    import pkg_resources
    import sys

    site_pkg = pkg_resources.get_distribution("alembic").location
    sys.path.insert(0, site_pkg)
    command = importlib.import_module("alembic.command")
    Config = importlib.import_module("alembic.config").Config
    sys.path.remove(site_pkg)

    root = Path(__file__).resolve().parents[1]
    config = Config()
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(config, revision)


def ensure_schema(db_url: str) -> None:
    """Ensure the database schema exists and is up to date."""
    try:
        run_migrations(db_url)
    except Exception:
        if db_url.startswith("sqlite"):
            engine = sa.create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=sa.pool.StaticPool,
            )
        else:
            engine = sa.create_engine(db_url)
        Base.metadata.create_all(engine, checkfirst=True)
        engine.dispose()


def log_rl_feedback(
    text: str,
    feedback: str,
    score: float = 1.0,
    *,
    processed: bool = False,
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> None:
    """Persist a feedback record for RL modules."""

    if session_factory is None:
        session_factory = create_session(db_url)

    from .sql_db import RLFeedback  # self import for type checking

    Session = session_factory
    with Session() as s:
        s.add(
            RLFeedback(
                text=text, feedback=feedback, score=score, processed=processed
            )
        )
        s.commit()


__all__ = [
    "create_session",
    "Base",
    "UserProfile",
    "UserPreference",
    "ConversationMessage",
    "ArchetypeStats",
    "MatchHistory",
    "SelfLearningEvent",
    "SelfLearningState",
    "ReactionPair",
    "PreferenceMessage",
    "EmbeddingMessage",
    "EmotionEntry",
    "RewardEntry",
    "RLFeedback",
    "log_rl_feedback",
    "run_migrations",
    "ensure_schema",
    "RLPolicyWeight",
    "ReplayExperience",
]
