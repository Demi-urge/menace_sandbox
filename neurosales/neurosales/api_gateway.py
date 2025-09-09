from __future__ import annotations

from typing import Dict, List, Optional
import os
import logging
import json
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Response, Depends, Request
import time
from pydantic import BaseModel
from .metrics import metrics
from .security import get_api_key, RateLimiter

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dep
    redis = None  # type: ignore

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
from .sql_db import create_session, run_migrations

from .rl_training import schedule_periodic_training, schedule_feedback_export
from .db_backup import schedule_database_backup
from .user_preferences import PreferenceProfile
from .orchestrator import SandboxOrchestrator
from context_builder_util import create_context_builder

load_dotenv()

logger = logging.getLogger(__name__)


class ArchetypeCache:
    def __init__(self):
        if redis is not None:
            url = os.getenv("NEURO_REDIS_URL", "redis://localhost:6379/0")
            try:
                self.client = redis.Redis.from_url(url)
            except Exception:  # pragma: no cover - redis not running
                logger.exception("Redis connection failed")
                self.client = None
        else:
            self.client = None
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        if self.client is not None:
            try:
                val = self.client.get(key)
                return val.decode() if val else None
            except Exception:  # pragma: no cover - redis failure
                logger.exception("Redis get failed")
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int = 3600) -> None:
        if self.client is not None:
            try:
                self.client.set(key, value, ex=ex)
                return
            except Exception:  # pragma: no cover - redis failure
                logger.exception("Redis set failed")
        self._store[key] = value


class ChatRequest(BaseModel):
    user_id: str
    line: str


class ChatResponse(BaseModel):
    reply: str
    confidence: float


class MemoryQuery(BaseModel):
    user_id: str
    query: str
    top_k: int = 3


class HarvestRequest(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None
    selector: str = "article"


class UserUpdate(BaseModel):
    profile: Optional[PreferenceProfile] = None
    elo: Optional[float] = None
    triggers: Optional[List[str]] = None


def create_app(session_factory=None, memory_session_factory=None) -> FastAPI:
    """Create FastAPI app with metrics endpoint."""
    logging.basicConfig(level=logging.INFO)

    if not os.getenv("NEURO_API_KEY"):
        logger.error("NEURO_API_KEY environment variable not set")
        raise RuntimeError("NEURO_API_KEY must be configured")

    session_factory = session_factory or create_session()

    # define local schema for gateway specific tables
    Base = declarative_base()

    class UserModel(Base):
        __tablename__ = "users"
        id = sa.Column(sa.String, primary_key=True)
        profile = sa.Column(sa.JSON)
        elo = sa.Column(sa.Float, default=1000.0)
        triggers = sa.Column(sa.JSON, default=list)

    engine = session_factory.kw.get("bind")
    if engine is not None:
        try:
            Base.metadata.create_all(engine, checkfirst=True)
        except Exception:
            pass

    dataset_path = "rl_feedback_dataset.json"
    env_int = os.getenv("NEURO_AUTO_TRAIN_INTERVAL")
    interval = 3600
    if env_int is not None:
        try:
            interval = int(env_int)
        except ValueError:
            pass

    schedule_feedback_export(
        interval=interval,
        dataset_path=dataset_path,
        session_factory=session_factory,
    )

    schedule_periodic_training(
        interval=interval,
        dataset_path=dataset_path,
        session_factory=session_factory,
    )

    backup_env = os.getenv("NEURO_BACKUP_INTERVAL")
    backup_interval = 0
    if backup_env is not None:
        try:
            backup_interval = int(backup_env)
        except ValueError:
            pass
    if backup_interval > 0:
        schedule_database_backup(
            interval=backup_interval,
            backup_path=os.getenv("NEURO_BACKUP_PATH", "backup.json"),
            db_url=os.getenv("NEURO_DB_URL"),
        )

    orch_session_factory = memory_session_factory or session_factory
    orchestrator = SandboxOrchestrator(
        context_builder=create_context_builder(),
        persistent=memory_session_factory is not None,
        session_factory=orch_session_factory,
    )
    cache = ArchetypeCache()
    rate_limiter = RateLimiter()

    app = FastAPI()

    @app.on_event("startup")
    async def apply_migrations() -> None:
        db_url = os.getenv("NEURO_DB_URL")
        if not db_url:
            return
        try:
            run_migrations(db_url)
            logger.info("Database migrations applied")
        except Exception:
            logger.exception("Database migration failed")

    @app.middleware("http")
    async def log_and_metrics(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        log = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "path": request.url.path,
            "status": response.status_code,
            "duration": round(duration, 3),
        }
        logger.info(json.dumps(log))
        metrics.record_request(request.url.path, response.status_code, duration)
        return response

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        req: ChatRequest,
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        reply, conf = orchestrator.handle_chat(req.user_id, req.line)
        if orchestrator.preferences.get_profile(req.user_id).archetype:
            cache.set(req.user_id, orchestrator.preferences.get_profile(req.user_id).archetype)
        else:
            cached = cache.get(req.user_id)
            if cached:
                orchestrator.preferences.get_profile(req.user_id).archetype = cached
        return ChatResponse(reply=reply, confidence=conf)

    @app.get("/user/{user_id}")
    def get_user(
        user_id: str,
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        Session = session_factory
        with Session() as s:
            user = s.get(UserModel, user_id)
            if not user:
                raise HTTPException(status_code=404, detail="user not found")
            return {
                "id": user.id,
                "profile": user.profile,
                "elo": user.elo,
                "triggers": user.triggers,
            }

    @app.put("/user/{user_id}")
    def put_user(
        user_id: str,
        update: UserUpdate,
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        Session = session_factory
        with Session() as s:
            user = s.get(UserModel, user_id)
            if not user:
                user = UserModel(id=user_id, profile={}, elo=1000.0, triggers=[])
                s.add(user)
            if update.profile is not None:
                user.profile = update.profile.__dict__  # type: ignore
            if update.elo is not None:
                user.elo = update.elo
            if update.triggers is not None:
                user.triggers = update.triggers
            s.commit()
            return {"status": "ok"}

    @app.post("/memory/search")
    def memory_search(
        query: MemoryQuery,
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        mem = orchestrator.memories.get(query.user_id) or orchestrator._get_memory(query.user_id)
        matches = mem.most_similar(query.query, top_k=query.top_k)
        return {"matches": [m.content for m in matches]}

    @app.get("/metrics")
    def metrics_endpoint(
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        return Response(metrics.exposition(), media_type="text/plain")

    @app.post("/harvest")
    def harvest(
        req: HarvestRequest,
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        data = orchestrator.harvest_content(
            req.url,
            username=req.username,
            password=req.password,
            selector=req.selector,
        )
        return {"content": data}

    @app.get("/orchestrator/users")
    def list_users(
        api_key: str = Depends(get_api_key),
        _: None = Depends(rate_limiter),
    ):
        return {"users": list(orchestrator.memories.keys())}

    return app


__all__ = ["create_app"]
