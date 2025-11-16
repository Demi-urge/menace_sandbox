from __future__ import annotations

import hashlib
import os
from . import config as cfg
from typing import Any, Dict, List, Optional

try:
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover - optional dep
    MongoClient = None  # type: ignore

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional dep
    psycopg2 = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dep
    redis = None  # type: ignore

try:
    import pylibmc  # type: ignore
except Exception:  # pragma: no cover - optional dep
    pylibmc = None  # type: ignore

from .sql_db import create_session
from .vector_db import VectorDB, DatabaseVectorDB
from .embedding_memory import EmbeddingConversationMemory, DatabaseEmbeddingMemory
from .external_integrations import InfluenceGraphUpdater
from .neuro_etl import InMemoryPostgres, InMemoryMongo, InMemoryNeo4j
from .influence_graph import InfluenceGraph


class DatabaseConnectionManager:
    """Central manager for database and cache connections."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}

        # Load balancing can be toggled via config or environment variable
        if "enable_load_balancing" in config:
            self.enable_load_balancing = config["enable_load_balancing"]
        else:
            lb_env = os.getenv("NEURO_ENABLE_DB_LOAD_BALANCING")
            self.enable_load_balancing = bool(
                lb_env and lb_env.lower() in {"1", "true", "yes"}
            )

        # -- Postgres connections ------------------------------------------------
        self.pg_conns: List[Any] = []
        pool_size = None
        max_overflow = None
        if "pool_size" in config:
            pool_size = config["pool_size"]
        else:
            env_pool = os.getenv("NEURO_DB_POOL_SIZE")
            if env_pool:
                try:
                    pool_size = int(env_pool)
                except ValueError:
                    pool_size = None
        if "max_overflow" in config:
            max_overflow = config["max_overflow"]
        else:
            env_overflow = os.getenv("NEURO_DB_MAX_OVERFLOW")
            if env_overflow:
                try:
                    max_overflow = int(env_overflow)
                except ValueError:
                    max_overflow = None
        if "postgres_urls" in config:
            pg_urls = config["postgres_urls"]
        else:
            raw_pg = os.getenv("NEURO_POSTGRES_URLS")
            if raw_pg:
                pg_urls = [u.strip() for u in raw_pg.split(",") if u.strip()]
            else:
                pg_urls = ["memory"]
        for url in pg_urls:
            if psycopg2 is None or url == "memory":
                self.pg_conns.append(InMemoryPostgres())
            else:  # pragma: no cover - real database
                kwargs = {}
                if pool_size is not None:
                    kwargs["pool_size"] = pool_size
                if max_overflow is not None:
                    kwargs["max_overflow"] = max_overflow
                Session = create_session(url, **kwargs)
                self.pg_conns.append(Session())

        # -- Mongo connections ---------------------------------------------------
        self.mongo_conns: List[Any] = []
        if "mongo_urls" in config:
            mongo_urls = config["mongo_urls"]
        else:
            raw_m = os.getenv("NEURO_MONGO_URLS")
            if raw_m:
                mongo_urls = [u.strip() for u in raw_m.split(",") if u.strip()]
            else:
                mongo_urls = ["memory"]
        for url in mongo_urls:
            if MongoClient is None or url == "memory":
                self.mongo_conns.append(InMemoryMongo())
            else:  # pragma: no cover - real database
                self.mongo_conns.append(MongoClient(url))

        # -- Vector DB ---------------------------------------------------------
        vconf = config.get("vector_db", {})
        vurl = config.get("vector_db_url")
        if vurl:
            self.vector_db = DatabaseVectorDB(
                db_url=vurl,
                pinecone_index=vconf.get("index"),
                pinecone_key=vconf.get("key"),
                pinecone_env=vconf.get("env"),
                sync_interval=vconf.get("sync_interval", 5),
                max_messages=vconf.get("max_messages", 10),
                ttl_seconds=vconf.get("ttl_seconds"),
                decay_factor=vconf.get("decay_factor", 0.99),
            )
        else:
            self.vector_db = VectorDB(
                max_messages=vconf.get("max_messages", 10),
                ttl_seconds=vconf.get("ttl_seconds"),
                decay_factor=vconf.get("decay_factor", 0.99),
                pinecone_index=vconf.get("index"),
                pinecone_key=vconf.get("key"),
                pinecone_env=vconf.get("env"),
                sync_interval=vconf.get("sync_interval", 5),
            )

        # -- Embedding memory ---------------------------------------------------
        mem_url = config.get("embedding_db_url")
        if mem_url:
            self.embedding_memory = DatabaseEmbeddingMemory(db_url=mem_url)
        else:
            self.embedding_memory = EmbeddingConversationMemory()

        # -- Neo4j ---------------------------------------------------------------
        nconf = config.get("neo4j")
        uri = None
        auth = None
        if nconf:
            uri = nconf.get("uri")
            if nconf.get("auth"):
                auth = tuple(nconf.get("auth"))
        else:
            cfg_env = cfg.load_config()
            uri = cfg_env.neo4j_uri
            if cfg_env.neo4j_user and cfg_env.neo4j_pass:
                auth = (cfg_env.neo4j_user, cfg_env.neo4j_pass)

        if InfluenceGraphUpdater is not None and uri and auth:
            self.neo4j = InfluenceGraphUpdater(uri, auth)
        else:
            self.neo4j = InMemoryNeo4j(InfluenceGraph())

        # -- Redis -------------------------------------------------------------
        self.redis = None
        if redis is not None:
            cfg_env = cfg.load_config()
            url = cfg_env.redis_url or "redis://localhost:6379/0"
            try:  # pragma: no cover - requires redis server
                self.redis = redis.Redis.from_url(url)
            except Exception:  # pragma: no cover - redis failure
                self.redis = None

        # -- Memcached ---------------------------------------------------------
        self.memcached = None
        if pylibmc is not None:
            servers = cfg.load_config().memcached_servers
            try:  # pragma: no cover - requires memcached server
                self.memcached = pylibmc.Client(servers)
            except Exception:
                self.memcached = None

        # -- Preference engine -----------------------------------------------
        pref_url = config.get("preference_db_url")
        if pref_url:
            from .user_preferences import DatabasePreferenceEngine

            self.preference_engine = DatabasePreferenceEngine(db_url=pref_url)
        else:
            from .user_preferences import PreferenceEngine

            self.preference_engine = PreferenceEngine()

    # ------------------------------------------------------------------
    def _select(self, conns: List[Any], shard_key: Optional[str]) -> Any:
        if not conns:
            raise RuntimeError("no connections available")
        if self.enable_load_balancing and shard_key is not None:
            idx = int(hashlib.sha256(shard_key.encode()).hexdigest(), 16) % len(conns)
            return conns[idx]
        return conns[0]

    # ------------------------------------------------------------------
    def get_postgres(self, shard_key: Optional[str] = None) -> Any:
        """Return a connection/session to Postgres."""
        return self._select(self.pg_conns, shard_key)

    # ------------------------------------------------------------------
    def get_mongo(self, shard_key: Optional[str] = None) -> Any:
        """Return a connection to MongoDB."""
        return self._select(self.mongo_conns, shard_key)

    # ------------------------------------------------------------------
    def get_vector_db(self) -> VectorDB:
        return self.vector_db

    # ------------------------------------------------------------------
    def get_embedding_memory(self) -> EmbeddingConversationMemory:
        return self.embedding_memory

    # ------------------------------------------------------------------
    def get_neo4j(self) -> Any:
        return self.neo4j

    # ------------------------------------------------------------------
    def get_redis(self) -> Any:
        return self.redis

    # ------------------------------------------------------------------
    def get_memcached(self) -> Any:
        return self.memcached

    # ------------------------------------------------------------------
    def get_preference_engine(self):
        return self.preference_engine


__all__ = ["DatabaseConnectionManager"]
