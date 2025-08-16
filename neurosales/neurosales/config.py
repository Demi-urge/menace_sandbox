import os
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for external service integrations."""

    openai_key: Optional[str]
    pinecone_index: Optional[str]
    pinecone_key: Optional[str]
    pinecone_env: Optional[str]
    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_pass: Optional[str]
    redis_url: Optional[str]
    memcached_servers: List[str]
    proxy_list: List[str]


def load_config() -> ServiceConfig:
    """Load configuration from environment variables."""
    memcached = os.getenv("NEURO_MEMCACHED_SERVERS", "127.0.0.1").split(",")
    proxy_raw = os.getenv("NEURO_PROXY_LIST", "")
    proxies = [p.strip() for p in proxy_raw.split(',') if p.strip()]
    return ServiceConfig(
        openai_key=os.getenv("NEURO_OPENAI_KEY"),
        pinecone_index=os.getenv("NEURO_PINECONE_INDEX"),
        pinecone_key=os.getenv("NEURO_PINECONE_KEY"),
        pinecone_env=os.getenv("NEURO_PINECONE_ENV"),
        neo4j_uri=os.getenv("NEURO_NEO4J_URI"),
        neo4j_user=os.getenv("NEURO_NEO4J_USER"),
        neo4j_pass=os.getenv("NEURO_NEO4J_PASS"),
        redis_url=os.getenv("NEURO_REDIS_URL"),
        memcached_servers=memcached,
        proxy_list=proxies,
    )


def is_openai_enabled(cfg: Optional[ServiceConfig] = None) -> bool:
    cfg = cfg or load_config()
    if not cfg.openai_key:
        logger.warning("OpenAI disabled: NEURO_OPENAI_KEY missing")
        return False
    return True


def is_pinecone_enabled(cfg: Optional[ServiceConfig] = None) -> bool:
    cfg = cfg or load_config()
    if not (cfg.pinecone_index and cfg.pinecone_key and cfg.pinecone_env):
        logger.warning("Pinecone disabled: NEURO_PINECONE_* missing")
        return False
    return True


def is_neo4j_enabled(cfg: Optional[ServiceConfig] = None) -> bool:
    cfg = cfg or load_config()
    if not (cfg.neo4j_uri and cfg.neo4j_user and cfg.neo4j_pass):
        logger.warning("Neo4j disabled: NEURO_NEO4J_* missing")
        return False
    return True


def is_redis_enabled(cfg: Optional[ServiceConfig] = None) -> bool:
    cfg = cfg or load_config()
    return cfg.redis_url is not None


def is_memcached_enabled(cfg: Optional[ServiceConfig] = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.memcached_servers)


def get_proxy_list(cfg: Optional[ServiceConfig] = None) -> List[str]:
    cfg = cfg or load_config()
    return cfg.proxy_list

__all__ = [
    "ServiceConfig",
    "load_config",
    "is_openai_enabled",
    "is_pinecone_enabled",
    "is_neo4j_enabled",
    "is_redis_enabled",
    "is_memcached_enabled",
    "get_proxy_list",
]
