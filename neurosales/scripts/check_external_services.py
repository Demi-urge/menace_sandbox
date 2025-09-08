import os
import sys
from dotenv import load_dotenv
from neurosales import config
from billing.openai_wrapper import chat_completion_create
from context_builder_util import create_context_builder

load_dotenv()


def check_openai(cfg: config.ServiceConfig) -> bool:
    if not config.is_openai_enabled(cfg):
        print("OpenAI disabled")
        return True
    try:
        builder = create_context_builder()
    except Exception as e:
        print(f"ContextBuilder initialization error: {e}")
        return False
    try:
        os.environ.setdefault("OPENAI_API_KEY", cfg.openai_key)
        chat_completion_create(
            [{"role": "user", "content": "ping"}],
            model="gpt-3.5-turbo",
            context_builder=builder,
        )
        print("OpenAI reachable")
        return True
    except Exception as e:
        print(f"OpenAI error: {e}")
        return False


def check_pinecone(cfg: config.ServiceConfig) -> bool:
    if not config.is_pinecone_enabled(cfg):
        print("Pinecone disabled")
        return True
    try:
        import pinecone  # type: ignore
    except Exception as e:
        print(f"Pinecone library missing: {e}")
        return False
    try:
        pinecone.init(api_key=cfg.pinecone_key, environment=cfg.pinecone_env)
        index = pinecone.Index(cfg.pinecone_index)
        index.describe_index_stats()
        print(f"Pinecone index '{cfg.pinecone_index}' reachable")
        return True
    except Exception as e:
        print(f"Pinecone error: {e}")
        return False


def check_neo4j(cfg: config.ServiceConfig) -> bool:
    if not config.is_neo4j_enabled(cfg):
        print("Neo4j disabled")
        return True
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as e:
        print(f"Neo4j library missing: {e}")
        return False
    try:
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_pass))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("Neo4j reachable")
        return True
    except Exception as e:
        print(f"Neo4j error: {e}")
        return False


def check_redis(cfg: config.ServiceConfig) -> bool:
    if not config.is_redis_enabled(cfg):
        print("Redis disabled")
        return True
    try:
        import redis  # type: ignore
    except Exception as e:
        print(f"Redis library missing: {e}")
        return False
    try:
        r = redis.Redis.from_url(cfg.redis_url)
        r.ping()
        print("Redis reachable")
        return True
    except Exception as e:
        print(f"Redis error: {e}")
        return False


def check_memcached(cfg: config.ServiceConfig) -> bool:
    if not config.is_memcached_enabled(cfg):
        print("Memcached disabled")
        return True
    try:
        import pylibmc  # type: ignore
    except Exception as e:
        print(f"Memcached library missing: {e}")
        return False
    try:
        client = pylibmc.Client(cfg.memcached_servers)
        client.set("_ping", "1")
        client.get("_ping")
        print("Memcached reachable")
        return True
    except Exception as e:
        print(f"Memcached error: {e}")
        return False


def main() -> None:
    cfg = config.load_config()
    checks = [
        check_openai(cfg),
        check_pinecone(cfg),
        check_neo4j(cfg),
        check_redis(cfg),
        check_memcached(cfg),
    ]
    if all(checks):
        print("All external service checks passed")
        sys.exit(0)
    print("Some services failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
