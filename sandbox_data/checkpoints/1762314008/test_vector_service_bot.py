from vector_service import SharedVectorService
from bot_vectorizer import BotVectorizer


def test_vectorise_bot_embedding_length():
    svc = SharedVectorService()
    bot = {
        "type": "assistant",
        "status": "active",
        "tasks": [],
        "estimated_profit": 0,
    }
    vec = svc.vectorise("bot", bot)
    assert len(vec) == BotVectorizer().dim
