import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.sentiment_bot as sb


def test_classification():
    pos = sb.classify_sentiment("I love this great product")
    neg = sb.classify_sentiment("I hate this awful bug")
    assert pos > neg
    assert sb.label_from_score(pos) == "positive"
    assert sb.label_from_score(neg) == "negative"


def test_db_roundtrip(tmp_path):
    db = sb.SentimentDB(tmp_path / "s.db")
    item = sb.FeedbackItem(text="Good", product="menace", source="x", sentiment=0.5, label="positive")
    db.add(item)
    fetched = db.fetch("menace")
    assert fetched and fetched[0].text == "Good"


def test_shift_detection(tmp_path):
    db = sb.SentimentDB(tmp_path / "s.db")
    bot = sb.SentimentBot(db)
    for _ in range(3):
        db.add(sb.FeedbackItem(text="Good", product="menace", source="x", sentiment=0.9, label="positive"))
    for _ in range(3):
        db.add(sb.FeedbackItem(text="Bad", product="menace", source="x", sentiment=-0.9, label="negative"))
    assert bot.detect_shift("menace", threshold=0.5, window=3)


class DummyPred:
    def predict(self, _vec):
        return 0.9


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["p"]


class DummyStrategy:
    def __init__(self):
        self.items = []

    def receive_sentiment(self, item):
        self.items.append(item)


def test_prediction_and_strategy_integration(tmp_path):
    manager = StubManager(DummyPred())
    strategy = DummyStrategy()
    bot = sb.SentimentBot(
        db=sb.SentimentDB(tmp_path / "s.db"),
        prediction_manager=manager,
        strategy_bot=strategy,
    )
    posts = [sb.FeedbackItem(text="Great", product="m", source="x")]
    bot.fetch_posts = lambda urls: posts
    analysed = bot.process(["u"])
    assert analysed[0].predicted > analysed[0].sentiment
    assert analysed[0].profitability >= 0
    assert strategy.items and strategy.items[0].text == "Great"

