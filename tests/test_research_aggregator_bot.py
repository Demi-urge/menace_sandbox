import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import time  # noqa: E402
import menace.research_aggregator_bot as rab  # noqa: E402
import menace.chatgpt_research_bot as crb  # noqa: E402
import menace.chatgpt_idea_bot as cib  # noqa: E402
import menace.chatgpt_enhancement_bot as ceb  # noqa: E402
import menace.chatgpt_prediction_bot as cpb  # noqa: E402
import sqlite3  # noqa: E402
import types  # noqa: E402


def _builder():
    return types.SimpleNamespace(
        refresh_db_weights=lambda *a, **k: None,
        build=lambda *a, **k: "",
    )


def test_memory_decay(monkeypatch):
    mem = rab.ResearchMemory()
    old = rab.ResearchItem(topic="t", content="c", timestamp=0)
    mem.add(old)
    mem.add(rab.ResearchItem(topic="t", content="d", timestamp=time.time()))
    monkeypatch.setattr(time, "time", lambda: 1000)
    mem.decay()
    assert all(it.content != "c" for it in mem.short)


def test_process_uses_memory(monkeypatch):
    mem = rab.ResearchMemory()
    item = rab.ResearchItem(topic="Topic", content="cached", timestamp=time.time())
    mem.add(item)
    bot = rab.ResearchAggregatorBot(["Topic"], memory=mem, context_builder=_builder())
    called = False

    def fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not call")

    monkeypatch.setattr(bot, "_gather_online", fail)
    bot.process("Topic", energy=1)
    assert not called


def test_refine_removes_duplicates():
    items = [
        rab.ResearchItem(topic="a", content="one", timestamp=0),
        rab.ResearchItem(topic="a", content="one", timestamp=1),
        rab.ResearchItem(topic="a", content="two", timestamp=2),
    ]
    refined = rab.ResearchAggregatorBot._refine(items)
    texts = [i.content for i in refined]
    assert texts == ["one", "two"]


def test_interactive_loop(monkeypatch, tmp_path):
    calls = []

    text_bot = rab.TextResearchBot()

    def fake_process(urls, files, ratio=0.2):
        calls.append(urls[0])
        return [rab.text_research_bot.TextSource(content=f"data {urls[0]}")]

    monkeypatch.setattr(text_bot, "process", fake_process)
    info_db = rab.InfoDB(tmp_path / "info.db")
    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    bot = rab.ResearchAggregatorBot(
        ["A", "B"],
        text_bot=text_bot,
        info_db=info_db,
        enhancements_db=enh_db,
        context_builder=_builder(),
    )
    result = bot.process("A", energy=3)
    topics = {r.topic for r in result}
    assert {"A", "B"} <= topics
    assert calls == ["A", "B"]


def test_chatgpt_integration(monkeypatch):
    builder = _builder()
    client = cib.ChatGPTClient("key", context_builder=builder)

    def fake_ask(messages):
        return {"choices": [{"message": {"content": "Some info"}}]}

    monkeypatch.setattr(client, "ask", fake_ask)
    chat_bot = crb.ChatGPTResearchBot(builder, client)

    text_bot = rab.TextResearchBot()
    monkeypatch.setattr(
        text_bot,
        "process",
        lambda urls, files, ratio=0.2: [rab.text_research_bot.TextSource(content="detail")],
    )

    collected = []
    bot = rab.ResearchAggregatorBot(
        ["Topic"], text_bot=text_bot, chatgpt_bot=chat_bot, context_builder=builder
    )
    bot.receive_chatgpt = lambda conv, summary: collected.append(summary)
    chat_bot.send_callback = bot.receive_chatgpt
    bot.process("Topic", energy=3)
    assert collected and collected[0]


def test_enhancement_prediction(monkeypatch, tmp_path):
    enh_bot = ceb.ChatGPTEnhancementBot(
        None,
        db=ceb.EnhancementDB(tmp_path / "e.db"),
        context_builder=_builder(),
    )
    monkeypatch.setattr(
        enh_bot,
        "propose",
        lambda instruction, num_ideas=1, context="": [
            ceb.Enhancement(idea="imp", rationale="fast")
        ],
    )
    pred_bot = cpb.ChatGPTPredictionBot.__new__(cpb.ChatGPTPredictionBot)
    pred_bot.evaluate_enhancement = lambda i, r: cpb.EnhancementEvaluation(
        description="d", reason="r", value=0.5
    )
    bot = rab.ResearchAggregatorBot(
        ["Topic"],
        enhancement_bot=enh_bot,
        prediction_bot=pred_bot,
        memory=rab.ResearchMemory(),
        context_builder=_builder(),
    )
    bot._maybe_enhance("Topic", "reason")
    item = bot.memory.medium[-1]
    assert item.summary == "d"
    assert item.notes == "r"
    assert item.quality == 0.5


def test_enhancement_links(monkeypatch, tmp_path):
    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    enh_bot = ceb.ChatGPTEnhancementBot(None, db=enh_db, context_builder=_builder())
    monkeypatch.setattr(
        enh_bot,
        "propose",
        lambda instruction, num_ideas=1, context="": [
            ceb.Enhancement(idea="imp", rationale="fast")
        ],
    )
    info_db = rab.InfoDB(tmp_path / "info.db")
    info_db.set_current_model(1)
    bot = rab.ResearchAggregatorBot(
        ["Topic"],
        enhancement_bot=enh_bot,
        info_db=info_db,
        enhancements_db=enh_db,
        context_builder=_builder(),
    )
    bot._maybe_enhance("Topic", "reason")
    with sqlite3.connect(enh_db.path) as conn:  # noqa: SQL001
        row = conn.execute("SELECT id, triggered_by FROM enhancements").fetchone()
    assert row is not None and row[1] == "ResearchAggregatorBot"
    eid = row[0]
    assert enh_db.models_for(eid) == [1]


def test_info_db_persistence(monkeypatch, tmp_path):
    text_bot = rab.TextResearchBot()
    monkeypatch.setattr(
        text_bot,
        "process",
        lambda urls, files, ratio=0.2: [
            rab.text_research_bot.TextSource(content="info", url="http://src")
        ],
    )
    info_db = rab.InfoDB(tmp_path / "info.db")
    bot = rab.ResearchAggregatorBot(
        ["Topic"], text_bot=text_bot, info_db=info_db, context_builder=_builder()
    )
    bot.process("Topic", energy=3)
    items = info_db.search("Topic")
    assert any(i.source_url == "http://src" for i in items)
    assert items[0].summary


def test_infodb_summary_and_depth(tmp_path):
    db = rab.InfoDB(tmp_path / "info.db")
    text = "One. Two. Three. Four."
    it = rab.ResearchItem(topic="T", content=text, timestamp=time.time())
    db.add(it)
    stored = db.search("T")[0]
    assert stored.data_depth > 0
    assert stored.summary


def test_enhancement_generates_followup_research(monkeypatch, tmp_path):
    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    enh_bot = ceb.ChatGPTEnhancementBot(None, db=enh_db, context_builder=_builder())
    monkeypatch.setattr(
        enh_bot,
        "propose",
        lambda instruction, num_ideas=1, context="": [
            ceb.Enhancement(idea="imp", rationale="fast")
        ],
    )

    text_bot = rab.TextResearchBot()
    monkeypatch.setattr(
        text_bot,
        "process",
        lambda urls, files, ratio=0.2: [
            rab.text_research_bot.TextSource(content="detail", url="u")
        ],
    )

    class V:
        def process(self, q, ratio=0.2):
            return [type("VI", (), {"summary": "vid", "url": "v"})()]

    class C:
        def process(self, inst, depth=1, ratio=0.2):
            return rab.chatgpt_research_bot.ResearchResult([], "chat")

    info_db = rab.InfoDB(tmp_path / "info.db")
    bot = rab.ResearchAggregatorBot(
        ["Topic"],
        enhancement_bot=enh_bot,
        text_bot=text_bot,
        video_bot=V(),
        chatgpt_bot=C(),
        info_db=info_db,
        enhancements_db=enh_db,
        context_builder=_builder(),
    )
    bot._maybe_enhance("Topic", "reason")
    items = info_db.search("imp")
    assert any(i.content == "detail" for i in items)
    with sqlite3.connect(info_db.path) as conn:  # noqa: SQL001
        count = conn.execute("SELECT COUNT(*) FROM info_enhancements").fetchone()[0]
    assert count > 1


def test_workflow_reusable_tagging(tmp_path):
    db = rab.InfoDB(tmp_path / "info.db")
    full = rab.ResearchItem(
        topic="Topic",
        content="x" * 600,
        timestamp=time.time(),
        title="Topic",
        tags=["workflow", "topic", "extra"],
        category="workflow",
        data_depth=1.0,
    )
    db.add(full)
    partial = rab.ResearchItem(
        topic="Topic",
        content="short",
        timestamp=time.time(),
        title="Topic",
        tags=["workflow", "topic"],
        category="workflow",
        data_depth=0.2,
    )
    db.add(partial)

    bot = rab.ResearchAggregatorBot(["Topic", "extra"], info_db=db, context_builder=_builder())
    items = bot._collect_topic("Topic", energy=1)

    tag_map = {i.content: i.tags for i in items}
    assert "reusable" in tag_map[full.content]
    assert "partial_reusable" not in tag_map[full.content]
    assert "partial_reusable" in tag_map[partial.content]


def test_delegate_sub_bots_respects_missing(monkeypatch):
    class T:
        def __init__(self):
            self.calls = []

        def process(self, urls, files, ratio=0.2):
            self.calls.append(urls[0])
            return [rab.text_research_bot.TextSource(content="text")]

    class V:
        def __init__(self):
            self.calls = []

        def process(self, q, ratio=0.2):
            self.calls.append(q)
            return [type("VI", (), {"summary": "vid", "url": "v"})()]

    class C:
        def __init__(self):
            self.calls = []

        def process(self, inst, depth=1, ratio=0.2):
            self.calls.append(inst)
            return rab.chatgpt_research_bot.ResearchResult([], "chat")

    text_bot = T()
    video_bot = V()
    chat_bot = C()

    mem = rab.ResearchMemory()
    mem.add(rab.ResearchItem(topic="T", content="x", timestamp=time.time(), type_="text"))

    bot = rab.ResearchAggregatorBot(
        ["T"],
        memory=mem,
        text_bot=text_bot,
        video_bot=video_bot,
        chatgpt_bot=chat_bot,
        context_builder=_builder(),
    )
    bot.process("T", energy=3)

    assert not text_bot.calls
    assert video_bot.calls and chat_bot.calls


def test_cache_ttl_refresh(monkeypatch):
    current = [0]
    monkeypatch.setattr(time, "time", lambda: current[0])

    bot = rab.ResearchAggregatorBot(["T"], cache_ttl=10, context_builder=_builder())
    calls = {"n": 0}

    def gather(topic, energy):
        calls["n"] += 1
        return [rab.ResearchItem(topic=topic, content="d", timestamp=time.time())]

    monkeypatch.setattr(bot, "_collect_topic", gather)

    bot.process("T")
    assert calls["n"] == 1

    current[0] = 5
    bot.process("T")
    assert calls["n"] == 1

    current[0] = 15
    bot.process("T")
    assert calls["n"] == 2
