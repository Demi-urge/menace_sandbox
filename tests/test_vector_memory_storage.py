import menace.memory_bot as mb
from pathlib import Path


def test_vector_add_and_query(tmp_path: Path):
    store = mb.VectorMemoryStorage(tmp_path / 'mem.json.gz', embedder=None)
    bot = mb.MemoryBot(store)
    bot.log('u1', 'hello world')
    bot.log('u2', 'another message')
    res_text = bot.search('hello')
    assert res_text[0].user == 'u1'
    res_vec = store.query_vector('hello')
    assert res_vec
