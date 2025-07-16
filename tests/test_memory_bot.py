import menace.memory_bot as mb


def test_log_and_search(tmp_path):
    storage = mb.MemoryStorage(tmp_path / 'mem.json.gz')
    bot = mb.MemoryBot(storage)
    bot.log('user1', 'hello world')
    bot.log('user2', 'another message')
    results = bot.search('hello')
    assert len(results) == 1
    assert results[0].user == 'user1'


def test_cache(tmp_path):
    storage = mb.MemoryStorage(tmp_path / 'mem.json.gz')
    bot = mb.MemoryBot(storage)
    bot.log('u', 'hi there')
    first = bot.search('hi')
    assert bot.search('hi') is first
