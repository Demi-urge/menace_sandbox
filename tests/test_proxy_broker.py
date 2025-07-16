import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import asyncio
import menace.proxy_broker as pb


async def test_session_pinning():
    pool = pb.ProxyPool([
        {"ip": "1.1.1.1", "port": 80, "city": "A"},
        {"ip": "2.2.2.2", "port": 81, "city": "B"},
    ])
    broker = pb.ProxyBroker([pool])
    p1 = await broker.acquire(session="s1")
    p2 = await broker.acquire(session="s1")
    assert p1 is p2
    await broker.release(p1, session="s1", latency=0.5)


async def test_geo_selection():
    pool = pb.ProxyPool([
        {"ip": "4.4.4.4", "port": 83, "city": "Y"}
    ])
    broker = pb.ProxyBroker([pool], selector=pb.GeoSelector(db_path=""))
    proxy = await broker.acquire(city="Y")
    assert proxy and proxy.city == "Y"


async def test_health_scoring():
    info = pb.ProxyInfo(ip="5.5.5.5", port=84)
    scorer = pb.HealthScorer(alpha=0.5)
    scorer.update(info, latency=1.0)
    first = info.score
    scorer.update(info, latency=2.0, failed=True)
    assert info.score > first


def test_run():
    asyncio.run(test_session_pinning())
    asyncio.run(test_geo_selection())
    asyncio.run(test_health_scoring())
