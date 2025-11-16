import asyncio
import menace.captcha_system as cs


def test_detector_simple():
    det = cs.CaptchaDetector()
    assert det.detect('<div>captcha</div>')
    assert not det.detect('<p>hello</p>')


class DummyPage:
    async def screenshot(self):
        return b''

    async def content(self):
        return '<html></html>'


def test_manager_local(monkeypatch):
    monkeypatch.setattr(cs, 'boto3', None)
    mgr = cs.CaptchaManager(bucket='dummy', redis_url='redis://localhost:1/0')
    page = DummyPage()
    q = mgr.subscribe()
    asyncio.run(mgr.snapshot_and_pause(page, 'job'))
    evt = asyncio.run(q.get())
    assert evt['type'] == 'blocked'
    assert mgr.local_state['job']['state'] == 'BLOCKED'
    mgr.mark_resolved('job', 't')
    evt2 = asyncio.run(q.get())
    assert evt2['type'] == 'solved'
    assert mgr.local_state['job']['state'] == 'SOLVED'
    token = asyncio.run(mgr.wait_for_solution('job', poll_interval=0))
    assert token == 't'
