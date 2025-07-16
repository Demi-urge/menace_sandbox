import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import asyncio
import menace.captcha_pipeline as cp
import menace.captcha_system as cs

class DummyPage:
    def __init__(self, html: str):
        self._html = html
        self.context = self
    async def content(self):
        return self._html
    async def screenshot(self):
        return b''
    async def close(self):
        pass
    # tracing methods for ReplayEngine compatibility
    class tracing:
        @staticmethod
        async def start(*a, **k):
            pass
        @staticmethod
        async def stop(*a, **k):
            pass
    async def route(self, *a, **k):
        pass
    async def add_init_script(self, *a, **k):
        pass
    async def new_page(self):
        return self


def test_pipeline_waits_and_resumes():
    mgr = cs.CaptchaManager(bucket='dummy', redis_url='redis://localhost:1/0')
    mgr.s3 = None
    pipe = cp.CaptchaPipeline(manager=mgr)
    page = DummyPage('<div>captcha</div>')

    async def runner():
        t = asyncio.create_task(pipe.run(page, 'job', timeout=1))
        await asyncio.sleep(0.1)
        mgr.mark_resolved('job', 'tok')
        await t
    asyncio.run(runner())
    assert mgr.local_state['job']['state'] == 'SOLVED'


def test_pipeline_autosolves(monkeypatch):
    monkeypatch.setenv('ANTICAPTCHA_API_KEY', 'dummy')
    monkeypatch.setattr(cs.AntiCaptchaClient, '_remote_solve', lambda self, p: ('tok', None))
    mgr = cs.CaptchaManager(bucket='dummy', redis_url='redis://localhost:1/0')
    mgr.s3 = None
    pipe = cp.CaptchaPipeline(manager=mgr)
    page = DummyPage('<div>captcha</div>')

    async def runner():
        await pipe.run(page, 'job2', timeout=1)
    asyncio.run(runner())
    assert mgr.local_state['job2']['state'] == 'SOLVED'
    assert mgr.local_state['job2']['token'] == 'tok'
