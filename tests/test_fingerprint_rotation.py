import sys
import types
import pytest

pytest.importorskip("selenium")

class DummyDriver:
    def __init__(self, *, options=None, use_subprocess=False, **_):
        self.options = options
        self.cdp = []
    def implicitly_wait(self, _):
        pass
    def execute_cdp_cmd(self, name, args):
        self.cdp.append((name, args))

def test_start_driver_randomises(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyautogui', types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        'menace.clipped.proxy_manager',
        types.SimpleNamespace(
            get_available_proxy=lambda: None,
            release_proxy=lambda proxy: None,
        ),
    )
    import menace.clipped.poster_base as pb
    monkeypatch.setattr('undetected_chromedriver.Chrome', DummyDriver)

    called = {}
    def fake_stealth(driver, **kwargs):
        called['args'] = kwargs
    monkeypatch.setattr(pb, 'stealth', fake_stealth)

    bot = pb.PosterBot('schedule.json')
    bot.schedule = []
    bot.accounts = {}
    bot.start_driver()
    args = bot.driver.options.arguments

    assert any(a.startswith('--user-agent=') for a in args)
    assert any(a.startswith('--window-size=') for a in args)
    assert any(cmd[0] == 'Page.addScriptToEvaluateOnNewDocument' for cmd in bot.driver.cdp)
    assert 'args' in called
    assert 'languages' in called['args']
    assert 'platform' in called['args']
