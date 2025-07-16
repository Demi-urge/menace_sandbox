import menace.anticaptcha_stub as ac


def test_stub_returns_result():
    client = ac.AntiCaptchaClient(api_key="dummy")
    res = client.solve("image.png")
    assert isinstance(res, ac.SolveResult)
    assert res.text is None
    assert res.used_remote
    assert res.error
