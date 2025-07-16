import json
import importlib
import menace.proxy_manager as pm


def test_get_proxy_from_env(monkeypatch):
    monkeypatch.setenv("PROXIES", "1.1.1.1:80,2.2.2.2:81")
    proxy = pm.get_proxy()
    assert proxy in {"1.1.1.1:80", "2.2.2.2:81"}


def test_get_proxy_from_file(tmp_path, monkeypatch):
    proxies_file = tmp_path / "proxies.txt"
    proxies_file.write_text("3.3.3.3:82\n4.4.4.4:83")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROXIES", raising=False)
    proxy = pm.get_proxy()
    assert proxy in {"3.3.3.3:82", "4.4.4.4:83", None}


def test_proxy_cli(tmp_path, capsys):
    data = [
        {"ip": "5.5.5.5", "port": 84, "status": "active"},
        {"ip": "6.6.6.6", "port": 85, "status": "inactive"},
    ]
    path = tmp_path / "proxies.json"
    path.write_text(json.dumps(data))
    pm_cli = importlib.import_module("menace.clipped.proxy_manager")
    pm_cli.cli(["--file", str(path)])
    out = capsys.readouterr().out.strip()
    assert out == "5.5.5.5:84"
