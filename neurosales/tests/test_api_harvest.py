import os
import sys
import importlib.util
import types
import logging
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamic_path_router import resolve_path

spec = importlib.util.spec_from_file_location(
    "neurosales.api_harvest",
    str(resolve_path("neurosales/api_harvest.py")),
)
api_harvest = importlib.util.module_from_spec(spec)
sys.modules.setdefault("neurosales", types.ModuleType("neurosales"))
sys.modules["neurosales.api_harvest"] = api_harvest
spec.loader.exec_module(api_harvest)
sys.modules["neurosales"].api_harvest = api_harvest
APIScraper = api_harvest.APIScraper


def test_fetch_pubmed_xml():
    scraper = APIScraper()
    resp = MagicMock()
    resp.text = "<xml></xml>"
    with patch("neurosales.api_harvest.requests.Session.get", return_value=resp) as p:
        xml = scraper.fetch_pubmed_xml(["1", "2"])
    assert "xml" in xml
    p.assert_called_once()
    assert p.call_args[1]["params"]["id"] == "1,2"


def test_fetch_citation_context():
    scraper = APIScraper()
    resp = MagicMock()
    resp.json.return_value = {"data": {"paper": {"title": "x"}}}
    with patch("neurosales.api_harvest.requests.Session.post", return_value=resp) as p:
        data = scraper.fetch_citation_context("abc")
    assert data["data"]["paper"]["title"] == "x"
    p.assert_called_once()


def test_fetch_crossref_metadata():
    scraper = APIScraper()
    resp = MagicMock()
    resp.json.return_value = {"message": {"title": "y"}}
    with patch("neurosales.api_harvest.requests.Session.get", return_value=resp) as p:
        meta = scraper.fetch_crossref_metadata("10.1/abc")
    assert meta["message"]["title"] == "y"
    p.assert_called_once()


def test_download_kaggle_dataset_success():
    scraper = APIScraper()
    with patch("neurosales.api_harvest.subprocess.check_call") as p:
        assert scraper.download_kaggle_dataset("me/ds", "/tmp")
        p.assert_called_once()


def test_download_kaggle_dataset_failure_logs(caplog):
    scraper = APIScraper()
    with patch(
        "neurosales.api_harvest.subprocess.check_call",
        side_effect=RuntimeError("boom"),
    ) as p, caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            scraper.download_kaggle_dataset("me/ds", "/tmp")
    assert p.called
    assert any("Failed to download Kaggle dataset" in r.getMessage() for r in caplog.records)


def test_fetch_fmri_atlas_json_with_proxy():
    resp = MagicMock()
    resp.json.return_value = {"atlas": 1}
    with patch("neurosales.api_harvest.requests.Session.get", return_value=resp) as p:
        scraper = APIScraper(proxies=["http://proxy"])
        data = scraper.fetch_fmri_atlas_json("http://ex")
    assert data["atlas"] == 1
    proxies = p.call_args[1]["proxies"]
    assert proxies["http"].startswith("http://")


def test_env_proxy_list_parsed(monkeypatch):
    monkeypatch.setenv("NEURO_PROXY_LIST", "http://a,http://b")
    scraper = APIScraper()
    assert set(scraper.proxies) == {"http://a", "http://b"}
