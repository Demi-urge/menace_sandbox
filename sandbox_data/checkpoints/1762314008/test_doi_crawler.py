import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pytest
from neurosales.doi_crawler import DOICrawler
from unittest.mock import patch, MagicMock

PAGE1 = """
<html>
<body>
<div>10.1234/abc1</div>
<a href="page2">Next</a>
</body>
</html>
"""

PAGE2 = """
<html>
<body>
<div>10.1234/abc2</div>
<a href="page3">next</a>
</body>
</html>
"""

PAGE3 = """
<html><body>10.1234/abc2</body></html>
"""

ARTICLE1 = """
<html>
<body>
References:
<div>10.1234/ref1</div>
<div>10.1234/ref2</div>
</body>
</html>
"""

RESPONSES = {
    "http://example.com/search": PAGE1,
    "http://example.com/page2": PAGE2,
    "http://example.com/page3": PAGE3,
    "https://doi.org/10.1234/abc1": ARTICLE1,
    "https://doi.org/10.1234/abc2": "",
    "https://doi.org/10.1234/ref1": "",
    "https://doi.org/10.1234/ref2": "",
}


def fake_get(url, timeout=10):
    resp = MagicMock()
    resp.text = RESPONSES.get(url, "")
    return resp


def test_doi_crawler_pagination_and_refs(tmp_path):
    crawler = DOICrawler(max_depth=2)
    out_file = tmp_path / "out.csv"
    with patch("neurosales.doi_crawler.requests.Session.get", side_effect=fake_get):
        crawler.crawl(["http://example.com/search"], str(out_file))

    text = out_file.read_text(encoding="utf-8")
    assert "10.1234/abc1" in text
    assert "10.1234/abc2" in text
    assert "10.1234/ref1" in text
    assert "10.1234/ref2" in text


def test_doi_crawler_logs_failure(tmp_path, caplog):
    crawler = DOICrawler(max_depth=1)
    out_file = tmp_path / "out.csv"
    with patch(
        "neurosales.doi_crawler.requests.Session.get",
        side_effect=Exception("boom"),
    ), caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            crawler.crawl(["http://bad"], str(out_file))

    assert any(
        "Failed to fetch search page" in r.getMessage() for r in caplog.records
    )

