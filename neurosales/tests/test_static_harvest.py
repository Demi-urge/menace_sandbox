import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.static_harvest import StaticHarvester
from unittest.mock import patch, MagicMock


SAMPLE_HTML = """
<html>
<head><title>Neuro Study</title></head>
<body>
    <h2>Methods</h2>
    <p>We tracked dopamine spikes across subjects.</p>
    <table class="roi"><tr><td>ROI 1</td></tr></table>
    <img src="dopamine1.png" alt="dopamine spike graph" />
    <div>buy-button clicks: 42</div>
</body>
</html>
"""


def test_static_harvest_parses_html(tmp_path):
    harvester = StaticHarvester()
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_HTML
    with patch("neurosales.static_harvest.requests.Session.get", return_value=mock_resp):
        out_file = tmp_path / "out.csv"
        harvester.crawl(["http://example.com"], str(out_file))
    text = out_file.read_text(encoding="utf-8")
    assert "Neuro Study" in text
    assert "dopamine" in text
    assert "ROI 1" in text
    assert "buy-button" in text

