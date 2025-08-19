import governed_retrieval as gr


def test_govern_retrieval_blocks_license():
    text = "This project is licensed under the GNU General Public License"
    assert gr.govern_retrieval(text, {}) is None


def test_govern_retrieval_redacts_and_alerts():
    meta = {"token": "AKIA1234567890ABCD12"}
    reason = "api_key=ABCD1234567890EFGH1234"
    text = "eval('data') plus secret AKIA1234567890ABCD12"
    governed = gr.govern_retrieval(text, meta, reason)
    assert governed is not None
    meta_out, reason_out = governed
    assert meta_out["token"] == "[REDACTED]"
    assert reason_out == "[REDACTED]"
    alerts = meta_out.get("semantic_alerts")
    assert alerts and any("eval" in a[1] for a in alerts)
    assert meta_out.get("redacted")
