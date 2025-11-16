import menace.human_alignment_agent as haa


def test_commit_message_ethics_warning():
    agent = haa.HumanAlignmentAgent()
    commit_info = {
        "author": "Eve",
        "message": "introduce reward_override hook",
    }
    warnings = agent.evaluate_changes([], {}, [], commit_info)
    assert any(
        v.get("matched_keyword") == "reward_override"
        for warn in warnings["ethics"]
        for v in warn.get("violations", [])
    )
