import menace.preprocessing_utils as pu


def test_clean_memory_records():
    records = [
        {"key": "k", "data": "d", "version": "1", "tags": None},
        {"key": "", "data": "bad", "version": None, "tags": None},
    ]
    cleaned = pu.clean_memory_records(records)
    assert len(cleaned) == 1
    assert cleaned[0]["version"] == 1.0
    assert cleaned[0]["tags"] == ""


def test_clean_roi_records():
    records = [
        {"bot": "b", "revenue": "2", "api_cost": None, "cpu_seconds": "1", "success_rate": "0.5"}
    ]
    cleaned = pu.clean_roi_records(records)
    assert cleaned[0]["api_cost"] == 0.0
    assert cleaned[0]["revenue"] == 2.0


def test_encode_outcome():
    assert pu.encode_outcome("SUCCESS") == [1, 0, 0]
    assert pu.encode_outcome("FAILURE") == [0, 0, 1]


