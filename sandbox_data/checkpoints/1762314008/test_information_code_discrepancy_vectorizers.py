from vector_service import SharedVectorService


def test_information_code_discrepancy_vectorizers():
    svc = SharedVectorService()
    info = {"content": "abc", "summary": "d"}
    code = {"language": "python", "content": "print('hi')\n"}
    disc = {"description": "hi", "severity": 5}

    info_vec = svc.vectorise("information", info)
    code_vec = svc.vectorise("code", code)
    disc_vec = svc.vectorise("discrepancy", disc)

    assert info_vec == [0.0003, 0.001]
    assert code_vec == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
    assert disc_vec == [0.5, 0.002]
