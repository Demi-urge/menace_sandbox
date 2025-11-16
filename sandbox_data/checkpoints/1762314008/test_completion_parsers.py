from completion_parsers import parse_json, extract_code_block


def test_parse_json():
    assert parse_json('{"x": 1}') == {"x": 1}


def test_extract_code_block():
    text = 'before\n```python\nprint("hi")\n```\nafter'
    assert extract_code_block(text) == 'print("hi")'
