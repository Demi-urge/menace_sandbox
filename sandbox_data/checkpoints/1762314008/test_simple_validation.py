import pytest

from menace.simple_validation import SimpleSchema, fields, ValidationError

class Child(SimpleSchema):
    name = fields.Str(required=True, min_length=1)

class Parent(SimpleSchema):
    child = fields.Nested(Child, required=True)
    nums = fields.List(fields.Int(min=0), required=True, min_length=1)


def test_nested_and_list():
    schema = Parent()
    data = {"child": {"name": "ok"}, "nums": [1, 2, 3]}
    assert schema.load(data) == {"child": {"name": "ok"}, "nums": [1, 2, 3]}


def test_validation_errors():
    schema = Parent()
    with pytest.raises(ValidationError):
        schema.load({"child": {"name": "ok"}, "nums": []})
    with pytest.raises(ValidationError):
        schema.load({"child": {"name": ""}, "nums": [1]})
