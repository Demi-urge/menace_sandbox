from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type


class ValidationError(Exception):
    """Validation error raised when data is invalid."""

    def __init__(self, messages: Dict[str, list[str]]) -> None:
        super().__init__("Validation failed")
        self.messages = messages


@dataclass
class SimpleField:
    """Basic field with type conversion, nesting and validation."""

    type: Optional[Type] = str
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    nested: Optional[Type["SimpleSchema"]] = None
    many: bool = False
    inner: Optional["SimpleField"] = None

    def deserialize(self, value: Any):
        if value is None:
            if self.required:
                raise ValueError("missing")
            return None

        if self.many:
            if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
                raise ValueError("invalid")
            result = [self.inner.deserialize(v) for v in value]
            length = len(result)
            if self.min_length is not None and length < self.min_length:
                raise ValueError("min_length")
            if self.max_length is not None and length > self.max_length:
                raise ValueError("max_length")
            return result

        if self.nested is not None:
            schema = self.nested() if isinstance(self.nested, type) else self.nested
            if not isinstance(value, dict):
                raise ValueError("invalid")
            return schema.load(value)

        try:
            result = self.type(value) if self.type is not None else value
        except Exception as exc:  # pragma: no cover - conversion errors
            raise ValueError("invalid") from exc

        if isinstance(result, (int, float)):
            if self.min_value is not None and result < self.min_value:
                raise ValueError("min")
            if self.max_value is not None and result > self.max_value:
                raise ValueError("max")

        if isinstance(result, str):
            if self.min_length is not None and len(result) < self.min_length:
                raise ValueError("min_length")
            if self.max_length is not None and len(result) > self.max_length:
                raise ValueError("max_length")

        return result


class SimpleSchema:
    """Very small schema system used when marshmallow isn't available."""

    _declared_fields: Dict[str, SimpleField] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._declared_fields = {
            name: val
            for name, val in cls.__dict__.items()
            if isinstance(val, SimpleField)
        }

    def load(self, data: Dict[str, Any]):
        errors: Dict[str, list[str]] = {}
        result: Dict[str, Any] = {}
        for name, field in self._declared_fields.items():
            if name in data:
                try:
                    result[name] = field.deserialize(data[name])
                except ValueError as e:
                    errors.setdefault(name, []).append(str(e))
            elif field.required:
                errors.setdefault(name, []).append("missing")
        if errors:
            raise ValidationError(errors)
        return result


class fields:
    @staticmethod
    def Str(*, required: bool = False, min_length: int | None = None, max_length: int | None = None):
        return SimpleField(str, required=required, min_length=min_length, max_length=max_length)

    @staticmethod
    def Int(*, required: bool = False, min: int | None = None, max: int | None = None):
        return SimpleField(int, required=required, min_value=min, max_value=max)

    @staticmethod
    def Float(*, required: bool = False, min: float | None = None, max: float | None = None):
        return SimpleField(float, required=required, min_value=min, max_value=max)

    @staticmethod
    def Bool(*, required: bool = False):
        return SimpleField(bool, required=required)

    @staticmethod
    def List(inner: SimpleField, *, required: bool = False, min_length: int | None = None, max_length: int | None = None):
        return SimpleField(required=required, many=True, inner=inner, min_length=min_length, max_length=max_length)

    @staticmethod
    def Nested(schema: Type[SimpleSchema] | SimpleSchema, *, required: bool = False, many: bool = False):
        if many:
            return fields.List(fields.Nested(schema), required=required)
        return SimpleField(required=required, nested=schema)


__all__ = ["ValidationError", "SimpleField", "SimpleSchema", "fields"]
