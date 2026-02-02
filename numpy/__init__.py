"""Lightweight numpy stub for environments without the real dependency."""
from __future__ import annotations

from collections.abc import Iterable
import math
import statistics
from typing import Any, List


class ndarray(list):
    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        if len(shape) != 2:
            return self
        rows, cols = shape
        data = list(self)
        if rows == -1 and cols not in (-1, 0):
            rows = len(data) // cols
        if cols == -1 and rows not in (-1, 0):
            cols = len(data) // rows
        if rows in (-1, 0) or cols in (-1, 0):
            return self
        out: list[list[Any]] = []
        idx = 0
        for _ in range(rows):
            row: list[Any] = []
            for _ in range(cols):
                row.append(data[idx] if idx < len(data) else None)
                idx += 1
            out.append(row)
        return ndarray(out)

    def mean(self) -> float:
        return float(mean(self))

    def std(self) -> float:
        return float(std(self))

    def var(self) -> float:
        return float(var(self))

    def sum(self) -> float:
        return float(sum(_flatten(self)))

    def any(self) -> bool:
        return any(bool(x) for x in _flatten(self))

    def __truediv__(self, other: float) -> "ndarray":
        return ndarray([x / other for x in _flatten(self)])

    def __eq__(self, other: object) -> "ndarray":  # type: ignore[override]
        other_list = list(_flatten(other)) if isinstance(other, Iterable) else []
        return ndarray([a == b for a, b in zip(_flatten(self), other_list)])


def _flatten(values: Any) -> Iterable[Any]:
    if isinstance(values, ndarray):
        values = list(values)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        for item in values:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub in _flatten(item):
                    yield sub
            else:
                yield item
    else:
        yield values


def array(values: Any, dtype: Any | None = None) -> ndarray:
    if isinstance(values, ndarray):
        data = list(values)
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        data = list(values)
    else:
        data = [values]
    if dtype in (float, "float"):
        data = [float(v) for v in data]
    return ndarray(data)


def asarray(values: Any) -> ndarray:
    return array(values)


def arange(stop: int) -> ndarray:
    return ndarray(list(range(stop)))


def mean(values: Any) -> float:
    data = list(_flatten(values))
    if not data:
        return 0.0
    return float(statistics.mean(data))


def std(values: Any) -> float:
    data = list(_flatten(values))
    if len(data) < 2:
        return 0.0
    return float(statistics.pstdev(data))


def var(values: Any) -> float:
    data = list(_flatten(values))
    if len(data) < 2:
        return 0.0
    return float(statistics.pvariance(data))


def percentile(values: Any, q: float) -> float:
    data = sorted(float(v) for v in _flatten(values))
    if not data:
        return 0.0
    if q <= 0:
        return data[0]
    if q >= 100:
        return data[-1]
    k = (len(data) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def abs(values: Any) -> ndarray:  # type: ignore[override]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return ndarray([math.fabs(v) for v in _flatten(values)])
    return ndarray([math.fabs(values)])


def dot(a: Any, b: Any) -> float:
    a_list = list(_flatten(a))
    b_list = list(_flatten(b))
    return float(sum(x * y for x, y in zip(a_list, b_list)))


def hstack(arrays: List[Any]) -> ndarray:
    if not arrays:
        return ndarray([])
    first = arrays[0]
    if isinstance(first, Iterable) and first and isinstance(list(first)[0], Iterable):
        rows = len(first)
        combined: list[list[Any]] = [[] for _ in range(rows)]
        for arr in arrays:
            for idx, row in enumerate(arr):
                combined[idx].extend(list(row))
        return ndarray(combined)
    out: list[Any] = []
    for arr in arrays:
        out.extend(list(_flatten(arr)))
    return ndarray(out)


def nanmean(values: Any) -> float:
    data = [v for v in _flatten(values) if not isinstance(v, float) or not math.isnan(v)]
    if not data:
        return float("nan")
    return float(statistics.mean(data))


def isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


nan = float("nan")
inf = float("inf")

__all__ = [
    "abs",
    "any",
    "arange",
    "array",
    "asarray",
    "dot",
    "hstack",
    "inf",
    "isnan",
    "mean",
    "nan",
    "nanmean",
    "ndarray",
    "percentile",
    "std",
    "var",
]
