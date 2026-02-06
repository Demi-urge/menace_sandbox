"""Minimal pipeline utilities."""
from __future__ import annotations

from typing import Iterable, List, Sequence


class Pipeline:
    def __init__(self, steps: Sequence[object]) -> None:
        self.steps = list(steps)

    def fit(self, X: Iterable, y: Iterable | None = None) -> "Pipeline":
        data = X
        for step in self.steps[:-1]:
            data = self._fit_transform_step(step, data, y)
        if self.steps:
            last = self.steps[-1]
            if hasattr(last, "fit"):
                last.fit(data, y)
        return self

    def transform(self, X: Iterable) -> List:
        data = X
        for step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError("Pipeline step does not support transform")
            data = step.transform(data)
        return list(data)

    def predict(self, X: Iterable) -> List:
        data = X
        for step in self.steps[:-1]:
            data = step.transform(data)
        if not self.steps:
            raise AttributeError("Pipeline has no steps")
        last = self.steps[-1]
        if not hasattr(last, "predict"):
            raise AttributeError("Pipeline final step does not support predict")
        return list(last.predict(data))

    def fit_transform(self, X: Iterable, y: Iterable | None = None) -> List:
        data = X
        for step in self.steps[:-1]:
            data = self._fit_transform_step(step, data, y)
        if not self.steps:
            return list(data)
        last = self.steps[-1]
        if hasattr(last, "fit_transform"):
            return list(last.fit_transform(data, y))
        if hasattr(last, "fit") and hasattr(last, "transform"):
            last.fit(data, y)
            return list(last.transform(data))
        last.fit(data, y)
        return list(data)

    def _fit_transform_step(self, step: object, data: Iterable, y: Iterable | None) -> Iterable:
        if hasattr(step, "fit_transform"):
            return step.fit_transform(data, y)
        if hasattr(step, "fit") and hasattr(step, "transform"):
            step.fit(data, y)
            return step.transform(data)
        if hasattr(step, "fit"):
            step.fit(data, y)
            return data
        raise AttributeError("Pipeline step does not support fit")


def make_pipeline(*steps: object) -> Pipeline:
    estimators = [step[1] if isinstance(step, tuple) else step for step in steps]
    return Pipeline(estimators)
