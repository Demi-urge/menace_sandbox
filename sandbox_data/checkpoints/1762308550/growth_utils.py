def growth_score(category: str) -> int:
    """Return numeric score for ROI growth categories.

    Higher scores indicate improvements with greater compounding
    potential. "exponential" growth ranks highest, followed by
    "linear" and then "marginal".
    """
    return {"exponential": 2, "linear": 1, "marginal": 0}.get(category, 0)


__all__ = ["growth_score"]
