"""Minimal jinja2 stub used by error_bot templates."""


class Template:
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, **_kwargs: object) -> str:
        return self.text
