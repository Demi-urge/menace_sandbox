"""Minimal sqlalchemy.engine stub."""


class URL:
    def __init__(self, raw_url: str) -> None:
        self.raw_url = raw_url

    def get_backend_name(self) -> str:
        if "://" in self.raw_url:
            return self.raw_url.split("://", 1)[0]
        return "sqlite"


class Engine:
    def __init__(self, raw_url: str) -> None:
        self.url = URL(raw_url)


def create_engine(raw_url: str) -> Engine:
    return Engine(raw_url)


def make_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url:
        from .exc import ArgumentError

        raise ArgumentError("Invalid database URL")
    return raw_url
