import os
import random

def get_proxy() -> str | None:
    """Return a proxy string in host:port format or None if no proxy is available."""
    env_proxies = os.getenv("PROXIES")
    if env_proxies:
        proxies = [p.strip() for p in env_proxies.split(",") if p.strip()]
    else:
        try:
            with open("proxies.txt", "r", encoding="utf-8") as f:
                proxies = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            proxies = []
    return random.choice(proxies) if proxies else None


if __name__ == "__main__":
    from clipped.proxy_manager import cli

    cli()
