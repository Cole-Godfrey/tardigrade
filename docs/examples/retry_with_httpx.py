from __future__ import annotations

import time

from tardigrade import RetryConfig, armor


class HTTPStatusError(RuntimeError):
    pass


class ReadTimeout(TimeoutError):
    pass


attempts = 0


@armor(
    name="fetch_json",
    retry=RetryConfig(
        max_attempts=4,
        base_delay=0.2,
        jitter=False,
        retryable_exceptions=(HTTPStatusError, ReadTimeout),
    ),
)
def fetch_json(url: str) -> dict[str, str]:
    global attempts
    attempts += 1
    time.sleep(0.1)
    if attempts < 3:
        raise ReadTimeout(f"timed out calling {url}")
    return {"url": url, "status": "ok"}


if __name__ == "__main__":
    print(fetch_json("https://api.example.com/data"))
