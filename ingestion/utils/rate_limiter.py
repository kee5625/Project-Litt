import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        self._rate = rate          # tokens per second
        self._burst = float(burst) # max tokens that can accumulate
        self._tokens = float(burst)
        self._last = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)
