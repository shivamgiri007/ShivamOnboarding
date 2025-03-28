from fastapi import Request
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis
import time
from contextlib import contextmanager
from typing import Callable
from app.core.config import settings

@contextmanager
def catch_time() -> any:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

async def add_process_time_header(request: Request, call_next: Callable):
    with catch_time() as time_taken:
        response = await call_next(request)
    
    response.headers["X-Process-Time"] = str(time_taken())
    return response

async def setup_rate_limiter():
    redis_connection = redis.from_url(settings.REDIS_URL)
    await FastAPILimiter.init(redis_connection)

def get_rate_limiter(times: int = 100, seconds: int = 60):
    return RateLimiter(times=times, seconds=seconds)