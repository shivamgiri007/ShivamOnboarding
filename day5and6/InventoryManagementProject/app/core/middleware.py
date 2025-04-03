# from fastapi import Request
# from fastapi_limiter import FastAPILimiter
# from fastapi_limiter.depends import RateLimiter
# import redis
# import time
# from contextlib import contextmanager
# from typing import Callable
# from app.core.config import settings
# # from fakeredis import FakeRedis

# # @contextmanager
# # def catch_time() -> any:
# #     start = time.perf_counter()
# #     yield lambda: time.perf_counter() - start

# # async def add_process_time_header(request: Request, call_next: Callable):
# #     with catch_time() as time_taken:
# #         response = await call_next(request)
    
# #     response.headers["X-Process-Time"] = str(time_taken())
# #     return response

# # async def setup_rate_limiter():
# #     try:
# #         # Try real Redis first
# #         redis_connection = redis.from_url(settings.REDIS_URL)
# #         await redis_connection.ping()
# #         print("Using real Redis for rate limiting")
# #     except (redis.ConnectionError, ValueError):
# #         # Fall back to a simpler rate limiter when Redis isn't available
# #         print("Using in-memory rate limiting (no Redis available)")
# #         from fastapi_limiter import FastAPILimiter
# #         from fastapi_limiter.depends import RateLimiter
# #         FastAPILimiter.init()  # Initialize with in-memory storage
# #         return

# #     await FastAPILimiter.init(redis_connection)
# # def get_rate_limiter(times: int = 100, seconds: int = 60):
# #     return RateLimiter(times=times, seconds=seconds)