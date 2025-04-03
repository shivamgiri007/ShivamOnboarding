from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from app.routes import auth, items, users, reports 
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
# from app.core.middleware import add_process_time_header, setup_rate_limiter, get_rate_limiter
from app.db.session import create_db_and_tables
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, create_db_and_tables)
    # try:
    #     await setup_rate_limiter()  # Will fail gracefully if Redis isn't available
    # except Exception as e:
    #     print(f"Rate limiter initialization failed: {e}")
    yield  # Cleanup tasks can be added here if needed

app = FastAPI(
    title = "Inventory Management System",
    description = "Compherensive inventory management system",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.middleware("http")(add_process_time_header)

# Create database tables and setup rate limiter on startup
# @app.on_event("startup")
# async def on_startup():
#     create_db_and_tables()
#     await setup_rate_limiter()
# Use Insted below code as on_event("depriciated")



app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(items.router, prefix="/api/v1/items", tags=["items"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}