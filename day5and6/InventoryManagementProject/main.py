from fastapi import FastAPI 
from app.routes import auth, items, users, reports 
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.middleware import add_process_time_header, setup_rate_limiter
from app.db.session import create_db_and_tables

app = FastAPI(
    title = "Inventory Management System",
    description = "Compherensive inventory management system",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(add_process_time_header)

# Create database tables and setup rate limiter on startup
@app.on_event("startup")
async def on_startup():
    create_db_and_tables()
    await setup_rate_limiter()



app.include_router(auth.router, prefix="api/v1/auth" tags=["Authentication"],dependencies=[Depends(get_rate_limiter(times=10, seconds=60))])
app.include_router(items.router, prefix="api/v1/users" tags=["items"],dependencies=[Depends(get_rate_limiter(times=30, seconds=60))])
app.include_router(users.router, prefix="api/v1/users", tags=["users"],dependencies=[Depends(get_rate_limiter(times=60, seconds=60))])
app.include_router(reports.router, prefix="api/v1/reports", tags=["reports"],dependencies=[Depends(get_rate_limiter(times=5, seconds=60))])