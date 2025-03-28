from typing import Annotated
from fastapi import Depends, Query, Header, HTTPException, status
from sqlmodel import Session
from app.db.session import get_session
from app.core.security import get_current_active_user
from app.db.models import User

def get_db_session():
    with get_session() as session:
        yield session

# Pagination parameters
def pagination_params(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, le=1000, description="Maximum number of items to return")
):
    return {"skip": skip, "limit": limit}

# Common query parameters for items
def item_query_params(
    name: str | None = Query(None, min_length=1, max_length=50, description="Filter by name"),
    min_price: float | None = Query(None, ge=0, description="Minimum price filter"),
    max_price: float | None = Query(None, ge=0, description="Maximum price filter"),
    in_stock: bool | None = Query(None, description="Filter by stock availability")
):
    return {
        "name": name,
        "min_price": min_price,
        "max_price": max_price,
        "in_stock": in_stock
    }

# User-agent header dependency
def get_user_agent(user_agent: Annotated[str | None, Header()] = None):
    if not user_agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User-Agent header is required"
        )
    return user_agent

# Admin-only dependency
def admin_only(current_user: Annotated[User, Depends(get_current_active_user)]):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Common response headers
def common_response_headers():
    return {
        "X-API-Version": "1.0",
        "Cache-Control": "no-cache, no-store, must-revalidate"
    }