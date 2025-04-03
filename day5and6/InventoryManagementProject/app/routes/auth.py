from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import Annotated
from sqlmodel import Session, select

from app.core.security import (
    create_access_token,
    get_password_hash,
    authenticate_user,
    get_current_active_user
)

from app.core.config import settings
from app.db.models import User, UserCreate, UserRead
from app.db.session import get_session

router = APIRouter()

@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session)
):
    user = await authenticate_user(session, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name
        }
    }

@router.post("/register")
async def register_user(
    user: UserCreate,
    session: Session = Depends(get_session)
):
    # Check if username already exists
    existing_user = session.exec(select(User).where(User.username == user.username)).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = session.exec(select(User).where(User.email == user.email)).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        disabled=False  # New users are active by default
    )
    
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    
    return {"message": "User created successfully", "user_id": db_user.id}

@router.get("/me", response_model=UserRead)
async def read_user_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return current_user