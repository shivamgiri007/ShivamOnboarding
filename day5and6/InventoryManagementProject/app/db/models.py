from typing import Optional, List
from sqlmodel import SQLModel , Field, Relationship
from datetime import timedelta
from pydantic import EmailStr
from sqlmodel import Session
from typing import Annotated

class UserBase(SQLModel):
    username: str = Field(
        index=True, 
        unique=True, 
        min_length=3, 
        max_length=50, 
        regex="^[a-zA-Z0-9_-]+$"
    )
    email: EmailStr = Field(index=True, unique=True)
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    disabled: bool = Field(default=False)
    is_admin: bool = Field(default=False)

class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str = Field(min_length=8, max_length=128)
    
    items: List["Item"] = Relationship(back_populates="owner")

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)

class UserRead(UserBase):
    id: int

class UserUpdate(SQLModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    disabled: Optional[bool] = None
    is_admin: Optional[bool] = None

class ItemBase(SQLModel):
    name: str = Field(index=True, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=255)

class Item(ItemBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id")

    owner: "User" = Relationship(back_populates="items")

class ItemCreate(ItemBase):
    pass

class ItemRead(ItemBase):
    id: int
    owner_id: int
