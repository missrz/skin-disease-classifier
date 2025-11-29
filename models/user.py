from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
import re

from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)

    @validator("full_name")
    def clean_full_name(cls, v):
        if v is None:
            return v
        v = v.strip()
        if len(v) == 0:
            raise ValueError("full_name cannot be empty if provided")
        # Allow letters, spaces, hyphens and apostrophes
        if not re.match(r"^[\p{L}0-9 '\-\.]+$", v, flags=re.UNICODE):
            # Fallback: only basic check to avoid rejecting many names
            return v
        return v


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserDB(UserBase):
    id: Optional[str]
    hashed_password: str


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str]


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
