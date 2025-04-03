from pydantic import Field, field_validator, computed_field, BaseModel, SecretStr
from typing import Optional,List
from dotenv import load_dotenv
import os
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    raise FileNotFoundError(f"No .env file found at: {env_path}")

class Settings(BaseModel):
    SECRET_KEY: str = os.environ['SECRET_KEY']  # Will raise KeyError if missing
    DATABASE_URL: str = os.environ['DATABASE_URL']
    
    # Optional fields with defaults
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    CORS_ORIGINS: list[str] = ["*"]

    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key should be at least 32 characters")
        return v

# Debug: Print environment variables being loaded
print("Environment variables found:")
print(f"SECRET_KEY exists: {'SECRET_KEY' in os.environ}")
print(f"DATABASE_URL exists: {'DATABASE_URL' in os.environ}")

try:
    settings = Settings()
    print("Settings initialized successfully!")
except Exception as e:
    print(f"Error initializing settings: {e}")
    raise