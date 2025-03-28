from pydantic import BaseSettings,Field, validator 

class Settings(BaseSettings):
    SECRET_KEY:str = Field(..., env="SECRET_KEY")
    ALGORITHM:str ="HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:int=120

    # Database settings
    DB_ENGINE: str = Field("postgresql", env="DB_ENGINE")  # postgresql, mysql, sqlserver
    DB_HOST: str = Field("localhost", env="DB_HOST")
    DB_PORT: str = Field("5432", env="DB_PORT")  # 5432 for PostgreSQL, 3306 for MySQL, 1433 for SQL Server
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_NAME: str = Field(..., env="DB_NAME")
    
    # Redis for rate limiting
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    CORS_ORIGINS:list[str] = ["*"]

    @property
    def DATABASE_URL(self) -> str:
        if self.DB_ENGINE == "postgresql":
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        elif self.DB_ENGINE == "mysql":
            return f"mysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        elif self.DB_ENGINE == "sqlserver":
            return f"mssql+pyodbc://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            raise ValueError(f"Unsupported database engine: {self.DB_ENGINE}")

    class Config:
        env_file=".env"
        extra="forbid"

    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key should at least 32 characters")
        return v    
settings=Settings()