from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
print(DATABASE_URL)
# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Session Local
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base model
Base = declarative_base()

# Define a model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    

# Base.metadata.create_all(engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI with PostgreSQL"}

@app.post("/users/")
def create_user(name: str, age: int, db: Session = Depends(get_db)):
    new_user = User(name=name, age=age)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
