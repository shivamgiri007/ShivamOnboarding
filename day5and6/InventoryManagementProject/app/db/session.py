from sqlmodel import create_engine, SQLModel, Session
from app.core.config import settings
from contextlib import contextmanager

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Check connection health before using
    pool_size=5,  # Number of connections to keep open
    max_overflow=10,  # Number of connections to create beyond pool_size
    echo=True  # Log SQL queries (disable in production)
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@contextmanager
def get_session():
    session = Session(engine)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()