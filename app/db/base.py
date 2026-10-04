"""SQLAlchemy base and session management."""
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

Base = declarative_base()

_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """Create or return the SQLAlchemy engine."""
    global _engine
    url = database_url or settings.DATABASE_URL
    if _engine is None or str(_engine.url) != url:
        if url.startswith("sqlite"):
            # Ensure SQLite parent directory exists
            if "///" in url:
                file_path = url.split("///")[1]
                path = Path(file_path).resolve().parent
                path.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(
                url, connect_args={"check_same_thread": False}
            )
        else:
            _engine = create_engine(url)
    return _engine


def get_session_factory(
    database_url: str | None = None,
) -> sessionmaker[Session]:
    """Get the sessionmaker for the current engine."""
    global _sessionmaker
    engine = get_engine(database_url)
    if _sessionmaker is None or _sessionmaker.kw.get("bind") != engine:
        init_db(engine)
        _sessionmaker = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
    return _sessionmaker


def init_db(engine: Engine | None = None) -> None:
    """Create all tables in the database."""
    import app.db.models  # noqa: F401
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)


@contextmanager
def get_db_session(
    database_url: str | None = None,
) -> Generator[Session, None, None]:
    """Yield a database session and ensure clean close."""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
    finally:
        session.close()


get_session = get_db_session

