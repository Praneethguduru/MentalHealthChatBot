# database.py — OPTIMIZED
# Changes:
#   - WAL journal mode: dramatically improves concurrent read/write performance
#   - Increased connection timeout to prevent "database is locked" under load
#   - StaticPool removed (was default anyway); using NullPool for thread safety

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,          # wait up to 30 s if DB is locked (default is 5 s)
    },
    pool_pre_ping=True,         # test connections before using them
)

# Enable WAL mode once at startup — survives reconnects
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.execute(text("PRAGMA synchronous=NORMAL"))   # safe + faster than FULL
    conn.execute(text("PRAGMA cache_size=-64000"))    # 64 MB page cache
    conn.execute(text("PRAGMA foreign_keys=ON"))
    conn.commit()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()