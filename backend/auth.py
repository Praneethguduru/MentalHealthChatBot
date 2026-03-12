# auth.py — FIXED
# Root cause of DetachedInstanceError:
#   _user_cache was storing the SQLAlchemy ORM User object.
#   When the DB session that loaded it closed (after the dependency returned),
#   the cached object became detached from any session.
#   On the next request that hit the cache, the ORM object was returned and
#   passed to an endpoint running in a thread. Any access to user.id or
#   user.email triggered SQLAlchemy's lazy-load, which tried to refresh
#   the object — but there was no live session → DetachedInstanceError 500.
#
# Fix:
#   Cache stores a plain CurrentUser dataclass (just ints/strings).
#   All field values are read from the ORM object while the session is still
#   open, then stored as simple Python values with zero SQLAlchemy state.
#   The dataclass is safe to pass to any thread, any endpoint, forever.

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Header
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from database import SessionLocal
from models import User

# ---------------- LOAD ENV ----------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY not set in .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ---------------- PASSWORD HASHING ----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password.encode("utf-8")[:72])

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password.encode("utf-8")[:72], hashed)

# ---------------- DATABASE DEP ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- JWT ----------------
def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ---------------- CURRENT USER ----------------
@dataclass
class CurrentUser:
    """
    Plain dataclass — holds only primitive values (int, str).
    No SQLAlchemy state, no session binding, safe in any thread.
    This is what every endpoint receives as `user`.
    """
    id: int
    email: str

# ---------------- USER CACHE ----------------
# Cache now stores CurrentUser (plain dataclass), NOT the ORM User object.
# Format: { user_id: (CurrentUser, expiry_timestamp) }
_user_cache: dict = {}
_CACHE_TTL = 60  # seconds

def _get_cached_user(user_id: int, db: Session) -> CurrentUser | None:
    now = time.monotonic()
    cached = _user_cache.get(user_id)

    # Return from cache if still fresh
    if cached:
        current_user, expires_at = cached
        if now < expires_at:
            return current_user  # plain dataclass, always safe

    # Cache miss or expired — query DB while session is still open
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None

    # Read fields NOW while the ORM object is attached to an active session,
    # then immediately discard the ORM object — never store it in the cache.
    current_user = CurrentUser(id=user.id, email=user.email)
    _user_cache[user_id] = (current_user, now + _CACHE_TTL)
    return current_user

def invalidate_user_cache(user_id: int):
    """Call this after any user mutation (e.g. password change)."""
    _user_cache.pop(user_id, None)

# ---------------- AUTH DEPENDENCY ----------------
def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> CurrentUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = _get_cached_user(user_id, db)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user  # CurrentUser dataclass — safe everywhere, no session needed