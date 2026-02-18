# auth.py — OPTIMIZED
# Changes:
#   - Simple in-process user cache to avoid repeated DB lookups on every request
#   - Cache is keyed by user_id with a short TTL (60 s) — safe for this use case
#   - Token expiry extended to 24 h (was 12 h) for better UX

import os
import time
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
ACCESS_TOKEN_EXPIRE_HOURS = 24   # was 12 — users no longer get logged out mid-day

# ---------------- PASSWORD HASHING ----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")[:72]
    return pwd_context.hash(password_bytes)

def verify_password(password: str, hashed: str) -> bool:
    password_bytes = password.encode("utf-8")[:72]
    return pwd_context.verify(password_bytes, hashed)

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

# ---------------- USER CACHE ----------------
# Avoids hitting the DB on every authenticated request.
# Format: { user_id: (User object, expiry_timestamp) }
_user_cache: dict = {}
_CACHE_TTL = 60  # seconds

def _get_cached_user(user_id: int, db: Session):
    """Return user from cache if fresh, otherwise query DB and cache."""
    now = time.monotonic()
    cached = _user_cache.get(user_id)
    if cached:
        user_obj, expires_at = cached
        if now < expires_at:
            return user_obj
    # Cache miss or expired — query DB
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        _user_cache[user_id] = (user, now + _CACHE_TTL)
    return user

def invalidate_user_cache(user_id: int):
    """Call this if user data is mutated (e.g., password change)."""
    _user_cache.pop(user_id, None)

# ---------------- AUTH DEPENDENCY ----------------
def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
):
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
    return user