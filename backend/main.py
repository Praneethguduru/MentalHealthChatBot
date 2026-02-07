from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine
from models import User
from auth import (
    get_db, hash_password, verify_password,
    create_token, get_current_user
)
from rag import get_response

# ✅ CREATE APP FIRST
app = FastAPI()

# ✅ DB INIT
Base.metadata.create_all(bind=engine)

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SCHEMAS ----------
class Signup(BaseModel):
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class Chat(BaseModel):
    message: str
    new_session: bool = False  # ✅ IMPORTANT

# ---------- ROUTES ----------
@app.get("/")
def home():
    return {"status": "Mental Health Chatbot Running"}

@app.post("/signup")
def signup(data: Signup, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        email=data.email,
        hashed_password=hash_password(data.password)
    )
    db.add(user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(data: Login, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user.id)
    return {"access_token": token}

@app.post("/chat")
def chat(req: Chat, user=Depends(get_current_user)):
    response = get_response(
        message=req.message,
        new_session=req.new_session  # ✅ PASS FLAG
    )
    return {"response": response}
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine
from models import User
from auth import (
    get_db, hash_password, verify_password,
    create_token, get_current_user
)
from rag import get_response

# ✅ CREATE APP FIRST
app = FastAPI()

# ✅ DB INIT
Base.metadata.create_all(bind=engine)

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SCHEMAS ----------
class Signup(BaseModel):
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class Chat(BaseModel):
    message: str
    new_session: bool = False  # ✅ IMPORTANT

# ---------- ROUTES ----------
@app.get("/")
def home():
    return {"status": "Mental Health Chatbot Running"}

@app.post("/signup")
def signup(data: Signup, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        email=data.email,
        hashed_password=hash_password(data.password)
    )
    db.add(user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(data: Login, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user.id)
    return {"access_token": token}

@app.post("/chat")
def chat(req: Chat, user=Depends(get_current_user)):
    response = get_response(
        message=req.message,
        new_session=req.new_session  # ✅ PASS FLAG
    )
    return {"response": response}
