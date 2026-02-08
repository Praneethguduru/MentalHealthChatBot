from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine
from models import User, Conversation, Message
from auth import (
    get_db, hash_password, verify_password,
    create_token, get_current_user
)
from rag import get_response

# ---------------- APP ----------------
app = FastAPI()

# ---------------- DB INIT ----------------
Base.metadata.create_all(bind=engine)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMAS ----------------
class Signup(BaseModel):
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class Chat(BaseModel):
    message: str

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"status": "Mental Health Chatbot Running"}

# ---------- AUTH ----------
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

# ---------- CONVERSATIONS ----------
@app.post("/conversations")
def create_conversation(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    convo = Conversation(user_id=user.id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return {"conversation_id": convo.id}

@app.get("/conversations")
def list_conversations(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    convos = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return [
        {"id": c.id, "title": c.title, "created_at": c.created_at}
        for c in convos
    ]

# ---------- CHAT (MEMORY-BASED) ----------
@app.post("/chat/{conversation_id}")
def chat(
    conversation_id: int,
    req: Chat,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=req.message
    )
    db.add(user_msg)
    db.commit()

    # Get assistant response USING conversation memory
    response = get_response(
        message=req.message,
        conversation_id=conversation_id,
        db=db
    )

    # Save assistant message
    ai_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response
    )
    db.add(ai_msg)
    db.commit()

    return {"response": response}
from models import User, Conversation, Message, PHQ8Assessment
from phq8 import calculate_phq8, PHQ8_QUESTIONS, PHQ8_OPTIONS
from pydantic import BaseModel
from typing import Dict

# Add this schema
class PHQ8Response(BaseModel):
    no_interest: int
    depressed: int
    sleep: int
    tired: int
    appetite: int
    failure: int
    concentrating: int
    moving: int

# ---------- PHQ-8 ROUTES ----------

@app.get("/phq8/questions")
def get_phq8_questions():
    """Get PHQ-8 questionnaire"""
    return {
        "questions": PHQ8_QUESTIONS,
        "options": PHQ8_OPTIONS,
        "instructions": "Over the last 2 weeks, how often have you been bothered by the following problems?"
    }

@app.post("/phq8/submit")
def submit_phq8(
    responses: PHQ8Response,
    conversation_id: int = None,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit PHQ-8 assessment and calculate score"""
    
    # Convert to dict
    response_dict = responses.dict()
    
    # Calculate score
    result = calculate_phq8(response_dict)
    
    # Save to database
    assessment = PHQ8Assessment(
        user_id=user.id,
        conversation_id=conversation_id,
        no_interest=response_dict['no_interest'],
        depressed=response_dict['depressed'],
        sleep=response_dict['sleep'],
        tired=response_dict['tired'],
        appetite=response_dict['appetite'],
        failure=response_dict['failure'],
        concentrating=response_dict['concentrating'],
        moving=response_dict['moving'],
        total_score=result['total_score'],
        severity=result['severity'],
        binary=result['binary']
    )
    
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    
    return {
        "assessment_id": assessment.id,
        "score": result['total_score'],
        "severity": result['severity'],
        "binary": result['binary'],
        "depressed": result['depressed'],
        "interpretation": get_interpretation(result['total_score'])
    }

@app.get("/phq8/history")
def get_phq8_history(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's PHQ-8 assessment history"""
    assessments = (
        db.query(PHQ8Assessment)
        .filter(PHQ8Assessment.user_id == user.id)
        .order_by(PHQ8Assessment.created_at.desc())
        .all()
    )
    
    return [
        {
            "id": a.id,
            "score": a.total_score,
            "severity": a.severity,
            "binary": a.binary,
            "date": a.created_at
        }
        for a in assessments
    ]

@app.get("/phq8/latest")
def get_latest_phq8(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's most recent PHQ-8 score"""
    latest = (
        db.query(PHQ8Assessment)
        .filter(PHQ8Assessment.user_id == user.id)
        .order_by(PHQ8Assessment.created_at.desc())
        .first()
    )
    
    if not latest:
        return {"message": "No assessments found"}
    
    return {
        "score": latest.total_score,
        "severity": latest.severity,
        "date": latest.created_at
    }

def get_interpretation(score: int) -> str:
    """Get interpretation text based on score"""
    if score <= 4:
        return "Your responses suggest minimal or no depression. Keep taking care of yourself!"
    elif score <= 9:
        return "Your responses suggest mild depression. Consider monitoring your mood and practicing self-care."
    elif score <= 14:
        return "Your responses suggest moderate depression. It may be helpful to speak with a mental health professional."
    elif score <= 19:
        return "Your responses suggest moderately severe depression. I strongly recommend consulting a mental health professional."
    else:
        return "Your responses suggest severe depression. Please seek professional help as soon as possible. If you're in crisis, contact a crisis helpline immediately."
from pydantic import BaseModel

# Add this schema
class ConversationUpdate(BaseModel):
    title: str

# ---------- CONVERSATION MANAGEMENT ----------

@app.put("/conversations/{conversation_id}/title")
def update_conversation_title(
    conversation_id: int,
    data: ConversationUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update conversation title"""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
        .first()
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.title = data.title
    db.commit()
    
    return {"message": "Title updated", "title": data.title}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation"""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
        .first()
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted"}
@app.get("/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages from a conversation"""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id
        )
        .first()
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp.asc())
        .all()
    )
    
    return [
        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
        for m in messages
    ]