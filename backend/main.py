# main.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import base64
import os

from database import Base, engine
from models import User, Conversation, Message, PHQ8Assessment
from auth import (
    get_db, hash_password, verify_password,
    create_token, get_current_user
)
from rag import get_response
from phq8 import calculate_phq8, PHQ8_QUESTIONS, PHQ8_OPTIONS
from phq8_therapeutic_responses import get_phq8_follow_up_prompt
from voice_depression_detector import voice_detector

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

class PHQ8Response(BaseModel):
    no_interest: int
    depressed: int
    sleep: int
    tired: int
    appetite: int
    failure: int
    concentrating: int
    moving: int

class ConversationUpdate(BaseModel):
    title: str

class VoiceAudio(BaseModel):
    audio_data: str       # base64 encoded WAV
    conversation_id: int  # active conversation
    transcript: str       # speech-to-text result

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"status": "Mental Health Chatbot Running"}

# ==================== AUTH ====================

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

# ==================== CONVERSATIONS ====================

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

@app.get("/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: int,
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

@app.put("/conversations/{conversation_id}/title")
def update_conversation_title(
    conversation_id: int,
    data: ConversationUpdate,
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

    conversation.title = data.title
    db.commit()
    return {"message": "Title updated", "title": data.title}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
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

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted"}

# ==================== CHAT (TEXT) ====================

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

    # Get assistant response
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

# ==================== VOICE CHAT WITH DEPRESSION DETECTION ====================

@app.post("/voice/chat")
def voice_chat(
    req: VoiceAudio,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Full voice pipeline:
    1. Receive base64 audio + transcript from frontend
    2. Analyze audio for depression markers using trained model
    3. Pass voice context invisibly to RAG for smarter responses
    4. Return chat response + voice analysis result
    """

    # Verify conversation belongs to user
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == req.conversation_id,
            Conversation.user_id == user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # ---- Step 1: Analyze audio for depression markers ----
    voice_analysis = None
    voice_context = ""

    try:
        audio_bytes = base64.b64decode(req.audio_data)
        voice_analysis = voice_detector.analyze_audio_bytes(audio_bytes)
        voice_context = voice_detector.get_therapeutic_context(voice_analysis)
        
        if voice_analysis and not voice_analysis.get('error'):
            print(f"Voice Analysis - Risk: {voice_analysis.get('risk_level')} | "
                  f"Probability: {voice_analysis.get('probability', 0):.2%}")
    except Exception as e:
        print(f"Voice analysis error (non-critical): {e}")
        voice_analysis = None
        voice_context = ""

    # ---- Step 2: Save user message (transcript) ----
    user_msg = Message(
        conversation_id=req.conversation_id,
        role="user",
        content=req.transcript
    )
    db.add(user_msg)
    db.commit()

    # ---- Step 3: Get therapeutic response with voice context ----
    response = get_response(
        message=req.transcript,
        conversation_id=req.conversation_id,
        db=db,
        voice_context=voice_context
    )

    # ---- Step 4: Save assistant response ----
    ai_msg = Message(
        conversation_id=req.conversation_id,
        role="assistant",
        content=response
    )
    db.add(ai_msg)
    db.commit()

    # ---- Step 5: Prepare voice result for frontend ----
    voice_result = None
    if voice_analysis and not voice_analysis.get('error'):
        prob = voice_analysis.get('probability', 0)
        risk = voice_analysis.get('risk_level', 'Low')

        voice_result = {
            "depression_detected": voice_analysis.get('depression_detected', False),
            "probability": round(prob * 100, 1),
            "risk_level": risk,
            "confidence": round(voice_analysis.get('confidence', 0) * 100, 1),
            "badge_color": (
                "#ef4444" if risk == "High"
                else "#f59e0b" if risk == "Moderate"
                else "#10b981"
            )
        }

    return {
        "response": response,
        "voice_analysis": voice_result
    }

# ==================== PHQ-8 ROUTES ====================

@app.get("/phq8/questions")
def get_phq8_questions():
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
    response_dict = responses.dict()
    result = calculate_phq8(response_dict)

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

    interpretation = get_interpretation(result['total_score'])

    if not conversation_id:
        new_conversation = Conversation(
            user_id=user.id,
            title="PHQ-8 Assessment"
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id
        assessment.conversation_id = conversation_id
        db.commit()

    result_message = f"""PHQ-8 Assessment Results:

Score: {result['total_score']} out of 24
Severity: {result['severity']}
Status: {'Depression Detected' if result['depressed'] else 'No Depression Detected'}

{interpretation}"""

    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content="I just completed a PHQ-8 assessment"
    )
    db.add(user_msg)

    assessment_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=result_message
    )
    db.add(assessment_msg)
    db.commit()

    follow_up_prompt = get_phq8_follow_up_prompt(result['total_score'], result['severity'])

    therapeutic_response = get_response(
        message=follow_up_prompt,
        conversation_id=conversation_id,
        db=db
    )

    response_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=therapeutic_response
    )
    db.add(response_msg)
    db.commit()

    return {
        "assessment_id": assessment.id,
        "conversation_id": conversation_id,
        "score": result['total_score'],
        "severity": result['severity'],
        "binary": result['binary'],
        "depressed": result['depressed'],
        "interpretation": interpretation,
        "therapeutic_response": therapeutic_response
    }

@app.get("/phq8/history")
def get_phq8_history(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
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

# ==================== HELPER FUNCTIONS ====================

def get_interpretation(score: int) -> str:
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