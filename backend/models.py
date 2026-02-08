from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# ---------------- USER (UNCHANGED) ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    conversations = relationship("Conversation", back_populates="user")
    assessments = relationship("PHQ8Assessment", back_populates="user")  # NEW

# ---------------- CONVERSATION ----------------
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Conversation")
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

# ---------------- MESSAGE ----------------
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

# ---------------- PHQ-8 ASSESSMENT (NEW) ----------------
class PHQ8Assessment(Base):
    __tablename__ = "phq8_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    
    # Individual scores (0-3 each)
    no_interest = Column(Integer)
    depressed = Column(Integer)
    sleep = Column(Integer)
    tired = Column(Integer)
    appetite = Column(Integer)
    failure = Column(Integer)
    concentrating = Column(Integer)
    moving = Column(Integer)
    
    # Calculated fields
    total_score = Column(Integer)
    severity = Column(String)
    binary = Column(Integer)  # 0 or 1
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="assessments")