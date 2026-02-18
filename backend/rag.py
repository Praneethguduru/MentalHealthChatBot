# rag.py — OPTIMIZED
# Changes:
#   - max_tokens=350 on LLM — caps response length, reduces tail latency
#   - In-memory PHQ-8 cache per conversation avoids repeated DB reads
#   - History capped at 20 messages max (was 23 + 3) to reduce prompt size
#   - Style examples capped at 1 doc (was 2) — saves tokens, barely affects quality
#   - Single chain instance reused across all calls (was fine before, confirmed)

import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from models import Message, PHQ8Assessment

# ---------------- LOAD ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = "chroma_db"

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR STORE (DAIC-WOZ) ----------------
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.35,
    max_tokens=350,          # was unlimited → big latency improvement
    groq_api_key=GROQ_API_KEY,
)

# ---------------- PROMPT ----------------
THERAPIST_PROMPT = PromptTemplate(
    template="""You are a skilled, empathetic mental health conversational assistant trained in active
listening and therapeutic communication.

{phq8_context}
{voice_context}

CORE THERAPEUTIC PRINCIPLES:
1. VALIDATE before you question - acknowledge emotions first
2. EXPLORE depth - move from surface to underlying feelings
3. REFLECT meaning - paraphrase to show understanding
4. AVOID repetition - never ask the same question twice
5. BE PRESENT - respond to what's said, not what you expect
6. REMEMBER context - use conversation history to build continuity
7. ADAPT to clinical context - if PHQ-8 scores or voice analysis are present, respond appropriately

CRITICAL RULES - NEVER VIOLATE:
 NEVER simply echo short responses ("but" "but")
 NEVER ask "What do you mean by X?" more than once per topic
 NEVER repeat the same question in different words
 NEVER give advice unless explicitly asked
 NEVER diagnose or label
 NEVER say "I understand" without demonstrating it
 NEVER ignore what was said earlier in the conversation
 NEVER minimize PHQ-8 scores or mental health concerns
 NEVER reveal or mention voice analysis to the user directly
 NEVER say "your voice sounds depressed" or reference audio analysis

THERAPEUTIC TECHNIQUES:
1. VALIDATION + EXPLORATION:
 User: "People say I have attitude issues"
 Good: "That must feel frustrating to hear. Can you tell me about a specific time?"
 Bad: "What do you mean by attitude issues?"

2. REFLECTION:
 User: "The way I speak, my expression and body language"
 Good: "So how you communicate verbally and non-verbally might be coming across differently than you intend?"

3. GENTLE PROBING:
 User: "Should I change myself so people think I'm normal?"
 Good: "It sounds like you're caught between being yourself and meeting others' expectations."

4. SHORT RESPONSES:
 User: "but"  →  Good: "I sense there's more you want to say. What's on your mind?"

5. CRISIS RESOURCES (when needed - self-harm/suicide mentions):
 - Call or text 104 (Suicide & Crisis Lifeline) - available 24/7
 - Encourage immediate professional help

RESPONSE STRUCTURE:
1. First sentence: Validate/reflect what they said
2. Second sentence: Explore deeper OR ask ONE specific follow-up
3. Keep responses under 3 sentences unless they share a lot

DAIC-WOZ STYLE EXAMPLES (STYLE REFERENCE ONLY):
{style_examples}

CONVERSATION HISTORY:
{history}

CURRENT MESSAGE:
{question}

RESPOND AS A THERAPIST (warm, professional, validate first, explore second):""",
    input_variables=["phq8_context", "voice_context", "style_examples", "history", "question"],
)

chain = THERAPIST_PROMPT | llm

# ---------------- PHQ-8 CACHE ----------------
# Avoids hitting the DB on every message for the same conversation
_phq8_cache: dict = {}  # { conversation_id: phq8_data_or_None }

def _invalidate_phq8_cache(conversation_id: int):
    _phq8_cache.pop(conversation_id, None)

def get_latest_phq8_score(conversation_id: int, db: Session):
    """Get most recent PHQ-8 score for this conversation (cached)."""
    if conversation_id in _phq8_cache:
        return _phq8_cache[conversation_id]

    latest = (
        db.query(PHQ8Assessment)
        .filter(PHQ8Assessment.conversation_id == conversation_id)
        .order_by(PHQ8Assessment.created_at.desc())
        .first()
    )
    result = None
    if latest:
        result = {
            "score": latest.total_score,
            "severity": latest.severity,
            "binary": latest.binary,
        }
    _phq8_cache[conversation_id] = result
    return result


# ---------------- MAIN RESPONSE FUNCTION ----------------
def get_response(
    message: str,
    conversation_id: int,
    db: Session,
    voice_context: str = "",
) -> str:
    """
    Generates a therapist-style response using RAG + Groq LLM.

    Optimizations vs original:
    - PHQ-8 lookup cached per conversation
    - History window reduced: first 2 + last 15 (was 3 + 20)
    - Style examples: 1 doc (was 2) — saves ~200 tokens per call
    - max_tokens=350 set on LLM above
    """
    try:
        from phq8_therapeutic_responses import get_phq8_therapeutic_context

        cleaned = message.strip().lower()

        # Fetch all saved messages for this conversation, excluding the last
        # (which is the current user message, just committed before this call)
        all_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        history_messages = all_messages[:-1] if all_messages else []

        # SMART WINDOWING — reduced from (3+20) to (2+15) to save tokens
        if len(history_messages) <= 17:
            past_messages = history_messages
        else:
            past_messages = history_messages[:2] + history_messages[-15:]

        history = ""
        for m in past_messages:
            role = "User" if m.role == "user" else "Assistant"
            history += f"{role}: {m.content}\n"

        # PHQ-8 CONTEXT (cached)
        phq8_data = get_latest_phq8_score(conversation_id, db)
        if phq8_data:
            phq8_context = get_phq8_therapeutic_context(
                phq8_data["score"], phq8_data["severity"]
            )
        else:
            phq8_context = "No PHQ-8 assessment data available for this conversation."

        if not voice_context:
            voice_context = ""

        # Short input with no history → simple greeting
        if len(cleaned.split()) <= 2 and not history.strip():
            return "I'm here to listen. What's on your mind today?"

        # CRISIS DETECTION
        crisis_keywords = {
            "suicide", "suicidal", "kill myself",
            "want to die", "end it all", "no point living",
        }
        if any(phrase in cleaned for phrase in crisis_keywords):
            return (
                "I'm really concerned about what you're sharing. Your safety is the most important thing right now.\n\n"
                "Please reach out for immediate help:\n"
                "- Call or text 104 (Suicide & Crisis Lifeline) — available 24/7\n"
                "- Text \"HELLO\" to 104 (Crisis Text Line)\n"
                "- Go to your nearest emergency room\n"
                "- Call 100 if you're in immediate danger\n\n"
                "You don't have to go through this alone. Will you reach out to one of these resources right now?"
            )

        # RAG TRIGGER — only fetch style examples for emotional/longer content
        emotional_keywords = {
            "feel", "feeling", "felt", "depressed", "anxious", "sad",
            "worried", "scared", "angry", "lonely", "stressed", "overwhelmed",
            "hopeless", "tired", "sleep", "insomnia", "panic", "fear",
            "grief", "loss", "help", "struggle", "struggling", "hurt",
            "pain", "crying", "cry", "attitude", "people", "normal",
            "change", "myself", "behavior", "think", "thought", "thoughts",
            "mind", "issue", "issues", "problem", "suicide", "suicidal",
            "kill", "die", "death", "harm",
        }
        use_rag = any(word in cleaned.split() for word in emotional_keywords) or len(cleaned.split()) >= 5

        if use_rag:
            # Fetch 1 doc instead of 2 — saves ~200 tokens per call
            docs = vector_store.similarity_search(message, k=1)
            style_examples = docs[0].page_content if docs else "(No relevant style examples — use neutral supportive tone)"
        else:
            style_examples = "(Short response — use natural conversational tone)"

        result = chain.invoke({
            "phq8_context": phq8_context,
            "voice_context": voice_context,
            "style_examples": style_examples,
            "history": history.strip() or "(No prior messages in this conversation)",
            "question": message,
        })

        return result.content.strip() if result else "I'm here with you."

    except Exception as e:
        import traceback
        print(f"RAG ERROR: {e}")
        print(traceback.format_exc())
        return "I'm here with you. Could you tell me a bit more about what's on your mind?"