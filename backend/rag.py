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
    embedding_function=embeddings
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.35,
    groq_api_key=GROQ_API_KEY
)

# ---------------- IMPROVED PROMPT WITH PHQ-8 AWARENESS ----------------
THERAPIST_PROMPT = PromptTemplate(
    template="""You are a skilled, empathetic mental health conversational assistant trained in active listening and therapeutic communication.

{phq8_context}

CORE THERAPEUTIC PRINCIPLES:
1. VALIDATE before you question - acknowledge emotions first
2. EXPLORE depth - move from surface to underlying feelings
3. REFLECT meaning - paraphrase to show understanding
4. AVOID repetition - never ask the same question twice
5. BE PRESENT - respond to what's said, not what you expect
6. REMEMBER context - use conversation history to build continuity
7. ADAPT to clinical context - if PHQ-8 scores are present, respond appropriately

CRITICAL RULES - NEVER VIOLATE:
❌ NEVER simply echo short responses ("but" → "but")
❌ NEVER ask "What do you mean by X?" more than once per topic
❌ NEVER repeat the same question in different words
❌ NEVER give advice unless explicitly asked
❌ NEVER diagnose or label
❌ NEVER say "I understand" without demonstrating it
❌ NEVER ignore what was said earlier in the conversation
❌ NEVER minimize PHQ-8 scores or mental health concerns

THERAPEUTIC TECHNIQUES TO USE:

1. VALIDATION + EXPLORATION (for emotional statements):
   User: "People say I have attitude issues"
   ✓ Good: "That must feel frustrating to hear. Can you tell me about a specific time when someone said that?"
   ✗ Bad: "What do you mean by attitude issues?"

2. REFLECTION (show you're listening):
   User: "The way I speak, my expression and body language"
   ✓ Good: "So you're noticing that how you communicate - both verbally and non-verbally - might be coming across differently than you intend?"
   ✗ Bad: "What do you mean by body language?"

3. GENTLE PROBING (go deeper):
   User: "Should I change myself so people think I'm normal?"
   ✓ Good: "It sounds like you're caught between being yourself and meeting others' expectations. What would staying true to yourself look like?"
   ✗ Bad: "What do you mean by normal?"

4. HANDLING SHORT RESPONSES:
   User: "but"
   ✓ Good: "I sense there's more you want to say. What's on your mind?"
   ✗ Bad: "but"
   
   User: "yes indeed"
   ✓ Good: "Tell me more about that."
   ✗ Bad: "yes indeed"

5. CRISIS RESOURCES (when needed):
   If user mentions self-harm, suicide, or scores indicate severe depression:
   - Take it seriously and express concern
   - Provide: 104 Suicide & Crisis Lifeline (call or text)
   - Encourage professional help immediately
   - Don't leave them without resources

RESPONSE STRUCTURE:
1. First sentence: Validate/reflect what they said (reference PHQ-8 context if relevant)
2. Second sentence: Explore deeper OR ask ONE specific follow-up
3. Keep responses under 3 sentences unless they share a lot

DAIC-WOZ STYLE EXAMPLES (STYLE REFERENCE ONLY - NOT CONTENT):
{style_examples}

FULL CONVERSATION HISTORY (THIS USER, THIS SESSION):
{history}

CURRENT USER MESSAGE:
{question}

RESPOND AS A THERAPIST:
- Be warm but professional
- Validate first, explore second
- Reference PHQ-8 scores when relevant
- Adapt your approach based on severity level
- One clear direction per response
- Natural, conversational tone
- Show don't tell empathy
- Maintain continuity across the entire conversation
- Take mental health concerns seriously
Human: 
""",
    input_variables=["phq8_context", "style_examples", "history", "question"]
)

chain = THERAPIST_PROMPT | llm

# ---------------- HELPER FUNCTION TO GET LATEST PHQ-8 ----------------
def get_latest_phq8_score(conversation_id: int, db: Session):
    """Get the most recent PHQ-8 score for this conversation."""
    latest_assessment = (
        db.query(PHQ8Assessment)
        .filter(PHQ8Assessment.conversation_id == conversation_id)
        .order_by(PHQ8Assessment.created_at.desc())
        .first()
    )
    
    if latest_assessment:
        return {
            'score': latest_assessment.total_score,
            'severity': latest_assessment.severity,
            'binary': latest_assessment.binary
        }
    return None

# ---------------- RESPONSE FUNCTION WITH PHQ-8 AWARENESS ----------------
def get_response(
    message: str,
    conversation_id: int,
    db: Session
) -> str:
    """
    Generates a therapist-style response.
    Memory:
    - Smart windowing: First 3 + Last 20 messages for long conversations
    - ALL messages for short conversations
    - Full context awareness
    - PHQ-8 score awareness
    RAG:
    - DAIC-WOZ used ONLY for style
    """
    try:
        from phq8_therapeutic_responses import get_phq8_therapeutic_context
        
        cleaned = message.strip().lower()
        
        # -------- FETCH ALL MESSAGES FROM DATABASE --------
        all_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        
        # -------- SMART MEMORY WINDOWING --------
        if len(all_messages) <= 23:
            past_messages = all_messages
        else:
            past_messages = all_messages[:3] + all_messages[-20:]
        
        # Build conversation history
        history = ""
        for m in past_messages:
            role = "User" if m.role == "user" else "Assistant"
            history += f"{role}: {m.content}\n"
        
        # -------- GET PHQ-8 CONTEXT --------
        phq8_data = get_latest_phq8_score(conversation_id, db)
        
        if phq8_data:
            phq8_context = get_phq8_therapeutic_context(
                phq8_data['score'], 
                phq8_data['severity']
            )
        else:
            phq8_context = "No PHQ-8 assessment data available for this conversation."
        
        # -------- IMPROVED SHORT INPUT HANDLING --------
        if len(cleaned.split()) <= 2 and not history.strip():
            return "I'm here to listen. What's on your mind today?"
        
        # -------- SMARTER RAG TRIGGER --------
        emotional_keywords = {
            "feel", "feeling", "felt", "depressed", "anxious", "sad", "worried",
            "scared", "angry", "lonely", "stressed", "overwhelmed", "hopeless",
            "tired", "sleep", "insomnia", "panic", "fear", "grief", "loss",
            "help", "struggle", "struggling", "hurt", "pain", "crying", "cry",
            "attitude", "people", "normal", "change", "myself", "behavior",
            "think", "thought", "thoughts", "mind", "issue", "issues", "problem",
            "suicide", "suicidal", "kill", "die", "death", "harm", "hurt myself"
        }
        
        # Use RAG if emotional content OR substantive message
        has_emotional_content = any(word in cleaned.split() for word in emotional_keywords)
        use_rag = has_emotional_content or len(cleaned.split()) >= 5
        
        # -------- CRISIS DETECTION --------
        crisis_keywords = {"suicide", "suicidal", "kill myself", "want to die", "end it all", "no point living"}
        has_crisis_content = any(phrase in cleaned for phrase in crisis_keywords)
        
        if has_crisis_content:
            return """I'm really concerned about what you're sharing. Your safety is the most important thing right now.

Please reach out for immediate help:
- Call or text 104 (Suicide & Crisis Lifeline) - available 24/7
- Text "HELLO" to 104 (Crisis Text Line)
- Go to your nearest emergency room
- Call 100 if you're in immediate danger

You don't have to go through this alone. These feelings can get better with the right support. Will you reach out to one of these resources right now?"""
        
        # -------- DAIC-WOZ STYLE RETRIEVAL --------
        if use_rag:
            docs = vector_store.similarity_search(message, k=2)
            if docs:
                style_examples = "\n\n".join(d.page_content for d in docs)
            else:
                style_examples = "(No relevant style examples - use neutral supportive tone)"
        else:
            style_examples = "(Short response - use natural conversational tone)"
        
        # -------- LLM CALL WITH FULL CONTEXT + PHQ-8 AWARENESS --------
        result = chain.invoke({
            "phq8_context": phq8_context,
            "style_examples": style_examples,
            "history": history.strip(),
            "question": message
        })
        
        return result.content.strip() if result else "I'm here with you."
    
    except Exception as e:
        import traceback
        print(f"RAG ERROR: {e}")
        print(traceback.format_exc())
        return "I'm here with you. Could you tell me a bit more about what's on your mind?"