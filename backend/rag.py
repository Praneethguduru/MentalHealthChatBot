import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from models import Message

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

# ---------------- IMPROVED PROMPT ----------------
THERAPIST_PROMPT = PromptTemplate(
    template="""You are a skilled, empathetic mental health conversational assistant trained in active listening and therapeutic communication.

CORE THERAPEUTIC PRINCIPLES:
1. VALIDATE before you question - acknowledge emotions first
2. EXPLORE depth - move from surface to underlying feelings
3. REFLECT meaning - paraphrase to show understanding
4. AVOID repetition - never ask the same question twice
5. BE PRESENT - respond to what's said, not what you expect
6. REMEMBER context - use conversation history to build continuity

CRITICAL RULES - NEVER VIOLATE:
❌ NEVER simply echo short responses ("but" → "but")
❌ NEVER ask "What do you mean by X?" more than once per topic
❌ NEVER repeat the same question in different words
❌ NEVER give advice unless explicitly asked
❌ NEVER diagnose or label
❌ NEVER say "I understand" without demonstrating it
❌ NEVER ignore what was said earlier in the conversation

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

5. CURIOSITY WITHOUT INTERROGATION:
   Instead of: "What do you mean?" or "Can you explain?"
   Use: "Help me understand..." or "Say more about..." or "What comes up for you when..."

6. EMPATHIC CONFRONTATION (when needed):
   User: "I think whatever I do is normal but..."
   ✓ Good: "There seems to be a 'but' there - like part of you wonders if others see it differently. What's that about?"
   ✗ Bad: "What do you mean by normal?"

7. BUILDING ON PREVIOUS CONTEXT:
   If user mentioned "attitude issues" earlier and now says "should I change":
   ✓ Good: "You mentioned people saying you have attitude issues. It sounds like you're wondering if changing yourself is the answer. What feels right to you?"
   ✗ Bad: Ignoring the earlier context

RESPONSE STRUCTURE:
1. First sentence: Validate/reflect what they said (reference earlier context when relevant)
2. Second sentence: Explore deeper OR ask ONE specific follow-up
3. Keep responses under 3 sentences unless they share a lot

FORBIDDEN PHRASES:
- "What do you mean by..." (if already asked about that topic)
- "Can you clarify..."
- "I understand" (without showing how)
- "That's interesting"
- "Tell me more" (more than once in a row)

DAIC-WOZ STYLE EXAMPLES (STYLE REFERENCE ONLY - NOT CONTENT):
{style_examples}

FULL CONVERSATION HISTORY (THIS USER, THIS SESSION):
{history}

CURRENT USER MESSAGE:
{question}

RESPOND AS A THERAPIST:
- Be warm but professional
- Validate first, explore second
- Reference earlier parts of conversation when relevant
- One clear direction per response
- Natural, conversational tone
- Show don't tell empathy
- Maintain continuity across the entire conversation
Human: 
""",
    input_variables=["style_examples", "history", "question"]
)

chain = THERAPIST_PROMPT | llm

# ---------------- RESPONSE FUNCTION WITH SMART MEMORY ----------------
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
    RAG:
    - DAIC-WOZ used ONLY for style
    """
    try:
        cleaned = message.strip().lower()
        
        # -------- FETCH ALL MESSAGES FROM DATABASE --------
        all_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        
        # -------- SMART MEMORY WINDOWING --------
        # Strategy: Keep first 3 messages (context) + last 20 messages (recent conversation)
        if len(all_messages) <= 23:
            # Short conversation - use ALL messages
            past_messages = all_messages
        else:
            # Long conversation - keep beginning context + recent messages
            past_messages = all_messages[:3] + all_messages[-20:]
        
        # Build conversation history
        history = ""
        for m in past_messages:
            role = "User" if m.role == "user" else "Assistant"
            history += f"{role}: {m.content}\n"
        
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
            "think", "thought", "thoughts", "mind", "issue", "issues", "problem"
        }
        
        # Use RAG if emotional content OR substantive message
        has_emotional_content = any(word in cleaned.split() for word in emotional_keywords)
        use_rag = has_emotional_content or len(cleaned.split()) >= 5
        
        # -------- DAIC-WOZ STYLE RETRIEVAL --------
        if use_rag:
            docs = vector_store.similarity_search(message, k=2)
            if docs:
                style_examples = "\n\n".join(d.page_content for d in docs)
            else:
                style_examples = "(No relevant style examples - use neutral supportive tone)"
        else:
            style_examples = "(Short response - use natural conversational tone)"
        
        # -------- LLM CALL WITH FULL CONTEXT --------
        result = chain.invoke({
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