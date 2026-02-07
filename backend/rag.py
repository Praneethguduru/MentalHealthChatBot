import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# ---------------- LOAD ENV ----------------
load_dotenv()

CHROMA_DIR = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR STORE ----------------
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.35,  # slightly lower = less hallucination
    groq_api_key=GROQ_API_KEY
)

# ---------------- THERAPIST PROMPT ----------------
THERAPIST_PROMPT = PromptTemplate(
    template="""
You are a neutral, supportive mental health conversational assistant inspired by clinical interview style.

IMPORTANT:
The text below contains **examples from unrelated clinical interviews**.
They are provided ONLY to guide conversational tone and pacing.
They are NOT about the user.
They do NOT describe the user's past, feelings, or experiences.

STRICT RULES:
- Never treat examples as user history.
- Never imply prior conversations.
- Never say "earlier", "before", or "previously".
- Never assume emotions or events not stated by the user.
- Do NOT give advice, diagnoses, or summaries.

Style:
- Calm, grounded, natural.
- Reflection preferred over questions.
- If unsure, respond with a simple neutral acknowledgment.

UNRELATED INTERVIEW EXAMPLES (style reference only):
{examples}

User:
{question}

Assistant:
""",
    input_variables=["examples", "question"]
)

chain = THERAPIST_PROMPT | llm


# ---------------- RESPONSE FUNCTION ----------------
def get_response(message: str, new_session: bool = False) -> str:
    try:
        cleaned = message.strip().lower()

        # ---- very short / vague replies → NO RAG ----
        if cleaned in [
            "yes", "yeah", "ok", "okay",
            "a lot", "yes a lot", "maybe", "kind of"
        ]:
            return "Take your time. We can go at whatever pace feels right."

        # ---- disable RAG for greetings / identity ----
        no_rag_triggers = [
            "hi", "hello", "hey",
            "i am", "i'm", "my name is",
            "this is"
        ]

        use_rag = (
            not new_session
            and len(cleaned.split()) >= 6
        )

        for trigger in no_rag_triggers:
            if cleaned.startswith(trigger):
                use_rag = False
                break

        if use_rag:
            docs = vector_store.similarity_search(message, k=2)
            examples = "\n\n".join(d.page_content for d in docs)
        else:
            examples = ""

        result = chain.invoke({
            "examples": examples,
            "question": message
        })

        if result is None:
            return "I’m here with you."

        return result.content.strip()

    except Exception as e:
        print("RAG ERROR:", e)
        return "I’m here and listening."
