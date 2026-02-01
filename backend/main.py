import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import SupabaseVectorStore
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your website can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace * with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD SECRETS FROM ENVIRONMENT VARIABLES
# (We will set these in the Render Dashboard later)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. SETUP MODELS
# We use a lightweight embedding model that can run on small cloud instances
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# We use Llama 3 70B via Groq (Runs on the internet, not your PC)
llm = ChatGroq(
    temperature=0.6, 
    model_name="llama3-70b-8192", 
    groq_api_key=GROQ_API_KEY
)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    message: str

@app.get("/")
def home():
    return {"status": "Mental Health API is running"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # RAG: Search DAIZ-WOZ context
    try:
        docs = vector_store.similarity_search(req.message, k=3)
        context_text = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"Vector search failed: {e}")
        context_text = "No specific clinical context available."

    # Prompt
    template = """
    You are a compassionate, non-judgmental mental health support assistant. 
    You are trained on clinical dialogues (DAIZ-WOZ).
    
    GUIDELINES:
    1. Validate the user's feelings first.
    2. Use the context below to inform your advice, but keep it natural.
    3. If the user seems suicidal, provide emergency resources immediately.
    4. Keep responses concise (under 3 sentences) unless asked for more.
    
    CONTEXT FROM DATABASE:
    {context}

    USER: {question}
    ASSISTANT:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response_text = chain.run({"context": context_text, "question": req.message})

    # Save to Supabase (Secure History)
    supabase.table("messages").insert({
        "chat_id": req.chat_id,
        "role": "user",
        "content": req.message
    }).execute()

    supabase.table("messages").insert({
        "chat_id": req.chat_id,
        "role": "assistant",
        "content": response_text
    }).execute()

    return {"response": response_text}