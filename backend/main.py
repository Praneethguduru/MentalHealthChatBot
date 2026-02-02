import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from rag import get_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

class Msg(BaseModel):
    user_id: str
    chat_id: str
    message: str

@app.get("/")
def home(): return {"status": "Online"}

@app.post("/chat")
async def chat(req: Msg):
    ai_text = get_response(req.message)
    
    # Save History
    supabase.table("messages").insert({"chat_id": req.chat_id, "role": "user", "content": req.message}).execute()
    supabase.table("messages").insert({"chat_id": req.chat_id, "role": "assistant", "content": ai_text}).execute()
    
    return {"response": ai_text}