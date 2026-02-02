import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from supabase import create_client

# These load from Render Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN5Y29pcGFoYXZzb291a2xmcWpzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTkzNzY1MywiZXhwIjoyMDg1NTEzNjUzfQ.xDu5SR4g_7SkqZ3iIp9tSHlABGY6Z4YV1tD4TbZ6PMw" 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = SupabaseVectorStore(
    client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents"
)

llm = ChatGroq(temperature=0.6, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

def get_response(message: str):
    # 1. Search DB for similar text from transcripts
    docs = vector_store.similarity_search(message, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 2. Prompt Engineering
    template = """
    You are an empathetic mental health assistant.
    Base your advice on the following therapy transcripts (context).
    
    Context: {context}
    
    User: {question}
    Assistant:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context, "question": message})