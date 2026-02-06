import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.6,
    groq_api_key=GROQ_API_KEY
)


def get_response(message: str) -> str:
    docs = vector_store.similarity_search(message, k=3)
    context = "\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        template="""
You are an empathetic mental health assistant.
Use the context but do not give medical diagnosis.

Context:
{context}

User:
{question}

Assistant:
""",
        input_variables=["context", "question"]
    )

    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "question": message
    })

    return result.content
