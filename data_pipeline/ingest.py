import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 📌 BASE DIRECTORY = data_pipeline/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📁 Transcripts live INSIDE data_pipeline
DATA_FOLDER = os.path.join(BASE_DIR, "clean_transcripts")

# 📁 Vector DB stored in backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHROMA_DIR = os.path.join(PROJECT_ROOT, "backend", "chroma_db")

print("📁 Using transcript folder:", DATA_FOLDER)
print("📁 Using chroma directory:", CHROMA_DIR)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest():
    documents = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, file)
            print(f"📄 Reading {file}")

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file}
                    )
                )

    if not documents:
        raise ValueError("❌ No valid documents found. Check transcript files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # 🔒 SAFETY FILTER
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        raise ValueError("❌ No valid chunks after splitting.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"✅ Successfully ingested {len(chunks)} chunks into ChromaDB")

if __name__ == "__main__":
    ingest()
