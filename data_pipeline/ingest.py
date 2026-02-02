import os
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Service Role Key needed for writing!
DATA_FOLDER = "clean_transcripts"
# ---------------------

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest():
    print(f"Scanning '{DATA_FOLDER}' for .txt files...")
    all_docs = []

    # Walk through all subfolders
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Reading: {file_path}")
                
                try:
                    # Read the text file
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()

                    # Skip empty files
                    if not text_content.strip():
                        continue

                    # Create Document with metadata (so we know which patient it is)
                    doc = Document(
                        page_content=text_content, 
                        metadata={"source": file, "folder": os.path.basename(root)}
                    )
                    all_docs.append(doc)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if len(all_docs) == 0:
        print("No .txt files found! Check your folder structure.")
        return

    # Split Text into Chunks (Important for RAG)
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(all_docs)

    # Upload
    print(f"Uploading {len(split_docs)} chunks to Supabase...")
    SupabaseVectorStore.from_documents(
        split_docs, 
        embeddings, 
        client=supabase, 
        table_name="documents", 
        query_name="match_documents"
    )
    print("✅ Success! Data is in the cloud.")

if __name__ == "__main__":
    ingest()