from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import chromadb
import os
import time

# ----------------- Config -----------------
PDF_PATH = "data/HRPolicyManual.pdf"
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # for genai.Client()

if GEMINI_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not set! Use .env or environment variables.")

# ----------------- Initialize GenAI Client -----------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------- Ingest PDF -----------------
def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("🌐 Connecting to Chroma Cloud...")
    chroma_client = chromadb.HttpClient(
        host="api.trychroma.com",
        port=443,
        ssl=True,
        headers={"x-chroma-token": CHROMA_API_KEY},
        tenant="a0168b05-e999-4d12-b417-e9c7e27c41ec",
        database="hr_manual"
    )

    collection = chroma_client.get_or_create_collection(
        name="hr_policy_collection_gemini"
    )

    print("📤 Uploading embeddings to Chroma Cloud...")

    BATCH_SIZE = 90
    start_idx = 900
    for i in range(start_idx, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]

        # ----------------- Generate embeddings with Gemini -----------------
        embeddings_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )

        vectors = [e.values for e in embeddings_result.embeddings]  # list of embeddings

        # ----------------- Add to Chroma -----------------
        for j, doc in enumerate(batch):
            collection.add(
                ids=[str(i + j)],
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                embeddings=[vectors[j]]
            )

        print(f"Uploaded {i + len(batch)}/{len(chunks)} chunks")
        time.sleep(60)  # short pause to avoid rate limits

    print("✅ Upload completed!")

if __name__ == "__main__":
    ingest_pdf()
