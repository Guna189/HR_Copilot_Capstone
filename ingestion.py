from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb

PDF_PATH = "data/HRPolicyManual.pdf"
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")

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

    print("🧠 Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("🌐 Connecting to Chroma Cloud...")
    client = chromadb.HttpClient(
        host="api.trychroma.com",
        port=443,
        ssl=True,
        headers={"x-chroma-token": CHROMA_API_KEY},
        tenant="a0168b05-e999-4d12-b417-e9c7e27c41ec",
        database="hr_manual"
    )

    collection = client.get_or_create_collection(
        name="hr_policy_collection"
    )

    print("📤 Uploading embeddings to Chroma Cloud...")
    for i, doc in enumerate(chunks):
        vector = embeddings.embed_query(doc.page_content)

        collection.add(
            ids=[str(i)],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            embeddings=[vector]
        )

        if i % 50 == 0:
            print(f"Uploaded {i}/{len(chunks)}")

    print("✅ Upload completed!")

if __name__ == "__main__":
    ingest_pdf()
