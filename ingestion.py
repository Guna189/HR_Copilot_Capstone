from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import time

PDF_PATH = "data/HRPolicyManual.pdf"
PERSIST_DIR = "vectorstore"

def ingest_pdf():
    print("📄 Loading PDF...", end=" ")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print("Done ✅")

    print("✂️ Splitting documents into chunks...", end=" ")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Done ✅ Total chunks created: {len(chunks)}")

    print("🧠 Initializing embeddings...", end=" ")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    print("Done ✅")

    print("📦 Creating vector store...", end=" ")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("Done ✅")

    print("💾 Persisting vector store to disk...", end=" ")
    print("Done ✅ PDF ingestion complete 🎉")

if __name__ == "__main__":
    ingest_pdf()
