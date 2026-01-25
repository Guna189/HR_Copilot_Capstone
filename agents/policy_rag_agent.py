from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "vectorstore"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 12,
        "lambda_mult": 0.15
    }
)

def retrieve_policies(query: str):
    docs = retriever.invoke(query)
    return docs
