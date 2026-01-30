import re
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from llm import llm

# ---------------- CONFIG ----------------
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = "hr_manual"
COLLECTION_NAME = "hr_policy_collection"

EMBED_MODEL = "nomic-embed-text"

# ---------------- EMBEDDINGS ----------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# ---------------- CHROMA CLOUD CLIENT ----------------
client = chromadb.HttpClient(
    host="api.trychroma.com",
    port=443,
    ssl=True,
    headers={"x-chroma-token": CHROMA_API_KEY},
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collection = client.get_collection(COLLECTION_NAME)

# ---------------- RERANK PROMPT ----------------
rerank_prompt = PromptTemplate(
    input_variables=["query", "docs"],
    template="""
You are a semantic reranker.

User Query:
{query}

Candidate Policy Chunks:
{docs}

Select the 10 most relevant chunks.
Return ONLY their indices as comma-separated numbers.
"""
)

rerank_chain = rerank_prompt | llm

# ---------------- SAFE LLM RERANK ----------------
def llm_rerank(query, docs, top_n=10):
    """
    Reranks documents using LLM.
    Always safe against hallucinated indices.
    """
    if not docs:
        return []

    text = "\n\n".join(
        f"[{i}] {d.page_content}" for i, d in enumerate(docs)
    )

    resp = rerank_chain.invoke({
        "query": query,
        "docs": text
    })

    output = resp.content if hasattr(resp, "content") else str(resp)

    raw_indices = re.findall(r"\d+", output)

    valid_indices = []
    for i in raw_indices:
        idx = int(i)
        if 0 <= idx < len(docs):
            valid_indices.append(idx)

    if not valid_indices:
        return docs[:top_n]

    return [docs[i] for i in valid_indices[:top_n]]

# ---------------- CLOUD RETRIEVAL ----------------
def retrieve_policies(query: str, final_k=5):
    """
    Retrieval pipeline (Chroma Cloud):
    1. Vector similarity search
    2. Convert to LangChain Documents
    3. LLM reranking
    4. Return top-k
    """

    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=20
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    documents = [
        Document(
            page_content=text,
            metadata=meta or {}
        )
        for text, meta in zip(docs, metas)
    ]

    reranked = llm_rerank(query, documents, top_n=10)

    return reranked[:final_k]
