# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings

# PERSIST_DIR = "vectorstore"

# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text"
# )

# vectordb = Chroma(
#     persist_directory=PERSIST_DIR,
#     embedding_function=embeddings
# )

# retriever = vectordb.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "fetch_k": 12,
#         "lambda_mult": 0.15
#     }
# )

# def retrieve_policies(query: str):
#     docs = retriever.invoke(query)
#     return docs

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from llm import llm

PERSIST_DIR = "vectorstore"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# Stage 1: broad recall
broad_retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)

# LLM Rerank prompt
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


def llm_rerank(query, docs, top_n=10):
    text = "\n\n".join(
        f"[{i}] {d.page_content}" for i, d in enumerate(docs)
    )

    resp = rerank_chain.invoke({
        "query": query,
        "docs": text
    })

    indices = [
        int(i.strip())
        for i in resp.split(",")
        if i.strip().isdigit()
    ]

    return [docs[i] for i in indices[:top_n]]


def retrieve_policies(query: str, final_k=5):
    # Stage 1
    top20 = broad_retriever.invoke(query)

    # Stage 2
    top10 = llm_rerank(query, top20, top_n=10)

    # Stage 3 (MMR)
    final = vectordb.similarity_search_by_vector(
        embedding=embeddings.embed_query(query),
        k=final_k,
        filter=None
    )

    return final
