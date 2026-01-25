# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import getpass
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI


# file_path = "HRPolicyManual.pdf"
# loader = PyPDFLoader(file_path)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500, chunk_overlap=100, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))

from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")

sample_text = "Hello world!"
vector = emb.embed_query(sample_text)

print("Vector length:", len(vector))
