from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="phi3:mini",
    temperature=0.2
)

# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
