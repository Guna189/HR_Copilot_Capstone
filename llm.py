# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(
#     model="phi3:mini",
#     temperature=0.2
# )

# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Set up your API Key
# Replace 'your_api_key_here' with your actual Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq model
# Common models: 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)