from langchain_core.prompts import PromptTemplate
from llm import llm

prompt = PromptTemplate(
    input_variables=["decision"],
    template="""
Evaluate the reliability of the HR answer.

Answer:
{decision}

STRICT RULES:
- Return ONLY valid raw JSON
- Do NOT use markdown
- Do NOT use ``` or backticks
- Do NOT include any text outside JSON

JSON format:
{{
  "confidence_score": number between 0 and 1,
  "risk_level": "Low" | "Medium" | "High",
  "explanation": string
}}

Rule:
- Return raw json as text
"""
)

chain = prompt | llm

def evaluate_confidence(decision: str):
    return chain.invoke({"decision": decision})
