from langchain_core.prompts import PromptTemplate
from llm import llm

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an HR Case Analysis Assistant.

Extract employee-specific details ONLY if they are explicitly mentioned.
If information is missing, clearly list it.

Query:
{query}

Return valid JSON ONLY in this format:
{{
  "employee_type": string | null,
  "joining_date": string | null,
  "department": string | null,
  "location": string | null,
  "missing_info": [string]
}}
"""
)

chain = prompt | llm

def analyze_case(query: str):
    response = chain.invoke({"query": query})
    return response.content if hasattr(response, "content") else str(response)

