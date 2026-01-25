from langchain_core.prompts import PromptTemplate
from llm import llm

prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an HR query classifier.

Classify the user query into exactly ONE of the following categories:

- policy_explanation : User is asking to explain an HR policy or process
- policy_lookup      : User is asking for a specific factual value from policy
- decision_support   : User is asking for a decision based on employee context
- exception_case     : User is asking about an exception or special case
- insufficient_info  : Query cannot be answered from HR policy

Query:
{query}

Rules:
- Return ONLY the label
- Do not add explanations
"""
)

# LCEL chain (replaces LLMChain)
chain = prompt | llm


def classify_intent(query: str) -> str:
    response = chain.invoke({"query": query})
    label = response.content if hasattr(response, "content") else response
    return label.strip().lower().split()[0]
