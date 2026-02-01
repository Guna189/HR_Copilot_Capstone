from langchain_core.prompts import PromptTemplate
from llm import llm

prompt = PromptTemplate(
    input_variables=["query", "policies", "case", "chat_context"],
    template="""
You are an HR Policy Assistant.
Answer the user's question using ONLY the policy excerpts provided.
Conversation Context (for continuity only):
{chat_context}

User Query:
{query}

Relevant Policy Excerpts:
{policies}

Instructions:
- Maintain conversational continuity
- Do NOT repeat previously answered facts unless required
- Be concise and professional
- Explain clearly in professional HR language
- Do NOT include unrelated benefits, schemes, or policies
- Do NOT mention missing employee details
- Do NOT show internal reasoning steps
- Structure the answer in short paragraphs or bullet points
- Do NOT infer timelines or conditions
- Do NOT use similar policies (e.g., promotion vs recruitment)
- If the answer is not explicitly stated, say:
  "The policy does not explicitly specify this."
"""
)


# LCEL chain (this replaces LLMChain)
chain = prompt | llm


def reason_decision(query, policies, case, chat_context=""):
    policy_text = "\n\n".join(
        f"- {p.page_content}" for p in policies
    )

    response = chain.invoke(
        {
            "query": query,
            "policies": policy_text,
            "chat_context": chat_context
        }
    )

    return response.content if hasattr(response, "content") else response
