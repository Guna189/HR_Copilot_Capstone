# from agents.intent_agent import classify_intent
# from agents.policy_rag_agent import retrieve_policies
# from agents.case_agent import analyze_case
# from agents.reasoning_agent import reason_decision
# from agents.confidence_agent import evaluate_confidence

# def filter_wrong_selection_docs(query, docs):
#     q = query.lower()

#     # If asking about interview PROCESS (not promotion)
#     if "interview" in q and "promotion" not in q:
#         filtered = [
#             d for d in docs
#             if "promotion" not in d.page_content.lower()
#             and "eligibility score" not in d.page_content.lower()
#             and "merit list" not in d.page_content.lower()
#         ]
#         return filtered if filtered else docs

#     return docs


# def handle_query(query: str):
#     intent = classify_intent(query)
#     print(intent)

#     # policies = retrieve_policies(query)
#     policies = retrieve_policies(query)
#     policies = filter_wrong_selection_docs(query, policies)

#     print(policies)

#     # Always produce an answer, never raw docs
#     case = analyze_case(query)
#     print(case)

#     decision = reason_decision(
#         query=query,
#         policies=policies,
#         case=case
#     )

#     confidence = evaluate_confidence(decision)

#     return {
#     "intent": intent,
#     "answer": decision,
#     "confidence": confidence,
#     "sources": list({p.metadata.get("page", "N/A") for p in policies})
#     }

from langgraph.graph import StateGraph, END
from graph import HRState
from graph_nodes import *

def build_graph():
    graph = StateGraph(HRState)

    graph.add_node("intent", intent_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("case", case_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("confidence", confidence_node)

    graph.set_entry_point("intent")

    graph.add_edge("intent", "retrieval")
    graph.add_edge("retrieval", "case")
    graph.add_edge("case", "reasoning")
    graph.add_edge("reasoning", "confidence")
    graph.add_edge("confidence", END)

    return graph.compile()

hr_graph = build_graph()

def handle_query(query: str):
    result = hr_graph.invoke({"query": query})
    return {
        "intent": result["intent"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"]
    }
