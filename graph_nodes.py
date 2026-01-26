from agents.intent_agent import classify_intent
from agents.policy_rag_agent import retrieve_policies
from agents.case_agent import analyze_case
from agents.reasoning_agent import reason_decision
from agents.confidence_agent import evaluate_confidence

def intent_node(state):
    intent = classify_intent(state["query"])
    return {"intent": intent}

def retrieval_node(state):
    policies = retrieve_policies(state["query"])
    return {"policies": policies}

def case_node(state):
    case = analyze_case(state["query"])
    return {"case": case}

def reasoning_node(state):
    answer = reason_decision(
        query=state["query"],
        policies=state["policies"],
        case=state["case"]
    )
    sources = list({p.metadata.get("page", "N/A") for p in state["policies"]})
    return {"answer": answer, "sources": sources}

def confidence_node(state):
    confidence = evaluate_confidence(state["answer"])
    return {"confidence": confidence}
