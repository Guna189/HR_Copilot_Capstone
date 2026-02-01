from typing import TypedDict, List, Any

class HRState(TypedDict):
    query: str
    chat_history: List[dict]
    intent: str
    policies: List[Any]
    case: str
    answer: str
    confidence: dict
    sources: list
