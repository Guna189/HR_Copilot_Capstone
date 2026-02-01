import streamlit as st
from orchestrator import handle_query

st.set_page_config(page_title="HR CoPilot", layout="centered")
st.title("💼 HR CoPilot – Policy Reasoning Agent")

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Ask an HR question...")

if user_query:
    # Store user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = handle_query(
                query=user_query,
                chat_history=st.session_state.chat_history
            )

        # Format response
        if isinstance(result, dict):
            answer = result.get("answer", "No answer generated")

            response_text = f"""
**Answer:**  
{answer}

**Confidence:**  
- Score: {result['confidence'].get('confidence_score')}
- Risk: {result['confidence'].get('risk_level')}

**Sources:**  
{', '.join(map(str, result.get('sources', [])))}
"""
        else:
            response_text = str(result)

        st.markdown(response_text)

    # Store assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text
    })
