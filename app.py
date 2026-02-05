import streamlit as st
from orchestrator import handle_query
import traceback

# ----------------------------------
# Page Setup
# ----------------------------------
st.set_page_config(
    page_title="HR CoPilot",
    page_icon="💼",
    layout="centered"
)

st.title("💼 HR CoPilot")
st.caption("Policy Reasoning Agent for HR Queries")

# ----------------------------------
# Session State
# ----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------
# Render Chat History
# ----------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------
# Chat Input
# ----------------------------------
user_query = st.chat_input("Ask an HR policy question…")

if user_query:
    # User bubble (RIGHT)
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant bubble (LEFT)
    with st.chat_message("assistant"):
        with st.spinner("HR CoPilot is thinking…"):
            try:
                result = handle_query(
                    query=user_query,
                    chat_history=st.session_state.chat_history
                )

                if not isinstance(result, dict):
                    raise ValueError("Invalid response from reasoning engine")

                answer = result.get("answer", "No answer generated.")
                confidence = result.get("confidence", {})
                sources = result.get("sources", [])

                # -----------------------------
                # Copilot Response Layout
                # -----------------------------
                st.markdown("### 🤖 HR CoPilot")

                st.markdown(answer)

                st.success(
                    f"""
**Confidence**
- Score: **{confidence.get('confidence_score', 'N/A')}**
- Risk: **{confidence.get('risk_level', 'N/A')}**
"""
                )

                st.info(
                    f"**Sources:** {', '.join(map(str, sources)) if sources else 'Internal policy documents'}"
                )

                response_text = f"""
🤖 **HR CoPilot**

{answer}

Confidence:
Score: {confidence.get('confidence_score')}
Risk: {confidence.get('risk_level')}

Sources: {sources}
"""

            except Exception as e:
                error_message = str(e)
                traceback.print_exc()

                st.error(
                    f"""
🚫 **Service Unavailable**

Currently services are not running.  
Please contact **Gunavardhan** and mention:

`{error_message}`
"""
                )

                response_text = f"Service failure: {error_message}"

    # Save assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text
    })