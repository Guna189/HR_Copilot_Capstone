import streamlit as st
from orchestrator import handle_query

st.set_page_config(page_title="HR CoPilot", layout="centered")
st.title("HR CoPilot – Policy Reasoning Agent")

query = st.text_area(
    "Enter employee query",
    placeholder="e.g. How many casual leaves are allowed?"
)

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            result = handle_query(query)
            print("Result:\n\n\n\n",result)

        st.subheader("Response")

        # If backend returns structured dict
        if isinstance(result, dict):
            st.write("**Answer:**")
            st.write(result.get("answer", "No answer generated"))

            st.write("**Confidence:**")
            st.json(result.get("confidence", {}))

            st.write("**Sources (page numbers):**")
            st.write(result.get("sources", []))

        # If backend returns plain text
        else:
            st.write(result)
