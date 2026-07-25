import streamlit as st
from ollama import Client

# Create Client instance
client = Client(host="http://localhost:11434")

st.set_page_config(
    page_title="Custom LLM model by Prakash Senapati - Ollama",
    layout="centered",
)

st.title("Mr Vinay Agrawal - Ollama App")

prompt = st.text_area("Enter you prompt:", height=200)

if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            response = client.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": "user", "content": prompt}],
            )

        # Extract assistant text in a version-safe way
        text = None
        if isinstance(response, dict):
            text = (
                response.get("message", {}).get("content")
                or response.get("content")
                or response.get("response")
            )
        else:
            text = getattr(getattr(response, "message", None), "content", None)

        st.success("Response generated successfully!")
        st.subheader("Response:")
        st.write(text if text else response)

