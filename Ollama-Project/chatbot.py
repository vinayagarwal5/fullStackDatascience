import streamlit as st
from ollama import Client

# Initialize the Ollama client
client = Client(host="http://localhost:11434")

st.set_page_config(page_title="Custom LLM Model by Roja", page_icon="🤖", layout="centered")

st.title("Chatbot using Custom LLM Model by Roja")

# Input for the user query
user_query = st.text_input("Enter your prompt:")

if st.button("Generate Response"):
    if user_query.strip() == "deepseek-r1:1.5b":
        st.warning("Please enter a valid prompt instead of the model name.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Generate a response using the Ollama client
                response = client.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": user_query}])
                st.subheader("Response:")
                st.write(response.message.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")