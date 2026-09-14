import os
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Gemini apps", page_icon="🤖", layout="wide")
st.title("Prompt Engineering with Gemini")

# Gemini API key input
api_key = st.text_input("Enter your Gemini API Key:", type="password")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client(api_key=api_key)

    def retriever_info(query):
        return "about prime minister of india"

    def rag_query(query):
        retrieved_info = retriever_info(query)
        augmented_prompt = f"User query: {query}. Retrieved information: {retrieved_info}"

        response = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents=augmented_prompt,
            config=types.GenerateContentConfig(
                temperature=1,
                max_output_tokens=1000,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                top_k=50,
                stop_sequences=["End"],
            ),
        )
        return response.text.strip()

    user_input = st.text_input("Enter your query:")
    if user_input:
        with st.spinner("Generating response..."):
            answer = rag_query(user_input)
            st.write(answer)

        