import streamlit as st


st.title("Sales Analysis")
st.set_page_config(layout="wide")

home_page = st.Page("sales.py", title="Sales Data", url_path="sales")
des_page = st.Page("descriptive.py", title="Sales Data Descriptive Analysis", url_path="descriptive")
inf_page = st.Page("inferential-stats.py", title="Sales Data Inferential Analysis", url_path="inferential-stats")

nav = st.navigation([home_page, des_page,inf_page])
nav.run()
