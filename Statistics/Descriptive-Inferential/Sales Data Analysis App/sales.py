import data as data
import streamlit as st 

st.subheader("Data for Sales Analysis")
sales_data = data.generate_data()
st.dataframe(sales_data,hide_index=True)