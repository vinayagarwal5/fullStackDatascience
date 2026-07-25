import data as data
import streamlit as st 

sales_data = data.generate_data()

# Descriptive Statistics
st.subheader("Descriptive Statistics")
ds = sales_data['sale_qty'].describe()
st.dataframe(ds)
def des_stats():
    sales_mean = sales_data['sale_qty'].mean()
    sales_median = sales_data['sale_qty'].median()
    sales_mode = sales_data['sale_qty'].mode()[0]
    return list((sales_mean,sales_median,sales_mode))

des_stats = des_stats()
st.write(f"Mean Units Sold: {des_stats[0]}")
st.write(f"Median Units Sold: {des_stats[1]}")
st.write(f"Mode Units Sold: {des_stats[2]}")

# Group statistics by category
def group_cat():
    cat_statistics = sales_data.groupby('category')['sale_qty'].agg(['sum','mean','std']).reset_index()
    cat_statistics.columns = ['Category', 'Total Units Sold', 'Average Units Sold', 'Std Dev of Units Sold']
    return cat_statistics


st.subheader("Group statistics by category")

st.dataframe(group_cat())

