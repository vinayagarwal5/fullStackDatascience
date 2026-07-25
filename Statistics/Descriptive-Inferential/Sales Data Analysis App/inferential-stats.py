# Inferential Statistics

import data as data
import streamlit as st 
import numpy as np
import scipy.stats as stats
import descriptive as ds
import matplotlib.pyplot as plt
import seaborn as sns

sales_data = data.generate_data()


sales_mean = sales_data['sale_qty'].mean()
sales_median = sales_data['sale_qty'].median()
sales_mode = sales_data['sale_qty'].mode()[0]

sales_std = sales_data['sale_qty'].std()
sales_sqrt_len = np.sqrt(len(sales_data['sale_qty']))
df = len(sales_data['sale_qty'])-1

# t-score for the confidence level
confidence_level = 0.95
sample_standard_error = sales_std/sales_sqrt_len
t_score = stats.t.ppf((1+confidence_level)/2 , df)
margin_of_error = t_score * sample_standard_error
confidence_interval = [sales_mean - margin_of_error , sales_mean + margin_of_error]

st.subheader("Confidence Interval for Mean Units Sold")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Lower Bound", value=confidence_interval[0])
with col2:
    st.metric(label="Upper Bound", value=confidence_interval[1])

# Hypothesis Testing

t_statistic, p_value = stats.ttest_1samp(sales_data['sale_qty'], 20)

st.subheader("Hypothesis Testing (t-test)")
st.write(f"T-statistic: {t_statistic}, P-value: {p_value}")

if p_value < 0.05:
    st.write("Reject the null hypothesis: The mean units sold is significantly different from 20.")
else:
    st.write("Fail to reject the null hypothesis: The mean units sold is not significantly different from 20.")


# Visualizations
st.subheader("Visualizations")
cat_ds = ds.group_cat()

col1, col2 = st.columns(2)
with col1 : 
    # Histogram of units sold
    plt.figure(figsize=(10, 6))
    sns.histplot(sales_data['sale_qty'], bins=10, kde=True)
    plt.axvline(sales_mean, color='red', linestyle='--', label='Mean')
    plt.axvline(sales_median, color='blue', linestyle='--', label='Median')
    plt.axvline(sales_mode, color='green', linestyle='--', label='Mode')
    plt.title('Distribution of Qty Sold')
    plt.xlabel('Units Sold')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(plt)
with col2 :
    # Boxplot for units sold by category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y='sale_qty', data=sales_data)
    plt.title('Boxplot of Units Sold by Category')
    plt.xlabel('Category')
    plt.ylabel('Units Sold')
    st.pyplot(plt)

with col1:
    # Bar plot for total units sold by category

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Category', y='Total Units Sold', data=cat_ds)
    plt.title('Total Units Sold by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Units Sold')
    st.pyplot(plt)

