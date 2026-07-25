import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tips = sns.load_dataset("tips")

# function create and display plot 
def display_plot(title, plot_func):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_func(ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    
# plot 

def scatter_plot(ax):
    sns.scatterplot(data=tips, x="total_bill", y="tip",hue="time", size="size", palette="deep",ax=ax)
    ax.set_title("Scatter plot of total bill vs tip")
        
def line_plot(ax):
    sns.lineplot(data=tips, x= 'size', y='total_bill', hue='sex',markers='o',ax=ax)
    ax.set_title("Line plot of total bill vs tip")

def bar_plot(ax):
    sns.barplot(data=tips, x='day', y='total_bill', hue = 'sex',palette='muted',ax=ax)
    ax.set_title("Barplot of Total Bill by Day")

def box_plot(ax):
    sns.boxplot(data=tips, x='day', y='tip', hue='smoker', palette='Set2',ax=ax)
    ax.set_title("Boxplot of Tips by Day and Smoker Status")
    
def violin_plot(ax):
    sns.violinplot(data=tips, x='day', y='total_bill', hue='time', split=True, palette='pastel',ax=ax)
    ax.set_title("Violin Plot of Total Bill by Day and Time")  

def count_plot(ax):
    sns.countplot(data=tips, x='day', hue='smoker', palette='dark',ax=ax)
    ax.set_title("Count Plot of Days by Smoker Status")
    
def reg_plot(ax):
    sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'s':50}, line_kws={'color':'red'},ax=ax)
    ax.set_title("Regression Plot of Total Bill vs Tip") 

def hist_plot(ax):
    sns.histplot(data=tips, x='total_bill', bins=20, kde=True, color='blue',ax=ax)
    ax.set_title("Histogram of Total Bill with KDE")
    
def strip_plot(ax):
    sns.stripplot(data=tips, x='day', y='tip', hue='sex', jitter=True, palette='Set1',ax=ax)
    ax.set_title("strip plot: Tips by data and gender")

def kde_plot(ax):
    sns.kdeplot(data=tips, x='total_bill',hue='sex', fill=True, palette='tab10',ax=ax)
    ax.set_title("kde plot:Total bill density by gender")


sns.set_theme(style="whitegrid")
st.set_page_config(layout="wide")

st.title("📊 Seaborn Learning Dashboard")


data = {
    "Question / Goal": [
        "See Distribution",
        "Find Outliers",
        "Compare Categories",
        "Count Categories",
        "Check Correlation",
        "Analyze Two Variables",
        "Add Trend Line",
        "Study All Features",
        "Analyze Time Trends",
        "Compare Distributions"
    ],
    "Recommended Plot": [
        "Histogram, KDE",
        "Box Plot",
        "Bar Plot",
        "Count Plot",
        "Heatmap",
        "Scatter Plot",
        "Regression Plot",
        "Pair Plot",
        "Line Plot",
        "Violin Plot"
    ],
    "Function": [
        "sns.histplot(), sns.kdeplot()",
        "sns.boxplot()",
        "sns.barplot()",
        "sns.countplot()",
        "sns.heatmap()",
        "sns.scatterplot()",
        "sns.regplot()",
        "sns.pairplot()",
        "sns.lineplot()",
        "sns.violinplot()"
    ]
}

df = pd.DataFrame(data)



with st.container(border=True):

    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

col1, col2, col3 = st.columns(3)
with st.container(border=False):
    with col1:
        display_plot("Scatter Plot", scatter_plot)
    with col2:
        display_plot("Line Plot", line_plot)
    with col3:
        display_plot("Bar Plot", bar_plot)

with st.container(border=False):
    with col1:
        display_plot("Box Plot", box_plot)
    with col2:
        display_plot("Violin Plot", violin_plot)
    with col3:
        display_plot("Count Plot", count_plot)
with st.container(border=False):
    with col1:
        display_plot("Regression Plot", reg_plot)
    with col2:
        display_plot("Histogram Plot", hist_plot)
    with col3:
        display_plot("Strip Plot", strip_plot)
with st.container(border=False):
    with col1:
        display_plot("KDE Plot", kde_plot)