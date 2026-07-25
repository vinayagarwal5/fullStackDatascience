import pandas as pd 
import numpy as np 
import streamlit as st 

@st.cache_data
## Generate Synthetic Data
def generate_data():
    np.random.seed(42)
    sales_data = {
        'product_sku' : [f'SKU_{i}' for i in range(1,21)],
        'product_name' : [f'Product {i}' for i in range(1,21)],
        'category' : np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'],20),
        'sale_qty' : np.random.poisson(lam=20,size=20),
        'sale_date': pd.date_range(start='2026-01-01',periods=20,freq='D').date
    }
    return pd.DataFrame(sales_data)