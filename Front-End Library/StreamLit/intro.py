import streamlit as st

# Add title and text to the app
st.title("Welcome to the Calculator App")

# Add some text to the app
st.write("This app calculates the cube of a number.")

# Add header
st.header("Select a number")

# Create a slider to select a number
number = st.slider("Choose a number", 0, 100, 1)

# Display the selected number
st.subheader(f"You selected: {number}")
cube = number ** 3
st.write(f"The cube of {number} is: {cube}")