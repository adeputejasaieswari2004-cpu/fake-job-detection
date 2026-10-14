import streamlit as st
import pandas as pd

st.title("AI-Based Fake Job Detection System")

st.write("Upload a job dataset to analyze job postings.")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(data.head())

    st.write("Dataset Information")
    st.write("Total Rows:", data.shape[0])
    st.write("Total Columns:", data.shape[1])