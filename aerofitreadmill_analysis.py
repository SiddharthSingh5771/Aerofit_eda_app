# Importing Libraries 
import numpy as np
import pandas as pd 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt 
import io

# Setting the Page Configuration of Streamlit Dashboard
st.set_page_config(page_title="Aerofit Treadmill Analysis", layout="wide")
st.title("Aerofit Treadmill Data Analysis Dashboard")

# Upload the Dataset
uploaded_file = st.file_uploader("Please upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic Data Analysis 
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Shape of the Dataset
    st.subheader("Shape of the Dataset")
    st.write("Number of rows and columns in the dataset are:", df.shape)
    st.write("Column names of my dataset are:", df.columns.tolist())

    # Checkboxes
    st.subheader("Statistics of the Dataset")
    data_info = st.checkbox("Show data Information")
    missing_value = st.checkbox("Show Missing Values")
    statistics = st.checkbox("Show the Statistical Summary of the dataset")

    if data_info:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    if missing_value:
        st.write("Missing values of the Dataset are: ", df.isna().sum())

    if statistics:
        st.write("Dataset Statistics are:", df.describe())

    # Visual Analysis
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    st.write("Numeric Columns:", numeric_cols)
    st.write("Categorical Columns:", categorical_cols)

    # Count Plot (Numeric)
    st.subheader("Count Plot for Numeric Column")
    selected_cols = st.selectbox("Select a numeric column: ", numeric_cols)
    fig, ax = plt.subplots()
    sns.countplot(x=df[selected_cols], ax=ax)
    st.pyplot(fig)

    # Count Plot (Categorical)
    st.subheader("Count Plot for Categorical Column")
    cat_cols = st.selectbox("Select a categorical column:", categorical_cols)
    fig, ax = plt.subplots()
    sns.countplot(x=df[cat_cols], ax=ax)
    st.pyplot(fig)

    # Box Plot
    st.subheader("Box Plot for Numeric Columns")
    selected_numeric_column = st.selectbox("Select a numeric column for box plot", numeric_cols)
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=df[selected_numeric_column], ax=ax_box)
    st.pyplot(fig_box)

    # Histogram
    st.subheader("Histogram for Numeric Columns")
    selected_hist_column = st.selectbox("Select a numeric column for histogram", numeric_cols)
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[selected_hist_column], kde=True, ax=ax_hist)
    st.pyplot(fig_hist)

    # Bi-variate Analysis
    st.subheader("Bi-variate Analysis: Categorical vs Numeric")
    num_col = st.selectbox("Select a numeric column:", numeric_cols, key="num_col")
    cat_col = st.selectbox("Select a categorical column:", categorical_cols, key="cat_col")
    fig_bi, ax_bi = plt.subplots()
    sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax_bi)
    st.pyplot(fig_bi)

    #Multi -Variate Analysis
    #Heatmap of Our dataset to check the co-relation
    st.subheader("Co-relation Heatmap")
    fig, ax= plt.subplots(figsize =(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot = True, cmap = "magma", ax=ax)
    st.pyplot(fig)

    #Pair - Plot
    st.subheader("Pair Plot of our Dataset")
    fig= sns.pairplot(df[numeric_cols])
    st.pyplot(fig)

else: 
    st.write("Please upload the dataset first for the exploratory data analysis")
