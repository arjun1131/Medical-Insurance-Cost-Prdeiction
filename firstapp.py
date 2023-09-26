import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import streamlit as st

# Streamlit App Header
st.title('Machine Learning Model Deployment')

# Load your dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the dataset:")
    st.write(data.head())

    # Data preprocessing
    # Add your data preprocessing code here, e.g., handling missing values, feature engineering.

    # Split the data into training and testing sets
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with your target variable name
    y = data['target_column']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    st.write("Training a Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"Accuracy: {accuracy}")
    st.write(f"R-squared: {r2}")
    st.write(f"Mean Absolute Error: {mae}")
