import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open("breast_cancer_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Breast Cancer Prediction App")
st.write("Predict whether a tumor is Malignant or Benign using Logistic Regression.")

# Initialize data
data = None

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file with features", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    if st.button("Predict on Uploaded Data"):
        predictions = model.predict(data)
        data["Prediction"] = ["Malignant" if p == 0 else "Benign" for p in predictions]
        st.write("Predictions:")
        st.dataframe(data)

        st.download_button(
            label="Download Predictions as CSV",
            data=data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

# Manual input
st.subheader("Manual Input for Prediction")

default_features = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_worst","texture_worst","perimeter_worst",
    "area_worst","smoothness_worst","compactness_worst","concavity_worst",
    "concave points_worst","symmetry_worst","fractal_dimension_worst"
]

features = list(data.columns) if data is not None else default_features

input_data = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

if st.button("Predict Manual Input"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    pred_label = "Malignant" if prediction[0] == 0 else "Benign"
    st.write(f"Prediction: {pred_label}")

# Visualization
if data is not None and "Prediction" in data.columns:
    st.subheader("Prediction Distribution")
    plt.figure(figsize=(6,4))
    sns.countplot(x="Prediction", data=data)
    plt.title("Prediction Distribution")
    st.pyplot(plt)
