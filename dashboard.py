import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the trained model
with open("breast_cancer_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Breast Cancer Prediction App")
st.write("Predict whether a tumor is Malignant or Benign using logistic regression.")

uploaded_file = st.file_uploader("Upload CSV file with features", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    if st.button("Predict on Uploaded Data"):#button to trigger prediction
        predictions = model.predict(data)
        data["Prediction"] = ["Malignant" if p==0 else "Benign" for p in predictions]
        st.write("Predictions:")
        st.dataframe(data)

        st.download_button(
            label="Download Predictions as CSV",
            data=data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )


st.subheader("Manual Input for Prediction")


features = list(data.columns) if uploaded_file else [
    "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se",
    "concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
    "area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
    "symmetry_worst","fractal_dimension_worst"
]

input_data = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

if st.button("Predict Manual Input"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    pred_label = "Malignant" if prediction[0]==0 else "Benign"
    st.write(f"Prediction: {pred_label}")

#visualizations
st.subheader("Prediction Distribution (CSV Upload)")
if uploaded_file:
    sns.countplot(x="Prediction", data=data)
    plt.title("Prediction Distribution")
    st.pyplot(plt)
