import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open("models/breast_cancer.pkl", "rb"))
scaler = pickle.load(open("models/breast_cancer_scaler.pkl", "rb"))

# Function to make predictions
def predict_diagnosis(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Malignant (M)" if prediction[0] == 1 else "Benign (B)"

def app():
    st.title("Breast Cancer Prediction Web App")
    st.write("Enter the features to get a prediction.")

    with st.form("breast_cancer_form"):
        texture_mean = st.number_input("Texture Mean", min_value=0.0, format="%.6f")
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, format="%.6f")
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.6f")
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.6f")
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, format="%.6f")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, format="%.6f")
        texture_se = st.number_input("Texture Se", min_value=0.0, format="%.6f")
        area_se = st.number_input("Area Se", min_value=0.0, format="%.6f")
        smoothness_se = st.number_input("Smoothness Se", min_value=0.0, format="%.6f")
        compactness_se = st.number_input("Compactness Se", min_value=0.0, format="%.6f")
        concavity_se = st.number_input("Concavity Se", min_value=0.0, format="%.6f")
        concave_points_se = st.number_input("Concave Points Se", min_value=0.0, format="%.6f")
        symmetry_se = st.number_input("Symmetry Se", min_value=0.0, format="%.2f")
        fractal_dimension_se = st.number_input("Fractal Dimension Se", min_value=0.0, format="%.6f")
        texture_worst = st.number_input("Texture Worst", min_value=0.0, format="%.6f")
        area_worst = st.number_input("Area Worst", min_value=0.0, format="%.6f")
        smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, format="%.6f")
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0, format="%.6f")
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, format="%.6f")
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, format="%.6f")
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, format="%.6f")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, format="%.6f")

        features = [texture_mean, smoothness_mean, compactness_mean,
                    concave_points_mean, symmetry_mean, fractal_dimension_mean,
                    texture_se, area_se, smoothness_se, compactness_se,
                    concavity_se, concave_points_se, symmetry_se,
                    fractal_dimension_se, texture_worst, area_worst,
                    smoothness_worst, compactness_worst, concavity_worst,
                    concave_points_worst, symmetry_worst, fractal_dimension_worst]

        submitted = st.form_submit_button("Predict")

    if submitted:
        diagnosis = predict_diagnosis(features)
        if diagnosis == "Malignant (M)":
            st.error("Unfortunately, you have malignant.")
        else:
            st.success("Do not worry, you are healthy.")

# Run the app
if __name__ == "__main__":
    app()