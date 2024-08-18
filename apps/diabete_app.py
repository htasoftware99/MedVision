import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open("models/diabetes.pkl", "rb"))
scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))

def app():
    st.title("Diabetes Prediction")

    # Create a form for user input
    with st.form(key='diabetes_form'):
        # User input via sliders
        pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=0)
        glucose = st.slider('Glucose Level', min_value=0, max_value=200, value=0)
        blood_pressure = st.slider('Blood Pressure', min_value=0, max_value=200, value=0)
        skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=0)
        insulin = st.slider('Insulin Level', min_value=0, max_value=900, value=0)
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=0.0)
        dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0)
        age = st.slider('Age', min_value=0, max_value=120, value=0)

        # Add a submit button
        submit_button = st.form_submit_button("Predict")

    # Prepare the input array for prediction
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Scale the input data
    user_input_scaled = scaler.transform(user_input)

    # Prediction
    if submit_button:
        prediction = model.predict(user_input_scaled)
        
        if prediction[0] == 1:
            st.error("Unfortunately, there is diabetes.")
        else:
            st.success("Do not worry, there is no diabetes.")

if __name__ == "__main__":
    app()