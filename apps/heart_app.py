import streamlit as st
import pandas as pd
import pickle
import json
import requests
from sklearn.preprocessing import StandardScaler

# Ollama API üzerinden gemma modeline istek gönderme
def get_gemma_explanation(prompt):
    try:
        ollama_endpoint = "http://localhost:11434/api/generate"
        payload = json.dumps({"model": "gemma:2b", "prompt": prompt, "stream": False})
        response = requests.post(ollama_endpoint, data=payload)
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama API: {str(e)}"

# Define the function for the Streamlit app
def app():
    # Set the title of the app
    st.title('Heart Disease Prediction App')

    # ELI5 seçeneği ekleyin
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")

    # Load the model and scaler
    model = pickle.load(open("models/heart_disease.pkl", "rb"))
    scaler = pickle.load(open("models/heart_disease_scaler.pkl", "rb"))

    # Create a form for user input
    with st.form(key='prediction_form'):
        st.header('Enter Your Data')
        
        age = st.slider("Age", min_value=1, max_value=80, value=30)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol = st.slider("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
        restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
        thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
        oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (1-3)", options=[1, 2, 3])

        # Add a submit button
        submit_button = st.form_submit_button('Make Prediction')

    # Process the input when the button is clicked
    if submit_button:
        # Convert categorical inputs to numerical values
        sex = 1 if sex == 'Male' else 0
        fbs = 1 if fbs == 'Yes' else 0
        exang = 1 if exang == 'Yes' else 0

        # Convert user input to a DataFrame
        user_input = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        # Scale the data
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input_scaled)

        # Display the result and prepare explanation prompt
        if prediction[0] == 1:
            st.error("Result: There is a risk of heart disease.")
            if eli5_mode:
                prompt = "The prediction indicates a risk of heart disease. Please explain like I'm 5."
            else:
                prompt = "The prediction indicates a risk of heart disease. Please provide a detailed explanation."
        else:
            st.success("Result: No risk of heart disease.")
            if eli5_mode:
                prompt = "No risk of heart disease detected. Please explain like I'm 5."
            else:
                prompt = "No risk of heart disease detected. Please provide a detailed explanation."

        # Get explanation from Gemma model
        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

# Call the function to run the app
if __name__ == "__main__":
    app()
