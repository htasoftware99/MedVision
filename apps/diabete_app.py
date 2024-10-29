import streamlit as st
import numpy as np
import pickle
import json
import requests

# Load the trained model and scaler
model = pickle.load(open("models/diabetes.pkl", "rb"))
scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))

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

def app():
    st.title("Diabetes Prediction")

    # ELI5 seçeneği ekleyin
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")

    # Kullanıcı formu
    with st.form(key='diabetes_form'):
        pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=0)
        glucose = st.slider('Glucose Level', min_value=0, max_value=200, value=0)
        blood_pressure = st.slider('Blood Pressure', min_value=0, max_value=200, value=0)
        skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=0)
        insulin = st.slider('Insulin Level', min_value=0, max_value=900, value=0)
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=0.0)
        dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0)
        age = st.slider('Age', min_value=0, max_value=120, value=0)
        
        submit_button = st.form_submit_button("Predict")

    # Tahmin için veriyi hazırlama
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)

    # Tahmin ve açıklama
    if submit_button:
        prediction = model.predict(user_input_scaled)

        if prediction[0] == 1:
            result_message = "Unfortunately, there is diabetes."
            st.error(result_message)
            if eli5_mode:
                prompt = "A diabetes diagnosis was made based on the test results. Please explain it like I'm 5."
            else:
                prompt = "A diabetes diagnosis was made based on the test results. Please provide a detailed explanation."
        else:
            result_message = "Do not worry, there is no diabetes."
            st.success(result_message)
            if eli5_mode:
                prompt = "No diabetes detected based on the test results. Please explain it like I'm 5."
            else:
                prompt = "No diabetes detected based on the test results. Please provide a detailed explanation."

        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

if __name__ == "__main__":
    app()
