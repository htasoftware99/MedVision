import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import json
import requests
from sklearn.ensemble import RandomForestClassifier
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

# Eğitimli modeli ve scaler'ı yükleme
with open('models/covid.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('models/covid_scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

def covid_tahmini_yap(model, resim, scaler=None):
    # Resmi oku ve özellik vektörüne dönüştür
    resim = resim.convert("L")
    resim = resim.resize((28, 28))
    resim_duzen = np.array(resim).flatten()
    
    if scaler:
        resim_duzen = scaler.transform([resim_duzen])  # Veri normalizasyonu
    
    # Tahmin yap
    tahmin = model.predict(resim_duzen)
    return tahmin[0]

def app():
    st.title("COVID-19 Detection")
    st.write("Please upload an x-ray image")
    
    # ELI5 seçeneğini ekleyin
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Tahmin
        label = covid_tahmini_yap(rf, image, sc)
        
        if label == 0:
            st.error("Unfortunately, you are infected by COVID-19")
            if eli5_mode:
                prompt = "COVID-19 infection detected in the x-ray image. Please explain it like I'm 5."
            else:
                prompt = "COVID-19 infection detected in the x-ray image. Please provide a detailed explanation."
        else:
            st.success("Everything is OK")
            if eli5_mode:
                prompt = "No sign of COVID-19 infection in the x-ray image. Please explain it like I'm 5."
            else:
                prompt = "No sign of COVID-19 infection in the x-ray image. Please provide a detailed explanation."

        # Açıklamayı Gemma modelinden al
        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

if __name__ == "__main__":
    app()
