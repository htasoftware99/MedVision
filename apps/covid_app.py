import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    st.write("Please upload x-ray image")
    
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Tahmin
        label = covid_tahmini_yap(rf, image, sc)
        if label == 0:
            st.error("Unfortunately, you are infected by covid 19")
        else:
            st.success("Everything is OK")

if __name__ == "__main__":
    app()
