import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import json
import requests

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

# Modeli yükleyin
model_path = "models/vgg_unfrozen.keras"  # Model yolunu belirtin
model = tf.keras.models.load_model(model_path)

# Sınıf etiketlerini tanımlayın
class_labels = {0: 'PNEUMONIA', 1: 'NORMAL'}

def app():
    st.title("Pneumonia vs Normal X-ray Classifier")
    
    # ELI5 seçeneğini ekleyin
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")
    
    if uploaded_file is not None:
        # Yüklenen resmi gösterin
        img = image.load_img(uploaded_file, target_size=(128, 128), color_mode='rgb')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Resmi tahmin için hazırlayın
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Tahmin yapın
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Tahmin sonucuna göre açıklama istemi oluşturun
        if predicted_label == 'NORMAL':
            st.success(f"Prediction: {predicted_label}")
            if eli5_mode:
                prompt = "The X-ray image shows no signs of pneumonia. Please explain it like I'm 5 years old."
            else:
                prompt = "The X-ray image shows no signs of pneumonia. Please provide a detailed explanation."
        else:
            st.error(f"Prediction: {predicted_label}")
            if eli5_mode:
                prompt = "The X-ray image shows signs of pneumonia. Please explain it like I'm 5 years old."
            else:
                prompt = "The X-ray image shows signs of pneumonia. Please provide a detailed explanation."

        # Açıklamayı Gemma modelinden al
        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

if __name__ == "__main__":
    app()
