import streamlit as st
import tensorflow as tf
from PIL import Image
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

def app():
    # Modeli yükle
    model = tf.keras.models.load_model('models/Brain Tumors Classifier.h5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Sınıf etiketlerini tanımlayın
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    # Görüntüyü ön işleme fonksiyonu
    def preprocess_image(image):
        # Görüntünün RGB modunda olduğundan emin olun
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Görüntüyü yeniden boyutlandırın
        img = image.resize((224, 224))

        # Görüntüyü numpy dizisine çevirin
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Görüntüyü normalize edin
        img_array = img_array / 255.0  

        # Batch boyutu ekleyin
        img_array = np.expand_dims(img_array, 0)

        return img_array

    # Streamlit uygulaması
    st.title("MRI Brain Tumor Detection")

    # ELI5 seçeneği
    eli5_mode = st.checkbox("Explain like I'm 5 (ELI5)")

    # Dosya yükleyici
    uploaded_file = st.file_uploader("Upload a MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Görüntü dosyasını açın
        image = Image.open(uploaded_file)

        # Görüntüyü gösterin
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Görüntüyü ön işleme tabi tutun
        img_array = preprocess_image(image)

        # Tahminleri yapın
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # Sonucu dönüştür
        if predicted_class == 'No Tumor':
            display_class = 'Tumor does not exist'
            st.success(f"Result: **{display_class}**")
            prompt = "The diagnosis is 'No Tumor'. Please explain like I'm 5 years old." if eli5_mode else "The diagnosis is 'No Tumor'. Please provide a detailed explanation."
        else:
            display_class = 'Tumor exists'
            st.error(f"Result: **{display_class}**")
            prompt = f"The diagnosis is '{predicted_class}'. Please explain like I'm 5 years old." if eli5_mode else f"The diagnosis is '{predicted_class}'. Please provide a detailed explanation."

        # Gemma modelinden açıklama al
        explanation = get_gemma_explanation(prompt)
        st.write(explanation)

# Uygulamayı çalıştır
if __name__ == "__main__":
    app()
