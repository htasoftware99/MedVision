import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np

# Load the pre-trained model
model_path = "models/vgg_unfrozen.keras"  # Path to your model
model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = {0: 'PNEUMONIA', 1: 'NORMAL'}

def app():
    st.title("Pneumonia vs Normal X-ray Classifier")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        img = image.load_img(uploaded_file, target_size=(128, 128), color_mode='rgb')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Prepare the image for prediction
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Display result with appropriate message
        if predicted_label == 'NORMAL':
            st.success(f"Prediction: {predicted_label}")
        else:
            st.error(f"Prediction: {predicted_label}")
