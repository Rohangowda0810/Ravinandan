import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ✅ Load the trained model
model_path = "C:/Users/RAVINANDAN/model.h5"  # Update with the correct path if needed
model = load_model(model_path)

# ✅ Define class labels
labels = {0: 'Healthy', 1: 'Powdery Mildew', 2: 'Rust'}

# ✅ Function to make predictions
def predict_disease(image):
    img = load_img(image, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for prediction
    
    predictions = model.predict(img_array)[0]  # Get predictions
    predicted_class = np.argmax(predictions)  # Get highest probability class
    confidence = predictions[predicted_class] * 100  # Convert to percentage
    
    return labels[predicted_class], confidence

# ✅ Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to check for disease.")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    predicted_label, confidence = predict_disease(uploaded_file)
    
    # Show results
    st.success(f"Prediction: **{predicted_label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
