import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ✅ Sidebar for model uploads
st.sidebar.title("Upload & Merge Model Files")

# Upload both model parts
part1 = st.sidebar.file_uploader("Upload model_part1.h5", type="h5")
part2 = st.sidebar.file_uploader("Upload model_part2.h5", type="h5")

# Check if both files are uploaded
if part1 and part2:
    st.sidebar.success("Both parts uploaded! Merging now...")

    # Save uploaded files
    with open("model_part1.h5", "wb") as f1:
        f1.write(part1.read())

    with open("model_part2.h5", "wb") as f2:
        f2.write(part2.read())

    # Merge model parts
    with open("model.h5", "wb") as outfile:
        for part in ["model_part1.h5", "model_part2.h5"]:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    st.sidebar.success("Files merged into model.h5 successfully!")

# ✅ Load and compile model
model_path = "model.h5"
model = None

if os.path.exists(model_path):
    model = load_model(model_path)
    model.compile()  # ✅ Ensure model is compiled
    st.sidebar.success("Model loaded and compiled successfully!")
else:
    st.sidebar.error("Model file not found. Please upload both parts.")

# ✅ Class labels
labels = {0: 'Healthy', 1: 'Powdery Mildew', 2: 'Rust'}

# ✅ Define preprocessing function outside the prediction function to avoid retracing
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)  # Convert to array
    
    # ✅ Try different normalizations
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)  # Standard normalization

    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    return img_array

# ✅ Prediction function
def predict_disease(image):
    img_array = preprocess_image(image)  # Use the preprocessed image
    predictions = model.predict(img_array)[0]  # Get predictions

    # Debugging: Print raw model output
    st.sidebar.text(f"Raw Model Predictions: {predictions}")

    predicted_class = np.argmax(predictions)  # Get highest probability class
    confidence = predictions[predicted_class] * 100  # Convert to percentage

    return labels[predicted_class], confidence

# ✅ Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to check for disease.")

# ✅ File uploader for image prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    if model:
        predicted_label, confidence = predict_disease(uploaded_file)
        st.success(f"Prediction: **{predicted_label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("Model not found. Please upload model files first.")
