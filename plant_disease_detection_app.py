import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ✅ Sidebar for model file uploads (New Feature)
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

    # Merge the split files into model.h5
    with open("model.h5", "wb") as outfile:
        for part in ["model_part1.h5", "model_part2.h5"]:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    st.sidebar.success("Files merged into model.h5 successfully!")

# ✅ Load the trained model only if it exists
model_path = "model.h5"
if os.path.exists(model_path):
    try:
        model = load_model(model_path, compile=False)  # Prevent unnecessary compilation warnings
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model = None
else:
    st.sidebar.error("Model file not found. Please upload both parts.")
    model = None

# ✅ Define class labels
labels = {0: 'Healthy', 1: 'Powdery Mildew', 2: 'Rust'}

# ✅ Function to make predictions
def predict_disease(image):
    try:
        img = load_img(image, target_size=(224, 224))  # Resize image
        img_array = img_to_array(img)  # Convert to array
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)  # Normalize correctly
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for prediction
        
        predictions = model.predict(img_array)[0]  # Get predictions
        st.write(f"🔍 Raw Model Predictions: {predictions}")  # Debugging line

        predicted_class = np.argmax(predictions)  # Get highest probability class
        confidence = predictions[predicted_class] * 100  # Convert to percentage
        
        return labels[predicted_class], confidence
    except Exception as e:
        return f"Error: {e}", 0

# ✅ Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to check for disease.")

# ✅ File uploader for image prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    if model is not None:
        predicted_label, confidence = predict_disease(uploaded_file)
        if "Error" in predicted_label:
            st.error(predicted_label)  # Show error message
        else:
            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("Model not loaded. Please check model file.")
