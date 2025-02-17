import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load MobileNetV2 pre-trained model
model = MobileNetV2(weights="imagenet")

# Title of the app
st.title("🌿 Plant Disease Detection App")

# File uploader for image (supports jpg, png, jpeg)
uploaded_image = st.file_uploader("📸 Upload a Plant Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Open the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize and normalize)
    image_resized = image.resize((224, 224))  # Resize image to model's input size
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Apply preprocessing for MobileNetV2

    # Display the processed image
    st.image(image_resized, caption="Processed Image (Resized)", use_column_width=True)

    # Edge Detection Filter (Optional)
    image_cv = np.array(image_resized)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)  # Apply Canny edge detection
    st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")

    # Prediction using the model (MobileNetV2)
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

    # Show predictions
    st.write("🔍 **Predictions:**")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}. {label} ({score*100:.2f}%)")

    # If you want to apply disease detection, this is where you'd load your own model
    # Example: 
    # model = load_model("your_plant_disease_model.h5")
    # disease_predictions = model.predict(image_array)
    # st.write(f"Detected Disease: {disease_predictions}")

