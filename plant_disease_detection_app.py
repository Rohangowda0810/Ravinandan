import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load your custom-trained plant disease model (example path)
# If you don't have a custom model, train one using a dataset like PlantVillage.
model = load_model("path_to_your_plant_disease_model.h5")

# Title of the app
st.title("🌿 Plant Disease Detection App")

# File uploader for image (supports jpg, png, jpeg)
uploaded_image = st.file_uploader("📸 Upload a Plant Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize and normalize)
    image_resized = image.resize((224, 224))  # Resize image to model's input size
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Display the processed image
    st.image(image_resized, caption="Processed Image (Resized)", use_column_width=True)

    # Edge Detection Filter (Optional)
    image_cv = np.array(image_resized)
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)  # Apply Canny edge detection
    st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")

    # Prediction using the custom model
    predictions = model.predict(image_array)
    
    # Assuming the model returns class indices for Rust, Powdery Mildew, Healthy
    # Example: [Rust, Powdery Mildew, Healthy] = [0, 1, 2]
    class_labels = ["Healthy", "Rust", "Powdery Mildew"]
    
    # Get the predicted class index (assuming output is in softmax probabilities)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    predicted_confidence = np.max(predictions) * 100  # Confidence of the prediction

    # Show the predicted result
    st.write(f"🔍 **Prediction Result**: {predicted_label} ({predicted_confidence:.2f}%)")

