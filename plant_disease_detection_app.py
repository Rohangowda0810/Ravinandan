import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

# Title of the app
st.title("Plant Disease Detection App")

# File uploader for model architecture (.h5)
model_file = st.file_uploader("Upload Model Architecture (.h5)", type=["h5"])

# File uploader for model weights (.h5)
weights_file = st.file_uploader("Upload Model Weights (.h5)", type=["h5"])

# Button to trigger model loading after files are uploaded
if model_file and weights_file:
    try:
        # Load the model architecture without weights
        model = load_model(model_file, compile=False)

        # Load the weights separately
        model.load_weights(weights_file)

        # Output message
        st.write("Model and weights loaded successfully!")

        # Model summary (for debugging and confirmation)
        st.text_area("Model Summary", model.summary(), height=200)

        # Example: You can now use the model to make predictions if you have input data
        # For example, an image upload for prediction:

        uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

        if uploaded_image:
            # Preprocess the image for prediction (resize, normalization, etc.)
            # Assuming image preprocessing code is defined here:
            # You can use PIL or OpenCV to preprocess the image
            from PIL import Image
            import numpy as np

            image = Image.open(uploaded_image)
            image = image.resize((224, 224))  # Resize to model input shape (adjust as needed)
            image = np.array(image) / 255.0  # Normalize if needed
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Predict using the loaded model
            predictions = model.predict(image)

            # Show the predictions
            st.write(f"Prediction: {predictions}")
        
    except Exception as e:
        st.error(f"Error loading model or weights: {e}")

