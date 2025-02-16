 
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels (update based on your dataset)
CLASS_NAMES = ["Healthy", "Powdery Mildew", "Rust"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((225, 225))  # Resize as per your model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload an image of a plant leaf to detect possible diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Show prediction result
    st.subheader(f"🔍 Prediction: **{predicted_class}**")
