import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pothole_cnn_model.h5")

model = load_model()

st.title("🛣️ Road Damage & Pothole Detection System")
st.write("Upload a road image to check whether it's a normal road or contains a pothole.")

uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing the road surface...")
    
    # Preprocess image
    img = np.array(image)
    resized_img = cv2.resize(img, (128, 128)) / 255.0
    input_data = np.expand_dims(resized_img, axis=0)
    
    # Prediction
    prediction = model.predict(input_data)
    score = prediction[0][0]
    
    if score > 0.5:
        st.error(f"🚨 Result: Pothole Detected! (Damaged Road) [Confidence: {score:.2f}]")
    else:
        st.success(f"✅ Result: Normal Road Surface. [Confidence: {1 - score:.2f}]")