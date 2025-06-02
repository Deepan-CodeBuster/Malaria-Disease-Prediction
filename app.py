import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

# Load model
with open('malaria_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Malaria Detection from RBC Cell Images")
st.write("Upload a cell image to check for malaria.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((64, 64))  # Resize image to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)

    if st.button("Predict"):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        confidence = prediction * 100  # Convert to percentage

        if prediction > 0.5:
            st.success(f"No Malaria Detected ({confidence:.2f}% confidence)")
        else:
            st.error(f"Malaria Detected ({100 - confidence:.2f}% confidence)")
