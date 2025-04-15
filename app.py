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
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300, use_column_width=False)

    if st.button("Predict"):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.success("No Malaria Detected")
        else:
            st.error("Malaria Detected")
            
