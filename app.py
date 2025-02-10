import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2  # For resizing if needed

# Set page config for a more polished look
st.set_page_config(
    page_title="Waste Classifier",
    layout="wide",
    page_icon="♻️"
)

# Custom CSS styling for a modern design
st.markdown("""
    <style>
    .main {
        background: #F5F5F5;
        font-family: "Roboto", sans-serif;
    }
    .stTextInput > label {
        font-size: 1.1rem;
        color: #000000;
        font-weight: 500;
    }
    .stButton button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    # Update "MODEL_FILE_PATH" to your actual model file
    model = tf.keras.models.load_model(".waste_classification_model.h5")
    return model

model = load_model()

# Class labels — match with your training logic
CLASS_NAMES = ['Organic', 'Recyclable']

# Sidebar Navigation
st.sidebar.title("Waste Classifier Menu")
app_mode = st.sidebar.selectbox("Choose a page:", ["Home", "Predict"])

# Main Page Layout
def home_page():
    st.title("Waste Classification App ♻️")
    st.write("""
        This application classifies waste images as either Organic or Recyclable.
        Just upload an image to see the prediction!
    """)
    st.image(
        "https://i.ibb.co/28X3bPz/waste-classification.png",
        caption="Waste Classification",
        use_column_width=True
    )

def predict_page():
    st.title("Upload an Image to Classify")
    uploaded_file = st.file_uploader(
        "Select an image to upload", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False, width=400)
        
        # Convert and preprocess
        image = image.convert("RGB")
        image = image.resize((224, 224))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Inference
        prediction = model.predict(x)
        class_idx = np.argmax(prediction[0])
        class_label = CLASS_NAMES[class_idx]
        
        st.subheader(f"Prediction: {class_label}")
        st.write("Confidence Scores:")
        for i, label in enumerate(CLASS_NAMES):
            st.write(f"{label}: {prediction[0][i]*100:.2f}%")

# Routing logic
if app_mode == "Home":
    home_page()
elif app_mode == "Predict":
    predict_page()
