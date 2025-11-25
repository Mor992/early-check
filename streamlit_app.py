import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------
st.set_page_config(page_title="Simple Skin Cancer App", page_icon="🔬")

MODEL_ID = "1SvHgAenMvRpTZolC5GcagozNR_MprQ6-"   # your .keras file
MODEL_FILE = "resnet_model.keras"
IMG_SIZE = (224, 224)

# -----------------------------------------------------
# DOWNLOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
    return tf.keras.models.load_model(MODEL_FILE)

model = load_model()

st.title("Simple Skin Cancer Classifier")

# -----------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------
def predict(model, pil_img):
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    cls = 1 if prob > 0.5 else 0
    return cls, prob

# -----------------------------------------------------
# UI
# -----------------------------------------------------
uploaded = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=300)

    if st.button("Analyze"):
        cls, prob = predict(model, img)

        st.subheader("Result")

        if cls == 1:
            st.error(f"⚠️ Cancer Detected — Confidence {prob*100:.2f}%")
        else:
            st.success(f"✓ No Cancer Detected — Confidence {(1-prob)*100:.2f}%")

st.caption("Simple version — no Grad-CAM, no extra processing.")
