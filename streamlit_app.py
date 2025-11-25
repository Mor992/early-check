import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import zipfile
import os
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection")

DRIVE_ID = "14WsxVymQ6wHc7t7DmiLlVFpVm5cxu90F"   # your zip file
ZIP_FILE = "model.zip"
MODEL_FILE = "model.keras"   # change if your extracted file has a different name

# --------------------- DOWNLOAD + UNZIP ---------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):

        if not os.path.exists(ZIP_FILE):
            st.info("Downloading model zip from Google Drive...")
            url = f"https://drive.google.com/uc?id={DRIVE_ID}"
            gdown.download(url, ZIP_FILE, quiet=False)

        st.info("Extracting model...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as z:
            z.extractall()

    st.success("Model loaded!")
    return tf.keras.models.load_model(MODEL_FILE)


# --------------------- PREDICT ---------------------
def predict(model, img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    cls = 1 if prob > 0.5 else 0
    return cls, prob


# --------------------- UI ---------------------
st.title("Skin Cancer Detection")

model = load_model()

img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, width=300)

    if st.button("Analyze"):
        cls, prob = predict(model, img)

        if cls == 1:
            st.error(f"⚠️ Cancer detected — {prob*100:.2f}%")
        else:
            st.success(f"✓ No cancer detected — {(1-prob)*100:.2f}%")

