import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DRIVE_ID = "1XlZArIYbtkG3_NRyP2hsRBViY0C5T67f"
MODEL_PATH = "model.weights.h5"  # model filename

st.title("Skin Lesion Classifier")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -----------------------------
# IMAGE UPLOADER
# -----------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    # Predict
    class_names = [
        'Melanoma → Cancer (malignant)',
        'Melanocytic Nevus → Usually benign (moles, not cancer)',
        'Basal Cell Carcinoma (BCC) → Cancer (skin cancer, usually slow-growing)',
        'Actinic Keratosis (AK) → Pre-cancerous (can turn into squamous cell carcinoma if untreated)']
    pred = model.predict(arr)[0]

    st.subheader("Prediction Result")

    if pred.shape[0] > 1:
        idx = int(np.argmax(pred))
        confidence = float(pred[idx])
        st.write(f"Top class: **{class_names[idx]}** ({confidence:.4f})")
        st.write("**Full probabilities:**")
        for i, p in enumerate(pred):
            st.write(f"{class_names[i]}: {p:.4f}")

        st.subheader("Simple Report")
        st.write(f"The model predicts **{class_names[idx]}** with **{confidence:.2%}** confidence.")
    else:
        p = float(pred[0])
        st.write(f"Probability: {p:.4f}")

# -----------------------------
# INSTRUCTIONS WHEN NO IMAGE
# -----------------------------
else:
    st.info("""
    ### How to use:
    1. Upload a clear skin lesion image (jpg, jpeg, png)
    2. Wait for the model to predict the class
    3. Review the prediction result and confidence
    """)
