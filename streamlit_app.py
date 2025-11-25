import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "1resnet_model.h5"  # make sure your model file is in same folder

st.title("Skin Lesion Classifier — Simple App")

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Image uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    # Predict
    class_names = ["melanoma", "nevus", "bcc", "ak"]  # EDIT your class names here
    pred = model.predict(arr)[0]

    st.subheader("Prediction Result")

    if pred.shape[0] > 1:
        idx = int(np.argmax(pred))
        confidence = float(pred[idx])
        st.write(f"Top class: {class_names[idx]} ({confidence:.4f})")
        st.write("\n**Full probabilities:**")
        for i, p in enumerate(pred):
            st.write(f"{class_names[i]}: {p:.4f}")

        st.subheader("Simple Report")
        st.write(f"The model predicts **{class_names[idx]}** with **{confidence:.2%}** confidence.")
    else:
        p = float(pred[0])
        st.write(f"Probability: {p:.4f}")
