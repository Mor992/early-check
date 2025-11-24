import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
MODEL_DRIVE_ID = "1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q"   # YOUR .keras MODEL
MODEL_FILENAME = "final_resnet_model.keras"
INPUT_SIZE = (224, 224)

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬", layout="wide")

# =========================================================
# DOWNLOAD + LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model..."):
            try:
                gdown.download(url, MODEL_FILENAME, quiet=False)
            except Exception as e:
                st.error(f"❌ Download error: {e}")
                return None

    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model(MODEL_FILENAME)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None


# =========================================================
# GRAD‑CAM++
# =========================================================
def compute_gradcam_plus_plus(model, image, layer_name="conv5_block3_out"):
    try:
        conv_layer = model.get_layer(layer_name)
    except:
        # fallback last conv
        def find_last_conv_layer(model):
    # Search for any Conv2D layer from the end
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer.name
    
    # If model uses ResNet blocks, try known names
            for name in ["conv5_block3_out", "post_relu"]:
                try:
                    model.get_layer(name)
                    return name
                except:
                    pass

            raise ValueError("No convolution layer found in model.")

last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None

    # GradCAM++
    grads2 = grads ** 2
    grads3 = grads ** 3

    alpha_num = grads2
    alpha_denom = grads2 * 2 + tf.reduce_sum(conv_out * grads3, axis=(1, 2), keepdims=True)
    alpha_denom = tf.where(alpha_denom == 0, tf.ones_like(alpha_denom), alpha_denom)

    alpha = alpha_num / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))
    cam = tf.reduce_sum(weights[..., None, None, :] * conv_out, axis=-1)

    cam = tf.nn.relu(cam)
    cam = cam[0].numpy()

    if cam.max() != 0:
        cam /= cam.max()

    return cam


def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = plt.cm.jet(heatmap)[..., :3]
    overlay = heatmap_color * 0.5 + (image / 255.0)
    return np.clip(overlay, 0, 1)


# =========================================================
# PREDICT
# =========================================================
def predict(model, pil_img):
    img = pil_img.resize(INPUT_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    cls = 1 if prob > 0.5 else 0

    cam = compute_gradcam_plus_plus(model, arr)
    overlay = overlay_heatmap(np.array(img), cam) if cam is not None else None

    return cls, prob, overlay


# =========================================================
# UI
# =========================================================
st.title("🔬 Skin Cancer Detection – ResNet50 + Grad‑CAM++")

model = load_model()
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    if st.button("Analyze"):
        cls, prob, overlay = predict(model, img)

        st.subheader("Prediction Result")

        if cls == 1:
            st.error(f"⚠️ **Cancer Detected** — Confidence: {prob*100:.2f}%")
        else:
            st.success(f"✓ **No Cancer Detected** — Confidence: {(1-prob)*100:.2f}%")

        if overlay is not None:
            st.subheader("Grad‑CAM++")
            st.image(overlay, caption="Heatmap Overlay")


st.markdown("---")
st.caption("For educational purposes only. Not medical advice.")
