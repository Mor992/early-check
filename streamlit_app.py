import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Optional: to download from Google Drive
try:
    import gdown
    GDOWN_AVAILABLE = True
except Exception:
    GDOWN_AVAILABLE = False


# ---------------------
# Utility functions
# ---------------------

def load_model_from_local(path):
    """Load a Keras model from a local file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model = tf.keras.models.load_model(path)
    return model


def download_model_from_gdrive(file_id, dest_path):
    """Download a model file from Google Drive using gdown."""
    if not GDOWN_AVAILABLE:
        raise RuntimeError("gdown is not installed. Install it or add it to requirements.txt.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    return dest_path


def preprocess_input_image(img: Image.Image, target_size=(224, 224)):
    """Resize and preprocess an input image for ResNet-based models."""
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)

    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        arr = preprocess_input(arr)
    except Exception:
        arr = arr / 255.0

    return arr


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate a Grad-CAM heatmap."""
    img_tensor = tf.expand_dims(img_array, axis=0)

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_img: Image.Image, heatmap, alpha=0.4, cmap='jet'):
    """Overlay heatmap on the original image."""
    import matplotlib.cm as cm

    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        original_img.size, resample=Image.BILINEAR
    )

    heatmap_arr = np.array(heatmap_resized)
    colormap = cm.get_cmap(cmap)
    colored_heatmap = colormap(heatmap_arr / 255.0)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    colored_heatmap = Image.fromarray(colored_heatmap)
    overlay = Image.blend(original_img.convert('RGBA'), colored_heatmap.convert('RGBA'), alpha=alpha)
    return overlay


# ---------------------
# Streamlit UI
# ---------------------

st.set_page_config(page_title='Cancer Type Prediction with Grad-CAM', layout='wide')
st.title('Cancer Type Prediction with Grad-CAM')

st.markdown(
    "Upload a skin lesion image to classify cancer type and visualize the important regions using Grad-CAM."
)

DEFAULT_LOCAL_MODEL_PATH = "/mnt/data/final_resnet_model.keras"
DEFAULT_GDRIVE_ID = "1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q"
DEFAULT_CLASS_NAMES = "Actinic Keratoses,Basal Cell Carcinoma,Melanoma,Not Cancer"

use_download = st.checkbox("Download model from Google Drive", value=True)
model = None
model_load_error = None

# ---------------------
# Model Loading
# ---------------------

if use_download:
    file_id = st.text_input("Google Drive file ID", value=DEFAULT_GDRIVE_ID)
    dest = st.text_input("Destination path", value="/tmp/final_resnet_model.keras")

    if st.button("Download and Load Model"):
        try:
            st.info("Downloading model...")
            download_model_from_gdrive(file_id, dest)
            st.success("Download complete. Loading model...")
            model = load_model_from_local(dest)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

else:
    local_path = st.text_input("Local model path", value=DEFAULT_LOCAL_MODEL_PATH)
    if st.button("Load Local Model"):
        try:
            model = load_model_from_local(local_path)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# ---------------------
# Model Summary
# ---------------------

if model is not None:
    with st.expander("Show model summary"):
        buf = io.StringIO()
        try:
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.text(buf.getvalue())
        except Exception as e:
            st.text(f"Unable to show summary: {e}")

# ---------------------
# Image Upload
# ---------------------

st.header("Upload Image for Prediction")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

image = None
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# ---------------------
# Class Names
# ---------------------

class_names_input = st.text_input(
    "Comma-separated class names",
    value=DEFAULT_CLASS_NAMES
)
class_names = [c.strip() for c in class_names_input.split(",")]

# ---------------------
# Grad-CAM setup
# ---------------------

last_conv_default = st.text_input(
    "Last convolutional layer name",
    value="conv5_block3_out"
)

# ---------------------
# Prediction + Grad-CAM
# ---------------------

if st.button("Predict and Generate Grad-CAM"):
    if model is None:
        st.error("Load a model first.")
    elif image is None:
        st.error("Upload an image first.")
    else:
        try:
            target_size = (224, 224)
            preprocessed = preprocess_input_image(image, target_size)

            preds = model.predict(np.expand_dims(preprocessed, axis=0))
            top_k = min(3, preds.shape[-1])
            top_inds = preds[0].argsort()[-top_k:][::-1]

            st.subheader("Prediction Results")
            for i, idx in enumerate(top_inds):
                name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                st.write(f"**{i+1}. {name} — {preds[0][idx]:.2%}**")

            # Grad-CAM
            heatmap = make_gradcam_heatmap(
                preprocessed,
                model,
                last_conv_default,
                pred_index=top_inds[0]
            )

            overlay = overlay_heatmap_on_image(image, heatmap)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

            with st.expander("Raw Heatmap"):
                fig, ax = plt.subplots()
                ax.axis("off")
                ax.imshow(heatmap, cmap="jet")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction or Grad-CAM: {e}")

st.markdown("---")
st.markdown(
    "**Tips:**\n"
    "- Ensure the model path or Google Drive ID is correct.\n"
    "- Check the model summary to find the correct last convolutional layer.\n"
    "- Ensure class names match the training order."
)
