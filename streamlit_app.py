import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tempfile
import matplotlib.pyplot as plt
import cv2

# Optional: allow downloading the model from Google Drive using gdown
# If you prefer to place the model file manually, put it in the app folder and set MODEL_PATH accordingly.
GDRIVE_FILE_ID = "1MA6mvotqb_RkswHivYjkZXbba0HVWpYN"
DEFAULT_MODEL_FILENAME = "resnet_model.keras"  # change if your model is a SavedModel folder

st.set_page_config(page_title="Skin / Lesion Classifier", layout="wide")

st.title("Streamlit app — Load & run your model")

st.markdown(
    """
    **How to provide the model file**

    1. Place your Keras `.h5` model in the same folder as this app and name it `model.h5`, **or**
    2. Install `gdown` and let the app download the model from the Google Drive link (app will attempt this if the file is missing).

    If your model is a TensorFlow SavedModel directory (not an .h5), set `DEFAULT_MODEL_FILENAME` to that folder name.
    """
)

# Helper: download from Google Drive using gdown if available
def try_download_model_gdrive(file_id: str, out_path: str) -> bool:
    try:
        import gdown
    except Exception:
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception:
        return False


@st.cache_resource
def load_model(path: str):
    import tensorflow as tf
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        # sometimes SavedModel directory is provided
        try:
            model = tf.keras.models.load_model(path, compile=False)
            return model
        except Exception as e2:
            st.error(f"Failed to load model: {e}\n{e2}")
            return None


# Simple Grad-CAM implementation (works for typical conv nets)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    import tensorflow as tf
    grad_model = tf.keras.models.Model([
        model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(original_img.convert("RGB"))
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed)


# Ensure model file exists or offer to download
model_path = DEFAULT_MODEL_FILENAME
if not os.path.exists(model_path):
    st.warning(f"Model file '{model_path}' not found in app folder.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try download from provided Drive link"):
            ok = try_download_model_gdrive(GDRIVE_FILE_ID, model_path)
            if ok:
                st.success("Downloaded model to %s" % model_path)
            else:
                st.error("Could not download model. Install gdown or download manually and place the model file in the app folder.")
    with col2:
        uploaded_model = st.file_uploader("Or upload the model file (.h5)", type=["h5"], accept_multiple_files=False)
        if uploaded_model is not None:
            with open(model_path, "wb") as f:
                f.write(uploaded_model.read())
            st.success(f"Saved uploaded model to {model_path}")

model = load_model(model_path)
if model is None:
    st.info("No model loaded. Place the model file and rerun the app (or upload/download above).")
    st.stop()

# Inspect input shape from the model
input_shape = None
try:
    input_shape = tuple(s for s in model.input_shape if s is not None)
except Exception:
    input_shape = (224, 224, 3)

# Determine target size
if len(input_shape) == 3:
    target_size = (input_shape[0], input_shape[1])
else:
    target_size = (224, 224)

st.write(f"Loaded model. Expected input size (hxwxc): {target_size + (input_shape[2],) if len(input_shape)==3 else input_shape}")

# Uploader
uploaded_image = st.file_uploader("Upload image for prediction", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 1])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

    # Preprocess
    img_resized = image.resize(target_size)
    img_array = np.array(img_resized).astype(np.float32)

    # Try common preprocessing: scale 0-1
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # If model expects channels_first, convert
    try:
        if model.input_shape and len(model.input_shape) == 4 and model.input_shape[1] == 3:
            # channels_first
            img_batch = np.transpose(img_batch, (0, 3, 1, 2))
    except Exception:
        pass

    # Predict
    preds = model.predict(img_batch)

    # Present predictions
    with col1:
        st.subheader("Predictions")
        if preds.ndim == 2 and preds.shape[1] > 1:
            # multiclass
            import pandas as pd
            probs = preds[0]
            # If model has class names, user can edit this list
            class_names = [f"class_{i}" for i in range(len(probs))]
            df = pd.DataFrame({"class": class_names, "probability": probs})
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
            st.table(df)
        else:
            p = float(preds.ravel()[0])
            st.write(f"Output scalar / single-probability: {p:.4f}")
            st.progress(p)

    # Try Grad-CAM: find last conv layer automatically
    with col2:
        st.subheader("Grad-CAM (optional)")
        last_conv = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name or 'Conv' in layer.__class__.__name__:
                last_conv = layer.name
                break
        if last_conv is None:
            st.info("No convolutional layer found for Grad-CAM.")
        else:
            st.write(f"Using last convolutional layer: {last_conv}")
            try:
                heatmap = make_gradcam_heatmap(img_batch, model, last_conv, pred_index=None)
                overlayed = overlay_heatmap_on_image(heatmap, image, alpha=0.4)
                st.image(overlayed, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

    # Option to save results
    if st.button("Save prediction image and results"):
        out_dir = "predictions"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(uploaded_image.name))[0]
        out_img_path = os.path.join(out_dir, f"{base}_overlay.png")
        try:
            overlayed.save(out_img_path)
        except Exception:
            # fallback: save original
            image.save(out_img_path)
        st.success(f"Saved to {out_img_path}")


# Footer: requirements and run command
st.markdown("---")
st.header("Run & Requirements")
st.markdown(
    """
    **Install packages**

    ```bash
    pip install streamlit tensorflow pillow numpy opencv-python matplotlib gdown
    ```

    **Start the app**

    ```bash
    streamlit run streamlit_app.py
    ```

    If you prefer manual download of the model, in a terminal use:

    ```bash
    pip install gdown
    gdown --id 1MA6mvotqb_RkswHivYjkZXbba0HVWpYN -O model.h5
    ```
    """
)
