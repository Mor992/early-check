"""
Skin Cancer Detection App with Grad-CAM++ Visualization
Streamlit Application for Deployment
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import gdown
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #1f77b4;
    }
    .cancer-warning {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .no-cancer {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Drive model link - REPLACE WITH YOUR MODEL'S GOOGLE DRIVE FILE ID
MODEL_DRIVE_ID = "1uHgOzbvTY8hus4_ApzLlv7VO-Ye5uWpX"
MODEL_PATH = "best_resnet_model.h5"

# Lesion information database
LESION_DATABASE = {
    'melanoma': {
        'name': 'Melanoma',
        'risk': 'High',
        'description': 'Most serious type of skin cancer that develops in melanocytes. Can spread to other organs if not detected early.',
        'characteristics': [
            'Asymmetrical shape',
            'Irregular borders',
            'Multiple colors',
            'Diameter larger than 6mm',
            'Evolving over time'
        ],
        'recommendation': 'URGENT: Consult a dermatologist immediately for biopsy and treatment options.'
    },
    'basal_cell_carcinoma': {
        'name': 'Basal Cell Carcinoma',
        'risk': 'Moderate',
        'description': 'Most common form of skin cancer. Grows slowly and rarely spreads, but can be locally destructive.',
        'characteristics': [
            'Pearly or waxy bump',
            'Flat, flesh-colored lesion',
            'Bleeding or scabbing sore',
            'Pink growths with raised edges'
        ],
        'recommendation': 'Schedule an appointment with a dermatologist for evaluation and potential removal.'
    },
    'actinic_keratoses': {
        'name': 'Actinic Keratoses',
        'risk': 'Pre-cancerous',
        'description': 'Rough, scaly patches caused by sun damage. Can develop into squamous cell carcinoma if untreated.',
        'characteristics': [
            'Rough, dry, or scaly patch',
            'Flat to slightly raised',
            'Usually pink, red, or brown',
            'Often found on sun-exposed areas'
        ],
        'recommendation': 'Consult a dermatologist for treatment to prevent progression to cancer.'
    }
}


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Download and load the trained model from Google Drive"""
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive... This may take a minute.'):
            try:
                url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success('Model downloaded successfully!')
            except Exception as e:
                st.error(f'Error downloading model: {str(e)}')
                st.info('Please check your Google Drive file ID and sharing settings.')
                return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f'Error loading model: {str(e)}')
        return None


# ============================================================================
# GRAD-CAM++ IMPLEMENTATION
# ============================================================================

def compute_gradcam_plus_plus(model, image, layer_name='conv5_block3_out'):
    """Compute Grad-CAM++ heatmap"""
    
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except:
        # Fallback to last conv layer
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
        if conv_layers:
            layer_name = conv_layers[-1]
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[model.get_layer(layer_name).output, model.output]
            )
        else:
            return None
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        return None
    
    # Grad-CAM++ computation
    grads_squared = tf.square(grads)
    grads_cubed = grads_squared * grads
    
    alpha_num = grads_squared
    alpha_denom = grads_squared * 2.0 + tf.reduce_sum(
        conv_outputs * grads_cubed, axis=(1, 2), keepdims=True
    )
    alpha_denom = tf.where(
        alpha_denom != 0.0,
        alpha_denom,
        tf.ones_like(alpha_denom)
    )
    
    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))
    
    weights = tf.reshape(weights, [-1, 1, 1, tf.shape(conv_outputs)[-1]])
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)
    
    if tf.reduce_max(cam) > 0:
        cam = cam / tf.reduce_max(cam)
    
    return cam.numpy()[0]


def create_heatmap_overlay(original_image, heatmap):
    """Create overlay of heatmap on original image"""
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]
    
    # Normalize original image
    if original_image.max() > 1:
        original_image = original_image / 255.0
    
    # Create overlay
    overlay = heatmap_colored * 0.5 + original_image
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_and_visualize(image, model):
    """Make prediction and generate Grad-CAM++ visualization"""
    
    # Preprocess image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction_prob = model.predict(img_array, verbose=0)[0][0]
    prediction_class = int(prediction_prob > 0.5)
    
    # Generate Grad-CAM++
    heatmap = compute_gradcam_plus_plus(model, img_array)
    
    if heatmap is not None:
        overlay = create_heatmap_overlay(
            np.array(image.resize((224, 224))),
            heatmap
        )
    else:
        overlay = None
    
    return {
        'class': prediction_class,
        'probability': prediction_prob,
        'overlay': overlay
    }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🔬 Skin Cancer Detection System")
    st.markdown("### AI-Powered Melanoma Detection with Grad-CAM++ Visualization")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses deep learning (ResNet50) to detect skin cancer from images.
        
        **Features:**
        - Binary classification (Cancer/No Cancer)
        - Detailed lesion information
        - Grad-CAM++ visualization
        - Risk assessment
        
        **Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.
        """)
        
        st.header("📊 Model Info")
        st.info("""
        - **Model:** ResNet50
        - **Training Data:** HAM10000
        - **Accuracy:** ~90%+
        - **Classes:** Cancer vs Not Cancer
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check configuration.")
        st.stop()
    
    # File uploader
    st.header("📤 Upload Skin Lesion Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner('Analyzing image... Please wait.'):
                # Make prediction
                result = predict_and_visualize(image, model)
                
                prediction_class = result['class']
                prediction_prob = result['probability']
                overlay = result['overlay']
                
                # Display Grad-CAM++
                with col2:
                    st.subheader("🔥 Grad-CAM++ Heatmap")
                    if overlay is not None:
                        st.image(overlay, use_container_width=True)
                        st.caption("Red areas indicate regions most influential for the prediction")
                    else:
                        st.warning("Could not generate heatmap")
                
                st.markdown("---")
                
                # Display results
                if prediction_class == 1:
                    # Cancer detected
                    confidence = prediction_prob * 100
                    
                    st.markdown(f"""
                    <div class="cancer-warning">
                        <h2 style="color: #d32f2f;">⚠️ CANCER DETECTED</h2>
                        <p style="font-size: 18px;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📋 Detailed Analysis")
                    
                    # Display lesion information
                    tabs = st.tabs(["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"])
                    
                    for idx, (key, info) in enumerate(LESION_DATABASE.items()):
                        with tabs[idx]:
                            st.markdown(f"#### {info['name']}")
                            st.markdown(f"**Risk Level:** `{info['risk']}`")
                            st.markdown(f"**Description:** {info['description']}")
                            
                            st.markdown("**Common Characteristics:**")
                            for char in info['characteristics']:
                                st.markdown(f"- {char}")
                            
                            st.warning(f"**Recommendation:** {info['recommendation']}")
                    
                    st.error("""
                    ### ⚕️ IMPORTANT MEDICAL ADVICE
                    
                    **This AI tool provides preliminary screening only.**
                    
                    **Next Steps:**
                    1. Schedule an appointment with a dermatologist immediately
                    2. Bring this report and the original image
                    3. Request a professional biopsy if recommended
                    4. Do not delay treatment
                    
                    **Remember:** Early detection significantly improves treatment outcomes.
                    """)
                    
                else:
                    # No cancer detected
                    confidence = (1 - prediction_prob) * 100
                    
                    st.markdown(f"""
                    <div class="no-cancer">
                        <h2 style="color: #388e3c;">✓ NO CANCER DETECTED</h2>
                        <p style="font-size: 18px;"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("""
                    ### ✓ Good News!
                    
                    The AI model did not detect signs of skin cancer in this image.
                    
                    **Recommendations:**
                    - Continue regular self-examinations
                    - Monitor for any changes in size, color, or shape
                    - Protect skin from excessive sun exposure
                    - Schedule annual dermatology checkups
                    - Consult a doctor if you notice any concerning changes
                    
                    **Remember:** Regular monitoring is key to maintaining skin health.
                    """)
                
                # Confidence bar
                st.markdown("### 📊 Prediction Confidence")
                if prediction_class == 1:
                    st.progress(prediction_prob)
                    st.caption(f"Cancer: {prediction_prob*100:.1f}%")
                else:
                    st.progress(1 - prediction_prob)
                    st.caption(f"Not Cancer: {(1-prediction_prob)*100:.1f}%")
    
    else:
        # Instructions
        st.info("""
        ### 📝 How to Use:
        
        1. **Upload** a clear image of the skin lesion
        2. Click **Analyze Image** button
        3. Review the **AI prediction** and **Grad-CAM++ visualization**
        4. Read the **detailed report** and recommendations
        5. **Consult a dermatologist** for professional diagnosis
        
        ### 📸 Image Guidelines:
        - Use clear, well-lit photos
        - Lesion should be centered and in focus
        - Avoid shadows or reflections
        - JPG, JPEG, or PNG format
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>© 2025 Skin Cancer Detection System | For Educational Purposes Only</p>
        <p style="font-size: 12px;">
            <strong>Medical Disclaimer:</strong> This application is not a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult qualified healthcare providers with questions about medical conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
