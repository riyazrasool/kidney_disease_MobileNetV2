import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean and professional CSS
st.markdown("""
<style>
    /* Clean, medical-grade styling */
    .main {
        background: #f8f9fa;
    }
    
    /* Professional header */
    .header {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Clean cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2980b9;
        background: #e3f2fd;
    }
    
    /* Results display */
    .result-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .result-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .confidence-display {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
    }
    
    /* Probability bars */
    .prob-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .prob-label {
        font-weight: 600;
        color: white;
    }
    
    .prob-value {
        font-weight: 700;
        color: white;
    }
    
    /* Status indicators */
    .status-high {
        background: #27ae60;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .status-medium {
        background: #f39c12;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .status-low {
        background: #e74c3c;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching for better performance."""
    try:
        model_path = "kidney_disease_classifier_v2.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image for prediction.
    Args:
        image: PIL Image or numpy array
        target_size: Target size for resizing
    Returns:
        np.array: Preprocessed image
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image_class(image, model):
    """
    Predict the class label for the given image.
    Args:
        image: Preprocessed image
        model: Trained model
    Returns:
        tuple: (predicted_class, confidence_score, all_probabilities)
    """
    try:
        predictions = model.predict(image, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]
        
        # Ensure confidence is a valid number
        if not np.isfinite(confidence_score):
            confidence_score = 0.0
        
        # Class labels mapping
        class_labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
        predicted_class = class_labels.get(predicted_class_index, "Unknown")
        
        # Ensure probabilities are valid
        probabilities = predictions[0]
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        
        return predicted_class, confidence_score, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Clean header
    st.markdown("""
    <div class="header">
        <h1>🫁 Kidney Disease Classifier</h1>
        <p>AI-powered CT scan analysis for kidney disease detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📋 About</div>', unsafe_allow_html=True)
        st.markdown("""
        This application uses deep learning to classify kidney CT scans:
        
        **🔵 Cyst**: Fluid-filled sacs  
        **🟢 Normal**: Healthy tissue  
        **🟡 Stone**: Kidney stones  
        **🔴 Tumor**: Abnormal growths  
        
        **Accuracy**: 98.71%
        """)
        
        st.markdown('<div class="sidebar-title">🔧 Technical Details</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Model**: MobileNetV2
        - **Input**: 128×128 pixels
        - **Classes**: 4 categories
        - **Framework**: TensorFlow
        """)
        
        st.markdown('<div class="sidebar-title">⚠️ Disclaimer</div>', unsafe_allow_html=True)
        st.markdown("""
        For **educational purposes only**. 
        Always consult healthcare professionals.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 Upload CT Scan</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a CT scan image for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded CT Scan", use_column_width=True)
                image_array = np.array(image)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                image_array = None
        else:
            image_array = None
            st.info("👆 Please upload a CT scan image")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Analysis Results</div>', unsafe_allow_html=True)
        
        if image_array is not None:
            # Load model
            model = load_model()
            
            if model is not None:
                # Preprocess image
                preprocessed_image = preprocess_image(image_array)
                
                if preprocessed_image is not None:
                    # Make prediction
                    predicted_class, confidence, all_probabilities = predict_image_class(preprocessed_image, model)
                    
                    if predicted_class is not None:
                        # Clean results display
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<div class="result-title">🎯 Predicted Class</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="result-value">{predicted_class}</div>', unsafe_allow_html=True)
                        
                        # Safe confidence display
                        confidence_display = confidence if np.isfinite(confidence) else 0.0
                        st.markdown(f'<div class="confidence-display">Confidence: {confidence_display:.1%}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Class probabilities (no progress bars)
                        st.markdown("### 📊 Class Probabilities")
                        class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
                        
                        for i, (label, prob) in enumerate(zip(class_labels, all_probabilities)):
                            st.markdown('<div class="prob-bar">', unsafe_allow_html=True)
                            st.markdown(f'<span class="prob-label">{label}</span>', unsafe_allow_html=True)
                            st.markdown(f'<span class="prob-value">{prob:.1%}</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Status indicator
                        if confidence > 0.8:
                            st.markdown('<div class="status-high">✅ High Confidence Analysis</div>', unsafe_allow_html=True)
                        elif confidence > 0.6:
                            st.markdown('<div class="status-medium">⚠️ Medium Confidence Analysis</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-low">❌ Low Confidence Analysis</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to analyze image")
                else:
                    st.error("❌ Failed to process image")
            else:
                st.error("❌ Model not available")
        else:
            st.info("📤 Upload an image to see results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clean footer
    st.markdown("""
    <div class="footer">
        <p><strong>Built with Streamlit and TensorFlow</strong></p>
        <p>For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 