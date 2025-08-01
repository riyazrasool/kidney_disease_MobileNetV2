import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🫁",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .result-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .status-high { background: #27ae60; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0; }
    .status-medium { background: #f39c12; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0; }
    .status-low { background: #e74c3c; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0; }
    
    .error-box {
        background: #dc3545;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .warning-box {
        background: #ffc107;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model."""
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

def validate_ct_scan(image_array):
    """
    Validate if the uploaded image is likely a CT scan.
    This is a simple heuristic-based validation.
    """
    try:
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        # CT scan characteristics:
        # 1. Usually grayscale or with limited color variation
        # 2. High contrast between bone/tissue
        # 3. Specific intensity distribution
        
        # Check if image is mostly grayscale (low color variation)
        if len(image_array.shape) == 3:
            color_variation = np.std(image_array, axis=2)
            avg_color_variation = np.mean(color_variation)
            if avg_color_variation > 50:  # High color variation might indicate non-CT
                return False, "Image appears to have high color variation, which is unusual for CT scans."
        
        # Check contrast (CT scans typically have high contrast)
        contrast = np.std(gray)
        if contrast < 20:  # Very low contrast
            return False, "Image has very low contrast, which is unusual for CT scans."
        
        # Check intensity distribution (CT scans have specific histogram characteristics)
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        # CT scans typically have a bimodal or specific distribution
        # Simple check: if too many pixels are at extreme values, might not be CT
        dark_pixels = np.sum(hist[:50])  # Very dark pixels
        bright_pixels = np.sum(hist[200:])  # Very bright pixels
        total_pixels = np.sum(hist)
        
        if (dark_pixels + bright_pixels) / total_pixels > 0.8:
            return False, "Image intensity distribution doesn't match typical CT scan patterns."
        
        # Check aspect ratio (CT scans are usually square or close to square)
        height, width = gray.shape
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Image aspect ratio doesn't match typical CT scan dimensions."
        
        # Additional checks for common non-CT images
        # Check for text, logos, or UI elements
        if len(image_array.shape) == 3:
            # Look for saturated colors (common in photos, not CT scans)
            max_color_diff = np.max(image_array, axis=2) - np.min(image_array, axis=2)
            if np.mean(max_color_diff) > 100:
                return False, "Image appears to be a color photograph rather than a CT scan."
        
        return True, "Image appears to be a valid CT scan."
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the image for prediction."""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = image[:, :, :3]
        
        # Resize image using PIL (more reliable than cv2)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(target_size)
        image = np.array(pil_image)
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image_class(image, model):
    """Predict the class label for the given image."""
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
    # Header
    st.markdown("""
    <div class="header">
        <h1>🫁 Kidney Disease Classifier</h1>
        <p>AI-powered CT scan analysis for kidney disease detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.markdown("""
        This application uses deep learning to classify kidney CT scans:
        
        **🔵 Cyst**: Fluid-filled sacs  
        **🟢 Normal**: Healthy tissue  
        **🟡 Stone**: Kidney stones  
        **🔴 Tumor**: Abnormal growths  
        
        **Accuracy**: 98.71%
        """)
        
        st.markdown("## 🔧 Technical Details")
        st.markdown("""
        - **Model**: MobileNetV2
        - **Input**: 128×128 pixels
        - **Classes**: 4 categories
        - **Framework**: TensorFlow
        """)
        
        st.markdown("## ⚠️ Disclaimer")
        st.markdown("""
        For **educational purposes only**. 
        Always consult healthcare professionals.
        """)
        
        st.markdown("## 🔍 Image Validation")
        st.markdown("""
        The app validates uploaded images to ensure they are:
        - CT scan images (not photos)
        - Proper contrast and intensity
        - Appropriate aspect ratios
        - Medical imaging format
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 📤 Upload CT Scan")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a CT scan image for analysis"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
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
        st.markdown("## 🔍 Analysis Results")
        
        if image_array is not None:
            # First, validate if it's a CT scan
            is_valid_ct, validation_message = validate_ct_scan(image_array)
            
            if not is_valid_ct:
                st.markdown('<div class="error-box">❌ Invalid or unrelated image uploaded. Please upload a kidney CT scan.</div>', unsafe_allow_html=True)
            else:
                # Load model
                model = load_model()
                
                if model is not None:
                    # Preprocess image
                    preprocessed_image = preprocess_image(image_array)
                    
                    if preprocessed_image is not None:
                        # Make prediction
                        predicted_class, confidence, all_probabilities = predict_image_class(preprocessed_image, model)
                        
                        if predicted_class is not None:
                            # Check confidence threshold (0.7 = 70%)
                            confidence_threshold = 0.7
                            confidence_display = confidence if np.isfinite(confidence) else 0.0
                            
                            if confidence_display < confidence_threshold:
                                st.markdown('<div class="error-box">❌ Low confidence prediction. Please upload a clearer kidney CT scan image.</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                **Confidence Score: {confidence_display:.1%}** (Threshold: {confidence_threshold:.0%})
                                
                                The model is not confident enough to make a reliable prediction. This could be due to:
                                - Poor image quality
                                - Unclear or blurry CT scan
                                - Image not showing kidney region clearly
                                - Non-kidney CT scan
                                """)
                            else:
                                # Results display
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
                                st.markdown(f"### 📊 Confidence: **{confidence_display:.1%}**")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Class probabilities
                                st.markdown("### 📊 Class Probabilities")
                                class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
                                
                                for i, (label, prob) in enumerate(zip(class_labels, all_probabilities)):
                                    st.write(f"**{label}**: {prob:.1%}")
                                
                                # Status indicator
                                if confidence > 0.8:
                                    st.markdown('<div class="status-high">✅ High Confidence Analysis</div>', unsafe_allow_html=True)
                                elif confidence > 0.7:
                                    st.markdown('<div class="status-medium">✅ Confident Analysis</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="status-low">⚠️ Acceptable Confidence Analysis</div>', unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to analyze image")
                    else:
                        st.error("❌ Failed to process image")
                else:
                    st.error("❌ Model not available")
        else:
            st.info("📤 Upload an image to see results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Built with Streamlit and TensorFlow</strong></p>
        <p>For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
