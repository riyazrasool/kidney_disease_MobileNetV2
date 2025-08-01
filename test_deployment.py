#!/usr/bin/env python3
"""
Test script for Kidney Disease Classifier deployment
This script verifies that the model can be loaded and basic functionality works.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def test_model_loading():
    """Test if the model can be loaded successfully."""
    print("🔍 Testing model loading...")
    
    model_path = "kidney_disease_classifier_v2.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def test_preprocessing():
    """Test the preprocessing function with a dummy image."""
    print("\n🔍 Testing preprocessing...")
    
    try:
        # Create a dummy image (128x128 RGB)
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test preprocessing
        def preprocess_image(image, target_size=(128, 128)):
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            image = cv2.resize(image, target_size)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            return image
        
        preprocessed = preprocess_image(dummy_image)
        
        if preprocessed is not None and preprocessed.shape == (1, 128, 128, 3):
            print("✅ Preprocessing function works correctly")
            print(f"📊 Preprocessed shape: {preprocessed.shape}")
            return True
        else:
            print("❌ Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        return False

def test_prediction():
    """Test prediction with dummy data."""
    print("\n🔍 Testing prediction...")
    
    try:
        # Load model
        model = tf.keras.models.load_model("kidney_disease_classifier_v2.h5")
        
        # Create dummy input
        dummy_input = np.random.random((1, 128, 128, 3))
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Check output
        if prediction.shape == (1, 4):  # 4 classes
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction, axis=1)[0]
            
            class_labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
            predicted_label = class_labels.get(predicted_class, "Unknown")
            
            print("✅ Prediction function works correctly")
            print(f"📊 Predicted class: {predicted_label}")
            print(f"📊 Confidence: {confidence:.2%}")
            return True
        else:
            print("❌ Prediction output shape incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        'streamlit', 'tensorflow', 'numpy', 'cv2', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        return False
    else:
        print("✅ All dependencies available")
        return True

def main():
    """Run all tests."""
    print("🚀 Starting Kidney Disease Classifier Deployment Tests\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading),
        ("Preprocessing", test_preprocessing),
        ("Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment should work correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 