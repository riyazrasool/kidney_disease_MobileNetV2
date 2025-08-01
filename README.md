# 🫁 Kidney Disease Classifier

A Streamlit web application for classifying kidney CT scan images using deep learning. The application uses a MobileNetV2-based model trained on a dataset of kidney CT scans to classify images into four categories: Cyst, Normal, Stone, and Tumor.

## 🎯 Features

- **Image Upload**: Upload CT scan images in various formats (PNG, JPG, JPEG, BMP, TIFF)
- **Real-time Prediction**: Instant classification with confidence scores
- **Modern UI**: Clean, responsive interface with gradient design
- **Detailed Results**: Shows prediction probabilities for all classes
- **Error Handling**: Robust error handling for various edge cases
- **Mobile Responsive**: Works on desktop and mobile devices

## 📊 Model Performance

- **Test Accuracy**: 98.71%
- **Architecture**: MobileNetV2
- **Input Size**: 128x128 pixels
- **Classes**: 4 (Cyst, Normal, Stone, Tumor)

## 🚀 Quick Start

### Local Deployment

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - app.py
   # - requirements.txt
   # - kidney_disease_classifier_v2.h5
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Upload a kidney CT scan image
   - View the prediction results

### Streamlit Cloud Deployment

1. **Create a GitHub repository** with your project files
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set the main file path to `app.py`
   - Deploy!

## 📁 Project Structure

```
kidney_disease_classifier/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── kidney_disease_classifier_v2.h5 # Trained model file
├── mobileNetV4_1 (1).ipynb        # Training notebook
├── test_test.ipynb                 # Testing notebook
└── README.md                       # This file
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: MobileNetV2
- **Input Shape**: (128, 128, 3)
- **Output**: 4-class classification
- **Preprocessing**: Resize to 128x128, normalize to [0,1]

### Dependencies
- **Streamlit**: Web framework
- **TensorFlow**: Deep learning framework
- **OpenCV**: Image processing
- **Pillow**: Image handling
- **NumPy**: Numerical computations

### Class Labels
- **0**: Cyst (Fluid-filled sacs in the kidney)
- **1**: Normal (Healthy kidney tissue)
- **2**: Stone (Kidney stones)
- **3**: Tumor (Abnormal growths)

## 🎨 UI Features

### Sidebar Information
- **About**: Description of the application and classes
- **Technical Details**: Model architecture and specifications
- **Disclaimer**: Medical disclaimer

### Main Interface
- **Upload Area**: Drag-and-drop or click-to-upload images
- **Image Preview**: Display uploaded image
- **Prediction Results**: 
  - Predicted class with confidence
  - Progress bars for confidence levels
  - Detailed class probabilities
  - Color-coded confidence indicators

### Error Handling
- Model loading errors
- Image processing errors
- File format validation
- Graceful fallbacks

## ⚠️ Important Notes

### Medical Disclaimer
This application is for **educational and research purposes only**. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

### Model Limitations
- Trained on specific CT scan dataset
- May not generalize to all imaging conditions
- Requires high-quality, properly oriented images
- Confidence scores should be interpreted carefully

### File Requirements
- **Model File**: `kidney_disease_classifier_v2.h5` must be in the same directory as `app.py`
- **Image Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Image Quality**: Clear, well-lit CT scan images work best

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `kidney_disease_classifier_v2.h5` is in the correct directory
   - Check file permissions
   - Verify TensorFlow version compatibility

2. **Image upload errors**
   - Check file format (supported: PNG, JPG, JPEG, BMP, TIFF)
   - Ensure image file is not corrupted
   - Try different image formats

3. **Memory issues**
   - Reduce image size before uploading
   - Close other applications to free memory
   - Use smaller images for testing

4. **Deployment issues on Streamlit Cloud**
   - Verify all files are in the repository
   - Check requirements.txt compatibility
   - Ensure model file is included in the repository

## 📈 Performance Optimization

- **Model Caching**: Uses `@st.cache_resource` for efficient model loading
- **Image Processing**: Optimized preprocessing pipeline
- **Memory Management**: Efficient image handling and cleanup
- **Responsive Design**: Works on various screen sizes

## 🔄 Updates and Maintenance

### Version History
- **v1.0**: Initial release with basic classification
- **v2.0**: Enhanced UI with modern design
- **v2.1**: Added confidence indicators and error handling

### Future Enhancements
- Batch processing capabilities
- Export functionality for results
- Additional model architectures
- Enhanced visualization options

## 📞 Support

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the technical documentation
- Ensure all dependencies are correctly installed
- Verify file structure and permissions

## 📄 License

This project is for educational purposes. Please respect medical ethics and use responsibly.

---

**Built with ❤️ using Streamlit and TensorFlow** 