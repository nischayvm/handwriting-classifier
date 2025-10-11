# Handwriting Digit Classifier Web App

A modern web application that uses multiple machine learning models to classify handwritten digits (0-9) from uploaded images. The app provides predictions from three different ML algorithms and displays a consensus result.

## 🚀 Features

- **Multiple ML Models**: Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)
- **Real-time Predictions**: Upload images and get instant predictions
- **Modern UI**: Beautiful, responsive design with drag-and-drop functionality
- **Image Preprocessing**: Automatic image resizing, grayscale conversion, and normalization
- **Consensus Prediction**: Combines results from all models for better accuracy
- **Model Performance**: Shows accuracy metrics for each model

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn, TensorFlow
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, JavaScript
- **Data**: MNIST dataset for training

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🔧 Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd handwriting_classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload and predict**
   - Drag and drop an image or click to browse
   - Click "Predict Digit" to get results from all models
   - View the consensus prediction

## 📊 Model Performance

| Model | Accuracy | Description |
|-------|----------|-------------|
| Logistic Regression | ~92.64% | Linear classification model |
| K-Nearest Neighbors | ~97% | Instance-based learning |
| Support Vector Machine | ~98% | Kernel-based classification |

## 🎯 How It Works

### 1. Image Preprocessing
- Converts uploaded images to grayscale
- Resizes to 28x28 pixels (MNIST format)
- Inverts colors (white digits on black background)
- Normalizes pixel values to 0-1 range

### 2. Model Training
- All models are trained on the MNIST dataset
- 60,000 training images, 10,000 test images
- Models are trained when the Flask app starts

### 3. Prediction Process
- Preprocessed image is fed to all three models
- Each model returns a digit prediction (0-9)
- Consensus is calculated using majority voting
- Results are displayed with confidence metrics

## 📁 Project Structure

```
handwriting_classifier/
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── style.css         # CSS styling
└── handwriting_detection.ipynb  # Original notebook
```

## 🎨 Usage Guide

### Uploading Images
1. **Drag and Drop**: Simply drag an image file onto the upload area
2. **Click to Browse**: Click the upload area to open file browser
3. **Supported Formats**: JPG, PNG, GIF, BMP, and other common image formats

### Understanding Results
- **Processed Image**: Shows how your image was preprocessed
- **Model Predictions**: Individual predictions from each ML model
- **Consensus**: The most common prediction across all models
- **Best Prediction**: Highlighted when models agree

### Tips for Best Results
- Use clear, high-contrast images
- Ensure digits are well-separated
- Avoid heavily rotated or distorted digits
- Single digits work best (not multiple digits in one image)

## 🔧 Customization

### Adding New Models
1. Train your model in the `train_models()` function
2. Add it to the `models` dictionary
3. Update the frontend to display the new model's results

### Modifying UI
- Edit `templates/index.html` for layout changes
- Modify `static/style.css` for styling updates
- Update JavaScript in the HTML file for functionality

### Changing Model Parameters
- Modify model initialization in `app.py`
- Adjust training data size for faster/slower training
- Change preprocessing steps as needed

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**2. TensorFlow Issues**
```bash
pip install tensorflow-cpu  # For CPU-only systems
```

**3. Memory Issues**
- Reduce training data size in `train_models()`
- Use smaller model parameters

**4. Port Already in Use**
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📈 Performance Optimization

### For Production
- Use a production WSGI server (Gunicorn, uWSGI)
- Implement model caching
- Add request rate limiting
- Use a reverse proxy (Nginx)

### For Development
- Enable Flask debug mode
- Use smaller training datasets
- Implement model persistence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset creators
- scikit-learn and TensorFlow teams
- Flask community
- All open-source contributors

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Classifying! 🎯**

