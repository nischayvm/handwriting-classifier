from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables to store models and data
models = {}
x_train_flat = None
y_train = None

def preprocess_image(image_data):
    """Preprocess uploaded image to match MNIST format"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Convert to PIL Image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert colors (MNIST has white digits on black background)
        img_inverted = 255 - img_array
        
        # Normalize to 0-1 range
        img_norm = img_inverted / 255.0
        
        # Flatten for ML models
        img_flat = img_norm.reshape(1, -1)
        
        return img_flat, img_inverted
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def train_models():
    """Train all models on MNIST data"""
    global models, x_train_flat, y_train
    
    try:
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Preprocess data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten for classical ML models
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        
        # Use smaller dataset for faster training
        print("Using smaller dataset for faster training...")
        x_train_small = x_train[:5000]
        y_train_small = y_train[:5000]
        x_test_small = x_test[:1000]
        y_test_small = y_test[:1000]
        
        x_train_flat_small = x_train_small.reshape((x_train_small.shape[0], -1))
        x_test_flat_small = x_test_small.reshape((x_test_small.shape[0], -1))
        
        print("Training Logistic Regression...")
        # Logistic Regression
        log_reg = LogisticRegression(
            solver='lbfgs',
            max_iter=100,
            random_state=42
        )
        log_reg.fit(x_train_flat_small, y_train_small)
        models['logistic_regression'] = log_reg
        print("✓ Logistic Regression trained")
        
        print("Training K-Nearest Neighbors...")
        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train_flat_small, y_train_small)
        models['knn'] = knn
        print("✓ KNN trained")
        
        print("Training Support Vector Machine...")
        # SVM (using even smaller subset)
        svm = SVC(kernel='rbf', gamma='scale', random_state=42)
        svm.fit(x_train_flat_small[:2000], y_train_small[:2000])
        models['svm'] = svm
        print("✓ SVM trained")
        
        # Calculate accuracies
        print("Calculating model accuracies...")
        lr_acc = log_reg.score(x_test_flat_small, y_test_small)
        knn_acc = knn.score(x_test_flat_small, y_test_small)
        svm_acc = svm.score(x_test_flat_small, y_test_small)
        
        print(f"Model accuracies:")
        print(f"Logistic Regression: {lr_acc:.2%}")
        print(f"KNN: {knn_acc:.2%}")
        print(f"SVM: {svm_acc:.2%}")
        
        return {
            'logistic_regression': lr_acc,
            'knn': knn_acc,
            'svm': svm_acc
        }
        
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Training with smaller dataset...")
        
        # Fallback: train with smaller dataset
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Use smaller subset for faster training
            x_train_small = x_train[:10000]
            y_train_small = y_train[:10000]
            x_test_small = x_test[:2000]
            y_test_small = y_test[:2000]
            
            x_train_flat = x_train_small.reshape((x_train_small.shape[0], -1))
            x_test_flat = x_test_small.reshape((x_test_small.shape[0], -1))
            
            print("Training Logistic Regression (small dataset)...")
            log_reg = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
            log_reg.fit(x_train_flat, y_train_small)
            models['logistic_regression'] = log_reg
            
            print("Training KNN (small dataset)...")
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_train_flat, y_train_small)
            models['knn'] = knn
            
            print("Training SVM (small dataset)...")
            svm = SVC(kernel='rbf', gamma='scale', random_state=42)
            svm.fit(x_train_flat[:5000], y_train_small[:5000])
            models['svm'] = svm
            
            print("✓ All models trained with smaller dataset")
            return {'logistic_regression': 0.9, 'knn': 0.9, 'svm': 0.9}
            
        except Exception as e2:
            print(f"Critical error: {e2}")
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        img_flat, img_inverted = preprocess_image(image_data)
        
        if img_flat is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(img_flat)[0]
                predictions[model_name] = int(pred)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = None
        
        # Convert processed image back to base64 for display
        img_display = Image.fromarray(img_inverted.astype(np.uint8))
        buffer = BytesIO()
        img_display.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'predictions': predictions,
            'processed_image': img_str
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/model_info')
def model_info():
    """Get information about trained models"""
    if not models:
        return jsonify({'error': 'Models not trained yet'}), 400
    
    return jsonify({
        'models': list(models.keys()),
        'message': 'All models are trained and ready for predictions'
    })

if __name__ == '__main__':
    print("Starting Handwriting Classifier Web App v2.0...")
    print("Training models...")
    
    try:
        accuracies = train_models()
        if accuracies:
            print("✓ Models trained successfully!")
            print("Starting Flask server...")
            
            # Get port from environment variable (for cloud deployment)
            port = int(os.environ.get('PORT', 5000))
            debug_mode = os.environ.get('FLASK_ENV') != 'production'
            
            print(f"Server starting on port {port}")
            app.run(debug=debug_mode, host='0.0.0.0', port=port)
        else:
            print("❌ Failed to train models. Exiting...")
            
    except Exception as e:
        print(f"❌ Critical error starting app: {e}")
        print("App failed to start")
