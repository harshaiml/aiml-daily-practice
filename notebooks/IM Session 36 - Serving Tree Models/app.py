
from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import time

app = Flask(__name__)

# Load ONNX model at startup (ONCE!)
print("Loading ONNX model...")
session = rt.InferenceSession('iris_rf_model.onnx')
input_name = session.get_inputs()[0].name
print(f"Model loaded! Input name: {input_name}")

# Model metadata
MODEL_INFO = {
    'name': 'Iris Random Forest',
    'version': '1.0',
    'features': ['sepal length', 'sepal width', 'petal length', 'petal width'],
    'classes': ['setosa', 'versicolor', 'virginica']
}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': time.time()
    })

@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    return jsonify(MODEL_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Validate JSON
        if not request.json:
            return jsonify({'error': 'No JSON in request'}), 400

        # Validate features field
        if 'features' not in request.json:
            return jsonify({'error': 'Missing features field'}), 400

        # Parse features
        features = request.json['features']

        # Convert to numpy array
        try:
            features_array = np.array(features, dtype=np.float32)
        except (ValueError, TypeError):
            return jsonify({'error': 'Features must be numeric'}), 400

        # Validate feature count
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        if features_array.shape[1] != 4:
            return jsonify({
                'error': f'Expected 4 features, got {features_array.shape[1]}'
            }), 400

        # Make prediction
        start_time = time.time()
        prediction = session.run(None, {input_name: features_array})[0]
        inference_time = (time.time() - start_time) * 1000

        # Format response
        return jsonify({
            'prediction': int(prediction[0]),
            'class_name': MODEL_INFO['classes'][int(prediction[0])],
            'inference_time_ms': round(inference_time, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if 'features' not in request.json:
            return jsonify({'error': 'Missing features field'}), 400

        features_array = np.array(request.json['features'], dtype=np.float32)

        if features_array.ndim != 2 or features_array.shape[1] != 4:
            return jsonify({'error': 'Features must be 2D array with 4 columns'}), 400

        start_time = time.time()
        predictions = session.run(None, {input_name: features_array})[0]
        inference_time = (time.time() - start_time) * 1000

        return jsonify({
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'inference_time_ms': round(inference_time, 2),
            'per_sample_ms': round(inference_time / len(predictions), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
