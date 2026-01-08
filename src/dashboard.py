"""
Real-Time Threat Detection Dashboard
Windows-optimized Flask application with improved error handling
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import shap
from datetime import datetime
import os
import sys
import traceback

# Fix the template path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
template_folder = os.path.join(project_root, 'templates')

app = Flask(__name__, template_folder=template_folder)

# Global variables
model_data = None
explainer = None

def load_system():
    """Load trained model and initialize explainer"""
    global model_data, explainer
    
    model_path = os.path.join(project_root, 'threat_detector.pkl')
    
    if not os.path.exists(model_path):
        print(f"[!] Model file not found: {model_path}")
        print("[!] Please train the model first by running:")
        print("    python src\\threat_detector.py")
        return False
    
    try:
        model_data = joblib.load(model_path)
        explainer = shap.TreeExplainer(model_data['model'])
        print("[+] Model loaded successfully")
        print(f"[+] Features required: {len(model_data['feature_names'])}")
        print(f"[+] Classes: {list(model_data['label_encoder'].classes_)}")
        return True
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    if model_data is None:
        return "Model not loaded. Please train the model first.", 500
    return render_template('dashboard.html')

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information including required features"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'status': 'ready',
        'n_features': len(model_data['feature_names']),
        'features': model_data['feature_names'],
        'classes': list(model_data['label_encoder'].classes_)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict threat from network traffic features
    """
    if model_data is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({'error': 'No data provided in request body'}), 400
        
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" field in request',
                'example': {'features': {'Destination Port': 80, 'Flow Duration': 120000}}
            }), 400
        
        features = data['features']
        
        if not isinstance(features, dict):
            return jsonify({'error': 'Features must be a JSON object (key-value pairs)'}), 400
        
        if len(features) == 0:
            return jsonify({'error': 'Features object is empty'}), 400
        
        print(f"[*] Received {len(features)} features")
        print(f"[*] Sample features: {list(features.keys())[:5]}")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Get required features
        required_features = model_data['feature_names']
        
        # Check which features are missing
        missing_features = set(required_features) - set(df.columns)
        
        if missing_features:
            print(f"[*] Missing {len(missing_features)} features, filling with 0")
            for feat in missing_features:
                df[feat] = 0
        
        # Ensure correct feature order
        df = df[required_features]
        
        print("[*] Feature alignment complete")
        
        # Scale features
        X_scaled = model_data['scaler'].transform(df)
        
        # Predict
        prediction = model_data['model'].predict(X_scaled)[0]
        probabilities = model_data['model'].predict_proba(X_scaled)[0]
        
        # Get label
        predicted_label = model_data['label_encoder'].classes_[prediction]
        
        print(f"[+] Prediction: {predicted_label} (confidence: {max(probabilities):.2%})")
        
        # Calculate SHAP values
        try:
            shap_values = explainer.shap_values(X_scaled)
            
            # For multi-class, get values for predicted class
            if isinstance(shap_values, list):
                shap_vals = shap_values[prediction][0]
            else:
                shap_vals = shap_values[0]
            
            # Get top contributing features
            feature_contributions = []
            for idx, (feat, val) in enumerate(zip(required_features, shap_vals)):
                feature_contributions.append({
                    'feature': feat,
                    'value': float(df.iloc[0][feat]),
                    'shap_value': float(val),
                    'impact': 'Increases Risk' if val > 0 else 'Decreases Risk'
                })
            
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
        except Exception as e:
            print(f"[!] SHAP calculation failed: {e}")
            feature_contributions = []
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'prediction': predicted_label,
            'is_threat': predicted_label != 'BENIGN',
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(model_data['label_encoder'].classes_, probabilities)
            },
            'top_features': feature_contributions[:10],
            'threat_level': get_threat_level(max(probabilities), predicted_label),
            'features_used': len(required_features),
            'features_provided': len(features)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[!] Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'details': 'Check server console for full traceback'
        }), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict threats for batch of network flows
    """
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field in request'}), 400
        
        flows = data['data']
        
        if not isinstance(flows, list):
            return jsonify({'error': 'Data must be an array of flow objects'}), 400
        
        if len(flows) == 0:
            return jsonify({'error': 'Data array is empty'}), 400
        
        print(f"[*] Processing batch of {len(flows)} flows")
        
        # Convert to DataFrame
        df = pd.DataFrame(flows)
        
        # Get required features
        required_features = model_data['feature_names']
        
        # Fill missing features with 0
        for feat in required_features:
            if feat not in df.columns:
                df[feat] = 0
        
        # Ensure correct features
        df = df[required_features]
        
        # Scale and predict
        X_scaled = model_data['scaler'].transform(df)
        predictions = model_data['model'].predict(X_scaled)
        probabilities = model_data['model'].predict_proba(X_scaled)
        
        # Build response
        results = []
        for idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
            label = model_data['label_encoder'].classes_[pred]
            results.append({
                'index': idx,
                'prediction': label,
                'is_threat': label != 'BENIGN',
                'confidence': float(max(probs)),
                'threat_level': get_threat_level(max(probs), label)
            })
        
        # Summary statistics
        threat_count = sum(1 for r in results if r['is_threat'])
        
        return jsonify({
            'total_flows': len(results),
            'threats_detected': threat_count,
            'threat_percentage': (threat_count / len(results)) * 100,
            'results': results
        })
        
    except Exception as e:
        print(f"[!] Batch prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 400

def get_threat_level(confidence, label):
    """Determine threat severity level"""
    if label == 'BENIGN':
        return 'SAFE'
    elif confidence > 0.9:
        return 'CRITICAL'
    elif confidence > 0.7:
        return 'HIGH'
    elif confidence > 0.5:
        return 'MEDIUM'
    else:
        return 'LOW'

if __name__ == '__main__':
    print("="*60)
    print("THREAT DETECTION DASHBOARD")
    print("="*60)
    
    # Load model
    if not load_system():
        print("\n[!] Failed to load model. Exiting...")
        sys.exit(1)
    
    print("\n[+] Starting Flask server...")
    print("[+] Dashboard will be available at: http://localhost:5000")
    print("[+] Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)