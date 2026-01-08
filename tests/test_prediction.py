"""
Test the threat detection system
"""
import joblib
import pandas as pd

print("Loading model...")
model_data = joblib.load('threat_detector.pkl')

# Sample features
features = {
    "Destination Port": 80,
    "Flow Duration": 120000,
    "Total Fwd Packets": 8,
    "Total Backward Packets": 7,
    "Total Length of Fwd Packets": 480,
    "Total Length of Bwd Packets": 420,
    "Fwd Packet Length Max": 120,
    "Fwd Packet Length Min": 60,
    "Fwd Packet Length Mean": 60,
    "Fwd Packet Length Std": 20,
    "Bwd Packet Length Max": 100,
    "Bwd Packet Length Min": 50,
    "Bwd Packet Length Mean": 60,
    "Bwd Packet Length Std": 15,
    "Flow Bytes/s": 7500,
    "Flow Packets/s": 125,
    "Flow IAT Mean": 8571,
    "Flow IAT Std": 5000,
    "Flow IAT Max": 20000,
    "Flow IAT Min": 1000,
    "Fwd IAT Mean": 10000,
    "Fwd IAT Std": 6000,
    "Fwd IAT Max": 25000,
    "Fwd IAT Min": 1500,
    "Bwd IAT Mean": 7000,
    "Bwd IAT Std": 5500,
    "Bwd IAT Max": 22000,
    "Bwd IAT Min": 1200,
    "PSH Flag Count": 2,
    "SYN Flag Count": 1,
    "FIN Flag Count": 1,
    "ACK Flag Count": 15,
    "Average Packet Size": 60,
    "Fwd Avg Bytes/Bulk": 60,
    "Bwd Avg Bytes/Bulk": 60,
    "Protocol": 6
}

# Ensure all features are present
for feat in model_data['feature_names']:
    if feat not in features:
        features[feat] = 0

# Create DataFrame
df = pd.DataFrame([features])
df = df[model_data['feature_names']]

# Scale and predict
X_scaled = model_data['scaler'].transform(df)
prediction = model_data['model'].predict(X_scaled)[0]
probabilities = model_data['model'].predict_proba(X_scaled)[0]

# Get label
predicted_label = model_data['label_encoder'].classes_[prediction]
confidence = max(probabilities)

print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence:.2%}")
print(f"Is Threat: {predicted_label != 'BENIGN'}")
print("="*60)