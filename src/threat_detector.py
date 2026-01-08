"""
AI-Powered Threat Detection System with Explainability
Windows-optimized version
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for Windows
plt.switch_backend('Agg')

class ThreatDetectionSystem:
    """
    Main class for network threat detection with explainability
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.explainer = None
        
    def load_and_preprocess_data(self, filepath):
        """
        Load CICIDS2017 dataset and perform preprocessing
        """
        print("[*] Loading dataset...")
        print(f"[*] Reading file: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")
        
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Display dataset info
        print(f"[+] Dataset shape: {df.shape}")
        print(f"[+] Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Identify label column
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        
        print(f"[+] Label column: {label_col}")
        print(f"[+] Attack types distribution:")
        print(df[label_col].value_counts())
        
        return df, label_col
    
    def feature_engineering(self, df, label_col):
        """
        Extract and engineer features from raw data
        """
        print("\n[*] Performing feature engineering...")
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"[+] Features extracted: {len(self.feature_names)}")
        print(f"[+] Classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def train_model(self, X_train, y_train, n_estimators=100):
        """
        Train the threat detection model
        """
        print("\n[*] Training Random Forest model...")
        print(f"[*] Training samples: {len(X_train)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=0
        )
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        print("[+] Model training completed")
        
        return X_train_scaled
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        print("\n[*] Evaluating model...")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Threat Detection Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save to outputs folder
        output_path = 'outputs\\confusion_matrix.png'
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Confusion matrix saved to '{output_path}'")
        
        # Feature Importance
        self.plot_feature_importance()
        
        return X_test_scaled, y_pred, y_pred_proba
    
    def plot_feature_importance(self, top_n=20):
        """
        Visualize feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), 
                      [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.tight_layout()
            
            output_path = 'outputs\\feature_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Feature importance saved to '{output_path}'")
    
    def initialize_explainer(self, X_train_scaled):
        """
        Initialize SHAP explainer for model interpretability
        """
        print("\n[*] Initializing SHAP explainer...")
        
        # Use TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
        print("[+] SHAP explainer initialized")
        return self.explainer
    
    def explain_prediction(self, sample_index, X_test_scaled, y_test, y_pred):
        """
        Provide detailed explanation for a specific prediction
        """
        print(f"\n[*] Explaining prediction for sample {sample_index}...")
        
        # Get sample
        sample = X_test_scaled[sample_index:sample_index+1]
        
        # SHAP values
        shap_values = self.explainer.shap_values(sample)
        
        # True and predicted labels
        true_label = self.label_encoder.classes_[y_test[sample_index]]
        pred_label = self.label_encoder.classes_[y_pred[sample_index]]
        
        print(f"\n{'='*60}")
        print("PREDICTION EXPLANATION")
        print(f"{'='*60}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print(f"Prediction Correct: {true_label == pred_label}")
        
        # For multiclass, select SHAP values for predicted class
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_vals = shap_values[y_pred[sample_index]]
        else:
            shap_vals = shap_values
        
        # Waterfall plot
        try:
            plt.figure(figsize=(10, 6))
            expected_val = (self.explainer.expected_value[y_pred[sample_index]] 
                          if isinstance(self.explainer.expected_value, np.ndarray)
                          else self.explainer.expected_value)
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=expected_val,
                    data=sample[0],
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.tight_layout()
            
            output_path = f'outputs\\explanation_sample_{sample_index}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Explanation saved to '{output_path}'")
        except Exception as e:
            print(f"[!] Could not generate waterfall plot: {e}")
        
        return shap_vals
    
    def generate_shap_summary(self, X_test_scaled):
        """
        Generate summary plots for overall model behavior
        """
        print("\n[*] Generating SHAP summary plots...")
        
        # Calculate SHAP values for test set (sample for performance)
        sample_size = min(500, len(X_test_scaled))
        X_sample = X_test_scaled[:sample_size]
        shap_values = self.explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        
        try:
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Multi-class: plot for first attack class
                shap.summary_plot(shap_values[1], X_sample, 
                                feature_names=self.feature_names,
                                show=False)
            else:
                shap.summary_plot(shap_values, X_sample,
                                feature_names=self.feature_names,
                                show=False)
            
            plt.tight_layout()
            output_path = 'outputs\\shap_summary.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] SHAP summary saved to '{output_path}'")
        except Exception as e:
            print(f"[!] Could not generate SHAP summary: {e}")
    
    def save_model(self, filepath='threat_detector.pkl'):
        """
        Save trained model and preprocessing objects
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"[+] Model saved to '{filepath}'")
    
    def load_model(self, filepath='threat_detector.pkl'):
        """
        Load trained model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"[+] Model loaded from '{filepath}'")


def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("AI-POWERED THREAT DETECTION SYSTEM")
    print("Windows Edition")
    print("="*60)
    
    # Initialize system
    detector = ThreatDetectionSystem(model_type='random_forest')
    
    # Set data path - MODIFY THIS TO YOUR FILE LOCATION
    filepath = r'data\raw\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    
    # Alternative: Use sample data
    # filepath = r'data\raw\sample_data.csv'
    
    try:
        df, label_col = detector.load_and_preprocess_data(filepath)
    except FileNotFoundError:
        print("\n[!] Dataset not found!")
        print("[!] Please:")
        print("    1. Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("    2. Place CSV file in: data\\raw\\")
        print("    3. Or run: python src\\generate_sample_data.py")
        return
    
    # Limit dataset size for faster processing (optional)
    if len(df) > 100000:
        print(f"\n[*] Large dataset detected. Sampling 100,000 rows for faster processing...")
        df = df.sample(n=100000, random_state=42)
    
    # Feature engineering
    X, y = detector.feature_engineering(df, label_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n[+] Training set: {X_train.shape}")
    print(f"[+] Test set: {X_test.shape}")
    
    # Train model
    X_train_scaled = detector.train_model(X_train, y_train, n_estimators=100)
    
    # Evaluate
    X_test_scaled, y_pred, y_pred_proba = detector.evaluate_model(X_test, y_test)
    
    # Initialize explainer
    detector.initialize_explainer(X_train_scaled)
    
    # Explain specific predictions
    detector.explain_prediction(0, X_test_scaled, y_test, y_pred)
    detector.explain_prediction(10, X_test_scaled, y_test, y_pred)
    
    # Generate summary
    detector.generate_shap_summary(X_test_scaled)
    
    # Save model
    detector.save_model('threat_detector.pkl')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - threat_detector.pkl (trained model)")
    print("  - outputs\\confusion_matrix.png")
    print("  - outputs\\feature_importance.png")
    print("  - outputs\\explanation_sample_*.png")
    print("  - outputs\\shap_summary.png")
    print("\nNext step: Run the dashboard")
    print("  python src\\dashboard.py")


if __name__ == "__main__":
    main()
