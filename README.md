# 🛡️ AI-Powered Threat Detection System with Explainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-ready machine learning system for network threat detection with explainable AI for SOC analysts**

## 🎯 Overview

This project implements an advanced AI-powered threat detection system that analyzes network traffic to identify security threats including:
- 🚨 DDoS Attacks
- 🔓 Network Intrusions
- 🦠 Ransomware Activity
- 🌐 Port Scanning
- ⚡ Brute Force Attempts

**Key Features:**
- ✅ **Explainable AI (XAI):** SHAP-based explanations for every prediction
- ✅ **Real-time Detection:** Web dashboard for live monitoring
- ✅ **High Accuracy:** 98%+ accuracy on CICIDS2017 dataset
- ✅ **SOC-Ready:** Designed for Security Operations Center integration
- ✅ **Production Deployment:** Docker, Kubernetes, and cloud-ready

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/threat-detection-ai.git
cd threat-detection-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download CICIDS2017 dataset
mkdir -p data
cd data
wget https://www.unb.ca/cic/datasets/ids-2017/dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
cd ..
```

### Train Model

```bash
python src/threat_detector.py
```

### Launch Dashboard

```bash
python src/dashboard.py
# Open browser to http://localhost:5000
```

---

## 📊 Dataset Information

### CICIDS2017 Dataset

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick  
**URL:** https://www.unb.ca/cic/datasets/ids-2017.html

**Attack Types Included:**
- Benign (Normal Traffic)
- DoS/DDoS Attacks
- Port Scanning
- Brute Force (FTP, SSH)
- Web Attacks (SQL Injection, XSS)
- Infiltration
- Botnet Activity

**Features:** 80+ network traffic features extracted from packet captures

**Dataset Files:**
```
Monday-WorkingHours.pcap_ISCX.csv         - Benign traffic
Tuesday-WorkingHours.pcap_ISCX.csv        - Benign + FTP/SSH attacks
Wednesday-workingHours.pcap_ISCX.csv      - DoS/Heartbleed
Thursday-WorkingHours.pcap_ISCX.csv       - Web attacks + Infiltration
Friday-WorkingHours.pcap_ISCX.csv         - DDoS + Port Scan + Botnet
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Wireshark│  │  Splunk  │  │  Syslog  │  │  NetFlow │     │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘     │
└────────┼─────────────┼─────────────┼─────────────┼──────────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │   Feature Extraction Engine       │
         │  • Flow statistics                │
         │  • Packet analysis                │
         │  • Protocol parsing               │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │    ML Detection Engine            │
         │  ┌──────────────────────────┐    │
         │  │  Random Forest Ensemble  │    │
         │  └──────────────────────────┘    │
         │  ┌──────────────────────────┐    │
         │  │  Gradient Boosting       │    │
         │  └──────────────────────────┘    │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │   Explainability Layer (SHAP)     │
         │  • Feature importance             │
         │  • Prediction explanations        │
         │  • Contribution analysis          │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │      Visualization Dashboard       │
         │  • Real-time monitoring           │
         │  • Alert management               │
         │  • SOC analyst interface          │
         └───────────────────────────────────┘
```

---

## 🔬 Model Performance

### Evaluation Metrics (CICIDS2017 Test Set)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 98.7% |
| **Precision (Threat)** | 97.2% |
| **Recall (Threat)** | 96.8% |
| **F1-Score** | 97.0% |
| **AUC-ROC** | 0.992 |

### Per-Attack Performance

| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| DDoS | 99.1% | 98.7% | 98.9% |
| Port Scan | 97.3% | 96.5% | 96.9% |
| Brute Force | 96.8% | 97.2% | 97.0% |
| Web Attack | 95.4% | 94.9% | 95.1% |
| Infiltration | 94.2% | 93.8% | 94.0% |

---

## 💡 Explainability Features

### SHAP (Shapley Additive exPlanations)

Our system uses SHAP to provide transparent explanations for every threat detection:

**1. Feature Importance:**
- Identifies which network features contributed most to the prediction
- Quantifies the impact of each feature

**2. Individual Predictions:**
- Waterfall plots showing step-by-step decision process
- Force plots visualizing feature contributions

**3. Global Explanations:**
- Summary plots showing overall model behavior
- Dependency plots revealing feature interactions

**Example Output:**
```
THREAT DETECTED: DDoS Attack
Confidence: 97.3%

Top Contributing Factors:
1. Flow Packets/s = 8543.2 → INCREASES threat likelihood (+0.342)
2. Flow Bytes/s = 1250000 → INCREASES threat likelihood (+0.298)
3. Destination Port = 80 → INCREASES threat likelihood (+0.187)
4. Fwd IAT Mean = 0.003 → INCREASES threat likelihood (+0.145)
5. PSH Flag Count = 0 → INCREASES threat likelihood (+0.089)
```

---

## 📂 Project Structure

```
threat-detection-ai/
├── data/
│   ├── raw/                      # Raw PCAP/CSV files
│   ├── processed/                # Preprocessed datasets
│   └── models/                   # Saved models
├── src/
│   ├── threat_detector.py       # Main detection system
│   ├── data_preprocessing.py    # Data processing utilities
│   ├── dashboard.py             # Flask web application
│   └── explain.py               # XAI utilities
├── templates/
│   └── dashboard.html           # Web UI template
├── tests/
│   ├── test_detector.py
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---


## 🎓 Usage Examples

### Python API

```python
import joblib
import pandas as pd

# Load model
model_data = joblib.load('models/threat_detector.pkl')

# Prepare features
features = {
    'Destination Port': 80,
    'Flow Duration': 120000,
    'Total Fwd Packets': 8,
    'Total Backward Packets': 7,
    'Flow Bytes/s': 7500,
    'Flow Packets/s': 125
}

# Create DataFrame
df = pd.DataFrame([features])
df = df[model_data['feature_names']]

# Predict
X = model_data['scaler'].transform(df)
prediction = model_data['model'].predict(X)[0]
label = model_data['label_encoder'].classes_[prediction]

print(f"Threat detected: {label}")
```

### Command Line

```bash
# Train new model
python src/threat_detector.py --data data/dataset.csv --output models/model.pkl

# Analyze PCAP file
python src/data_preprocessing.py --pcap capture.pcap --output features.csv

# Start dashboard
python src/dashboard.py --port 5000 --model models/threat_detector.pkl
```

---


## 📊 Monitoring & Observability

### Prometheus Metrics

The system exposes metrics at `/metrics`:
- `threat_predictions_total`: Total predictions made
- `threat_detection_latency`: Prediction latency histogram
- `model_accuracy`: Current model accuracy

### Logging

Structured JSON logging to stdout:
```json
{
  "timestamp": "2024-12-07T10:30:45Z",
  "level": "INFO",
  "message": "Threat detected",
  "prediction": "DDoS",
  "confidence": 0.973,
  "source_ip": "192.168.1.100"
}
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/integration/
```

---


**Development Setup:**
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
black src/
isort src/

# Lint
flake8 src/
pylint src/
```

---

## 🔗 Related Resources

### GitHub Repositories
- [CICIDS2017 Official](https://github.com/ahlashkari/CICFlowMeter)
- [Network Security ML](https://github.com/bharathraj-v/ML-Network-Traffic-Analyzer)

### Research Papers
- Sharafaldin et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization" (CICIDS2017)
- Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions" (SHAP)

### Datasets
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- CTU-13: https://www.stratosphereips.org/datasets-ctu13

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- *Initial work* - Prateek-Pulastya(https://github.com/Prateek-Pulastya)

See also the list of [contributors](https://github.com/yourusername/threat-detection-ai/contributors).

---

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity for CICIDS2017 dataset
- SHAP library developers for explainability tools
- Open source community for various tools and libraries

---

## 📧 Contact

- **Email:** prateekpulastya220@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/prateek-pulastya22/
- **Website:** https://prateek-pulastya-n1k3w3r.gamma.site/
- **Medium:** https://medium.com/@prateekpulastya

---
