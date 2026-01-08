"""
Generate sample network traffic data for testing
"""
import numpy as np 
import pandas as pd 
import os

def generate_sample_data(n_samples=5000, output_file=r'data\raw\sample_data.csv'):
    """
    Generate synthetic network traffic data
    """
    print("[*] Generating sample data...")
    np.random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Features
    data = {
        'Destination Port': np.random.choice([80, 443, 22, 21, 3389], n_samples),
        'Flow Duration': np.random.exponential(50000, n_samples),
        'Total Fwd Packets': np.random.poisson(10, n_samples),
        'Total Backward Packets': np.random.poisson(8, n_samples),
        'Total Length of Fwd Packets': np.random.exponential(500, n_samples),
        'Total Length of Bwd Packets': np.random.exponential(400, n_samples),
        'Fwd Packet Length Max': np.random.exponential(150, n_samples),
        'Fwd Packet Length Min': np.random.exponential(60, n_samples),
        'Fwd Packet Length Mean': np.random.exponential(80, n_samples),
        'Fwd Packet Length Std': np.random.exponential(20, n_samples),
        'Bwd Packet Length Max': np.random.exponential(120, n_samples),
        'Bwd Packet Length Min': np.random.exponential(50, n_samples),
        'Bwd Packet Length Mean': np.random.exponential(70, n_samples),
        'Bwd Packet Length Std': np.random.exponential(15, n_samples),
        'Flow Bytes/s': np.random.exponential(10000, n_samples),
        'Flow Packets/s': np.random.exponential(100, n_samples),
        'Flow IAT Mean': np.random.exponential(10000, n_samples),
        'Flow IAT Std': np.random.exponential(5000, n_samples),
        'Flow IAT Max': np.random.exponential(20000, n_samples),
        'Flow IAT Min': np.random.exponential(1000, n_samples),
        'Fwd IAT Mean': np.random.exponential(12000, n_samples),
        'Fwd IAT Std': np.random.exponential(6000, n_samples),
        'Fwd IAT Max': np.random.exponential(25000, n_samples),
        'Fwd IAT Min': np.random.exponential(1500, n_samples),
        'Bwd IAT Mean': np.random.exponential(11000, n_samples),
        'Bwd IAT Std': np.random.exponential(5500, n_samples),
        'Bwd IAT Max': np.random.exponential(22000, n_samples),
        'Bwd IAT Min': np.random.exponential(1200, n_samples),
        'PSH Flag Count': np.random.poisson(2, n_samples),
        'SYN Flag Count': np.random.poisson(1, n_samples),
        'FIN Flag Count': np.random.poisson(1, n_samples),
        'ACK Flag Count': np.random.poisson(10, n_samples),
        'Average Packet Size': np.random.exponential(100, n_samples),
        'Fwd Avg Bytes/Bulk': np.random.exponential(80, n_samples),
        'Bwd Avg Bytes/Bulk': np.random.exponential(70, n_samples),
        'Protocol': np.random.choice([6, 17], n_samples)
    }
    
    # Labels - 70% BENIGN, 30% DDoS
    labels = np.random.choice(['BENIGN', 'DDoS'], n_samples, p=[0.7, 0.3])
    
    # Make DDoS samples have higher packet rates
    ddos_mask = labels == 'DDoS'
    data['Flow Packets/s'] = np.where(ddos_mask, 
                                       data['Flow Packets/s'] * 10, 
                                       data['Flow Packets/s'])
    data['Flow Bytes/s'] = np.where(ddos_mask, 
                                     data['Flow Bytes/s'] * 10, 
                                     data['Flow Bytes/s'])
    
    df = pd.DataFrame(data)
    df['Label'] = labels
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"[+] Generated {n_samples} samples")
    print(f"[+] Saved to: {output_file}")
    print(f"[+] Label distribution:")
    print(df['Label'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_sample_data(n_samples=5000)
    print("\n[+] Sample data generation complete!")
    print("[+] You can now run: python src\\threat_detector.py")