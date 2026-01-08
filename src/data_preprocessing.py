"""
Advanced Data Preprocessing for Network Traffic
Handles Wireshark PCAP and Splunk log parsing
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from collections import defaultdict
import json
from datetime import datetime

class NetworkTrafficParser:
    """
    Parse and extract features from raw network captures
    """
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'packets': [],
            'fwd_packets': [],
            'bwd_packets': [],
            'start_time': None,
            'end_time': None
        })
    
    def parse_pcap(self, pcap_file):
        """
        Parse PCAP file using Scapy
        
        Args:
            pcap_file: Path to PCAP file
        
        Returns:
            DataFrame with extracted features
        """
        print(f"[*] Parsing PCAP file: {pcap_file}")
        
        packets = rdpcap(pcap_file)
        print(f"[+] Loaded {len(packets)} packets")
        
        # Group packets into flows
        for pkt in packets:
            if IP in pkt:
                self._process_packet(pkt)
        
        # Extract features from flows
        features_list = []
        for flow_id, flow_data in self.flows.items():
            features = self._extract_flow_features(flow_id, flow_data)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        print(f"[+] Extracted {len(df)} flows")
        
        return df
    
    def _process_packet(self, pkt):
        """Process individual packet and add to flow"""
        ip_layer = pkt[IP]
        
        # Create flow identifier
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        # Get ports if TCP/UDP
        src_port = 0
        dst_port = 0
        protocol = ip_layer.proto
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            protocol = 6
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            protocol = 17
        
        # Bidirectional flow ID (sorted to group both directions)
        flow_id = tuple(sorted([
            (src_ip, src_port),
            (dst_ip, dst_port)
        ]) + [protocol])
        
        # Store packet info
        pkt_info = {
            'timestamp': float(pkt.time),
            'length': len(pkt),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'flags': self._get_tcp_flags(pkt) if TCP in pkt else {}
        }
        
        flow = self.flows[flow_id]
        
        # Initialize start time
        if flow['start_time'] is None:
            flow['start_time'] = pkt_info['timestamp']
        
        flow['end_time'] = pkt_info['timestamp']
        flow['packets'].append(pkt_info)
        
        # Determine direction (forward/backward)
        # Forward: first packet direction
        if len(flow['packets']) == 1:
            flow['fwd_direction'] = (src_ip, src_port)
        
        if (src_ip, src_port) == flow['fwd_direction']:
            flow['fwd_packets'].append(pkt_info)
        else:
            flow['bwd_packets'].append(pkt_info)
    
    def _get_tcp_flags(self, pkt):
        """Extract TCP flags"""
        if TCP in pkt:
            tcp = pkt[TCP]
            return {
                'FIN': int(tcp.flags.F),
                'SYN': int(tcp.flags.S),
                'RST': int(tcp.flags.R),
                'PSH': int(tcp.flags.P),
                'ACK': int(tcp.flags.A),
                'URG': int(tcp.flags.U)
            }
        return {}
    
    def _extract_flow_features(self, flow_id, flow_data):
        """
        Extract CICIDS2017-compatible features from flow
        """
        packets = flow_data['packets']
        fwd_packets = flow_data['fwd_packets']
        bwd_packets = flow_data['bwd_packets']
        
        # Basic flow info
        duration = (flow_data['end_time'] - flow_data['start_time']) * 1e6  # microseconds
        
        # Packet counts
        total_fwd_packets = len(fwd_packets)
        total_bwd_packets = len(bwd_packets)
        
        # Packet lengths
        fwd_lengths = [p['length'] for p in fwd_packets]
        bwd_lengths = [p['length'] for p in bwd_packets]
        
        # Inter-arrival times
        fwd_iat = self._calculate_iat(fwd_packets)
        bwd_iat = self._calculate_iat(bwd_packets)
        flow_iat = self._calculate_iat(packets)
        
        # TCP flags
        psh_count = sum(p['flags'].get('PSH', 0) for p in packets)
        syn_count = sum(p['flags'].get('SYN', 0) for p in packets)
        fin_count = sum(p['flags'].get('FIN', 0) for p in packets)
        ack_count = sum(p['flags'].get('ACK', 0) for p in packets)
        
        # Build feature dict
        features = {
            'Destination Port': packets[0]['dst_port'] if packets else 0,
            'Flow Duration': duration if duration > 0 else 1,
            'Total Fwd Packets': total_fwd_packets,
            'Total Backward Packets': total_bwd_packets,
            'Total Length of Fwd Packets': sum(fwd_lengths),
            'Total Length of Bwd Packets': sum(bwd_lengths),
            'Fwd Packet Length Max': max(fwd_lengths) if fwd_lengths else 0,
            'Fwd Packet Length Min': min(fwd_lengths) if fwd_lengths else 0,
            'Fwd Packet Length Mean': np.mean(fwd_lengths) if fwd_lengths else 0,
            'Fwd Packet Length Std': np.std(fwd_lengths) if fwd_lengths else 0,
            'Bwd Packet Length Max': max(bwd_lengths) if bwd_lengths else 0,
            'Bwd Packet Length Min': min(bwd_lengths) if bwd_lengths else 0,
            'Bwd Packet Length Mean': np.mean(bwd_lengths) if bwd_lengths else 0,
            'Bwd Packet Length Std': np.std(bwd_lengths) if bwd_lengths else 0,
            'Flow Bytes/s': (sum(fwd_lengths) + sum(bwd_lengths)) / (duration / 1e6) if duration > 0 else 0,
            'Flow Packets/s': len(packets) / (duration / 1e6) if duration > 0 else 0,
            'Flow IAT Mean': np.mean(flow_iat) if flow_iat else 0,
            'Flow IAT Std': np.std(flow_iat) if flow_iat else 0,
            'Flow IAT Max': max(flow_iat) if flow_iat else 0,
            'Flow IAT Min': min(flow_iat) if flow_iat else 0,
            'Fwd IAT Mean': np.mean(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Std': np.std(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Max': max(fwd_iat) if fwd_iat else 0,
            'Fwd IAT Min': min(fwd_iat) if fwd_iat else 0,
            'Bwd IAT Mean': np.mean(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Std': np.std(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Max': max(bwd_iat) if bwd_iat else 0,
            'Bwd IAT Min': min(bwd_iat) if bwd_iat else 0,
            'PSH Flag Count': psh_count,
            'SYN Flag Count': syn_count,
            'FIN Flag Count': fin_count,
            'ACK Flag Count': ack_count,
            'Average Packet Size': np.mean([p['length'] for p in packets]) if packets else 0,
            'Fwd Avg Bytes/Bulk': sum(fwd_lengths) / total_fwd_packets if total_fwd_packets > 0 else 0,
            'Bwd Avg Bytes/Bulk': sum(bwd_lengths) / total_bwd_packets if total_bwd_packets > 0 else 0,
            'Protocol': packets[0]['protocol'] if packets else 0
        }
        
        return features
    
    def _calculate_iat(self, packets):
        """Calculate inter-arrival times"""
        if len(packets) < 2:
            return []
        
        times = [p['timestamp'] for p in packets]
        iat = [(times[i+1] - times[i]) * 1e6 for i in range(len(times)-1)]
        return iat
    
    def parse_splunk_logs(self, log_file):
        """
        Parse Splunk-format logs
        
        Args:
            log_file: Path to Splunk log file (JSON or CSV)
        """
        print(f"[*] Parsing Splunk logs: {log_file}")
        
        if log_file.endswith('.json'):
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            df = pd.DataFrame(logs)
        else:
            df = pd.read_csv(log_file)
        
        print(f"[+] Loaded {len(df)} log entries")
        
        # Extract network features from logs
        # This depends on your Splunk log format
        # Example: parse firewall logs
        
        return self._extract_features_from_logs(df)
    
    def _extract_features_from_logs(self, df):
        """
        Extract features from Splunk log format
        Customize based on your log structure
        """
        # Example feature extraction
        features = pd.DataFrame()
        
        # Map common Splunk fields to features
        field_mapping = {
            'dest_port': 'Destination Port',
            'bytes_out': 'Total Length of Fwd Packets',
            'bytes_in': 'Total Length of Bwd Packets',
            'duration': 'Flow Duration'
        }
        
        for splunk_field, feature_name in field_mapping.items():
            if splunk_field in df.columns:
                features[feature_name] = df[splunk_field]
        
        return features


def preprocess_cicids2017(input_file, output_file):
    """
    Preprocess CICIDS2017 dataset
    """
    print("[*] Preprocessing CICIDS2017 dataset...")
    
    df = pd.read_csv(input_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    df = df.fillna(0)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove constant features
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(columns=cols_to_drop)
    
    print(f"[+] Dropped {len(cols_to_drop)} constant features")
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"[+] Saved to {output_file}")
    
    return df


def create_balanced_dataset(df, label_col='Label', sample_size=10000):
    """
    Create balanced dataset by sampling
    """
    print("[*] Creating balanced dataset...")
    
    labels = df[label_col].unique()
    samples_per_class = sample_size // len(labels)
    
    balanced_dfs = []
    for label in labels:
        subset = df[df[label_col] == label]
        if len(subset) > samples_per_class:
            subset = subset.sample(n=samples_per_class, random_state=42)
        balanced_dfs.append(subset)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[+] Balanced dataset shape: {balanced_df.shape}")
    print(f"[+] Class distribution:\n{balanced_df[label_col].value_counts()}")
    
    return balanced_df


if __name__ == "__main__":
    # Example 1: Parse PCAP file
    parser = NetworkTrafficParser()
    
    # Uncomment to parse your PCAP files
    # df = parser.parse_pcap('capture.pcap')
    # df.to_csv('extracted_features.csv', index=False)
    
    # Example 2: Preprocess CICIDS2017
    # processed_df = preprocess_cicids2017(
    #     'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    #     'processed_data.csv'
    # )
    
    # Example 3: Create balanced dataset
    # balanced_df = create_balanced_dataset(processed_df)
    # balanced_df.to_csv('balanced_data.csv', index=False)
    
    print("[+] Data preprocessing utilities ready")
