"""
Sample Dataset Generator for AI-NIDS.
Generates an authentic schema-conforming CICIDS2017 network traffic benchmark sample
(78 statistical flow features) for testing, CI/CD validation, and initial dashboard exploration
when full 50GB raw PCAP captures are not yet downloaded into data/raw/.
"""

from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd

from config.settings import (
    BENIGN_LABEL,
    CICIDS_COMMON_FEATURES,
    DEFAULT_DROP_COLUMNS,
    DEFAULT_RANDOM_STATE,
    SAMPLE_DATA_DIR,
)

logger = logging.getLogger("AI-NIDS.SampleData")


def generate_cicids_benchmark_sample(
    n_samples: int = 5000,
    malicious_ratio: float = 0.20,
    random_state: int = DEFAULT_RANDOM_STATE,
    output_filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates a realistic benchmark sample strictly conforming to the CICIDS2017 schema.
    Simulates statistical distributions of benign flows alongside realistic attack archetypes:
    - PortScan (high packet rate, low duration, SYN/RST flags)
    - DoS / DDoS (high flow packets/s, high forward packets, low inter-arrival time)
    - Brute Force / Web Attack (distinct packet length distributions, PSH flags)
    - Normal Benign Traffic (typical Poisson / log-normal web and enterprise traffic)
    """
    rng = np.random.RandomState(random_state)
    n_malicious = int(n_samples * malicious_ratio)
    n_benign = n_samples - n_malicious

    logger.info("Generating CICIDS2017 benchmark sample (%d rows: %d BENIGN, %d MALICIOUS)...", n_samples, n_benign, n_malicious)

    # 1. Generate Metadata / Non-predictive columns
    flow_ids = [f"192.168.10.{rng.randint(1, 254)}-192.168.10.{rng.randint(1, 254)}-{rng.randint(1024, 65535)}-{rng.choice([80, 443, 22, 21, 8080])}-6" for _ in range(n_samples)]
    src_ips = [f"192.168.10.{rng.randint(2, 250)}" for _ in range(n_samples)]
    src_ports = rng.randint(1024, 65535, size=n_samples)
    dst_ips = [f"192.168.10.{rng.choice([1, 50, 100, 200])}" for _ in range(n_samples)]
    dst_ports = rng.choice([80, 443, 21, 22, 53, 8080, 3389], size=n_samples)
    protocols = rng.choice([6, 17], p=[0.85, 0.15], size=n_samples)  # TCP / UDP
    timestamps = [f"07/07/2017 {rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}" for _ in range(n_samples)]

    # 2. Generate Flow Features
    # Benign distribution: log-normal flow durations, typical enterprise packet sizes
    benign_duration = rng.lognormal(mean=10.0, sigma=2.0, size=n_benign)
    benign_fwd_pkts = rng.poisson(lam=12, size=n_benign) + 1
    benign_bwd_pkts = rng.poisson(lam=10, size=n_benign) + 1
    benign_fwd_len = benign_fwd_pkts * rng.uniform(60, 600, size=n_benign)
    benign_bwd_len = benign_bwd_pkts * rng.uniform(100, 1400, size=n_benign)

    # Malicious distribution (e.g. DoS, PortScan, Patator)
    mal_duration = np.concatenate([
        rng.uniform(50, 5000, size=int(n_malicious * 0.4)),         # Port scans (fast, short)
        rng.lognormal(mean=14.0, sigma=1.0, size=n_malicious - int(n_malicious * 0.4)),  # DoS floods
    ])
    mal_fwd_pkts = np.concatenate([
        rng.choice([1, 2], size=int(n_malicious * 0.4)),             # Scans
        rng.poisson(lam=85, size=n_malicious - int(n_malicious * 0.4)),  # Floods
    ])
    mal_bwd_pkts = np.concatenate([
        np.zeros(int(n_malicious * 0.4), dtype=int),                 # Half-open scans (no bwd response)
        rng.poisson(lam=4, size=n_malicious - int(n_malicious * 0.4)),
    ])
    mal_fwd_len = mal_fwd_pkts * rng.uniform(40, 200, size=n_malicious)
    mal_bwd_len = mal_bwd_pkts * rng.uniform(0, 300, size=n_malicious)

    # Combine distributions
    flow_duration = np.concatenate([benign_duration, mal_duration])
    fwd_pkts = np.concatenate([benign_fwd_pkts, mal_fwd_pkts])
    bwd_pkts = np.concatenate([benign_bwd_pkts, mal_bwd_pkts])
    fwd_len = np.concatenate([benign_fwd_len, mal_fwd_len])
    bwd_len = np.concatenate([benign_bwd_len, mal_bwd_len])

    data = {
        "Flow ID": flow_ids,
        "Source IP": src_ips,
        "Source Port": src_ports,
        "Destination IP": dst_ips,
        "Destination Port": dst_ports,
        "Protocol": protocols,
        "Timestamp": timestamps,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_len,
        "Total Length of Bwd Packets": bwd_len,
        "Fwd Packet Length Max": np.clip(fwd_len / np.maximum(fwd_pkts, 1) * 1.5, 40, 1500),
        "Fwd Packet Length Min": np.clip(fwd_len / np.maximum(fwd_pkts, 1) * 0.5, 20, 600),
        "Fwd Packet Length Mean": fwd_len / np.maximum(fwd_pkts, 1),
        "Fwd Packet Length Std": rng.uniform(5, 120, size=n_samples),
        "Bwd Packet Length Max": np.clip(bwd_len / np.maximum(bwd_pkts, 1) * 1.8, 0, 1500),
        "Bwd Packet Length Min": np.clip(bwd_len / np.maximum(bwd_pkts, 1) * 0.3, 0, 400),
        "Bwd Packet Length Mean": bwd_len / np.maximum(bwd_pkts, 1),
        "Bwd Packet Length Std": rng.uniform(0, 200, size=n_samples),
        "Flow Bytes/s": np.clip((fwd_len + bwd_len) / np.maximum(flow_duration / 1e6, 1e-4), 0, 1e7),
        "Flow Packets/s": np.clip((fwd_pkts + bwd_pkts) / np.maximum(flow_duration / 1e6, 1e-4), 0, 1e6),
        "Flow IAT Mean": flow_duration / np.maximum(fwd_pkts + bwd_pkts, 1),
        "Flow IAT Std": rng.uniform(10, 5000, size=n_samples),
        "Flow IAT Max": flow_duration * 0.6,
        "Flow IAT Min": rng.uniform(0, 50, size=n_samples),
        "Fwd IAT Total": flow_duration * 0.8,
        "Fwd IAT Mean": (flow_duration * 0.8) / np.maximum(fwd_pkts, 1),
        "Fwd IAT Std": rng.uniform(10, 4000, size=n_samples),
        "Fwd IAT Max": flow_duration * 0.5,
        "Fwd IAT Min": rng.uniform(0, 30, size=n_samples),
        "Bwd IAT Total": flow_duration * 0.7,
        "Bwd IAT Mean": (flow_duration * 0.7) / np.maximum(bwd_pkts, 1),
        "Bwd IAT Std": rng.uniform(0, 3000, size=n_samples),
        "Bwd IAT Max": flow_duration * 0.4,
        "Bwd IAT Min": rng.uniform(0, 20, size=n_samples),
        "Fwd PSH Flags": np.concatenate([rng.binomial(1, 0.08, n_benign), rng.binomial(1, 0.45, n_malicious)]),
        "Bwd PSH Flags": np.zeros(n_samples),
        "Fwd URG Flags": np.zeros(n_samples),
        "Bwd URG Flags": np.zeros(n_samples),
        "Fwd Header Length": fwd_pkts * 32,
        "Bwd Header Length": bwd_pkts * 32,
        "Fwd Packets/s": fwd_pkts / np.maximum(flow_duration / 1e6, 1e-4),
        "Bwd Packets/s": bwd_pkts / np.maximum(flow_duration / 1e6, 1e-4),
        "Min Packet Length": np.clip(fwd_len / np.maximum(fwd_pkts, 1) * 0.4, 0, 500),
        "Max Packet Length": np.maximum(fwd_len, bwd_len) * 0.3 + 60,
        "Packet Length Mean": (fwd_len + bwd_len) / np.maximum(fwd_pkts + bwd_pkts, 1),
        "Packet Length Std": rng.uniform(10, 300, size=n_samples),
        "Packet Length Variance": rng.uniform(100, 90000, size=n_samples),
        "FIN Flag Count": np.concatenate([rng.binomial(1, 0.05, n_benign), rng.binomial(1, 0.35, n_malicious)]),
        "SYN Flag Count": np.concatenate([rng.binomial(1, 0.04, n_benign), rng.binomial(1, 0.60, n_malicious)]),
        "RST Flag Count": np.concatenate([rng.binomial(1, 0.02, n_benign), rng.binomial(1, 0.40, n_malicious)]),
        "PSH Flag Count": np.concatenate([rng.binomial(1, 0.09, n_benign), rng.binomial(1, 0.50, n_malicious)]),
        "ACK Flag Count": np.concatenate([rng.binomial(1, 0.90, n_benign), rng.binomial(1, 0.25, n_malicious)]),
        "URG Flag Count": np.zeros(n_samples),
        "CWE Flag Count": np.zeros(n_samples),
        "ECE Flag Count": np.zeros(n_samples),
        "Down/Up Ratio": np.clip(bwd_pkts / np.maximum(fwd_pkts, 1), 0, 10),
        "Average Packet Size": (fwd_len + bwd_len) / np.maximum(fwd_pkts + bwd_pkts, 1),
        "Avg Fwd Segment Size": fwd_len / np.maximum(fwd_pkts, 1),
        "Avg Bwd Segment Size": bwd_len / np.maximum(bwd_pkts, 1),
        "Fwd Header Length.1": fwd_pkts * 32,
        "Fwd Avg Bytes/Bulk": np.zeros(n_samples),
        "Fwd Avg Packets/Bulk": np.zeros(n_samples),
        "Fwd Avg Bulk Rate": np.zeros(n_samples),
        "Bwd Avg Bytes/Bulk": np.zeros(n_samples),
        "Bwd Avg Packets/Bulk": np.zeros(n_samples),
        "Bwd Avg Bulk Rate": np.zeros(n_samples),
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_len,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_len,
        "Init_Win_bytes_forward": np.concatenate([rng.choice([8192, 29200, 65535], size=n_benign), rng.choice([256, 1024, 0], size=n_malicious)]),
        "Init_Win_bytes_backward": np.concatenate([rng.choice([8192, 29200, 65535], size=n_benign), rng.choice([0, 512], size=n_malicious)]),
        "act_data_pkt_fwd": np.maximum(fwd_pkts - 1, 0),
        "min_seg_size_forward": rng.choice([20, 32], size=n_samples),
        "Active Mean": rng.uniform(0, 10000, size=n_samples),
        "Active Std": rng.uniform(0, 1000, size=n_samples),
        "Active Max": rng.uniform(0, 15000, size=n_samples),
        "Active Min": rng.uniform(0, 5000, size=n_samples),
        "Idle Mean": rng.uniform(0, 100000, size=n_samples),
        "Idle Std": rng.uniform(0, 5000, size=n_samples),
        "Idle Max": rng.uniform(0, 120000, size=n_samples),
        "Idle Min": rng.uniform(0, 80000, size=n_samples),
    }

    # Attack label assignments
    attack_types = [
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "FTP-Patator",
        "SSH-Patator",
        "Web Attack – Brute Force",
    ]
    malicious_subtypes = rng.choice(attack_types, size=n_malicious)
    raw_labels = [BENIGN_LABEL] * n_benign + list(malicious_subtypes)

    data["Label"] = raw_labels

    df = pd.DataFrame(data)

    # Shuffle records thoroughly
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_filepath is not None:
        output_filepath = Path(output_filepath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_filepath, index=False)
        logger.info("Saved CICIDS2017 benchmark sample to %s", output_filepath)

    return df


if __name__ == "__main__":
    target = SAMPLE_DATA_DIR / "CICIDS2017_Benchmark_Sample.csv"
    generate_cicids_benchmark_sample(n_samples=5000, output_filepath=target)
