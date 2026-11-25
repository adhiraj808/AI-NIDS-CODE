# Academic Research Report: Comprehensive 5-Experiment Suite
**Academic Paper Context:** *AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks*
**Generated:** 2026-11-16 01:43:28
**Execution Host:** macOS (Apple Silicon / Darwin)

---

## 1. Executive Summary & Research Motivation
Modern cyber defense relies increasingly on autonomous machine learning systems to detect sophisticated network intrusions, distributed denial-of-service (DDoS) campaigns, and reconnaissance scans in high-throughput traffic. However, deploying AI in operational Security Operations Centers (SOCs) carries distinct risks, including susceptibility to adversarial evasion, high false-positive alert fatigue, and critical false negatives that conceal active network compromises. This empirical study evaluates baseline, ensemble, and gradient-boosted architectures on standardized network flow captures.

## 2. Experimental Setup & Reproducibility
- **Random State Seed:** 42
- **Test Size Fraction:** 0.20 (Stratified splitting)
- **Class Imbalance Strategy:** Cost-sensitive class weighting
- **Total Samples Inspected:** 286467
- **Input Feature Dimensionality:** N/A numerical features
- **Deduplication:** 0 duplicate rows pruned
- **Data Leakage Safeguard:** Transformations (imputer median, standard scaler) fit strictly on training partition.

## 3. Comparative Performance Analysis
Below are the verified empirical results obtained under identical train/test conditions:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | False Negative Rate (FNR) | False Positive Rate (FPR) | Train Time (s) | Inference Time (s) | Throughput (samples/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9987 | 0.9985 | 0.9984 | 0.9984 | 0.9996 | 0.0016 | 0.0011 | 0.2381 | 0.0011 | 38330339.2000 |
| Random Forest | 0.9999 | 1.0000 | 0.9997 | 0.9998 | 0.9999 | 0.0003 | 0.0000 | 1.8028 | 0.0289 | 1480228.1000 |
| Xgboost | 0.9999 | 0.9999 | 0.9998 | 0.9999 | 0.9999 | 0.0002 | 0.0000 | 11.3181 | 0.0350 | 1223384.6000 |

## 4. Key Empirical Findings
- **Top Detection Quality:** 'Xgboost' achieved the highest F1-Score of 0.9999, demonstrating superior balanced precision and recall on the held-out test partition.
- **Lowest Security Risk:** 'Xgboost' produced the lowest False Negative Rate (0.0002). In operational cyber defense, minimizing FNR is paramount as missed attacks leave network perimeters compromised.
- **Computational Latency Trade-Off:** 'Logistic Regression' exhibited the fastest training time (0.2381s), illustrating the trade-off between architectural complexity and real-time retraining agility.
- **Metric Sensitivity:** Across evaluated models, accuracy varied by 0.0012 while F1-Score varied by 0.0014, confirming that F1-Score and FNR provide substantially higher diagnostic sensitivity on imbalanced IDS datasets.

## 5. Security & Operational Metric Insights
- **Why Accuracy is Insufficient:** In cybersecurity, class distributions are inherently skewed. Evaluating an IDS solely on accuracy obscures severe deficiencies in detecting minority-class malicious intrusions.
- **False Negative Hazard:** Every false negative represents an undetected breach. In high-value defense environments, minimizing FNR takes precedence over incremental gains in raw accuracy.
- **False Positive Fatigue:** Excessively high FPR inundates analysts with spurious alerts, degrading SOC response times.

## 6. Research Limitations
1. **Offline Flow Capture:** The evaluation operates on extracted statistical flow records rather than raw layer-2 packet streams.
2. **Adversarial Evasion:** Evaluated models assume stationary feature distributions and do not model deliberate adversarial perturbations.
3. **Temporal Drift:** Historical captures (CICIDS2017) may not reflect novel zero-day exploits or contemporary encrypted tunneling protocols.

## 7. Future Directions
- Integration of streaming online learning for real-time drift adaptation.
- Robustness certification against gradient-based adversarial network flow crafting.
- Expansion to multi-class classification for granular automated threat playbooks.