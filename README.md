# Kitsune (KitNET) Replication on Pre-Extracted CSVs

This project reimplements the **KitNET** anomaly detector from the Kitsune NIDS paper (Mirsky et al., NDSS 2018) and evaluates it on the Kitsune Network Attack Dataset using pre-extracted 115-dimensional feature CSVs (20% sampled subsets).

The repository supports:
- A **CSV-based replication path** — no pcap parsing required, uses pre-extracted feature CSVs directly.
- A **full Kitsune pipeline** — Feature Extractor + Feature Mapper + KitNET for raw-pcap experiments, available via the `core/` modules.

---

## Project Structure

```
kitsune_impl/
│
├── core/
│   ├── inc_stat.py                # 1D/2D damped incremental statistics (used by FE)
│   ├── feature_extractor.py       # 115-feature packet FE (raw pcap path, not used in CSV replication)
│   ├── feature_mapper.py          # Online correlation-based hierarchical clustering (FM)
│   ├── kitnet.py                  # KitNET: ensemble of autoencoders anomaly detector
│   └── kitsune.py                 # Full end-to-end Kitsune pipeline (FE + FM + AD)
│
├── evaluation/
│   ├── metrics.py                 # AUC, AUPRC, EER, TPR@FPR=0/0.001, F1, confusion matrix
│   └── plot_results.py            # ROC curves, PR curves, score distributions, timelines
│
├── dataset_reader.py              # Streams paired feature + label CSV rows
│                                  # Discovers *_dataset.csv / *_labels.csv pairs automatically
├── run_replication.py             # Main replication script for pre-extracted feature CSVs
├── check.py                       # Quick sanity check on saved scores and labels
└── checkmetric.py                 # Quick print of a saved metrics.json
```

---

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

For the raw pcap pipeline (optional, not needed for CSV replication):

```bash
pip install scapy dpkt
```

---

## Data Preparation

### Source Dataset

We use the **Kitsune Network Attack Dataset** (pre-extracted feature CSVs):

- Kaggle: `ymirsky/network-attack-dataset-kitsune`
- UCI: Kitsune Network Attack Dataset

Each attack scenario provides:
- `<AttackName>_dataset.csv` — feature matrix (115 features per row)
- `<AttackName>_labels.csv` — ground-truth labels aligned row-by-row (0 = benign, 1 = attack)
- `<AttackName>_pcap.pcap(ng)` — raw packet capture (not used in the CSV pipeline)

### 20% Sampled CSVs

This project assumes row-sampled subsets already exist for each attack. Place them under a single directory:

```
data/kitsune_20pct_samples/
    ARP_MitM_20pct_dataset.csv
    ARP_MitM_20pct_labels.csv
    Active_Wiretap_20pct_dataset.csv
    Active_Wiretap_20pct_labels.csv
    Fuzzing_20pct_dataset.csv
    Fuzzing_20pct_labels.csv
    Mirai_Botnet_20pct_dataset.csv
    Mirai_Botnet_20pct_labels.csv
    OS_Scan_20pct_dataset.csv
    OS_Scan_20pct_labels.csv
    SSDP_Flood_20pct_dataset.csv
    SSDP_Flood_20pct_labels.csv
    SSL_Renegotiation_20pct_dataset.csv
    SSL_Renegotiation_20pct_labels.csv
    SYN_DoS_20pct_dataset.csv
    SYN_DoS_20pct_labels.csv
    Video_Injection_20pct_dataset.csv
    Video_Injection_20pct_labels.csv
```

> **Mirai note:** if the Mirai dataset CSV has 116 columns due to an extra index column, drop the first column and resave before running. All feature files must have exactly 115 columns.

---

## How the Pipeline Works

For each dataset the runner performs these steps in order:

1. `PairedCSVDatasetReader` streams `(row_index, features, label)` from the paired `*_dataset.csv` and `*_labels.csv` files.
2. **FM training (FMgrace rows, default 5,000):** `FeatureMapper.update(x)` accumulates incremental correlation statistics.
3. **FM fitting:** `FeatureMapper.fit()` builds the correlation distance matrix and performs hierarchical clustering, yielding k feature groups.
4. KitNET is initialised with one autoencoder per feature group.
5. **AD training (ADgrace rows, default 50,000):** `FeatureMapper.transform(x)` splits each row into sub-instances, which are passed to `KitNET.train(...)`. The maximum training score φ is tracked throughout.
6. **Evaluation (all remaining rows):** `KitNET.execute(...)` scores each row without updating weights. Metrics are computed only on this post-warmup segment using the real labels from the CSV.

---

## How to Run

### 1. Clone and install

```bash
git clone <your-repo-url> kitsune_impl
cd kitsune_impl

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### 2. Run all sampled datasets

```bash
python run_replication.py \
    --sample_dir ./data/kitsune_20pct_samples \
    --output_dir ./results \
    --fm_grace 5000 \
    --ad_grace 50000 \
    --m 10
```

| Argument | Description |
|---|---|
| `--sample_dir` | Folder containing `*_dataset.csv` / `*_labels.csv` pairs |
| `--output_dir` | Where per-dataset results and the summary are saved |
| `--fm_grace` | Number of rows used to train the FeatureMapper |
| `--ad_grace` | Number of rows used to train KitNET after the mapper is fixed |
| `--m` | Maximum features per autoencoder input |
| `--skip_plots` | Skip plot generation at the end (useful when running datasets one at a time) |

### 3. Run a single dataset

Useful for debugging or re-running one attack scenario independently:

```bash
python run_replication.py \
    --features ./data/kitsune_20pct_samples/SSL_Renegotiation_20pct_dataset.csv \
    --labels   ./data/kitsune_20pct_samples/SSL_Renegotiation_20pct_labels.csv \
    --output_dir ./results \
    --fm_grace 5000 \
    --ad_grace 50000 \
    --m 10 \
    --skip_plots
```

> `--features` and `--labels` must always be provided together.

### 4. Generate plots

After all datasets have finished, run this once to generate all summary and per-dataset figures:

```bash
python -m evaluation.plot_results \
    --results_dir ./results \
    --output_dir  ./results/_plots
```

---

## Output Layout

After a complete run the `results/` directory looks like:

```
results/
  ARP_MitM_20pct/
    scores.csv              # row_index, label, score  (eval rows only)
    scores.npy              # scores as numpy array
    labels.npy              # labels aligned with scores
    metrics.json            # full metrics dictionary
    roc_curve.csv           # fpr, tpr columns
    pr_curve.csv            # recall, precision columns

  Active_Wiretap_20pct/
  Fuzzing_20pct/
  Mirai_Botnet_20pct/
  OS_Scan_20pct/
  SSDP_Flood_20pct/
  SSL_Renegotiation_20pct/
  SYN_DoS_20pct/
  Video_Injection_20pct/

  summary_metrics.csv       # one row per dataset, tabular summary
  summary_metrics.json      # same content in JSON

  _plots/
    summary_auc.png
    summary_auprc.png
    summary_runtime.png
    combined_roc.png
    combined_pr.png
    ARP_MitM_20pct_score_dist.png
    ARP_MitM_20pct_timeline.png
    Active_Wiretap_20pct_score_dist.png
    Active_Wiretap_20pct_timeline.png
    ...
```

### `metrics.json` fields

| Field | Description |
|---|---|
| `dataset` | Dataset name |
| `n_total`, `n_benign`, `n_malicious`, `attack_rate` | Eval set composition |
| `AUC` | Area under the ROC curve |
| `AUPRC` | Area under the precision-recall curve |
| `EER` | Equal error rate |
| `TPR_at_FPR_0` / `FNR_at_FPR_0` / `threshold_FPR_0` | Detection at zero false positive rate |
| `TPR_at_FPR_0001` / `FNR_at_FPR_0001` / `threshold_FPR_0001` | Detection at FPR = 0.001 |
| `F1_optimal`, `Precision_opt`, `Recall_opt`, `threshold_opt` | Best-F1 operating point |
| `TP`, `FP`, `FN`, `TN` | Confusion matrix at best-F1 threshold |
| `mean_score_benign` / `mean_score_attack` | Mean anomaly score per class |
| `median_score_benign` / `median_score_attack` | Median anomaly score per class |
| `std_score_benign` / `std_score_attack` | Standard deviation per class |
| `max_score`, `min_score` | Score range across the eval segment |
| `runtime_sec`, `rows_per_sec` | Throughput stats |
| `FMgrace`, `ADgrace`, `warmup_rows` | Grace period configuration |
| `max_cluster_size`, `n_clusters`, `cluster_sizes` | FeatureMapper clustering result |
| `phi_train_max` | Maximum anomaly score seen during AD training |

---

## Sanity Check

You can verify any finished dataset run without running the full eval again:

```python
import numpy as np
import json
from pathlib import Path

ds = Path("results/ARP_MitM_20pct")
scores = np.load(ds / "scores.npy")
labels = np.load(ds / "labels.npy")

print("scores:", scores.shape, "  labels:", labels.shape)
print("score range:", round(scores.min(), 6), "->", round(scores.max(), 6))

with open(ds / "metrics.json") as f:
    m = json.load(f)

print(f"Dataset : {m['dataset']}")
print(f"AUC     : {m['AUC']}")
print(f"AUPRC   : {m['AUPRC']}")
print(f"F1      : {m['F1_optimal']}")
```

If shapes match and metrics look reasonable, the pipeline ran correctly.

---

## Reference

> Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
> **Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection.**
> NDSS Symposium 2018. https://arxiv.org/abs/1802.09089
