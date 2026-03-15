# Kitsune Implementation

---

## Project Structure

```
kitsune_impl/
│
├── core/                          # Shared algorithmic components
│   ├── inc_stat.py                # Damped incremental statistics (1D & 2D)
│   ├── feature_extractor.py       # Network-layer FE: 115 features per packet
│   ├── feature_mapper.py          # Online hierarchical clustering (FM)
│   ├── kitnet.py                  # KitNET: ensemble autoencoder anomaly detector
│   └── kitsune.py                 # Full Kitsune pipeline (FE + FM + AD)
│
├──── dataset_reader.py          # read the dataset (no external dependencies)
│  
│──── run_replication.py         # Evaluation script for KitNET PCAP datasets
│
│
├── evaluation/
│   ├── metrics.py                 # TPR, FNR, AUC, EER, F1 computation
│   └── plot_results.py            # ROC curves, score distributions, AUC bar chart
│
└── utils/
    └── tests.py                   # Unit tests for all core components
```

---

## Dependencies

```
pip install numpy scipy scikit-learn matplotlib scapy dpkt
```

---

## Running Tests

```bash
cd kitsune_impl
python -m utils.tests
```

All 8 test suites should pass before running the full evaluation.

---

## Phase 1: Kitsune Replication

### Dataset preparation
Download the KitNET PCAP datasets from:
https://github.com/ymirsky/KitNET-py

Arrange the directory as:
```
datasets/
    OS_Scan/
        traffic.pcap
        labels.csv          # columns: packet_index,label  (0=benign, 1=malicious)
    Fuzzing/
        traffic.pcap
        labels.csv
    ARP_MitM/  ...
    SYN_DoS/   ...
    SSDP_Flood/ ...
    Mirai/     ...
    ...
```

### Run evaluation (m=10)
```bash
python -m phase1.run_replication \
    --dataset_dir ./datasets \
    --n_train 1000000 \
    --m 10 \
    --output_dir ./results/phase1_m10
```

### Run evaluation (m=1)
```bash
python -m phase1.run_replication \
    --dataset_dir ./datasets \
    --n_train 1000000 \
    --m 1 \
    --output_dir ./results/phase1_m1
```

### Plot results
```bash
python -m evaluation.plot_results \
    --results_dir ./results/phase1_m10 \
    --output_dir  ./results/plots
```

## Output Format

Each evaluation run produces a `results/<dataset>/` directory containing:
- `scores.npy`   — raw anomaly scores for all eval packets
- `labels.npy`   — ground-truth binary labels
- `metrics.json` — TPR@FPR=0, TPR@FPR=0.001, AUC, EER, F1

A `summary.json` aggregating all datasets is written to the top-level
results directory.
