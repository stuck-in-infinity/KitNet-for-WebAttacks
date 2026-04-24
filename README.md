# KitNET for Web Attacks

An extension of KitNET (Kitsune anomaly detector, Mirsky et al. NDSS 2018) for detecting web-layer attacks using HTTP-level feature engineering.

- **Phase 1** — Faithful replication of KitNET on the 9-scenario Kitsune dataset (115-dim pre-extracted CSVs)
- **Phase 2** — Novel extension to web attack detection (CIC-IDS-2018) comparing three feature strategies: Traditional (115-dim), HTTP-only (85-dim), Hybrid (200-dim)
- **Phase 3** — HTTP-tuned KitNET architecture with Leaky ReLU, EMA normalisation, and per-cluster RMSE z-scoring

---

## Project Structure

```
KitNet-for-WebAttacks/
│
├── core/                          Shared KitNET implementation
│   ├── kitnet.py                  Ensemble autoencoder anomaly detector
│   ├── feature_mapper.py          Correlation-based feature clustering
│   ├── feature_extractor.py       115-dim Kitsune network AfterImage features
│   ├── inc_stat.py                Incremental damped statistics (IncStat1D/2D)
│   └── _scipy_shim.py             Pure-numpy fallback (squareform/linkage/fcluster)
│
├── evaluation/                    Shared evaluation utilities
│   ├── metrics.py                 AUC, AUPRC, F1, EER, ROC/PR curves
│   ├── plot_results.py            Score distribution and timeline plots
│   └── _sklearn_shim.py           Pure-numpy fallback for sklearn metrics
│
├── dataset/                       Data (not in version control)
│   └── cse-cic-ids2018/
│       ├── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
│       ├── Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
│       └── CSV_Data/combined/     Combined per-day feature CSVs
│
├── phase1/                        Phase 1 — KitNET replication
│   ├── README.md
│   ├── dataset_reader.py
│   └── run_replication.py
│
├── phase2/                        Phase 2 — Web attack detection
│   ├── reader.py                  CIC-IDS-2018 CSV reader
│   ├── pipeline.py                Shared KitNET experiment pipeline
│   ├── http/                      HTTP feature extraction library
│   ├── traditional/               Experiment A — 115-dim network features
│   ├── http_only/                 Experiment B — 85-dim HTTP features
│   ├── hybrid/                    Experiment C — 200-dim (network + HTTP)
│   ├── csv_baseline/              Quick-run on original CIC CSVs (no PCAPs)
│   └── comparison/                Cross-experiment comparison
│
├── phase3/                        Phase 3 — HTTP-tuned KitNET architecture
│   ├── http_kitnet.py             HTTPFeatureMapper, HTTPAutoencoder, HTTPKitNET
│   ├── http_pipeline.py           Phase 3 pipeline with HTTP-tuned warmup
│   └── run_phase3_http.py         Standalone Kaggle-ready runner
│
├── results/                       Phase 1 results
├── results_phase2/                Phase 2 results
└── phase3results/                 Phase 3 results
```

---

## Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn

# For PCAP feature extraction only:
pip install dpkt scapy
```

---

## Phase 1 — KitNET Replication

Replicates the original Kitsune paper on 9 network attack scenarios using pre-extracted 115-dim AfterImage CSVs.

### Dataset

Download the Kitsune Network Attack Dataset:
- Kaggle: https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune

Place paired `*_dataset.csv` and `*_labels.csv` files under one directory:

```
data/kitsune_samples/
    ARP_MitM_dataset.csv          ARP_MitM_labels.csv
    Active_Wiretap_dataset.csv    Active_Wiretap_labels.csv
    Fuzzing_dataset.csv           Fuzzing_labels.csv
    Mirai_Botnet_dataset.csv      Mirai_Botnet_labels.csv
    OS_Scan_dataset.csv           OS_Scan_labels.csv
    SSDP_Flood_dataset.csv        SSDP_Flood_labels.csv
    SSL_Renegotiation_dataset.csv SSL_Renegotiation_labels.csv
    SYN_DoS_dataset.csv           SYN_DoS_labels.csv
    Video_Injection_dataset.csv   Video_Injection_labels.csv
```

### Run all 9 scenarios

```bash
python phase1/run_replication.py \
    --sample_dir data/kitsune_samples \
    --output_dir results \
    --fm_grace   5000 \
    --ad_grace   50000 \
    --m          10
```

### Run a single scenario

```bash
python phase1/run_replication.py \
    --features   data/kitsune_samples/ARP_MitM_dataset.csv \
    --labels     data/kitsune_samples/ARP_MitM_labels.csv \
    --output_dir results \
    --fm_grace   5000 \
    --ad_grace   50000 \
    --m          10
```

### Mirai Botnet — reduced warmup

Mirai has only ~24k benign rows before attacks begin. Use reduced grace periods:

```bash
python phase1/run_replication.py \
    --features   data/kitsune_samples/Mirai_Botnet_dataset.csv \
    --labels     data/kitsune_samples/Mirai_Botnet_labels.csv \
    --output_dir results \
    --fm_grace   2000 \
    --ad_grace   18000 \
    --m          10
```

### Generate plots after all runs

```bash
python -m evaluation.plot_results \
    --results_dir results \
    --output_dir  results/_plots
```

### Phase 1 arguments

| Argument | Default | Description |
|---|---|---|
| `--sample_dir` | — | Folder with `*_dataset.csv` / `*_labels.csv` pairs |
| `--features` | — | Single dataset CSV (use with `--labels`) |
| `--labels` | — | Single labels CSV (use with `--features`) |
| `--output_dir` | `results` | Where results are saved |
| `--fm_grace` | `5000` | Rows used to train the FeatureMapper |
| `--ad_grace` | `50000` | Rows used to train KitNET autoencoders |
| `--m` | `10` | Max features per autoencoder cluster |
| `--skip_plots` | `False` | Skip plot generation |

---

## Phase 2 — Web Attack Detection (CIC-IDS-2018)

Three independent KitNET detectors evaluated on Thursday 22-Feb-2018 and Friday 23-Feb-2018 from CIC-IDS-2018.

### Dataset

Download the CIC-IDS-2018 raw PCAPs and ground-truth CSVs:

```bash
# Thursday
aws s3 sync s3://cse-cic-ids2018/.../Thursday-22-02-2018/ \
    dataset/Thursday-22-02-2018/ --no-sign-request --region ca-central-1

# Friday
aws s3 sync s3://cse-cic-ids2018/.../Friday-23-02-2018/ \
    dataset/Friday-23-02-2018/ --no-sign-request --region ca-central-1
```

### Step 1 — Extract features from PCAPs

Processes all PCAPs and writes three CSV types per PCAP (`*_kitsune_115.csv`, `*_http_85.csv`, `*_hybrid_200.csv`):

```bash
python phase2/http/run_extraction.py \
    --pcap_dir   dataset/Thursday-22-02-2018 \
    --gt_csv     dataset/cse-cic-ids2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv \
    --output_dir dataset/cse-cic-ids2018/CSV_Data/thursday \
    --day        thursday

python phase2/http/run_extraction.py \
    --pcap_dir   dataset/Friday-23-02-2018 \
    --gt_csv     dataset/cse-cic-ids2018/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv \
    --output_dir dataset/cse-cic-ids2018/CSV_Data/friday \
    --day        friday
```

### Step 2 — Combine per-PCAP CSVs into day-level files

```bash
python phase2/combine_csvs.py \
    --input_dir  dataset/cse-cic-ids2018/CSV_Data \
    --output_dir dataset/cse-cic-ids2018/CSV_Data/combined
```

### Step 3 — Run the three experiments

**Experiment A — Traditional KitNET (115-dim network features)**

```bash
python phase2/traditional/run_traditional.py \
    --data_dir   dataset/cse-cic-ids2018/CSV_Data/combined \
    --output_dir results_phase2/traditional
```

**Experiment B — HTTP-only (85-dim HTTP-AfterImage features)**

```bash
python phase2/http_only/run_http.py \
    --data_dir   dataset/cse-cic-ids2018/CSV_Data/combined \
    --output_dir results_phase2/http_only
```

**Experiment C — Hybrid (200-dim = 115 network + 85 HTTP)**

```bash
python phase2/hybrid/run_hybrid.py \
    --data_dir   dataset/cse-cic-ids2018/CSV_Data/combined \
    --output_dir results_phase2/hybrid
```

### Step 4 — Cross-experiment comparison

```bash
python phase2/comparison/run_comparison.py \
    --results_dir results_phase2 \
    --output_dir  results_phase2/_comparison_plots
```

### Phase 2 quick-run (no PCAPs — original CIC CSVs only)

Runs KitNET directly on the CICFlowMeter flow features without any PCAP extraction:

```bash
python phase2/csv_baseline/run_csv_baseline.py \
    --thursday   dataset/cse-cic-ids2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv \
    --friday     dataset/cse-cic-ids2018/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv \
    --output_dir results_phase2/csv_baseline
```

### Phase 2 arguments (all three experiments share the same flags)

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `dataset/.../combined` | Directory with combined `*_kitsune_115.csv` / `*_http_85.csv` / `*_hybrid_200.csv` |
| `--output_dir` | `results_phase2/<method>` | Where results are saved |
| `--m` | `10` | Max cluster size for FeatureMapper |
| `--beta` | `0.75` | Hidden-layer compression ratio (n_hidden = ⌈β × n_visible⌉) |
| `--lr` | `0.1` | Autoencoder SGD learning rate |

---

## Phase 3 — HTTP-Tuned KitNET Architecture

Phase 3 introduces a custom architecture specifically tuned for HTTP traffic:
- **Leaky ReLU** hidden layer (α=0.01) instead of sigmoid
- **EMA-based input normalisation** (momentum=0.99) instead of min-max
- **Per-cluster RMSE z-score normalisation** before the output autoencoder
- **Forced minimum 15 clusters** (`max_cluster_size=5`, `min_clusters=15`)
- **HTTP-tuned warmup**: `fm_grace=min(8000, 20%×benign)`, `ad_grace=min(40000, 75%×benign)`

### Run locally

Requires the combined `*_http_85.csv` files produced by Phase 2 Step 2:

```bash
python phase3/run_phase3_http.py \
    --data_dir   dataset/cse-cic-ids2018/CSV_Data/combined \
    --output_dir phase3results
```

### Run on Kaggle

The standalone script `phase3/run_phase3_http.py` is self-contained (all classes inlined) and designed for Kaggle's CPU notebook environment. Edit the paths at the bottom of `main()`:

```python
COMBINED_DIR = Path("/kaggle/input/<your-dataset>/combined")
PHASE3_OUT   = Path("/kaggle/working/results_phase3")
PHASE2_OUT   = Path("/kaggle/input/<your-phase2-results>/results_phase2/full")
```

Then run the notebook cell. Expected runtime: ~20 minutes per day (~900 rows/s with 17 clusters).

### Phase 3 arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `dataset/.../combined` | Directory with `*_http_85.csv` files |
| `--output_dir` | `phase3results` | Where results are saved |
| `--max_cluster_size` | `5` | Max features per cluster (forces more clusters than Phase 2) |
| `--min_clusters` | `15` | Minimum number of sub-autoencoders |
| `--beta` | `0.75` | Hidden-layer compression ratio |
| `--lr` | `0.1` | Autoencoder SGD learning rate |

---

## Output Layout

All three phases write the same result format per day/dataset:

```
results_*/
  <day_or_dataset>/
    scores.csv              row_index, day, label, score, attack_type
    metrics.json            AUC, AUPRC, EER, F1, runtime, cluster info
    roc_curve.csv           fpr, tpr, threshold
    pr_curve.csv            precision, recall, threshold
    attack_breakdown.csv    per-attack-type detection rates at 5% FPR

  summary_metrics.csv       one row per day — tabular summary
  summary_metrics.json      same in JSON
  attack_breakdown_combined.csv  all days combined
```

---

## Key metrics.json fields

| Field | Description |
|---|---|
| `AUC` | Area under ROC curve (0.5 = random) |
| `AUPRC` | Area under precision-recall curve |
| `EER` | Equal error rate — lower is better |
| `F1_optimal` | Best F1 over all thresholds |
| `runtime_sec` / `rows_per_sec` | Wall-clock time and throughput |
| `n_clusters` / `cluster_sizes` | FeatureMapper clustering result |
| `fm_grace` / `ad_grace` | Warmup periods used |
| `feature_source` | `kitsune_network_115`, `http_afterimage_85`, `hybrid_200`, or `http_afterimage_85_phase3` |

---

## Reference

> Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
> **Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection.**
> NDSS Symposium 2018. https://arxiv.org/abs/1802.09089

> Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
> **Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.**
> ICISSP 2018.
