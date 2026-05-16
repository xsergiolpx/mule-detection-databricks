# `modeling/` — mule detection across the maturity ladder

End-to-end Databricks notebooks that build, train, score and benchmark **one model per tier** of the mule-detection maturity ladder described in [`../../mule_detection_research.md`](../../mule_detection_research.md). Every notebook reads from the same two Delta tables, writes per-account scores to its own table, and appends one row to a shared `tier_benchmark` Delta table so the closing visual in [`10_tier_comparison.py`](10_tier_comparison.py) shows the whole ladder side by side.

## Notebooks

| # | Notebook | Tier | What it shows |
|---|---|---|---|
| 00 | `00_data_generation.py` | — | Synthetic accounts + transactions, planted mule rings, EDA |
| 01 | `01_rules_engine.py` | 1 | Pure Spark, readability-first rules engine with a per-rule docstring + typology cite |
| 02 | `02_isolation_forest.py` | 2 | sklearn baseline **and** LinkedIn distributed Spark Isolation Forest side by side |
| 03 | `03_autoencoder.py` | 2 | PyTorch autoencoder via `TorchDistributor` |
| 04 | `04_xgboost_pu_learning.py` | 3 | `SparkXGBClassifier` + Elkan–Noto PU correction |
| 05 | `05_graphframes_features.py` | 4 | GraphFrames PageRank / two-hop / community feeding `SparkXGBClassifier` |
| 06 | `06_graphsage_gnn.py` | 4 | PyG GraphSAGE GNN on GPU via `TorchDistributor` + `pynvml` instrumentation |
| 07 | `07_lstm_sequence.py` | 5 | Per-account LSTM sequence model via `TorchDistributor` |
| 08 | `08_muletrack_markov.py` | 5 | MuleTrack-style Markov chain over behavioural states (Spark + numpy, CPU only) |
| 09 | `09_tgn_temporal_graph.py` | 5 | PyG Temporal Graph Network with memory + neighbour cache, `TorchDistributor` |
| 10 | `10_tier_comparison.py` | — | Demo finale: PR curves, recall lift, runtime per tier |

## How to reuse on a different workspace

**Change `config.py` and nothing else.** Catalog, schema, MLflow experiment, cluster id, workspace folder, data scale, seed, and the LinkedIn isolation-forest Maven coord all live there. Every modeling notebook does `%run ./_shared` → `%run ./config`, so the change propagates.

## How to deploy + run

Local Python (the driver uses the Databricks CLI):

```bash
# Authenticate once
databricks auth login --host https://adb-2690017451936431.11.azuredatabricks.net

# Install the LinkedIn isolation-forest Maven library on the target cluster
databricks libraries install \
  --cluster-id 0515-114053-9qo5ptwy \
  --maven-coordinates "com.linkedin.isolation-forest:isolation-forest_3.5.0_2.12:3.0.6"

# Upload + run all 11 notebooks sequentially
python deploy_and_run.py

# Or just upload, no run
python deploy_and_run.py --skip-run

# Or run a subset
python deploy_and_run.py --only 00,01,02
```

The driver submits each notebook as a one-off job against the cluster in `config.py`, polls until the run terminates, and prints a summary table with run URLs.

## Artefacts written

| Artefact | Location |
|---|---|
| Synthetic raw data | `{CATALOG}.{SCHEMA}.accounts`, `{CATALOG}.{SCHEMA}.transactions` |
| Per-tier scores | `{CATALOG}.{SCHEMA}.scores_<tier_name>` |
| Cross-tier benchmark | `{CATALOG}.{SCHEMA}.tier_benchmark` |
| Per-tier MLflow runs | One run per tier in `MLFLOW_EXPERIMENT` |

## Demo flow recommendation

1. Open `00` — generate data; show the EDA visuals.
2. Walk through `01` — one rule at a time. Compliance officer in the room understands every line.
3. Open `02` — anomaly detection without labels. Compare sklearn vs LinkedIn Spark side by side.
4. Skip to `04` — calibrated probability, PU correction.
5. `05` — graph features, interactive PyVis ring viz.
6. `06` — full GNN; show the GPU lighting up.
7. `08` — Markov chain. Auditable, runs on CPU, surprisingly strong.
8. `10` — the slide: every tier's PR curve overlaid, incremental recall lift.

`03` / `07` / `09` are deeper-dive bonuses for ML-curious audiences.
