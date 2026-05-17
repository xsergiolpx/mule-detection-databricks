# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 03 — Tier 2: Autoencoder anomaly detection (PyTorch + TorchDistributor)
# MAGIC
# MAGIC Same intent as Isolation Forest, different mechanism. The autoencoder learns a
# MAGIC low-dimensional manifold of "normal" account behaviour; accounts that reconstruct
# MAGIC poorly score high as anomalies.
# MAGIC
# MAGIC ### What's new vs notebook 02
# MAGIC
# MAGIC - **Non-linear manifold.** Isolation Forest assumes anomalies are isolated by axis-
# MAGIC   aligned splits; an autoencoder can pick up curved boundaries.
# MAGIC - **Distributed training.** Wrapped in `TorchDistributor` so it scales across the
# MAGIC   cluster's GPUs. At our toy scale (200k accounts) this is mostly pedagogical, but
# MAGIC   the pattern is the same one used at 100M+ scale.
# MAGIC
# MAGIC Reference: [Databricks — Spark PyTorch Distributor](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor)

# COMMAND ----------

# MAGIC %pip install torchinfo torchviz --quiet

# COMMAND ----------

# MAGIC %sh apt-get install -y graphviz > /dev/null 2>&1 && echo "✓ graphviz binary ready" || echo "graphviz install skipped (already present or no sudo)"

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import os
import time
import json
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.torch.distributor import TorchDistributor
from sklearn.metrics import average_precision_score

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build and persist a Parquet shard for the distributor
# MAGIC
# MAGIC `TorchDistributor` runs `train_fn` inside Spark task slots that do not share Python
# MAGIC objects with the driver. The standard pattern is: **write the training data to
# MAGIC durable storage on the driver, then have each worker read it inside `train_fn`**.

# COMMAND ----------

FEATURE_COLS = [
    "in_count", "in_amt_sum", "in_amt_mean", "in_distinct_src",
    "out_count", "out_amt_sum", "out_amt_mean", "out_distinct_dst",
    "passthrough_ratio", "fanin_ratio",
]

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Meet the dataset

# COMMAND ----------

show_dataset_overview(accounts, txns)

# COMMAND ----------

feat_sdf = build_account_features(accounts, txns)

assembler  = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
scaler     = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
prepared   = scaler.fit(assembler.transform(feat_sdf)).transform(assembler.transform(feat_sdf))

# Materialise feature arrays + label as a flat schema for fast Parquet IO -------
pdf = (prepared.select("account_id", "is_mule", "features")
                .rdd
                .map(lambda r: (int(r.account_id), bool(r.is_mule),
                                 [float(v) for v in r.features.toArray()]))
                .toDF(["account_id", "is_mule", "features"])
                .toPandas())

TRAIN_PATH = f"/dbfs/tmp/mule_demo/03_autoencoder/{CATALOG}_{SCHEMA}.parquet"
os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
pdf.to_parquet(TRAIN_PATH, index=False)
print(f"✓ Wrote {len(pdf):,} rows to {TRAIN_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Distributed training function
# MAGIC
# MAGIC Standard PyTorch DDP-style training loop. Rank 0 owns MLflow logging so we don't get
# MAGIC duplicate metric writes.

# COMMAND ----------

def train_autoencoder(
    train_path: str,
    n_features: int,
    epochs:     int = 30,
    batch_size: int = 4096,
    lr:         float = 1e-3,
    latent_dim: int = 4,
    run_id:     str | None = None,
):
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import pandas as pd
    import mlflow

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda    = torch.cuda.is_available()
    device      = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}"
                                if use_cuda else "cpu")

    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)

    # Each worker reads the full table and shards via DistributedSampler.
    pdf = pd.read_parquet(train_path)
    X   = torch.tensor(pdf["features"].tolist(), dtype=torch.float32)
    ds  = TensorDataset(X)

    sampler = (DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
               if world_size > 1 else None)
    loader  = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)

    class AutoEncoder(nn.Module):
        def __init__(self, d, latent=4):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, latent))
            self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, d))
        def forward(self, x): return self.dec(self.enc(x))

    model = AutoEncoder(n_features, latent=latent_dim).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model,
                    device_ids=[device.index] if use_cuda else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_seen     = 0
        model.train()
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
            n_seen     += xb.size(0)
        epoch_loss /= max(1, n_seen)
        losses.append(epoch_loss)
        if rank == 0:
            print(f"  epoch {epoch:02d}  loss = {epoch_loss:.5f}")
            if run_id:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch, run_id=run_id)

    # Persist the trained weights to durable storage so the driver can score with them.
    ckpt_path = f"/dbfs/tmp/mule_demo/03_autoencoder/model_rank{rank}.pt"
    if rank == 0:
        underlying = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        torch.save({
            "state_dict": underlying.state_dict(),
            "n_features": n_features,
            "latent_dim": latent_dim,
            "losses":     losses,
        }, ckpt_path)
        print(f"✓ rank 0 saved checkpoint to {ckpt_path}")

    if world_size > 1:
        dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Network architecture
# MAGIC
# MAGIC Before launching training, let's look at what we're about to train. Two views:
# MAGIC
# MAGIC - **`torchinfo`** — the spec sheet (layers, output shapes, parameter counts).
# MAGIC - **`torchviz`** — the picture (autograd graph rendered via Graphviz).

# COMMAND ----------

import torch
import torch.nn as nn
from torchinfo import summary

# Same class as defined inside train_autoencoder() — reproduced here so we can
# inspect the architecture without launching the distributed training.
class AutoEncoder(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, x): return self.dec(self.enc(x))

_viz_model = AutoEncoder(d=len(FEATURE_COLS), latent=4)
print(summary(_viz_model,
              input_size=(1, len(FEATURE_COLS)),
              col_names=("input_size", "output_size", "num_params"),
              depth=3))

# COMMAND ----------

# Visual architecture diagram via torchviz (autograd graph)
# Force everything to CPU — on DBR ML GPU runtimes PyTorch may default to CUDA,
# which causes "mat1 on cpu, other tensors on cuda:0" if we don't pin the device.
try:
    from torchviz import make_dot
    _viz_model_cpu = _viz_model.to("cpu")
    _x = torch.randn(1, len(FEATURE_COLS), device="cpu")
    _yhat = _viz_model_cpu(_x)
    dot = make_dot(_yhat, params=dict(_viz_model_cpu.named_parameters()),
                    show_attrs=False, show_saved=False)
    dot.attr(rankdir="TB", size="9,12")
    displayHTML(dot.pipe(format="svg").decode("utf-8"))
except Exception as e:
    print(f"torchviz rendering failed ({type(e).__name__}: {e})")
    print("If the error mentions 'dot' / 'graphviz', the system binary is missing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch via `TorchDistributor`

# COMMAND ----------

with mlflow.start_run(run_name="03_autoencoder") as run:
    mlflow.log_params({
        "epochs":     30,
        "batch_size": 4096,
        "lr":         1e-3,
        "latent_dim": 4,
        **{f"distributor_{k}": v for k, v in TORCH_DISTRIBUTOR_KWARGS.items()},
    })
    parent_run_id = run.info.run_id

    t0 = time.perf_counter()
    mode_used = run_distributed_or_local(
        train_autoencoder,
        TRAIN_PATH,
        len(FEATURE_COLS),
        30,
        4096,
        1e-3,
        4,
        parent_run_id,
    )
    runtime_train = time.perf_counter() - t0
    mlflow.log_metric("runtime_seconds_train", runtime_train)
    mlflow.log_param("training_mode", mode_used)
    print(f"\nTraining wall-clock = {runtime_train:.1f}s  (mode={mode_used})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score on the driver
# MAGIC
# MAGIC Load the rank-0 checkpoint and score every account. Reconstruction error = score.

# COMMAND ----------

ckpt = torch.load("/dbfs/tmp/mule_demo/03_autoencoder/model_rank0.pt", map_location="cpu")

class AutoEncoder(nn.Module):
    def __init__(self, d, latent=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, x): return self.dec(self.enc(x))

model = AutoEncoder(ckpt["n_features"], latent=ckpt["latent_dim"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

X = torch.tensor(pdf["features"].tolist(), dtype=torch.float32)
with torch.no_grad():
    recon       = model(X)
    recon_err   = ((recon - X) ** 2).mean(dim=1).numpy()
    latent_2d   = model.enc(X).numpy()  # for visualisation; latent_dim might be > 2

y = pdf["is_mule"].astype(int).values
pr_auc       = average_precision_score(y, recon_err)
p1, r_at_1pct = precision_recall_at_k(y, recon_err, 0.01)
p5, r_at_5pct = precision_recall_at_k(y, recon_err, 0.05)
print(f"PR-AUC={pr_auc:.3f}  P@5%={p5:.1%}  R@5%={r_at_5pct:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations
# MAGIC
# MAGIC 1. Training-loss curve (rank-0 perspective).
# MAGIC 2. 2-D latent-space scatter, mules in red.
# MAGIC 3. Reconstruction-error distribution by class.

# COMMAND ----------

# 1. Training loss -----------------------------------------------------------
fig = go.Figure(go.Scatter(x=list(range(len(ckpt["losses"]))), y=ckpt["losses"],
                            mode="lines+markers", line=dict(color="#1f77b4", width=3),
                            marker=dict(size=8)))
fig.update_layout(template="plotly_white", height=380,
                   title="Autoencoder training loss (rank 0)",
                   xaxis_title="epoch", yaxis_title="MSE loss",
                   margin=dict(l=20, r=20, t=60, b=40))
plotly_show(fig)

# COMMAND ----------

# 2. Latent space ------------------------------------------------------------
# Reduce to 2-D via PCA on the latent space if latent_dim > 2.
if latent_2d.shape[1] > 2:
    from sklearn.decomposition import PCA
    latent_2d = PCA(n_components=2, random_state=SEED).fit_transform(latent_2d)

mask_legit = y == 0
mask_mule  = y == 1

# Downsample legit for plot size safety
rng_lat = np.random.default_rng(SEED)
legit_idx = np.where(mask_legit)[0]
legit_sample = rng_lat.choice(legit_idx, size=min(4000, len(legit_idx)), replace=False)

fig = go.Figure()
fig.add_trace(go.Scatter(x=latent_2d[legit_sample, 0], y=latent_2d[legit_sample, 1],
                          mode="markers", marker=dict(size=4, color="#9aa0a6", opacity=0.35),
                          name="legit"))
fig.add_trace(go.Scatter(x=latent_2d[mask_mule, 0], y=latent_2d[mask_mule, 1],
                          mode="markers", marker=dict(size=7, color="#d62728", opacity=0.85),
                          name="mule"))
fig.update_layout(template="plotly_white", height=520,
                   title="Learned latent space — mules cluster apart from legit",
                   xaxis_title="latent 1", yaxis_title="latent 2",
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
plotly_show(fig)

# COMMAND ----------

# 3. Reconstruction error distribution --------------------------------------
upper = float(np.quantile(recon_err, 0.995))
fig = go.Figure()
fig.add_trace(go.Histogram(x=recon_err[mask_legit], nbinsx=80, histnorm="probability density",
                            marker_color="#9aa0a6", opacity=0.6, name="legit"))
fig.add_trace(go.Histogram(x=recon_err[mask_mule],  nbinsx=80, histnorm="probability density",
                            marker_color="#d62728", opacity=0.65, name="mule"))
fig.update_layout(template="plotly_white", barmode="overlay", height=420,
                   title="Reconstruction error by class",
                   xaxis_title="reconstruction error", yaxis_title="density",
                   xaxis=dict(range=[0, upper]),
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist scores and log to benchmark

# COMMAND ----------

scored_pd = pdf[["account_id", "is_mule"]].assign(recon_error=recon_err)
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("03_autoencoder")))

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("pr_auc",            pr_auc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r_at_5pct)

log_tier_metrics("03_autoencoder", 2,
                 pr_auc, p1, p5, r_at_5pct, runtime_train, parent_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test gates

# COMMAND ----------

run_gate("smoke",       len(recon_err) > 0,    "no scores produced")
run_gate("pr_auc",      pr_auc > 0.50,         f"PR-AUC {pr_auc:.3f} below 0.50 floor")
run_gate("loss_drops",  ckpt["losses"][-1] < ckpt["losses"][0],
         "training loss did not decrease")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Non-linear anomalies, no labels needed | Calibrated probability, graph structure | Once labels (HR-03 / SAR confirms) are available → Tier 3 (supervised XGBoost + PU) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
