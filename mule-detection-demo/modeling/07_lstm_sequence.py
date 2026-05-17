# Databricks notebook source
# MAGIC %md
# MAGIC # ⏱️ 07 — Tier 5: LSTM sequence model (PyTorch + TorchDistributor)
# MAGIC
# MAGIC Per-account temporal model. Each account becomes a sequence of `(amount, direction,
# MAGIC time)` tokens and an LSTM learns the temporal signature of a pass-through mule
# MAGIC (rapid inbound burst followed by immediate outbound).
# MAGIC
# MAGIC ### Why sequence over aggregates
# MAGIC
# MAGIC Two accounts can have the same monthly volume and the same pass-through ratio, but
# MAGIC look completely different in their per-day rhythm. The LSTM picks up the rhythm.
# MAGIC
# MAGIC ### Distribution
# MAGIC
# MAGIC `TorchDistributor` with `use_gpu=True`. Identical pattern to the autoencoder and
# MAGIC GraphSAGE notebooks.

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
from pyspark.sql import functions as F
from pyspark.ml.torch.distributor import TorchDistributor
from sklearn.metrics import average_precision_score, precision_recall_curve

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build sequences per account

# COMMAND ----------

SEQ_LEN = 20
LSTM_DATA_DIR = "/dbfs/tmp/mule_demo/07_lstm"
DATA_PATH     = f"{LSTM_DATA_DIR}/sequences.pt"
CKPT_PATH     = f"{LSTM_DATA_DIR}/model_rank0.pt"
os.makedirs(LSTM_DATA_DIR, exist_ok=True)

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Meet the dataset

# COMMAND ----------

show_dataset_overview(accounts, txns)

# COMMAND ----------

# Build (account_id, day, amount, direction=+1 inbound / -1 outbound) ---------
inbound  = txns.select(F.col("dst").alias("account_id"), "day", "amount",
                        F.lit( 1.0).alias("direction"))
outbound = txns.select(F.col("src").alias("account_id"), "day", "amount",
                        F.lit(-1.0).alias("direction"))
events   = inbound.unionByName(outbound)

# Last SEQ_LEN events per account, ordered by day, as a list-of-structs --------
from pyspark.sql.window import Window
w = Window.partitionBy("account_id").orderBy(F.col("day").desc())
events_ranked = events.withColumn("rk", F.row_number().over(w)).where(F.col("rk") <= SEQ_LEN)

# Materialise to pandas (200k accounts × 20 events ≈ 4M rows = ~100 MB) --------
events_pd = events_ranked.toPandas()
events_pd["log_amt"] = np.log1p(events_pd["amount"])
events_pd["day_n"]   = events_pd["day"] / events_pd["day"].max()

# Build a fixed-shape (N, SEQ_LEN, 3) tensor ----------------------------------
acc_pd = accounts.orderBy("account_id").toPandas()
id_to_idx = {int(a): i for i, a in enumerate(acc_pd["account_id"].values)}

X = np.zeros((len(acc_pd), SEQ_LEN, 3), dtype="float32")
for aid, grp in events_pd.groupby("account_id"):
    idx = id_to_idx.get(int(aid))
    if idx is None: continue
    g = grp.sort_values("rk", ascending=False).tail(SEQ_LEN)
    arr = g[["log_amt", "direction", "day_n"]].values.astype("float32")
    X[idx, -len(arr):, :] = arr

y = acc_pd["is_mule"].astype(int).values
print(f"X shape = {X.shape}    mule rate = {y.mean():.3%}")

torch.save({
    "X":  torch.tensor(X),
    "y":  torch.tensor(y, dtype=torch.long),
}, DATA_PATH)
print(f"✓ Saved sequences to {DATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Distributed training function

# COMMAND ----------

def train_lstm(
    data_path: str,
    ckpt_path: str,
    epochs:    int = 30,
    batch_size: int = 1024,
    hidden:    int = 32,
    lr:        float = 1e-3,
    seed:      int = 42,
    run_id:    str | None = None,
):
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    import mlflow

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda    = torch.cuda.is_available()
    device      = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}"
                                if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    blob = torch.load(data_path, map_location="cpu")
    X, y = blob["X"], blob["y"]

    n = len(y)
    torch.manual_seed(seed)
    perm = torch.randperm(n)
    train_idx = perm[: int(0.7 * n)]
    test_idx  = perm[int(0.7 * n):]
    train_ds  = TensorDataset(X[train_idx], y[train_idx])

    sampler = (DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                    shuffle=True)
               if world_size > 1 else None)
    loader  = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          shuffle=sampler is None)

    class MuleLSTM(nn.Module):
        def __init__(self, in_dim=3, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h.squeeze(0))

    model = MuleLSTM(in_dim=X.size(2), hidden=hidden).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model,
                    device_ids=[device.index] if use_cuda else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos = (y[train_idx] == 1).sum().item()
    neg = (y[train_idx] == 0).sum().item()
    w   = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)

    losses = []
    for epoch in range(epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss   = nn.functional.cross_entropy(logits, yb, weight=w)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0); n_seen += xb.size(0)
        epoch_loss /= max(1, n_seen)
        losses.append(epoch_loss)
        if rank == 0:
            print(f"  epoch {epoch:02d}  loss = {epoch_loss:.5f}")
            if run_id:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch, run_id=run_id)

    # Score the test set on rank 0 --------------------------------------------
    if rank == 0:
        model.eval()
        underlying = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        with torch.no_grad():
            X_all = X.to(device)
            proba = torch.softmax(underlying(X_all), dim=1)[:, 1].cpu()
        torch.save({
            "state_dict":  underlying.state_dict(),
            "proba":       proba,
            "test_idx":    test_idx,
            "train_idx":   train_idx,
            "losses":      losses,
        }, ckpt_path)
        print(f"✓ rank 0 saved {ckpt_path}")

    if world_size > 1:
        dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Network architecture

# COMMAND ----------

import torch
import torch.nn as nn
from torchinfo import summary

# Same class as inside train_lstm().
class MuleLSTM(nn.Module):
    def __init__(self, d_in=3, h=32):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, batch_first=True)
        self.head = nn.Linear(h, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h.squeeze(0))

_viz_model = MuleLSTM(d_in=3, h=32)
print(summary(_viz_model,
              input_size=(1, SEQ_LEN, 3),
              col_names=("input_size", "output_size", "num_params"),
              depth=3))

# COMMAND ----------

try:
    from torchviz import make_dot
    # Force model and input onto CPU — DBR ML GPU runtime defaults parameters
    # to cuda:0, which mismatches the CPU-side dummy input.
    _viz_model_cpu = _viz_model.to("cpu")
    _x = torch.randn(1, SEQ_LEN, 3, device="cpu")
    _yhat = _viz_model_cpu(_x)
    dot = make_dot(_yhat, params=dict(_viz_model_cpu.named_parameters()),
                    show_attrs=False, show_saved=False)
    dot.attr(rankdir="TB", size="9,14")
    displayHTML(dot.pipe(format="svg").decode("utf-8"))
except Exception as e:
    print(f"torchviz rendering failed ({type(e).__name__}: {e})")
    print("If the error mentions 'dot' / 'graphviz', the system binary is missing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch via TorchDistributor

# COMMAND ----------

with mlflow.start_run(run_name="07_lstm_sequence") as run:
    parent_run_id = run.info.run_id
    mlflow.log_params({"epochs": 30, "batch_size": 1024, "hidden": 32, "lr": 1e-3,
                        "seq_len": SEQ_LEN,
                        **{f"distributor_{k}": v for k, v in TORCH_DISTRIBUTOR_KWARGS.items()}})

    t0 = time.perf_counter()
    mode_used = run_distributed_or_local(
        train_lstm,
        DATA_PATH, CKPT_PATH,
        30, 1024, 32, 1e-3, SEED, parent_run_id,
    )
    runtime_train = time.perf_counter() - t0
    mlflow.log_metric("runtime_seconds_train", runtime_train)
    mlflow.log_param("training_mode", mode_used)
    print(f"\nTraining wall-clock = {runtime_train:.1f}s  (mode={mode_used})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load and evaluate

# COMMAND ----------

ckpt = torch.load(CKPT_PATH, map_location="cpu")
proba    = ckpt["proba"].numpy()
test_idx = ckpt["test_idx"].numpy()
y_test   = y[test_idx]
sc_test  = proba[test_idx]

auprc = average_precision_score(y_test, sc_test)
p1, _ = precision_recall_at_k(y_test, sc_test, 0.01)
p5, r5 = precision_recall_at_k(y_test, sc_test, 0.05)
print(f"LSTM AUPRC={auprc:.3f}  P@5%={p5:.1%}  R@5%={r5:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations

# COMMAND ----------

# 1. Training loss -----------------------------------------------------------
fig = go.Figure(go.Scatter(x=list(range(len(ckpt["losses"]))), y=ckpt["losses"],
                            mode="lines+markers", line=dict(color="#e377c2", width=3),
                            marker=dict(size=8)))
fig.update_layout(template="plotly_white", height=380,
                   title="LSTM training loss (rank 0)",
                   xaxis_title="epoch", yaxis_title="loss",
                   margin=dict(l=20, r=20, t=60, b=40))
plotly_show(fig)

# COMMAND ----------

# 2. PR curve ----------------------------------------------------------------
p, r, _ = precision_recall_curve(y_test, sc_test)
fig = go.Figure(go.Scatter(x=r, y=p, mode="lines", line=dict(color="#e377c2", width=3),
                            name=f"LSTM  AUPRC={auprc:.3f}"))
fig.update_layout(template="plotly_white", height=460,
                   title="Precision–Recall — Tier 5 (LSTM)",
                   xaxis_title="recall", yaxis_title="precision",
                   xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
plotly_show(fig)

# COMMAND ----------

# 3. Sequence heatmap for a true positive ------------------------------------
# Pick the highest-scored mule from the test set and show its sequence.
test_mules = test_idx[y[test_idx] == 1]
if len(test_mules) > 0:
    best_mule = test_mules[np.argsort(proba[test_mules])[::-1][0]]
    seq = X[best_mule]   # (SEQ_LEN, 3)
    seq_T = seq.T        # (3, SEQ_LEN)
    vmax = float(np.abs(seq).max())
    fig = go.Figure(go.Heatmap(
        z=seq_T,
        x=[f"event {i+1}" for i in range(seq_T.shape[1])],
        y=["log(amt)", "direction", "day"],
        zmin=-vmax, zmax=vmax,
        colorscale="RdBu_r",
        hovertemplate="%{y} @ %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title=""),
    ))
    fig.update_layout(template="plotly_white", height=320,
                       title=f"Top-scored mule (account_id={int(acc_pd['account_id'].values[best_mule])}, "
                              f"score={proba[best_mule]:.3f}) — last {SEQ_LEN} events",
                       margin=dict(l=20, r=20, t=60, b=40))
    plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist scores + MLflow + benchmark

# COMMAND ----------

scored_pd = pd.DataFrame({
    "account_id": acc_pd["account_id"].values,
    "is_mule":    y,
    "score":      proba,
})
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("07_lstm")))

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("pr_auc",            auprc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r5)

log_tier_metrics("07_lstm", 5, auprc, p1, p5, r5, runtime_train, parent_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test gates

# COMMAND ----------

run_gate("smoke",     proba.size > 0,         "no LSTM scores")
run_gate("pr_auc",    auprc > 0.50,           f"LSTM AUPRC {auprc:.3f} below 0.50 floor")
run_gate("loss_drops", ckpt["losses"][-1] < ckpt["losses"][0],
         "LSTM training loss did not decrease")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Per-account temporal rhythm (rapid pass-through), works on short histories | Graph context — the model sees one account at a time | Need ordering *and* neighbour context → Tier 5 TGN |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
