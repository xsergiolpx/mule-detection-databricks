# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 06 — Tier 4: GraphSAGE GNN (PyTorch Geometric + TorchDistributor)
# MAGIC
# MAGIC End-to-end graph neural network. The model now **learns** node embeddings that
# MAGIC incorporate information from each account's two-hop neighbourhood — it can detect
# MAGIC ring structure that no static feature captures.
# MAGIC
# MAGIC ### Distribution
# MAGIC
# MAGIC Training is wrapped in `TorchDistributor(use_gpu=True)`. The same code scales from
# MAGIC 1 GPU (this demo) to multi-node multi-GPU.
# MAGIC
# MAGIC ### GPU instrumentation
# MAGIC
# MAGIC Each epoch samples GPU memory + utilisation via `pynvml` and we plot the curves
# MAGIC alongside the training loss — handy for the demo to show the GPU actually working.
# MAGIC
# MAGIC Reference: GraphSAGE (Hamilton et al. 2017); DNB Norway production deployment at
# MAGIC 5M nodes (§6 of `mule_detection_research.md`).

# COMMAND ----------

# MAGIC %pip install torch-geometric pynvml torchinfo torchview --quiet

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import os
import time
import json
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from pyspark.sql import functions as F
from pyspark.ml.torch.distributor import TorchDistributor
from graphframes import GraphFrame
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.manifold import TSNE

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")
spark.sparkContext.setCheckpointDir("/tmp/graphframes_modeling_ckpt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the graph and persist tensors to DBFS
# MAGIC
# MAGIC `TorchDistributor` workers read the prepared tensors from DBFS — same pattern as
# MAGIC notebook 03.

# COMMAND ----------

GNN_DATA_DIR  = "/dbfs/tmp/mule_demo/06_graphsage"
GRAPH_PATH    = f"{GNN_DATA_DIR}/graph.pt"
CKPT_PATH     = f"{GNN_DATA_DIR}/model_rank0.pt"
METRICS_PATH  = f"{GNN_DATA_DIR}/metrics.json"
os.makedirs(GNN_DATA_DIR, exist_ok=True)

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

# Node features = lifetime aggregates (Tier-3 features, repurposed) -----------
feat_sdf = build_account_features(accounts, txns)
FEATURE_COLS = [
    "in_count", "in_amt_sum", "in_amt_mean", "in_distinct_src",
    "out_count", "out_amt_sum", "out_amt_mean", "out_distinct_dst",
    "passthrough_ratio", "fanin_ratio",
]
feat_pd = (feat_sdf.select("account_id", "is_mule", *FEATURE_COLS)
                    .orderBy("account_id")
                    .toPandas())

# Build edge_index from the transactions table -------------------------------
edges_pd = txns.select("src", "dst").toPandas()

n_nodes = len(feat_pd)
x = torch.tensor(feat_pd[FEATURE_COLS].values, dtype=torch.float32)
y = torch.tensor(feat_pd["is_mule"].astype(int).values, dtype=torch.long)

# Standardise features (mean=0, std=1) for stability -------------------------
mu, sigma = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
x = (x - mu) / sigma

edge_index = torch.tensor(edges_pd[["src", "dst"]].values.T, dtype=torch.long)
print(f"x={tuple(x.shape)}    edge_index={tuple(edge_index.shape)}    "
      f"mule rate = {y.float().mean().item():.3%}")

torch.save({"x": x, "edge_index": edge_index, "y": y}, GRAPH_PATH)
print(f"✓ Saved graph to {GRAPH_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Distributed training function

# COMMAND ----------

def train_graphsage(
    graph_path:  str,
    ckpt_path:   str,
    metrics_path: str,
    epochs:      int = 50,
    hidden:      int = 64,
    lr:          float = 5e-3,
    weight_decay: float = 1e-4,
    seed:        int = 42,
    run_id:      str | None = None,
):
    import os
    import time
    import json
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    import mlflow

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda    = torch.cuda.is_available()
    device      = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}"
                                if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    # GPU instrumentation (rank 0 only) ---------------------------------------
    gpu_log = []
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index) if use_cuda else None
    except Exception:
        handle = None

    blob = torch.load(graph_path, map_location="cpu")
    data = Data(x=blob["x"], edge_index=blob["edge_index"], y=blob["y"]).to(device)

    torch.manual_seed(seed + rank)
    n = data.num_nodes
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[: int(0.7 * n)]] = True
    test_mask  = ~train_mask
    data.train_mask = train_mask.to(device)
    data.test_mask  = test_mask.to(device)

    class MuleSAGE(torch.nn.Module):
        def __init__(self, d_in, hidden=64):
            super().__init__()
            self.c1 = SAGEConv(d_in, hidden)
            self.c2 = SAGEConv(hidden, hidden)
            self.head = torch.nn.Linear(hidden, 2)
        def forward(self, x, edge_index, return_embeddings=False):
            h1 = F.relu(self.c1(x, edge_index))
            h1 = F.dropout(h1, p=0.3, training=self.training)
            h2 = F.relu(self.c2(h1, edge_index))
            logits = self.head(h2)
            return (logits, h2) if return_embeddings else logits

    model = MuleSAGE(d_in=data.x.size(1), hidden=hidden).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[device.index] if use_cuda else None)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos = (data.y[train_mask] == 1).sum().item()
    neg = (data.y[train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)

    losses = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask],
                                weight=class_weights)
        loss.backward(); opt.step()
        losses.append(loss.item())

        gpu_mem_mb, gpu_util = 0.0, 0.0
        if handle is not None:
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_mem_mb = mem.used / 1024**2
            gpu_util   = util.gpu

        if rank == 0:
            gpu_log.append({"epoch": epoch, "loss": loss.item(),
                              "gpu_mem_mb": gpu_mem_mb, "gpu_util": gpu_util})
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch:02d}  loss = {loss.item():.4f}  "
                      f"gpu_mem={gpu_mem_mb:.0f}MB  gpu_util={gpu_util}%")
            if run_id:
                mlflow.log_metric("train_loss", loss.item(), step=epoch, run_id=run_id)
                mlflow.log_metric("gpu_mem_mb", gpu_mem_mb, step=epoch, run_id=run_id)
                mlflow.log_metric("gpu_util_pct", gpu_util,  step=epoch, run_id=run_id)

    # Score the test set + save embeddings ------------------------------------
    if rank == 0:
        model.eval()
        underlying = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        with torch.no_grad():
            logits, embeddings = underlying(data.x, data.edge_index, return_embeddings=True)
            proba = torch.softmax(logits, dim=1)[:, 1]
        torch.save({
            "state_dict":  underlying.state_dict(),
            "proba":       proba.cpu(),
            "embeddings":  embeddings.cpu(),
            "test_mask":   data.test_mask.cpu(),
            "losses":      losses,
        }, ckpt_path)
        with open(metrics_path, "w") as f:
            json.dump(gpu_log, f)
        print(f"✓ rank 0 wrote {ckpt_path} and {metrics_path}")

    if world_size > 1:
        dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Network architecture
# MAGIC
# MAGIC Two views before launching training:
# MAGIC
# MAGIC - **`torchinfo`** — layers + output shapes + parameter counts.
# MAGIC - **`torchview`** — boxes-and-arrows of the forward pass through both SAGE convs and the classification head.

# COMMAND ----------

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torchinfo import summary

# Same class as inside train_graphsage(). Reproduced here so we can inspect.
class MuleSAGE(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.c1 = SAGEConv(d_in, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 2)
    def forward(self, x, edge_index):
        h1 = F.relu(self.c1(x, edge_index))
        h1 = F.dropout(h1, p=0.3, training=self.training)
        h2 = F.relu(self.c2(h1, edge_index))
        return self.head(h2)

_viz_model = MuleSAGE(d_in=len(FEATURE_COLS), hidden=64)
_x_demo    = torch.randn(8, len(FEATURE_COLS))
_edge_demo = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
print(summary(_viz_model,
              input_data=[_x_demo, _edge_demo],
              col_names=("input_size", "output_size", "num_params"),
              depth=3))

# COMMAND ----------

try:
    from torchview import draw_graph
    g = draw_graph(_viz_model, input_data=(_x_demo, _edge_demo),
                    expand_nested=True, depth=3, graph_dir="TB",
                    graph_name="MuleSAGE")
    displayHTML(g.visual_graph.pipe(format="svg").decode("utf-8"))
except Exception as e:
    print(f"torchview rendering failed ({type(e).__name__}: {e})")
    print("If the error mentions 'dot' / 'graphviz', run:  %sh apt-get install -y graphviz")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch via TorchDistributor

# COMMAND ----------

with mlflow.start_run(run_name="06_graphsage_gnn") as run:
    parent_run_id = run.info.run_id
    mlflow.log_params({"epochs": 50, "hidden": 64, "lr": 5e-3, "weight_decay": 1e-4,
                        **{f"distributor_{k}": v for k, v in TORCH_DISTRIBUTOR_KWARGS.items()}})

    t0 = time.perf_counter()
    mode_used = run_distributed_or_local(
        train_graphsage,
        GRAPH_PATH, CKPT_PATH, METRICS_PATH,
        50, 64, 5e-3, 1e-4, SEED, parent_run_id,
    )
    runtime_train = time.perf_counter() - t0
    mlflow.log_metric("runtime_seconds_train", runtime_train)
    mlflow.log_param("training_mode", mode_used)
    print(f"\nTraining wall-clock = {runtime_train:.1f}s  (mode={mode_used})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load results and evaluate

# COMMAND ----------

ckpt = torch.load(CKPT_PATH, map_location="cpu")
with open(METRICS_PATH) as f:
    gpu_log = pd.DataFrame(json.load(f))

proba       = ckpt["proba"].numpy()
embeddings  = ckpt["embeddings"].numpy()
test_mask   = ckpt["test_mask"].numpy().astype(bool)

y_np = y.numpy()
auprc       = average_precision_score(y_np[test_mask], proba[test_mask])
p1, _       = precision_recall_at_k(y_np[test_mask], proba[test_mask], 0.01)
p5, r5      = precision_recall_at_k(y_np[test_mask], proba[test_mask], 0.05)
print(f"GraphSAGE AUPRC={auprc:.3f}  P@5%={p5:.1%}  R@5%={r5:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations

# COMMAND ----------

# 1. Training loss + GPU metrics ---------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(gpu_log["epoch"], gpu_log["loss"], color="#1f77b4", marker="o", label="loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(gpu_log["epoch"], gpu_log["gpu_mem_mb"], color="#d62728", linestyle="--", label="GPU MB")
ax2.plot(gpu_log["epoch"], gpu_log["gpu_util"],   color="#2ca02c", linestyle=":",  label="GPU util %")
ax2.set_ylabel("GPU mem (MB) / util (%)")
fig.suptitle("GraphSAGE training: loss vs GPU activity")
fig.tight_layout()

# COMMAND ----------

# 2. t-SNE of learned embeddings ---------------------------------------------
# Sample for speed.
sample_idx = np.random.default_rng(SEED).choice(len(embeddings),
                                                  size=min(5000, len(embeddings)),
                                                  replace=False)
print("Running t-SNE on a 5k-node sample …")
emb2d = TSNE(n_components=2, init="pca", random_state=SEED,
              perplexity=30, learning_rate="auto").fit_transform(embeddings[sample_idx])

fig, ax = plt.subplots(figsize=(8, 6))
labels_sub = y_np[sample_idx]
ax.scatter(emb2d[labels_sub == 0, 0], emb2d[labels_sub == 0, 1],
           s=4, alpha=0.2, color="#7f7f7f", label="legit")
ax.scatter(emb2d[labels_sub == 1, 0], emb2d[labels_sub == 1, 1],
           s=12, alpha=0.8, color="#d62728", label="mule")
ax.set_title("t-SNE of GraphSAGE node embeddings (5k sample)")
ax.legend()
fig.tight_layout()

# COMMAND ----------

# 3. PR curve ----------------------------------------------------------------
p, r, _ = precision_recall_curve(y_np[test_mask], proba[test_mask])
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r, p, color="#9467bd", label=f"GraphSAGE  AUPRC={auprc:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall — Tier 4 (GraphSAGE GNN)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# 4. Ring-level recall -------------------------------------------------------
# A "ring" is approximated by a connected component over the mule subgraph.
# How many rings have at least one member in the top 5% of model scores?
mule_ids = set(np.where(y_np == 1)[0].tolist())
mule_edges = (edges_pd[edges_pd["src"].isin(mule_ids) & edges_pd["dst"].isin(mule_ids)])
import networkx as nx
g_mule = nx.Graph()
g_mule.add_nodes_from(mule_ids)
g_mule.add_edges_from(mule_edges[["src", "dst"]].values.tolist())
rings = list(nx.connected_components(g_mule))

top5_pct_thresh = int(0.05 * len(proba))
top5_ids        = set(np.argsort(proba)[::-1][:top5_pct_thresh].tolist())
caught_rings    = sum(1 for r in rings if r & top5_ids)
ring_recall     = caught_rings / max(1, len(rings))
print(f"{len(rings)} rings → {caught_rings} caught at top-5% threshold "
      f"(ring-level recall = {ring_recall:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist scores + MLflow + benchmark

# COMMAND ----------

scored_pd = pd.DataFrame({
    "account_id": feat_pd["account_id"].values,
    "is_mule":    y_np,
    "score":      proba,
})
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("06_graphsage")))

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("pr_auc",            auprc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r5)
    mlflow.log_metric("ring_recall_top5",  ring_recall)

log_tier_metrics("06_graphsage", 4, auprc, p1, p5, r5, runtime_train, parent_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test gates

# COMMAND ----------

run_gate("smoke",     proba.size > 0,        "no GNN scores")
run_gate("pr_auc",    auprc > 0.60,          f"GraphSAGE AUPRC {auprc:.3f} below 0.60 floor")
run_gate("loss_drops", ckpt["losses"][-1] < ckpt["losses"][0],
         "GNN training loss did not decrease")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Learned ring-level signal from neighbour embeddings | Temporal ordering of transactions | Pass-through detection needs *order-of-events* → Tier 5 (LSTM / TGN) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
