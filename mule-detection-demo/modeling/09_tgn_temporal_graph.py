# Databricks notebook source
# MAGIC %md
# MAGIC # 🌀 09 — Tier 5: Temporal Graph Network (PyG TGN + TorchDistributor)
# MAGIC
# MAGIC The TGN combines a GNN with **per-node memory** that updates with every event. It
# MAGIC closes the loop on the maturity ladder:
# MAGIC
# MAGIC - GraphSAGE (06) sees the *static* graph.
# MAGIC - LSTM (07) sees the *temporal* sequence per account.
# MAGIC - **TGN (this notebook)** sees *both* — temporally-ordered events on an evolving graph.
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC - `TGNMemory` — per-node hidden state that updates as events arrive.
# MAGIC - `IdentityMessage` — feature representation of an event.
# MAGIC - `LastAggregator` — keeps only the most recent message per node.
# MAGIC - `LastNeighborLoader` — fixed-size cache of recent neighbours.
# MAGIC - `TransformerConv` — temporal-attention GNN that reads memory + neighbour msgs.
# MAGIC
# MAGIC Reference: Rossi et al. 2020; PyG TGN example: <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py>.

# COMMAND ----------

# MAGIC %pip install torch-geometric torchinfo --quiet

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
from sklearn.metrics import average_precision_score, precision_recall_curve

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the temporal event stream

# COMMAND ----------

TGN_DIR     = "/dbfs/tmp/mule_demo/09_tgn"
EVENT_PATH  = f"{TGN_DIR}/events.pt"
CKPT_PATH   = f"{TGN_DIR}/model_rank0.pt"
SNAP_PATH   = f"{TGN_DIR}/memory_snapshots.pt"
os.makedirs(TGN_DIR, exist_ok=True)

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

events_pd = (txns.orderBy("day").toPandas()
                  .rename(columns={"day": "t"})
                  .reset_index(drop=True))
events_pd["t"] = (events_pd["t"] * 86400).astype("int64")   # day → seconds

n_nodes = int(accounts.agg(F.max("account_id")).first()[0]) + 1
y_np = (accounts.orderBy("account_id")
                 .toPandas()["is_mule"].astype(int).values)

torch.save({
    "src": torch.tensor(events_pd["src"].values, dtype=torch.long),
    "dst": torch.tensor(events_pd["dst"].values, dtype=torch.long),
    "t":   torch.tensor(events_pd["t"].values,   dtype=torch.long),
    "msg": torch.tensor(np.log1p(events_pd[["amount"]].values).astype("float32")),
    "y":   torch.tensor(y_np, dtype=torch.long),
    "n_nodes": n_nodes,
}, EVENT_PATH)
print(f"✓ Saved {len(events_pd):,} events, {n_nodes:,} nodes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Distributed TGN training

# COMMAND ----------

def train_tgn(
    event_path: str,
    ckpt_path:  str,
    snap_path:  str,
    epochs:     int = 10,
    batch_size: int = 1024,
    memory_dim: int = 64,
    time_dim:   int = 16,
    lr:         float = 1e-3,
    seed:       int = 42,
    n_snapshots: int = 10,
    run_id:     str | None = None,
):
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch_geometric.nn import TGNMemory, TransformerConv
    from torch_geometric.nn.models.tgn import (
        IdentityMessage, LastAggregator, LastNeighborLoader, TimeEncoder,
    )
    import mlflow

    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda    = torch.cuda.is_available()
    device      = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}"
                                if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    blob = torch.load(event_path, map_location="cpu")
    src, dst, t, msg, y = blob["src"], blob["dst"], blob["t"], blob["msg"], blob["y"]
    n_nodes = blob["n_nodes"]
    raw_msg_dim = msg.size(1)

    # Move per-event tensors to the device once. neighbor_loader returns e_id on
    # the device, so any indexing into t/msg must also be on the device.
    t   = t.to(device)
    msg = msg.to(device)

    # Train/test split on EVENT index (chronological) -------------------------
    n_events    = src.size(0)
    n_train     = int(0.7 * n_events)
    torch.manual_seed(seed)

    memory = TGNMemory(
        num_nodes=n_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    neighbor_loader = LastNeighborLoader(num_nodes=n_nodes, size=10, device=device)

    gnn = TransformerConv(
        in_channels=memory_dim,
        out_channels=memory_dim,
        heads=2,
        concat=False,            # average across heads → output stays memory_dim
        dropout=0.1,
        edge_dim=time_dim + raw_msg_dim,
    ).to(device)
    head = nn.Linear(memory_dim, 2).to(device)
    time_encoder = TimeEncoder(time_dim).to(device)

    if world_size > 1:
        memory = nn.parallel.DistributedDataParallel(memory, device_ids=[device.index] if use_cuda else None)
        gnn    = nn.parallel.DistributedDataParallel(gnn,    device_ids=[device.index] if use_cuda else None)
        head   = nn.parallel.DistributedDataParallel(head,   device_ids=[device.index] if use_cuda else None)
        time_encoder = nn.parallel.DistributedDataParallel(time_encoder, device_ids=[device.index] if use_cuda else None)

    params = (list(memory.parameters()) + list(gnn.parameters()) + list(head.parameters())
              + list(time_encoder.parameters()))
    opt    = torch.optim.Adam(params, lr=lr)

    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    class_weight = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)

    losses, snap_acc = [], []
    snap_every = max(1, (n_train // batch_size) // n_snapshots)

    def get_module(m):
        return m.module if isinstance(m, nn.parallel.DistributedDataParallel) else m

    for epoch in range(epochs):
        get_module(memory).reset_state()
        neighbor_loader.reset_state()

        epoch_loss, n_seen, step = 0.0, 0, 0
        memory.train(); gnn.train(); head.train()

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            b_src = src[start:end].to(device)
            b_dst = dst[start:end].to(device)
            b_t   = t  [start:end].to(device)
            b_msg = msg[start:end].to(device)

            opt.zero_grad()

            # All unique nodes involved in this batch -------------------------
            nodes = torch.cat([b_src, b_dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(nodes)
            z, last_update = get_module(memory)(n_id)
            if e_id.numel() > 0:
                # Real edge attributes: (time-encoded delta since neighbour's last
                # memory update, raw transaction message). This is what makes the
                # GNN *temporal* — without it the model can't see the time signal.
                src_local       = edge_index[0]
                neighbour_last  = last_update[src_local]
                rel_t           = (t[e_id] - neighbour_last).float()
                t_emb           = get_module(time_encoder)(rel_t)
                edge_attr       = torch.cat([t_emb, msg[e_id]], dim=-1)
                z = get_module(gnn)(z, edge_index, edge_attr)

            # Classify the SRC and DST of this batch (supervise on both ends of
            # every transaction so 2× the supervision signal per batch).
            id_to_local = {int(n.item()): i for i, n in enumerate(n_id)}
            both       = torch.cat([b_src, b_dst])
            both_local = torch.tensor([id_to_local[int(n.item())] for n in both],
                                        device=device)
            logits = get_module(head)(z[both_local])
            labels = y[both.cpu()].to(device)
            loss   = nn.functional.cross_entropy(logits, labels, weight=class_weight)
            loss.backward(); opt.step()

            # Update memory + neighbour cache with this batch -----------------
            get_module(memory).update_state(b_src, b_dst, b_t, b_msg)
            neighbor_loader.insert(b_src, b_dst)

            # Detach memory state so the next iteration's forward pass does not
            # backprop through this batch's (already-freed) computation graph.
            get_module(memory).detach()

            epoch_loss += loss.item() * b_src.size(0); n_seen += b_src.size(0)

            # Snapshot the memory of one tracked mule for the demo viz --------
            if rank == 0 and step % snap_every == 0:
                # Snapshot the first 64 dims of memory state for node 200_000 if exists.
                tracked_node = min(200_000, n_nodes - 1)
                with torch.no_grad():
                    state, _ = get_module(memory)(torch.tensor([tracked_node], device=device))
                    snap_acc.append({
                        "epoch": epoch, "step": step,
                        "time":  int(b_t[-1].item()),
                        "state": state.squeeze(0).cpu().numpy().tolist(),
                    })
            step += 1

        epoch_loss /= max(1, n_seen)
        losses.append(epoch_loss)
        if rank == 0:
            print(f"  epoch {epoch}  loss = {epoch_loss:.4f}")
            if run_id:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch, run_id=run_id)

    # Per-event scoring with per-node max-pool ---------------------------------
    # Replay the entire stream. For every event we score the (src, dst) pair as
    # we did during training, then accumulate max(score) per node. This is what
    # the TGN architecture is really designed for — a node's mule-likelihood is
    # the maximum suspicion ever observed across its event history.
    if rank == 0:
        get_module(memory).eval(); get_module(gnn).eval(); get_module(head).eval()
        get_module(time_encoder).eval()
        # Restart from a clean memory state so the per-event scores reflect the
        # full event stream (not the train-only state).
        get_module(memory).reset_state()
        neighbor_loader.reset_state()

        proba = torch.full((n_nodes,), -1.0)  # max-pool start; nodes never touched stay at -1
        with torch.no_grad():
            for start in range(0, n_events, batch_size):
                end = min(start + batch_size, n_events)
                b_src = src[start:end].to(device)
                b_dst = dst[start:end].to(device)
                b_t   = t  [start:end].to(device)
                b_msg = msg[start:end].to(device)

                nodes = torch.cat([b_src, b_dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(nodes)
                z, last_update = get_module(memory)(n_id)
                if e_id.numel() > 0:
                    src_local       = edge_index[0]
                    rel_t           = (t[e_id] - last_update[src_local]).float()
                    t_emb           = get_module(time_encoder)(rel_t)
                    edge_attr       = torch.cat([t_emb, msg[e_id]], dim=-1)
                    z = get_module(gnn)(z, edge_index, edge_attr)

                id_to_local = {int(n.item()): i for i, n in enumerate(n_id)}
                both       = torch.cat([b_src, b_dst])
                both_local = torch.tensor([id_to_local[int(n.item())] for n in both],
                                            device=device)
                event_proba = torch.softmax(get_module(head)(z[both_local]), dim=1)[:, 1].cpu()

                # Vectorised per-node max-pool. scatter_reduce_ with amax keeps
                # the maximum suspicion ever observed for each node.
                proba.scatter_reduce_(0, both.cpu(), event_proba,
                                       reduce="amax", include_self=True)

                get_module(memory).update_state(b_src, b_dst, b_t, b_msg)
                neighbor_loader.insert(b_src, b_dst)

        # Nodes never seen → 0 (lowest score)
        proba[proba < 0] = 0.0

        torch.save({"proba": proba, "losses": losses}, ckpt_path)
        with open(snap_path, "wb") as f:
            torch.save(snap_acc, f)
        print(f"✓ rank 0 wrote {ckpt_path}")

    if world_size > 1:
        dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Network architecture
# MAGIC
# MAGIC TGN's forward pass is stateful (memory + neighbour cache + GNN + head, with
# MAGIC each batch updating memory in place). Auto-tools like `torchview` don't draw
# MAGIC this cleanly, so we use a hand-drawn block diagram + `torchinfo` on the
# MAGIC individual sub-modules.
# MAGIC
# MAGIC ```text
# MAGIC  ┌────────────────────┐    ┌─────────────────────┐
# MAGIC  │   Event stream     │    │   LastNeighborLoader│
# MAGIC  │ (src, dst, t, msg) │───▶│  (per-node cache)   │
# MAGIC  └─────────┬──────────┘    └──────────┬──────────┘
# MAGIC            │                          │
# MAGIC            ▼                          ▼
# MAGIC  ┌────────────────────┐    ┌─────────────────────┐
# MAGIC  │     TGNMemory      │───▶│   TransformerConv   │
# MAGIC  │ (per-node state z, │    │ (attention-weighted │
# MAGIC  │  last_update[v])   │    │ neighbour msg pass) │
# MAGIC  └────────────────────┘    └──────────┬──────────┘
# MAGIC            ▲                          │
# MAGIC            │                          ▼
# MAGIC            │            ┌─────────────────────┐
# MAGIC            │            │   Linear head       │
# MAGIC            │            │   → P(mule)         │
# MAGIC            │            └──────────┬──────────┘
# MAGIC            │                       │
# MAGIC            │                       ▼
# MAGIC            │            ┌─────────────────────┐
# MAGIC            └────────────│ update_state(...)   │
# MAGIC                          └─────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC The loop runs once per event-batch. `TimeEncoder` produces the time component
# MAGIC of the edge attributes that flow into `TransformerConv`. Component-by-component
# MAGIC sizes below.

# COMMAND ----------

import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage, LastAggregator, TimeEncoder,
)
from torchinfo import summary

_memory_dim, _time_dim, _raw_msg_dim = 64, 16, 1

_memory = TGNMemory(
    num_nodes=100,
    raw_msg_dim=_raw_msg_dim,
    memory_dim=_memory_dim,
    time_dim=_time_dim,
    message_module=IdentityMessage(_raw_msg_dim, _memory_dim, _time_dim),
    aggregator_module=LastAggregator(),
)
_gnn = TransformerConv(
    in_channels=_memory_dim, out_channels=_memory_dim,
    heads=2, concat=False, dropout=0.1,
    edge_dim=_time_dim + _raw_msg_dim,
)
_head         = nn.Linear(_memory_dim, 2)
_time_encoder = TimeEncoder(_time_dim)

print("=== TimeEncoder ===")
print(summary(_time_encoder, input_size=(8,), col_names=("input_size", "output_size", "num_params"), depth=2))
print("\n=== TransformerConv (the GNN block) ===")
print("In channels =", _memory_dim, "  out channels =", _memory_dim,
      "  heads = 2 (concat=False → out dim", _memory_dim, ")",
      "  edge_dim =", _time_dim + _raw_msg_dim)
print(f"  Trainable params: {sum(p.numel() for p in _gnn.parameters()):,}")
print("\n=== Linear head ===")
print(summary(_head, input_size=(1, _memory_dim), col_names=("input_size", "output_size", "num_params"), depth=2))
print(f"\n=== TGNMemory ===")
print(f"  num_nodes={_memory.num_nodes}, memory_dim={_memory.memory_dim}, raw_msg_dim={_memory.raw_msg_dim}")
print(f"  Trainable params: {sum(p.numel() for p in _memory.parameters()):,}")
print(f"\nTotal trainable params across TGN stack: "
      f"{sum(p.numel() for m in [_memory, _gnn, _head, _time_encoder] for p in m.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch via TorchDistributor

# COMMAND ----------

with mlflow.start_run(run_name="09_tgn_temporal") as run:
    parent_run_id = run.info.run_id
    mlflow.log_params({"epochs": 10, "batch_size": 1024, "memory_dim": 64,
                        "time_dim": 16, "lr": 1e-3,
                        **{f"distributor_{k}": v for k, v in TORCH_DISTRIBUTOR_KWARGS.items()}})
    t0 = time.perf_counter()
    mode_used = run_distributed_or_local(
        train_tgn,
        EVENT_PATH, CKPT_PATH, SNAP_PATH,
        10, 1024, 64, 16, 1e-3, SEED, 10, parent_run_id,
    )
    runtime_train = time.perf_counter() - t0
    mlflow.log_metric("runtime_seconds_train", runtime_train)
    mlflow.log_param("training_mode", mode_used)
    print(f"\nTraining wall-clock = {runtime_train:.1f}s  (mode={mode_used})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate

# COMMAND ----------

ckpt   = torch.load(CKPT_PATH, map_location="cpu")
proba  = ckpt["proba"].numpy()
losses = ckpt["losses"]

# Match scoring order: proba is indexed by account_id (since accounts run 0..N-1)
acc_pd = accounts.orderBy("account_id").toPandas()
ids    = acc_pd["account_id"].values
y_eval = acc_pd["is_mule"].astype(int).values
sc     = proba[ids]

# Random 30% test holdout, stratified on the label so both classes are present.
# (The previous version used the last 30% of account_ids — which put nearly all
#  mules into the test split because mules sit at the end of the id range.)
n      = len(ids)
rng_eval  = np.random.default_rng(SEED)
test_idx  = rng_eval.choice(n, size=int(0.3 * n), replace=False)
test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True

auprc = average_precision_score(y_eval[test_mask], sc[test_mask])
p1, _ = precision_recall_at_k(y_eval[test_mask], sc[test_mask], 0.01)
p5, r5 = precision_recall_at_k(y_eval[test_mask], sc[test_mask], 0.05)
print(f"TGN AUPRC={auprc:.3f}  P@5%={p5:.1%}  R@5%={r5:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations

# COMMAND ----------

# 1. Training loss -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, marker="o", color="#17becf")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title("TGN training loss")
ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# 2. Memory-state evolution for one tracked mule -----------------------------
with open(SNAP_PATH, "rb") as f:
    snaps = torch.load(f)
if len(snaps) > 0:
    states = np.array([s["state"][:16] for s in snaps])  # first 16 dims
    times  = [s["time"] for s in snaps]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(states.T, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(states).max(), vmax=np.abs(states).max())
    ax.set_xticks(range(len(times))[::max(1, len(times)//10)])
    ax.set_xticklabels([f"{t:,}s" for t in times[::max(1, len(times)//10)]], rotation=30)
    ax.set_ylabel("memory dim")
    ax.set_xlabel("event time (snapshots)")
    ax.set_title("Memory state evolution for a tracked node (first 16 dims)")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()

# COMMAND ----------

# 3. PR curve overlay vs GraphSAGE (if its scores table exists) --------------
p, r, _ = precision_recall_curve(y_eval[test_mask], sc[test_mask])
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r, p, color="#17becf", label=f"TGN  AUPRC={auprc:.3f}")

try:
    sage = (spark.table(scores_table("06_graphsage"))
                  .toPandas().sort_values("account_id"))
    n_sage = len(sage)
    mask_sage = np.zeros(n_sage, dtype=bool); mask_sage[int(0.7 * n_sage):] = True
    from sklearn.metrics import average_precision_score as ap
    p2, r2, _ = precision_recall_curve(sage["is_mule"][mask_sage].astype(int),
                                          sage["score"][mask_sage])
    ax.plot(r2, p2, color="#9467bd", linestyle="--", alpha=0.7,
             label=f"GraphSAGE  AUPRC={ap(sage['is_mule'][mask_sage].astype(int), sage['score'][mask_sage]):.3f}")
except Exception:
    pass
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall — TGN vs GraphSAGE")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist scores + MLflow + benchmark

# COMMAND ----------

scored_pd = pd.DataFrame({"account_id": ids, "is_mule": y_eval, "score": sc})
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("09_tgn")))

with mlflow.start_run(run_id=parent_run_id):
    mlflow.log_metric("pr_auc",            auprc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r5)

log_tier_metrics("09_tgn", 5, auprc, p1, p5, r5, runtime_train, parent_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test gates

# COMMAND ----------

run_gate("smoke",   sc.size > 0,        "no TGN scores")
run_gate("pr_auc",  auprc > 0.30,       f"TGN AUPRC {auprc:.3f} below 0.30 floor")
run_gate("loss_drops", losses[-1] < losses[0], "TGN loss did not decrease")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Temporal *and* graph signal in one model; closes the maturity ladder | Operational concerns: serving latency, cold-start newly-onboarded accounts | Stack ensembling: combine all tiers as features → meta-classifier (out of scope here) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
