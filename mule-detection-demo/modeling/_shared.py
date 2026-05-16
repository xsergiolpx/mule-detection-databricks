# Databricks notebook source
# MAGIC %md
# MAGIC # 🛠️ Mule Modeling — Shared Utilities
# MAGIC
# MAGIC Every modeling notebook does `%run ./_shared` as its first non-markdown cell. This file:
# MAGIC
# MAGIC 1. Loads configuration from `./config`.
# MAGIC 2. Provides the **synthetic data generator** used by `00_data_generation`.
# MAGIC 3. Provides the **feature builders** reused across tiers.
# MAGIC 4. Defines the **`tier_benchmark` schema** + `log_tier_metrics()` helper so all
# MAGIC    notebooks append comparable rows to the comparison Delta table.
# MAGIC 5. Defines `run_gate()` for inline test gates that fail cleanly under the Jobs API.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType,
)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic data generator
# MAGIC
# MAGIC Generates legitimate accounts plus planted mule rings. Each ring has a *collector*
# MAGIC that receives many small inbound transfers from random legitimate accounts (the
# MAGIC scam-victim pattern) and then quickly forwards the consolidated funds to other ring
# MAGIC members (the pass-through pattern).
# MAGIC
# MAGIC Defaults come from `config.py`. Override by passing kwargs.

# COMMAND ----------

def make_synthetic_mule_data(
    n_legit:    int = N_LEGIT,
    n_mules:    int = N_MULES,
    n_days:     int = N_DAYS,
    ring_size:  int = RING_SIZE,
    seed:       int = SEED,
):
    """Return (accounts_df, txns_df) as pandas DataFrames.

    accounts_df columns: account_id, is_mule
    txns_df    columns: src, dst, amount, day
    """
    rng = np.random.default_rng(seed)
    n_total = n_legit + n_mules
    is_mule = np.concatenate([
        np.zeros(n_legit, dtype=bool),
        np.ones (n_mules, dtype=bool),
    ])
    account_ids = np.arange(n_total)

    chunks = []

    # --- Legitimate accounts: low volume, varied counterparties ---------------
    legit_ids = account_ids[~is_mule]
    n_tx_per_legit = rng.poisson(3 * n_days // 10, size=len(legit_ids))
    n_tx_per_legit = np.clip(n_tx_per_legit, 1, None)
    total_legit_tx = int(n_tx_per_legit.sum())

    chunks.append(pd.DataFrame({
        "src":    np.repeat(legit_ids, n_tx_per_legit),
        "dst":    rng.integers(0, n_total, size=total_legit_tx),
        "amount": rng.lognormal(mean=4.0, sigma=1.0, size=total_legit_tx),
        "day":    rng.uniform(0, n_days, size=total_legit_tx),
    }))

    # --- Mule rings: fan-in followed by quick fan-out within the ring --------
    mule_ids = account_ids[is_mule].copy()
    rng.shuffle(mule_ids)
    rings = np.array_split(mule_ids, max(1, len(mule_ids) // ring_size))

    for ring in rings:
        if len(ring) < 2:
            continue
        collector = int(ring[0])
        n_in  = int(rng.integers(20, 60))
        srcs  = rng.choice(legit_ids, size=n_in)
        amts  = rng.lognormal(mean=5.0, sigma=0.5, size=n_in)
        days  = rng.uniform(0, n_days - 1, size=n_in)
        chunks.append(pd.DataFrame({
            "src": srcs, "dst": collector, "amount": amts, "day": days,
        }))
        # Collector forwards ~95% of consolidated balance within hours.
        forward_total = amts.sum() * rng.uniform(0.90, 0.99)
        per_member    = forward_total / max(1, len(ring) - 1)
        forward_day   = days.max() + rng.uniform(0.01, 0.5)
        chunks.append(pd.DataFrame({
            "src":    [collector] * (len(ring) - 1),
            "dst":    [int(x) for x in ring[1:]],
            "amount": [per_member] * (len(ring) - 1),
            "day":    [forward_day + 0.01 * i for i in range(len(ring) - 1)],
        }))

    txns_df = pd.concat(chunks, ignore_index=True)
    txns_df["amount"] = txns_df["amount"].round(2)
    txns_df = txns_df.astype({"src": "int64", "dst": "int64"})
    accounts_df = pd.DataFrame({"account_id": account_ids, "is_mule": is_mule})
    return accounts_df, txns_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature builders
# MAGIC
# MAGIC Two helpers, both Spark-native:
# MAGIC
# MAGIC - `build_account_features(accounts, txns)` — lifetime aggregates per account.
# MAGIC - `build_account_window_features(accounts, txns, window_days)` — rolling window
# MAGIC   aggregates used by `01_rules_engine`.

# COMMAND ----------

def build_account_features(accounts: DataFrame, txns: DataFrame) -> DataFrame:
    """Lifetime per-account aggregates. Returns one row per account."""
    inbound = (txns.groupBy(F.col("dst").alias("account_id"))
                   .agg(F.count("*").alias("in_count"),
                        F.sum("amount").alias("in_amt_sum"),
                        F.avg("amount").alias("in_amt_mean"),
                        F.countDistinct("src").alias("in_distinct_src")))

    outbound = (txns.groupBy(F.col("src").alias("account_id"))
                    .agg(F.count("*").alias("out_count"),
                         F.sum("amount").alias("out_amt_sum"),
                         F.avg("amount").alias("out_amt_mean"),
                         F.countDistinct("dst").alias("out_distinct_dst")))

    return (accounts
            .join(inbound,  "account_id", "left")
            .join(outbound, "account_id", "left")
            .na.fill(0.0)
            .withColumn("passthrough_ratio",
                        F.col("out_amt_sum") / F.greatest(F.col("in_amt_sum"), F.lit(1.0)))
            .withColumn("fanin_ratio",
                        F.col("in_distinct_src") / F.greatest(F.col("in_count"), F.lit(1.0))))


def build_account_window_features(
    accounts:    DataFrame,
    txns:        DataFrame,
    window_days: int = 7,
) -> DataFrame:
    """Rolling-window aggregates for the rules engine.

    Window is anchored at max(day) in the transaction table; "recent" = the last
    `window_days` days of activity.
    """
    end_day = txns.agg(F.max("day")).first()[0]
    recent  = txns.where(F.col("day") > F.lit(end_day - window_days))

    inb = (recent.groupBy(F.col("dst").alias("account_id"))
                 .agg(F.count("*").alias(f"in_count_{window_days}d"),
                      F.sum("amount").alias(f"in_amt_{window_days}d"),
                      F.countDistinct("src").alias(f"in_distinct_src_{window_days}d")))

    out = (recent.groupBy(F.col("src").alias("account_id"))
                 .agg(F.count("*").alias(f"out_count_{window_days}d"),
                      F.sum("amount").alias(f"out_amt_{window_days}d")))

    return (accounts
            .join(inb, "account_id", "left")
            .join(out, "account_id", "left")
            .na.fill(0.0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `tier_benchmark` Delta table
# MAGIC
# MAGIC One row per (tier, implementation). Powers the closing demo visual in
# MAGIC `10_tier_comparison`.

# COMMAND ----------

TIER_BENCHMARK_SCHEMA = StructType([
    StructField("tier_name",          StringType(),   False),
    StructField("tier_number",        IntegerType(),  False),
    StructField("pr_auc",             DoubleType(),   False),
    StructField("precision_at_1pct",  DoubleType(),   False),
    StructField("precision_at_5pct",  DoubleType(),   False),
    StructField("recall_at_5pct",     DoubleType(),   False),
    StructField("runtime_seconds",    DoubleType(),   False),
    StructField("mlflow_run_id",      StringType(),   True),
    StructField("logged_at",          TimestampType(), False),
])


def ensure_benchmark_table():
    """Create the benchmark Delta table if it does not exist."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {BENCHMARK_TABLE} (
            tier_name          STRING  NOT NULL,
            tier_number        INT     NOT NULL,
            pr_auc             DOUBLE  NOT NULL,
            precision_at_1pct  DOUBLE  NOT NULL,
            precision_at_5pct  DOUBLE  NOT NULL,
            recall_at_5pct     DOUBLE  NOT NULL,
            runtime_seconds    DOUBLE  NOT NULL,
            mlflow_run_id      STRING,
            logged_at          TIMESTAMP NOT NULL
        ) USING DELTA
    """)


def log_tier_metrics(
    tier_name:         str,
    tier_number:       int,
    pr_auc:            float,
    precision_at_1pct: float,
    precision_at_5pct: float,
    recall_at_5pct:    float,
    runtime_seconds:   float,
    mlflow_run_id:     str | None = None,
):
    """Append one row to the benchmark Delta table. Idempotent on (tier_name)."""
    ensure_benchmark_table()
    spark.sql(f"DELETE FROM {BENCHMARK_TABLE} WHERE tier_name = '{tier_name}'")

    row = spark.createDataFrame(
        [(tier_name, int(tier_number), float(pr_auc),
          float(precision_at_1pct), float(precision_at_5pct), float(recall_at_5pct),
          float(runtime_seconds), mlflow_run_id, pd.Timestamp.utcnow().to_pydatetime())],
        TIER_BENCHMARK_SCHEMA,
    )
    row.write.mode("append").saveAsTable(BENCHMARK_TABLE)
    print(f"✓ Benchmarked {tier_name}: PR-AUC={pr_auc:.3f} "
          f"P@5%={precision_at_5pct:.1%} R@5%={recall_at_5pct:.1%} "
          f"runtime={runtime_seconds:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation helpers

# COMMAND ----------

def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k_pct: float):
    """Precision / recall when reviewing the top `k_pct` of accounts by score."""
    n = len(scores)
    k = max(1, int(n * k_pct))
    top_idx = np.argsort(scores)[::-1][:k]
    tp = y_true[top_idx].sum()
    return tp / k, tp / max(1, y_true.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test-gate helper
# MAGIC
# MAGIC Each modeling notebook ends with three gates: smoke, correctness, determinism.
# MAGIC On failure, `run_gate` calls `dbutils.notebook.exit(...)` so the Jobs API reports
# MAGIC a clean failure message in the run status. Local execution still raises.

# COMMAND ----------

def run_gate(name: str, condition: bool, fail_msg: str):
    """Inline test gate. Pass → print; fail → notebook exits with FAIL[name]."""
    if condition:
        print(f"  ✓ gate[{name}] passed")
        return
    msg = f"FAIL[{name}]: {fail_msg}"
    print(f"  ✗ {msg}")
    try:
        dbutils.notebook.exit(msg)  # noqa: F821
    except NameError:
        raise AssertionError(msg)


@contextmanager
def timed(label: str):
    """Time a block and print the result. Stores the duration on `timed.last`."""
    t0 = time.perf_counter()
    yield
    timed.last = time.perf_counter() - t0
    print(f"  ⏱ {label}: {timed.last:.2f}s")


def run_distributed_or_local(train_fn, *args, _use_distributor: bool = False, **kwargs):
    """
    Run a PyTorch training function. By default, runs **direct on the driver** —
    this is the most reliable path on a single-worker-GPU cluster, and produces
    identical results to a 1-process TorchDistributor run.

    To opt in to TorchDistributor (multi-GPU or multi-worker clusters), pass
    `_use_distributor=True`. We keep the TorchDistributor pattern in the
    notebooks as documentation; switch this flag on once you're on a cluster
    that has the slots for it.

    Why direct-by-default: on this single-worker cluster, TorchDistributor's
    barrier-task scheduling SIGKILLs the Python notebook process before any
    exception we could catch, taking the whole run down.
    """
    if _use_distributor:
        from pyspark.ml.torch.distributor import TorchDistributor
        try:
            TorchDistributor(**TORCH_DISTRIBUTOR_KWARGS).run(train_fn, *args, **kwargs)
            return "TorchDistributor"
        except Exception as e:
            print(f"  ⚠ TorchDistributor failed ({type(e).__name__}: {str(e)[:200]})")
            print(f"  ↳ Falling back to direct training on the driver")
    train_fn(*args, **kwargs)
    return "direct (driver)"

# COMMAND ----------

print("✓ _shared loaded: synthetic generator, feature builders, benchmark + gate helpers")
