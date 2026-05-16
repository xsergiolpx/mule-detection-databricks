# Databricks notebook source
# MAGIC %md
# MAGIC # 🪜 08 — Tier 5: MuleTrack-style Markov chain (Spark + numpy)
# MAGIC
# MAGIC Reproduces the core idea of **MuleTrack** (Jambhrunkar, Sharma, Singla, Kailasam —
# MAGIC IWANN 2025): each account's day-by-day activity is bucketed into a small set of
# MAGIC behavioural states, and a per-account transition matrix is compared against the
# MAGIC population-level legitimate transition matrix.
# MAGIC
# MAGIC The point of the paper is that this **CPU-only, interpretable, sub-30-min batch
# MAGIC inference** approach hits surprisingly competitive performance on UPI / PromptPay
# MAGIC data — and produces transition matrices that auditors can read directly. No GPU,
# MAGIC no PyTorch, no `TorchDistributor`.
# MAGIC
# MAGIC ### Behavioural states
# MAGIC
# MAGIC | State | Definition |
# MAGIC |---|---|
# MAGIC | `dormant`        | no activity that day |
# MAGIC | `low_activity`   | small in + out volume |
# MAGIC | `burst_inbound`  | large in, no/low out |
# MAGIC | `pass_through`   | large in *and* large out same day |
# MAGIC | `cash_out`       | only out flows |
# MAGIC
# MAGIC The deviation score for an account is the Frobenius distance between its
# MAGIC transition matrix and the legitimate-population reference matrix.

# COMMAND ----------

# MAGIC %pip install plotly --quiet

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, IntegerType, DoubleType,
                                StringType, ArrayType, FloatType)
from sklearn.metrics import average_precision_score, precision_recall_curve

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

STATES = ["dormant", "low_activity", "burst_inbound", "pass_through", "cash_out"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Aggregate to (account, day) and label each day with a state

# COMMAND ----------

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

# Per-(account, day) inbound & outbound aggregates ---------------------------
in_agg  = (txns.groupBy(F.col("dst").alias("account_id"),
                          F.floor("day").cast("int").alias("day_idx"))
                .agg(F.sum("amount").alias("in_amt"),
                     F.count("*").alias("in_count")))
out_agg = (txns.groupBy(F.col("src").alias("account_id"),
                          F.floor("day").cast("int").alias("day_idx"))
                .agg(F.sum("amount").alias("out_amt"),
                     F.count("*").alias("out_count")))

daily = (accounts.select("account_id")
                  .crossJoin(spark.range(0, N_DAYS).withColumnRenamed("id", "day_idx"))
                  .join(in_agg,  ["account_id", "day_idx"], "left")
                  .join(out_agg, ["account_id", "day_idx"], "left")
                  .na.fill(0.0))

# Classify each (account, day) into a state ----------------------------------
LOW_AMT  = 5_000.0
HIGH_AMT = 30_000.0

state_expr = (
    F.when((F.col("in_amt") == 0) & (F.col("out_amt") == 0), F.lit("dormant"))
     .when((F.col("in_amt") >= HIGH_AMT) & (F.col("out_amt") >= HIGH_AMT),
           F.lit("pass_through"))
     .when((F.col("in_amt") >= HIGH_AMT) & (F.col("out_amt") < HIGH_AMT),
           F.lit("burst_inbound"))
     .when((F.col("in_amt") == 0) & (F.col("out_amt") > 0),
           F.lit("cash_out"))
     .otherwise(F.lit("low_activity"))
)
daily = daily.withColumn("state", state_expr).cache()
display(daily.groupBy("state").count().orderBy("state"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build per-account state sequences and transition matrices
# MAGIC
# MAGIC We collect each account's state sequence (ordered by day) and compute a 5×5
# MAGIC transition-count matrix in one Spark pandas-UDF call.

# COMMAND ----------

# Collect state sequences ----------------------------------------------------
seq_df = (daily.orderBy("account_id", "day_idx")
                .groupBy("account_id")
                .agg(F.collect_list("state").alias("states"))
                .join(accounts, "account_id"))

# COMMAND ----------

def states_to_transition_matrix(states):
    """Empirical transition counts → row-normalised transition matrix."""
    M = np.zeros((len(STATES), len(STATES)), dtype="float32")
    for a, b in zip(states[:-1], states[1:]):
        M[STATE_IDX[a], STATE_IDX[b]] += 1.0
    row_sums = M.sum(axis=1, keepdims=True)
    return np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)


from pyspark.sql.functions import pandas_udf

@pandas_udf(ArrayType(FloatType()))
def to_transition_flat(states_series: pd.Series) -> pd.Series:
    return states_series.apply(
        lambda s: states_to_transition_matrix(s).flatten().tolist()
    )

t0 = time.perf_counter()
matrices_sdf = seq_df.withColumn("trans", to_transition_flat("states"))
matrices_pd  = (matrices_sdf.select("account_id", "is_mule", "trans")
                              .toPandas())
runtime_build = time.perf_counter() - t0
print(f"✓ Per-account transition matrices in {runtime_build:.1f}s")

trans_arr = np.stack(matrices_pd["trans"].values).reshape(-1, len(STATES), len(STATES))
y         = matrices_pd["is_mule"].astype(int).values

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reference (legitimate) population matrix + deviation score

# COMMAND ----------

ref_matrix = trans_arr[y == 0].mean(axis=0)
print("Reference (legitimate-population) transition matrix:")
print(pd.DataFrame(ref_matrix, index=STATES, columns=STATES).round(3))

# Frobenius distance from the reference per account --------------------------
deviation = np.sqrt(((trans_arr - ref_matrix[None]) ** 2).sum(axis=(1, 2)))

auprc = average_precision_score(y, deviation)
p1, _ = precision_recall_at_k(y, deviation, 0.01)
p5, r5 = precision_recall_at_k(y, deviation, 0.05)
print(f"MuleTrack-style deviation AUPRC={auprc:.3f}  P@5%={p5:.1%}  R@5%={r5:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Visualisations

# COMMAND ----------

# 1. Sankey — legit vs mule population transitions ---------------------------
def sankey_for_population(M, title, color):
    sources, targets, values = [], [], []
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            if M[i, j] > 0.005:
                sources.append(i)
                targets.append(j + len(STATES))     # rhs nodes offset
                values.append(float(M[i, j]))
    node_labels = [f"{s} (t)" for s in STATES] + [f"{s} (t+1)" for s in STATES]
    fig = go.Figure(go.Sankey(
        node=dict(label=node_labels, pad=18, thickness=18,
                   color=[color]*len(node_labels)),
        link=dict(source=sources, target=targets, value=values),
    ))
    fig.update_layout(title_text=title, font_size=11, height=400)
    return fig

mule_matrix = trans_arr[y == 1].mean(axis=0)
displayHTML(sankey_for_population(ref_matrix,  "Legitimate population — average transitions", "#7f7f7f").to_html())
displayHTML(sankey_for_population(mule_matrix, "Mule population — average transitions",        "#d62728").to_html())

# COMMAND ----------

# 2. Side-by-side heatmaps ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, M, title in [(axes[0], ref_matrix,  "Legit reference"),
                      (axes[1], mule_matrix, "Mule average")]:
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=max(ref_matrix.max(), mule_matrix.max()))
    ax.set_xticks(range(len(STATES))); ax.set_xticklabels(STATES, rotation=30, ha="right")
    ax.set_yticks(range(len(STATES))); ax.set_yticklabels(STATES)
    ax.set_title(title)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i,j] > 0.4 else "black", fontsize=8)
fig.suptitle("Transition matrix: legit vs mule", fontsize=13)
fig.tight_layout()

# COMMAND ----------

# 3. Deviation score distribution --------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(deviation[y == 0], bins=80, alpha=0.6, color="#7f7f7f", label="legit", density=True)
ax.hist(deviation[y == 1], bins=80, alpha=0.7, color="#d62728", label="mule",  density=True)
ax.set_xlabel("Frobenius distance from legitimate reference")
ax.set_ylabel("density")
ax.set_title("Deviation score by class")
ax.legend()
fig.tight_layout()

# COMMAND ----------

# 4. PR curve ----------------------------------------------------------------
p, r, _ = precision_recall_curve(y, deviation)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r, p, color="#bcbd22", label=f"MuleTrack-style  AUPRC={auprc:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall — Tier 5 (Markov / MuleTrack-style)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persist scores + MLflow + benchmark

# COMMAND ----------

scored_pd = matrices_pd[["account_id", "is_mule"]].assign(deviation=deviation)
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("08_muletrack")))

with mlflow.start_run(run_name="08_muletrack_markov") as run:
    mlflow.log_params({"states": STATES,
                        "low_amount": LOW_AMT, "high_amount": HIGH_AMT,
                        "n_days": N_DAYS})
    mlflow.log_metric("pr_auc",            auprc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r5)
    mlflow.log_metric("runtime_seconds",   runtime_build)
    run_id = run.info.run_id

log_tier_metrics("08_muletrack", 5, auprc, p1, p5, r5, runtime_build, run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test gates

# COMMAND ----------

run_gate("smoke",     deviation.size > 0,                  "no deviation scores")
run_gate("pr_auc",    auprc > 0.40,                        f"MuleTrack AUPRC {auprc:.3f} below 0.40 floor")
run_gate("ref_normalised", abs(ref_matrix.sum(axis=1).mean() - 1.0) < 0.05,
         "reference transition matrix rows do not sum to ~1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Interpretable temporal pattern, CPU-only, sub-30-min batch | Cross-account neighbour signal, evolving graph state | Need full temporal-graph reasoning → Tier 5 TGN |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
