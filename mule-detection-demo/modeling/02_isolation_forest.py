# Databricks notebook source
# MAGIC %md
# MAGIC # 🌲 02 — Tier 2: Isolation Forest (sklearn + LinkedIn Spark)
# MAGIC
# MAGIC Unsupervised anomaly detection. The same algorithm, run two ways:
# MAGIC
# MAGIC 1. **sklearn `IsolationForest`** — single-node, pandas in / pandas out. The
# MAGIC    classic baseline from the research-doc Appendix §A.2.
# MAGIC 2. **LinkedIn `isolation-forest`** — distributed Spark/Scala implementation
# MAGIC    used in LinkedIn's anti-abuse stack. Same algorithm, same metrics, but it
# MAGIC    scales out across the cluster.
# MAGIC
# MAGIC Both run against the same Delta table and write **two rows** to
# MAGIC `tier_benchmark` (`02_isolation_forest_sklearn`, `02_isolation_forest_spark`)
# MAGIC so the closing demo visual shows the scale-out win.
# MAGIC
# MAGIC ### What this tier catches
# MAGIC
# MAGIC Accounts whose feature vector is statistically far from the bulk. No labels
# MAGIC required — surfaces unknown-unknowns that rules don't encode.
# MAGIC
# MAGIC ### What this tier misses
# MAGIC
# MAGIC Calibration (the score is a ranking, not a probability) and any signal that
# MAGIC lives in *who you transact with* rather than *what your aggregates look like*.

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import DataFrame as SparkDataFrame, functions as F
from pyspark.ml.feature import VectorAssembler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data and build features

# COMMAND ----------

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)
feat_sdf = build_account_features(accounts, txns).cache()

FEATURE_COLS = [
    "in_count", "in_amt_sum", "in_amt_mean", "in_distinct_src",
    "out_count", "out_amt_sum", "out_amt_mean", "out_distinct_dst",
    "passthrough_ratio", "fanin_ratio",
]
print(f"{feat_sdf.count():,} accounts × {len(FEATURE_COLS)} features")
display(feat_sdf.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Path A — sklearn `IsolationForest` (single-node baseline)
# MAGIC
# MAGIC Pull to pandas, fit on one machine. This is what most teams reach for first.

# COMMAND ----------

with timed("sklearn: toPandas"):
    feat_pd = feat_sdf.toPandas()

X = feat_pd[FEATURE_COLS].values
y = feat_pd["is_mule"].astype(int).values

with timed("sklearn: fit + score"):
    iso_sklearn = IsolationForest(
        n_estimators=200,
        contamination=N_MULES / (N_LEGIT + N_MULES),
        random_state=SEED,
        n_jobs=-1,
    )
    iso_sklearn.fit(X)
    scores_sklearn = -iso_sklearn.score_samples(X)   # higher = more anomalous

runtime_sklearn = timed.last

pr_auc_sk = average_precision_score(y, scores_sklearn)
p1_sk,  r1_sk  = precision_recall_at_k(y, scores_sklearn, 0.01)
p5_sk,  r5_sk  = precision_recall_at_k(y, scores_sklearn, 0.05)
print(f"\nsklearn   PR-AUC={pr_auc_sk:.3f}  P@5%={p5_sk:.1%}  R@5%={r5_sk:.1%}  "
      f"runtime={runtime_sklearn:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC # Path B — LinkedIn `isolation-forest` (distributed Spark/Scala)
# MAGIC
# MAGIC The library is attached to the cluster as a Maven dependency (see
# MAGIC `ISOLATION_FOREST_MAVEN_COORD` in `config.py`). It exposes a Spark ML
# MAGIC `Estimator`; the cleanest PySpark access pattern is via the JVM gateway, wrapped
# MAGIC in a small helper so the rest of the notebook stays Pythonic.
# MAGIC
# MAGIC Reference: <https://github.com/linkedin/isolation-forest>

# COMMAND ----------

def fit_linkedin_isolation_forest(df, features_col="features",
                                   n_estimators=200, contamination=0.04,
                                   bootstrap=False, seed=SEED):
    """Train LinkedIn's Spark IsolationForest via JVM gateway. Returns the JavaModel."""
    sc  = spark.sparkContext
    jvm = sc._jvm

    iso = (jvm.com.linkedin.relevance.isolationforest.IsolationForest()
              .setNumEstimators(n_estimators)
              .setContamination(contamination)
              .setBootstrap(bootstrap)
              .setRandomSeed(seed)
              .setFeaturesCol(features_col)
              .setPredictionCol("predicted_label")
              .setScoreCol("outlier_score"))
    return iso.fit(df._jdf)


def score_with_linkedin_isolation_forest(model, df):
    """Apply a trained model to a Spark DataFrame, return a Python Spark DataFrame."""
    java_predictions = model.transform(df._jdf)
    # Wrap the Java DataFrame back into a PySpark DataFrame.
    return SparkDataFrame(java_predictions, spark)

# COMMAND ----------

# Vectorise features --------------------------------------------------------
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
feat_vec  = assembler.transform(feat_sdf).select("account_id", "is_mule", "features").cache()
feat_vec.count()

t0 = time.perf_counter()
model_li  = fit_linkedin_isolation_forest(
    feat_vec,
    n_estimators=200,
    contamination=N_MULES / (N_LEGIT + N_MULES),
    seed=SEED,
)
scored_li = score_with_linkedin_isolation_forest(model_li, feat_vec)
scored_li.cache().count()
runtime_spark = time.perf_counter() - t0
print(f"LinkedIn Spark IF (fit + score) = {runtime_spark:.1f}s")

# COMMAND ----------

# Evaluate ------------------------------------------------------------------
scored_li_pd = (scored_li.select("account_id", "is_mule", "outlier_score")
                          .toPandas()
                          .sort_values("account_id"))
y_li        = scored_li_pd["is_mule"].astype(int).values
scores_li   = scored_li_pd["outlier_score"].values

pr_auc_li = average_precision_score(y_li, scores_li)
p1_li, r1_li = precision_recall_at_k(y_li, scores_li, 0.01)
p5_li, r5_li = precision_recall_at_k(y_li, scores_li, 0.05)
print(f"\nLinkedIn  PR-AUC={pr_auc_li:.3f}  P@5%={p5_li:.1%}  R@5%={r5_li:.1%}  "
      f"runtime={runtime_spark:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist scores

# COMMAND ----------

scores_combined = (
    feat_pd[["account_id", "is_mule"]]
        .assign(sklearn_score=scores_sklearn)
        .merge(scored_li_pd[["account_id", "outlier_score"]]
                .rename(columns={"outlier_score": "linkedin_score"}),
              on="account_id", how="left")
)
(spark.createDataFrame(scores_combined)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("02_isolation_forest")))
print(f"✓ Wrote {scores_table('02_isolation_forest')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisations
# MAGIC
# MAGIC 1. **Anomaly score distribution** — mules should sit in the right tail for both impls.
# MAGIC 2. **Side-by-side bar chart** — PR-AUC + runtime, sklearn vs LinkedIn-Spark.
# MAGIC 3. **PR curve overlay**.

# COMMAND ----------

# 1. Score histograms -------------------------------------------------------
# IMPORTANT: each (scores, labels) pair must be drawn from the *same row order*.
# `scores_sklearn` is aligned to `feat_pd` (so use `y`); `scores_li` was pulled
# back from Spark sorted by account_id (so use `y_li`). Mixing the two collapses
# the LinkedIn histogram to look identical for both classes — that's a bug, not
# a real result.
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, (name, scores_arr, y_arr) in zip(axes, [
    ("sklearn",        scores_sklearn, y),
    ("LinkedIn-Spark", scores_li,      y_li),
]):
    ax.hist(scores_arr[y_arr == 0], bins=60, alpha=0.6, color="#7f7f7f", label="legit", density=True)
    ax.hist(scores_arr[y_arr == 1], bins=60, alpha=0.7, color="#d62728", label="mule",  density=True)
    ax.set_title(f"{name} — anomaly score distribution")
    ax.set_xlabel("score (higher = more anomalous)"); ax.legend()
fig.tight_layout()

# COMMAND ----------

# 2. Side-by-side metrics ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
labels = ["sklearn\n(single node)", "LinkedIn Spark\n(distributed)"]

axes[0].bar(labels, [pr_auc_sk, pr_auc_li], color=["#1f77b4", "#2ca02c"])
for i, v in enumerate([pr_auc_sk, pr_auc_li]):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
axes[0].set_title("PR-AUC"); axes[0].set_ylim(0, 1.05)

axes[1].bar(labels, [runtime_sklearn, runtime_spark], color=["#1f77b4", "#2ca02c"])
for i, v in enumerate([runtime_sklearn, runtime_spark]):
    axes[1].text(i, v, f"{v:.1f}s", ha="center", va="bottom")
axes[1].set_title("Runtime (fit + score)"); axes[1].set_ylabel("seconds")
fig.suptitle("sklearn vs LinkedIn Spark Isolation Forest", fontsize=13)
fig.tight_layout()

# COMMAND ----------

# 3. PR curves overlay -------------------------------------------------------
from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(figsize=(7, 5))
for name, sc, color in [("sklearn", scores_sklearn, "#1f77b4"),
                          ("LinkedIn-Spark", scores_li, "#2ca02c")]:
    p, r, _ = precision_recall_curve(y, sc) if name == "sklearn" else \
              precision_recall_curve(y_li, sc)
    auprc = pr_auc_sk if name == "sklearn" else pr_auc_li
    ax.plot(r, p, color=color, label=f"{name}  AUPRC={auprc:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall — sklearn vs LinkedIn Spark")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow + benchmark logging
# MAGIC
# MAGIC Two MLflow runs, two rows in `tier_benchmark`.

# COMMAND ----------

with mlflow.start_run(run_name="02_isolation_forest_sklearn") as run_sk:
    mlflow.log_param("impl", "sklearn")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("pr_auc",            pr_auc_sk)
    mlflow.log_metric("precision_at_1pct", p1_sk)
    mlflow.log_metric("precision_at_5pct", p5_sk)
    mlflow.log_metric("recall_at_5pct",    r5_sk)
    mlflow.log_metric("runtime_seconds",   runtime_sklearn)
    run_sk_id = run_sk.info.run_id

with mlflow.start_run(run_name="02_isolation_forest_spark") as run_sp:
    mlflow.log_param("impl", "linkedin-spark")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("maven_coord",  ISOLATION_FOREST_MAVEN_COORD)
    mlflow.log_metric("pr_auc",            pr_auc_li)
    mlflow.log_metric("precision_at_1pct", p1_li)
    mlflow.log_metric("precision_at_5pct", p5_li)
    mlflow.log_metric("recall_at_5pct",    r5_li)
    mlflow.log_metric("runtime_seconds",   runtime_spark)
    run_sp_id = run_sp.info.run_id

log_tier_metrics("02_isolation_forest_sklearn", 2,
                 pr_auc_sk, p1_sk, p5_sk, r5_sk, runtime_sklearn, run_sk_id)
log_tier_metrics("02_isolation_forest_spark",   2,
                 pr_auc_li, p1_li, p5_li, r5_li, runtime_spark,   run_sp_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test gates

# COMMAND ----------

run_gate("smoke",                 len(scores_sklearn) > 0,                "no sklearn scores")
run_gate("sklearn_pr_auc",        pr_auc_sk > 0.50,                       f"sklearn PR-AUC {pr_auc_sk:.3f} below 0.50 floor")
run_gate("spark_pr_auc",          pr_auc_li > 0.50,                       f"LinkedIn Spark PR-AUC {pr_auc_li:.3f} below 0.50 floor")
run_gate("impls_agree_within_0.2", abs(pr_auc_sk - pr_auc_li) < 0.20,     f"sklearn vs LinkedIn AUPRC diverge by {abs(pr_auc_sk - pr_auc_li):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Anomalies not encoded as rules, no labels needed | Calibrated probability, network-level signal | Need to rank by `P(mule)` for capacity-constrained review → Tier 3 (supervised XGBoost + PU) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
