# Databricks notebook source
# MAGIC %md
# MAGIC # 🎯 04 — Tier 3: SparkXGBClassifier + PU-learning
# MAGIC
# MAGIC The first **supervised** tier in the ladder. Two passes over the same data:
# MAGIC
# MAGIC 1. **Standard supervised baseline** — `SparkXGBClassifier` assuming every unlabelled
# MAGIC    account is a negative.
# MAGIC 2. **Elkan–Noto PU correction** — recognises that the HR-03 / SAR list is *not* a
# MAGIC    complete enumeration of mules. The unlabelled set contains mules that haven't
# MAGIC    been caught yet. We train on a hidden-positives setup and divide the predicted
# MAGIC    probability by the estimated label propensity `c`.
# MAGIC
# MAGIC ### Why PU matters in production
# MAGIC
# MAGIC Treating unlabelled accounts as `y = 0` is the most common mistake in deployed AML
# MAGIC ML. The model learns to predict *accounts that look like recorded HR-03 cases*
# MAGIC rather than *accounts that exhibit mule behaviour*, and silently under-predicts on
# MAGIC novel typologies.
# MAGIC
# MAGIC Reference: §8.1 of `mule_detection_research.md`; Elkan & Noto (2008).

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import time
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
from sklearn.metrics import precision_recall_curve

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build features and assemble vector

# COMMAND ----------

FEATURE_COLS = [
    "in_count", "in_amt_sum", "in_amt_mean", "in_distinct_src",
    "out_count", "out_amt_sum", "out_amt_mean", "out_distinct_dst",
    "passthrough_ratio", "fanin_ratio",
]

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)
feat_sdf = build_account_features(accounts, txns).cache()

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
ml_df     = (assembler.transform(feat_sdf)
                       .select("account_id", "features",
                               F.col("is_mule").cast("int").alias("label_true")))
ml_df.cache().count()
print(f"{ml_df.count():,} rows ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/test split

# COMMAND ----------

train_df, test_df = ml_df.randomSplit([0.7, 0.3], seed=SEED)
print(f"train = {train_df.count():,}    test = {test_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Path A — supervised baseline
# MAGIC
# MAGIC Treat every unlabelled account as a negative. This is the wrong assumption in real
# MAGIC AML data — we run it to show what PU correction is fixing.

# COMMAND ----------

train_sup = train_df.withColumnRenamed("label_true", "label")
test_sup  = test_df.withColumnRenamed("label_true",  "label")

clf_sup = SparkXGBClassifier(
    features_col="features", label_col="label",
    num_workers=1,    # match Spark task-slot count on this cluster
    n_estimators=300, max_depth=5, learning_rate=0.1,
    tree_method="hist", eval_metric="aucpr",
)

t0 = time.perf_counter()
model_sup = clf_sup.fit(train_sup)
runtime_sup = time.perf_counter() - t0

pred_sup  = model_sup.transform(test_sup)
auprc_sup = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
).evaluate(pred_sup)

print(f"Supervised SparkXGBoost AUPRC = {auprc_sup:.3f}  (runtime {runtime_sup:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Path B — PU-learning (Elkan-Noto)
# MAGIC
# MAGIC Simulate the real-world HR-03 reality: half of the true mules are *not* on the list.
# MAGIC The model is trained on `s ∈ {0, 1}` (on-list vs not-on-list), then predictions are
# MAGIC divided by `c = P(s=1 | y=1)` estimated on a held-out slice of labelled positives.

# COMMAND ----------

# Hide half the mules in the training set --------------------------------------
rng_pu = np.random.default_rng(SEED)

train_pu = (train_df
    .withColumn("rand",  F.rand(seed=SEED))
    .withColumn(
        "label",
        F.when(
            (F.col("label_true") == 1) & (F.col("rand") < 0.5), F.lit(1)
        ).otherwise(F.lit(0))
    )
    .drop("rand")
    .select("account_id", "features", "label", "label_true")
    .cache()
)
print("PU training label distribution:")
train_pu.groupBy("label").count().show()

# COMMAND ----------

clf_pu = SparkXGBClassifier(
    features_col="features", label_col="label",
    num_workers=1,    # match Spark task-slot count on this cluster
    n_estimators=300, max_depth=5, learning_rate=0.1,
    tree_method="hist", eval_metric="aucpr",
)

t0 = time.perf_counter()
model_pu = clf_pu.fit(train_pu)
runtime_pu = time.perf_counter() - t0

# COMMAND ----------

# Estimate c = P(s=1 | y=1) on the held-out labelled positives ----------------
# Use the labelled (s=1) rows themselves — their predicted P(s=1) averaged is the
# Elkan-Noto estimator for c under SCAR.
def proba_pos(df):
    """Return P(s=1 | x) for each row as a Python list of floats."""
    from pyspark.ml.functions import vector_to_array
    pred = (model_pu.transform(df)
                     .select(vector_to_array("probability").alias("p")))
    return pred.select(F.col("p")[1].alias("p1")).toPandas()["p1"].values

labelled_pos_test = train_pu.where("label = 1").limit(1000)
c_est = proba_pos(labelled_pos_test).mean()
c_est = max(c_est, 1e-3)
print(f"Estimated propensity c = {c_est:.3f}")

# COMMAND ----------

# Score the test set and apply PU correction ---------------------------------
test_pu = test_df.withColumnRenamed("label_true", "label")
pred_pu = model_pu.transform(test_pu)

from pyspark.ml.functions import vector_to_array
pu_pd = (pred_pu
    .withColumn("p1", vector_to_array("probability")[1])
    .select("account_id", "label", "p1")
    .toPandas())
pu_pd["p_pu_corrected"] = np.clip(pu_pd["p1"] / c_est, 0.0, 1.0)

y_test   = pu_pd["label"].astype(int).values
proba_pu = pu_pd["p_pu_corrected"].values

from sklearn.metrics import average_precision_score
auprc_pu = average_precision_score(y_test, proba_pu)
p1_pu, _ = precision_recall_at_k(y_test, proba_pu, 0.01)
p5_pu, r5_pu = precision_recall_at_k(y_test, proba_pu, 0.05)
print(f"PU-corrected AUPRC = {auprc_pu:.3f}  P@5%={p5_pu:.1%}  R@5%={r5_pu:.1%}")

# Also pull supervised scores for the same test set for the PR-curve overlay --
sup_pd = (pred_sup
    .withColumn("p1", vector_to_array("probability")[1])
    .select("account_id", "label", "p1")
    .toPandas())
y_sup   = sup_pd["label"].astype(int).values
proba_sup = sup_pd["p1"].values
p1_sup, _ = precision_recall_at_k(y_sup, proba_sup, 0.01)
p5_sup, r5_sup = precision_recall_at_k(y_sup, proba_sup, 0.05)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisations

# COMMAND ----------

# 1. PR curves overlay --------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for name, y_arr, sc_arr, color in [
    ("supervised baseline",  y_sup, proba_sup, "#1f77b4"),
    ("PU-corrected",         y_test, proba_pu, "#2ca02c"),
]:
    p, r, _ = precision_recall_curve(y_arr, sc_arr)
    auc = auprc_sup if name == "supervised baseline" else auprc_pu
    ax.plot(r, p, color=color, label=f"{name}  AUPRC={auc:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Supervised vs PU-corrected SparkXGBoost")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# 2. Feature importance -------------------------------------------------------
importances = model_pu.get_booster().get_score(importance_type="gain")
imp_pd = (pd.DataFrame({"feature": list(importances.keys()),
                         "gain":    list(importances.values())})
            .sort_values("gain", ascending=True))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_pd["feature"], imp_pd["gain"], color="#2ca02c")
ax.set_xlabel("xgboost gain")
ax.set_title("Feature importance (PU-corrected model)")
fig.tight_layout()

# COMMAND ----------

# 3. Calibration curve --------------------------------------------------------
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(7, 5))
for name, sc, color in [
    ("supervised raw probability",  proba_sup, "#1f77b4"),
    ("PU-corrected probability",    proba_pu,  "#2ca02c"),
]:
    frac_pos, mean_pred = calibration_curve(y_test if name != "supervised raw probability" else y_sup,
                                            sc, n_bins=20, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "-o", color=color, label=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("fraction of positives")
ax.set_title(f"Calibration  (estimated c = {c_est:.3f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist scores + MLflow + benchmark

# COMMAND ----------

scored_pd = pu_pd[["account_id", "label", "p1", "p_pu_corrected"]].rename(
    columns={"label": "is_mule", "p1": "p_supervised"}
)
(spark.createDataFrame(scored_pd)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(scores_table("04_xgboost_pu")))

with mlflow.start_run(run_name="04_xgboost_pu_learning") as run:
    mlflow.log_params({
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.1,
        "num_workers":  2,    "label_hide_fraction": 0.5,
    })
    mlflow.log_metric("c_estimate",      c_est)
    mlflow.log_metric("auprc_supervised", auprc_sup)
    mlflow.log_metric("auprc_pu",         auprc_pu)
    mlflow.log_metric("pr_auc",            auprc_pu)
    mlflow.log_metric("precision_at_1pct", p1_pu)
    mlflow.log_metric("precision_at_5pct", p5_pu)
    mlflow.log_metric("recall_at_5pct",    r5_pu)
    mlflow.log_metric("runtime_seconds",   runtime_sup + runtime_pu)
    run_id = run.info.run_id

log_tier_metrics("04_xgboost_pu", 3,
                 auprc_pu, p1_pu, p5_pu, r5_pu, runtime_sup + runtime_pu, run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test gates

# COMMAND ----------

run_gate("smoke",              len(proba_pu) > 0,        "no PU predictions")
run_gate("pr_auc_pu",          auprc_pu > 0.60,          f"PU AUPRC {auprc_pu:.3f} below 0.60 floor")
run_gate("c_in_unit_interval", 0.0 < c_est <= 1.0,       f"c estimate out of range: {c_est:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Calibrated `P(mule)` from tabular features, learns from partial labels | Network / ring-level signal | When the false-positive cost of standalone-account scoring exceeds investigator capacity → Tier 4 (graph) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
