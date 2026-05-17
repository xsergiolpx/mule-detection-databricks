# Databricks notebook source
# MAGIC %md
# MAGIC # 📜 01 — Tier 1: Business-logic rules engine
# MAGIC
# MAGIC Bottom of the maturity ladder. Pure deterministic typology rules — every flag is
# MAGIC explainable in one sentence to a regulator.
# MAGIC
# MAGIC ### What this tier catches
# MAGIC
# MAGIC Account behaviours that match well-known mule typologies published by FATF, the Bank
# MAGIC of Thailand (BOT HR-03), Cifas (UK) and AUSTRAC.
# MAGIC
# MAGIC ### What this tier misses
# MAGIC
# MAGIC Anything outside the rule set. A novel typology slips through silently, and every
# MAGIC rule that fires creates an alert whether or not the account is actually a mule —
# MAGIC there is no probability, only a flag.
# MAGIC
# MAGIC ### Why this notebook is unusual
# MAGIC
# MAGIC This is the only notebook a compliance officer or regulator is realistically going
# MAGIC to read end-to-end, so it optimises for **code that reads like the typology
# MAGIC document**, not for compactness. Every rule is one named function, with a docstring
# MAGIC citing the typology it encodes. Adding a new rule = adding one function + one row
# MAGIC to the `RULES` registry.

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import time
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame, functions as F

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data and build window features
# MAGIC
# MAGIC The rolling 7-day window features come from `build_account_window_features()` in
# MAGIC `_shared`. Every rule below references these columns.

# COMMAND ----------

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Meet the dataset
# MAGIC
# MAGIC Before applying any rules, let's get a feel for what we're working with.

# COMMAND ----------

show_dataset_overview(accounts, txns)

# COMMAND ----------

features = build_account_window_features(accounts, txns, window_days=7)
features.printSchema()
display(features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The rules
# MAGIC
# MAGIC Each rule is a function `df → Column`. The function returns a boolean Spark
# MAGIC `Column` expressing the typology. No row-by-row iteration; the whole engine
# MAGIC compiles to a single Spark plan.

# COMMAND ----------

def rule_fanin(df: DataFrame):
    """
    Many distinct senders fund one account in 7 days.

    Typology: classic scam-collection pattern. The mule advertises bank details on
    Telegram / TikTok and receives small transfers from many victims.

    Reference: FATF, *Professional Money Laundering* (July 2018), p. 17.
    """
    return F.col("in_distinct_src_7d") >= 15


def rule_passthrough(df: DataFrame):
    """
    Outbound nearly equals inbound within the same 7-day window.

    Typology: pass-through laundering. The account is used as a transit point; funds
    rest for hours, not days. The defining signature of a money-mule account.

    Reference: BOT HR-03 typology #2 (`pass-through behaviour`).
    """
    return (F.col("out_amt_7d") /
            F.greatest(F.col("in_amt_7d"), F.lit(1.0))) > 0.70


def rule_burst_inflow(df: DataFrame):
    """
    Sudden large inflow from many small senders.

    Typology: recruited-mule activation. The account has been dormant or low-volume,
    then a single 7-day window shows > THB 50 000 inbound across > 10 senders.

    Reference: Cifas *Fraudscape 2025*, "first-party mule" archetype.
    """
    return (F.col("in_amt_7d")   > 50_000) & \
           (F.col("in_count_7d") > 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The registry
# MAGIC
# MAGIC The full set of rules the bank runs lives in **one list**. To add a typology, append
# MAGIC a row. To retire one, delete it. The registry is what an auditor would ask to see.

# COMMAND ----------

RULES = [
    ("rule_fanin",        rule_fanin),
    ("rule_passthrough",  rule_passthrough),
    ("rule_burst_inflow", rule_burst_inflow),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Apply the rules
# MAGIC
# MAGIC `apply_rules` folds the registry: it adds one boolean column per rule plus a
# MAGIC `rule_score` (sum of rules that fired). Everything is one Spark plan.

# COMMAND ----------

def apply_rules(df: DataFrame, rules):
    for name, rule_fn in rules:
        df = df.withColumn(name, rule_fn(df).cast("int"))
    score_expr = sum(F.col(name) for name, _ in rules)
    return df.withColumn("rule_score", score_expr)


with timed("apply rules (Spark)"):
    scored = apply_rules(features, RULES)
    scored.cache().count()   # materialise so timing is meaningful

runtime_s = timed.last
display(scored.select("account_id", "is_mule",
                       *[name for name, _ in RULES], "rule_score").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persist per-account scores

# COMMAND ----------

scores_target = scores_table("01_rules")
(scored.select("account_id", "is_mule", "rule_score",
               *[name for name, _ in RULES])
       .write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(scores_target))
print(f"✓ Wrote {scores_target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate
# MAGIC
# MAGIC Three metrics framed the way an investigator-capacity-constrained team thinks:
# MAGIC
# MAGIC - **PR-AUC** — overall ranking quality.
# MAGIC - **Precision@5%** — of the top 5% by score, how many are real mules?
# MAGIC - **Recall@5%** — what fraction of all mules sit in the top 5%?

# COMMAND ----------

from sklearn.metrics import average_precision_score

eval_pd = scored.select("is_mule", "rule_score").toPandas()
y       = eval_pd["is_mule"].astype(int).values
scores  = eval_pd["rule_score"].astype(float).values
# Break ties at random so top-K is well-defined when many rows share rule_score=0.
rng     = np.random.default_rng(SEED)
scores += rng.uniform(0, 1e-6, size=len(scores))

pr_auc          = average_precision_score(y, scores)
p1,  r_at_1pct  = precision_recall_at_k(y, scores, 0.01)
p5,  r_at_5pct  = precision_recall_at_k(y, scores, 0.05)
print(f"PR-AUC         : {pr_auc:.3f}")
print(f"Precision@1%   : {p1:.1%}    Recall@1% : {r_at_1pct:.1%}")
print(f"Precision@5%   : {p5:.1%}    Recall@5% : {r_at_5pct:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualisations
# MAGIC
# MAGIC Three demo visuals:
# MAGIC
# MAGIC 1. Accounts grouped by # rules hit, coloured by class.
# MAGIC 2. Rule co-firing heatmap (which rules tend to fire together).
# MAGIC 3. Precision / recall as the alert-threshold rises.

# COMMAND ----------

# 1. Hits-per-account ---------------------------------------------------------
hits = (scored.groupBy("rule_score", "is_mule").count()
              .orderBy("rule_score").toPandas())
hits["class"] = np.where(hits["is_mule"], "mule", "legit")

fig = px.bar(hits, x="rule_score", y="count", color="class", barmode="group",
              color_discrete_map={"legit": "#7f7f7f", "mule": "#d62728"},
              log_y=True, height=420,
              title="How many rules each account triggers (log scale)",
              labels={"rule_score": "# rules hit by an account",
                       "count":     "# accounts"})
fig.update_layout(template="plotly_white",
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                   margin=dict(l=20, r=20, t=60, b=40))
plotly_show(fig)

# COMMAND ----------

# 2. Rule co-firing heatmap --------------------------------------------------
rule_cols    = [name for name, _ in RULES]
cofire_pd    = scored.select(*rule_cols).toPandas().astype(int)
cofire_mat   = cofire_pd.T.dot(cofire_pd).values   # rule_i AND rule_j counts

fig = go.Figure(data=go.Heatmap(
    z=cofire_mat, x=rule_cols, y=rule_cols, colorscale="Reds",
    text=[[f"{v:,}" for v in row] for row in cofire_mat],
    texttemplate="%{text}", hovertemplate="%{y} ∩ %{x}: %{z:,}<extra></extra>",
))
fig.update_layout(template="plotly_white",
                   title="Rule co-firing (count of accounts hit by both rules)",
                   height=460, margin=dict(l=20, r=20, t=60, b=40),
                   xaxis=dict(tickangle=-30))
plotly_show(fig)

# COMMAND ----------

# 3. Precision / recall vs threshold ----------------------------------------
thresholds = list(range(0, len(RULES) + 1))
prec, rec  = [], []
for t in thresholds:
    flagged    = (eval_pd["rule_score"] >= t)
    n_flagged  = flagged.sum()
    tp         = (flagged & (eval_pd["is_mule"] == 1)).sum()
    prec.append(tp / max(1, n_flagged))
    rec.append (tp / max(1, eval_pd["is_mule"].sum()))

fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=prec, mode="lines+markers",
                          name="precision", line=dict(color="#1f77b4", width=3),
                          marker=dict(size=10)))
fig.add_trace(go.Scatter(x=thresholds, y=rec, mode="lines+markers",
                          name="recall",    line=dict(color="#ff7f0e", width=3),
                          marker=dict(size=10, symbol="square")))
fig.update_layout(template="plotly_white",
                   title="Precision and recall as the alert threshold rises",
                   height=420, yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                   xaxis=dict(title="rule_score threshold (≥)", tickvals=thresholds),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                   margin=dict(l=20, r=20, t=60, b=40))
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MLflow + benchmark logging

# COMMAND ----------

with mlflow.start_run(run_name="01_rules") as run:
    mlflow.log_param("n_rules",    len(RULES))
    mlflow.log_param("rules",      [name for name, _ in RULES])
    mlflow.log_param("window_days", 7)
    mlflow.log_metric("pr_auc",            pr_auc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r_at_5pct)
    mlflow.log_metric("runtime_seconds",   runtime_s)
    run_id = run.info.run_id

log_tier_metrics(
    tier_name         = "01_rules",
    tier_number       = 1,
    pr_auc            = pr_auc,
    precision_at_1pct = p1,
    precision_at_5pct = p5,
    recall_at_5pct    = r_at_5pct,
    runtime_seconds   = runtime_s,
    mlflow_run_id     = run_id,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Inference example — top-ranked vs lowest-ranked

# COMMAND ----------

top_5 = (scored.where("is_mule")
                .orderBy(F.col("rule_score").desc(), F.rand(SEED))
                .limit(5))
print("Top mule by rule_score:")
display(top_5.select("account_id", "is_mule", "rule_score",
                     *[name for name, _ in RULES]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Test gates

# COMMAND ----------

run_gate("smoke",        scored.count() > 0,
         "rules engine produced no rows")
run_gate("correctness",  pr_auc > 0.30,
         f"PR-AUC {pr_auc:.3f} below 0.30 floor for Tier-1")
top10_mule_ids = set(eval_pd.assign(score=scores)
                            .nlargest(10, "score")["is_mule"].astype(int).tolist())
run_gate("top10_has_mule", any(v == 1 for v in top10_mule_ids),
         "no mules in top-10 ranked accounts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Known typologies, fast to deploy, fully explainable | Novel patterns, calibrated probability, ring-level context | High false-positive rate or new typology appears in HR-03 backlog → Tier 2 (anomaly detection) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
