# Databricks notebook source
# MAGIC %md
# MAGIC # 🏁 10 — The maturity-ladder comparison (demo finale)
# MAGIC
# MAGIC Reads `tier_benchmark` (populated by notebooks 01–09) and renders the side-by-side
# MAGIC visuals that anchor the demo: *every tier adds something the previous one didn't
# MAGIC catch*.

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The benchmark table

# COMMAND ----------

bench = (spark.table(BENCHMARK_TABLE).orderBy("tier_number", "tier_name").toPandas())
display(spark.table(BENCHMARK_TABLE).orderBy("tier_number", "tier_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. PR-AUC, P@5%, R@5% by tier

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors_by_tier = {
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd", 5: "#bcbd22",
}
bar_colors = [colors_by_tier[t] for t in bench["tier_number"]]

for ax, col, title, fmt in [
    (axes[0], "pr_auc",            "PR-AUC by tier",            "{:.2f}"),
    (axes[1], "precision_at_5pct", "Precision @ top 5%",        "{:.1%}"),
    (axes[2], "recall_at_5pct",    "Recall @ top 5%",            "{:.1%}"),
]:
    ax.bar(bench["tier_name"], bench[col], color=bar_colors)
    for i, v in enumerate(bench[col]):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_xticklabels(bench["tier_name"], rotation=30, ha="right")
    ax.set_ylim(0, max(1.05, bench[col].max() * 1.15))
fig.suptitle("Mule detection — maturity ladder, side-by-side", fontsize=14)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Runtime by tier (CPU-only and GPU sit side by side)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(bench["tier_name"], bench["runtime_seconds"], color=bar_colors)
for i, v in enumerate(bench["runtime_seconds"]):
    ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
ax.set_xticklabels(bench["tier_name"], rotation=30, ha="right")
ax.set_ylabel("seconds (fit + score)")
ax.set_title("Runtime by tier")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PR curve overlay across every tier
# MAGIC
# MAGIC Read each tier's per-account score table, compute the PR curve, and overlay.

# COMMAND ----------

tier_score_tables = [
    ("01_rules",                       1, "rule_score"),
    ("02_isolation_forest", 2, "sklearn_score"),   # also write linkedin_score
    ("03_autoencoder",                 2, "recon_error"),
    ("04_xgboost_pu",                  3, "p_pu_corrected"),
    ("05_graphframes",                 4, "score"),
    ("06_graphsage",                   4, "score"),
    ("07_lstm",                        5, "score"),
    ("08_muletrack",                   5, "deviation"),
    ("09_tgn",                         5, "score"),
]

fig, ax = plt.subplots(figsize=(9, 7))
for name, tier_n, col in tier_score_tables:
    try:
        df = spark.table(scores_table(name)).toPandas()
        if col not in df.columns:
            print(f"  · {name} — no '{col}' column; skipping")
            continue
        y  = df["is_mule"].astype(int).values
        sc = df[col].values
        ap = average_precision_score(y, sc)
        p, r, _ = precision_recall_curve(y, sc)
        ax.plot(r, p, color=colors_by_tier[tier_n], alpha=0.85,
                 label=f"{name} (T{tier_n})  AUPRC={ap:.3f}")
    except Exception as e:
        print(f"  · {name} — skipped ({type(e).__name__}: {e})")

ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall across all tiers")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Incremental lift narrative
# MAGIC
# MAGIC The bar chart most useful for the slide deck: how much recall each tier *adds* over
# MAGIC the previous best.

# COMMAND ----------

# Best score per tier-number
best_per_tier = (bench.sort_values("recall_at_5pct", ascending=False)
                       .drop_duplicates(subset=["tier_number"])
                       .sort_values("tier_number")
                       .reset_index(drop=True))

prev_recall  = 0.0
incrementals = []
for _, row in best_per_tier.iterrows():
    inc = max(0.0, row["recall_at_5pct"] - prev_recall)
    incrementals.append(inc)
    prev_recall = max(prev_recall, row["recall_at_5pct"])
best_per_tier["incremental_recall"] = incrementals

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(best_per_tier["tier_name"],
               best_per_tier["recall_at_5pct"],
               color=[colors_by_tier[t] for t in best_per_tier["tier_number"]])
for b, inc, total in zip(bars, best_per_tier["incremental_recall"],
                           best_per_tier["recall_at_5pct"]):
    ax.text(b.get_x() + b.get_width()/2, total,
             f"total {total:.0%}\n+{inc:.0%}",
             ha="center", va="bottom", fontsize=9)
ax.set_xticklabels(best_per_tier["tier_name"], rotation=30, ha="right")
ax.set_ylabel("recall @ 5%")
ax.set_ylim(0, min(1.0, best_per_tier['recall_at_5pct'].max() * 1.25 + 0.1))
ax.set_title("Incremental recall lift up the maturity ladder")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Demo takeaway
# MAGIC
# MAGIC | Tier | What it bought | When to stop here |
# MAGIC |---|---|---|
# MAGIC | 1 — Rules | Explainability, no labels needed | Greenfield programmes, regulator-facing demos |
# MAGIC | 2 — Unsupervised | Catches novel patterns | When labels are sparse or untrusted |
# MAGIC | 3 — Supervised + PU | Calibrated probability | When investigator capacity is the bottleneck |
# MAGIC | 4 — Graph | Ring-level structural signal | When fraud is networked, not solo |
# MAGIC | 5 — Sequence + Temporal Graph | Order-of-events, evolving state | When pass-through latency matters |
# MAGIC
# MAGIC The right answer is rarely a single tier — production AML stacks combine tiers (rules
# MAGIC gate, ML score, graph ring-score, behavioural-biometrics tail) and use the union for
# MAGIC alerting + investigator routing.

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
