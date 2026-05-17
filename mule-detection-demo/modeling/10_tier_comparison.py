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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

colors_by_tier = {
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd", 5: "#bcbd22",
}
bar_colors = [colors_by_tier[t] for t in bench["tier_number"]]

# 3 metric bars side by side
fig = make_subplots(rows=1, cols=3,
                     subplot_titles=("PR-AUC by tier",
                                      "Precision @ top 5%",
                                      "Recall @ top 5%"))
for c, (col, fmt) in enumerate([("pr_auc",            "{:.2f}"),
                                  ("precision_at_5pct", "{:.1%}"),
                                  ("recall_at_5pct",    "{:.1%}")], start=1):
    fig.add_trace(go.Bar(x=bench["tier_name"], y=bench[col],
                          marker_color=bar_colors,
                          text=[fmt.format(v) for v in bench[col]],
                          textposition="outside",
                          showlegend=False), row=1, col=c)
    fig.update_yaxes(range=[0, max(1.05, bench[col].max() * 1.20)], row=1, col=c)
    fig.update_xaxes(tickangle=-30, row=1, col=c)
fig.update_layout(template="plotly_white", height=460,
                   title_text="<b>Mule detection — maturity ladder, side-by-side</b>",
                   margin=dict(l=20, r=20, t=80, b=80))
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Runtime by tier (CPU-only and GPU sit side by side)

# COMMAND ----------

fig = go.Figure(go.Bar(
    x=bench["tier_name"], y=bench["runtime_seconds"],
    marker_color=bar_colors,
    text=[f"{v:.1f}s" for v in bench["runtime_seconds"]],
    textposition="outside",
))
fig.update_layout(template="plotly_white", height=440,
                   title="Runtime by tier (seconds — fit + score)",
                   xaxis_tickangle=-30, yaxis_title="seconds",
                   margin=dict(l=20, r=20, t=60, b=80), showlegend=False)
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PR curve overlay across every tier
# MAGIC
# MAGIC Read each tier's per-account score table, compute the PR curve, and overlay.

# COMMAND ----------

tier_score_tables = [
    ("01_rules",            1, "rule_score"),
    ("02_isolation_forest", 2, "sklearn_score"),
    ("03_autoencoder",      2, "recon_error"),
    ("04_xgboost_pu",       3, "p_pu_corrected"),
    ("05_graphframes",      4, "score"),
    ("06_graphsage",        4, "score"),
    ("07_lstm",             5, "score"),
    ("08_muletrack",        5, "deviation"),
    ("09_tgn",              5, "score"),
]

fig = go.Figure()
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
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines",
                                  line=dict(color=colors_by_tier[tier_n], width=2.2),
                                  opacity=0.9,
                                  name=f"{name} (T{tier_n})  AUPRC={ap:.3f}"))
    except Exception as e:
        print(f"  · {name} — skipped ({type(e).__name__}: {e})")

fig.update_layout(template="plotly_white", height=560,
                   title="Precision–Recall across all tiers",
                   xaxis_title="recall", yaxis_title="precision",
                   xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="v", x=1.02, y=1, yanchor="top",
                                xanchor="left", font=dict(size=10)))
plotly_show(fig)

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

fig = go.Figure(go.Bar(
    x=best_per_tier["tier_name"], y=best_per_tier["recall_at_5pct"],
    marker_color=[colors_by_tier[t] for t in best_per_tier["tier_number"]],
    text=[f"total {tot:.0%}<br>+{inc:.0%}"
           for tot, inc in zip(best_per_tier["recall_at_5pct"],
                                best_per_tier["incremental_recall"])],
    textposition="outside",
))
fig.update_layout(template="plotly_white", height=460,
                   title="Incremental recall lift up the maturity ladder",
                   xaxis_tickangle=-30,
                   yaxis_title="recall @ 5%",
                   yaxis=dict(tickformat=".0%",
                               range=[0, min(1.05, best_per_tier['recall_at_5pct'].max() * 1.30 + 0.1)]),
                   margin=dict(l=20, r=20, t=60, b=80), showlegend=False)
plotly_show(fig)

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
