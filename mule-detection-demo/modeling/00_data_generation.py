# Databricks notebook source
# MAGIC %md
# MAGIC # 🌱 00 — Synthetic data generation
# MAGIC
# MAGIC Generates a population of legitimate accounts plus planted mule rings and persists
# MAGIC them as Delta tables. Every modeling notebook downstream reads from these two tables,
# MAGIC so the entire demo runs against **the same ground truth** and tier-by-tier metrics
# MAGIC are directly comparable.
# MAGIC
# MAGIC ### 📒 Data dictionary
# MAGIC
# MAGIC The notebook writes **two Delta tables** in Unity Catalog. Every modeling notebook
# MAGIC downstream reads from exactly these two — there is no other source of truth.
# MAGIC
# MAGIC #### `{CATALOG}.{SCHEMA}.accounts`
# MAGIC
# MAGIC One row per bank customer.
# MAGIC
# MAGIC | Column | Type | Range / meaning |
# MAGIC |---|---|---|
# MAGIC | `account_id` | `BIGINT` | `0 … N_LEGIT + N_MULES - 1`. Identifiers `0 … N_LEGIT-1` are legitimate accounts; `N_LEGIT … N_LEGIT+N_MULES-1` are planted mule accounts. |
# MAGIC | `is_mule` | `BOOLEAN` | Ground-truth label. `true` for accounts that participate in a mule ring; `false` otherwise. **This is the column every model is trying to predict.** |
# MAGIC
# MAGIC #### `{CATALOG}.{SCHEMA}.transactions`
# MAGIC
# MAGIC One row per money movement. The edges of the money-flow graph.
# MAGIC
# MAGIC | Column | Type | Range / meaning |
# MAGIC |---|---|---|
# MAGIC | `src` | `BIGINT` | Sending account_id. Foreign key into `accounts`. |
# MAGIC | `dst` | `BIGINT` | Receiving account_id. Foreign key into `accounts`. |
# MAGIC | `amount` | `DOUBLE` | Transaction amount in THB. Log-normally distributed; legit accounts ~exp(4±1), inbound scam-victim transfers ~exp(5±0.5). |
# MAGIC | `day` | `DOUBLE` | When the transaction happened, expressed as a float in `[0, N_DAYS)`. The integer part is the calendar day; the fractional part is time-of-day. |
# MAGIC
# MAGIC #### How the ground truth is constructed
# MAGIC
# MAGIC **Legitimate accounts** transact with a Poisson rate of ~3 transfers per 10 days,
# MAGIC each going to a uniformly random counterparty. Low volume, varied counterparties —
# MAGIC the boring everyday economy.
# MAGIC
# MAGIC **Mule rings** are planted with the two-step typology the rest of the demo is built to detect:
# MAGIC
# MAGIC 1. Each ring has one *collector* and ~5 *members*.
# MAGIC 2. The collector receives **20–60 small inbound transfers** from random legitimate
# MAGIC    accounts — the **scam-victim pattern**.
# MAGIC 3. Within hours, the collector forwards ~95% of the consolidated funds to the
# MAGIC    other ring members — the **pass-through pattern**.
# MAGIC
# MAGIC This is the FATF / BOT HR-03 mule signature in synthetic form. It is **deliberately
# MAGIC strong** so we can show what each maturity tier catches that the previous one
# MAGIC misses — not benchmark-grade, just pedagogical.

# COMMAND ----------

# MAGIC %pip install plotly pyvis --quiet

# COMMAND ----------

# MAGIC %run ./_shared

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Force a clean rebuild?
# MAGIC
# MAGIC Set the widget to `true` to drop and regenerate. Default is `false` so the table
# MAGIC is only built once per environment.

# COMMAND ----------

dbutils.widgets.dropdown("force_rebuild", "false", ["false", "true"])
FORCE_REBUILD = dbutils.widgets.get("force_rebuild") == "true"
print(f"force_rebuild = {FORCE_REBUILD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Schema bootstrap

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA  {SCHEMA}")
print(f"✓ Using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate or skip

# COMMAND ----------

tables_exist = (
    spark.catalog.tableExists(ACCOUNTS_TABLE)
    and spark.catalog.tableExists(TXNS_TABLE)
)

if tables_exist and not FORCE_REBUILD:
    print("✓ Tables already exist. Set force_rebuild=true to regenerate.")
else:
    print(f"Generating {N_LEGIT:,} legit + {N_MULES:,} mule accounts over {N_DAYS} days …")
    with timed("synthetic generation"):
        accounts_pd, txns_pd = make_synthetic_mule_data()

    accounts_sdf = (spark.createDataFrame(accounts_pd)
                         .withColumn("account_id", F.col("account_id").cast("long"))
                         .withColumn("is_mule",    F.col("is_mule").cast("boolean")))

    txns_sdf = (spark.createDataFrame(txns_pd)
                     .withColumn("src",    F.col("src").cast("long"))
                     .withColumn("dst",    F.col("dst").cast("long"))
                     .withColumn("amount", F.col("amount").cast("double"))
                     .withColumn("day",    F.col("day").cast("double")))

    (accounts_sdf.write.mode("overwrite")
                 .option("overwriteSchema", "true")
                 .saveAsTable(ACCOUNTS_TABLE))
    (txns_sdf.write.mode("overwrite")
             .option("overwriteSchema", "true")
             .saveAsTable(TXNS_TABLE))
    print(f"✓ Wrote {ACCOUNTS_TABLE} and {TXNS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sanity check

# COMMAND ----------

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

n_accounts = accounts.count()
n_txns     = txns.count()
n_mules    = accounts.where("is_mule").count()
mule_rate  = n_mules / n_accounts

print(f"accounts = {n_accounts:,}")
print(f"  mules  = {n_mules:,}  ({mule_rate:.2%})")
print(f"txns     = {n_txns:,}")

run_gate("smoke_accounts",  n_accounts >= N_LEGIT + N_MULES,
         f"expected >= {N_LEGIT + N_MULES:,} accounts, got {n_accounts:,}")
run_gate("smoke_txns",      n_txns > 0,           "no transactions written")
run_gate("smoke_mules",     n_mules == N_MULES,   f"expected {N_MULES} mules, got {n_mules}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. EDA — what the data looks like
# MAGIC
# MAGIC Three visuals that anchor every tier downstream:
# MAGIC
# MAGIC 1. **Transaction volume by day** — confirms the data spans `N_DAYS` and has no gaps.
# MAGIC 2. **Mule prevalence** — class imbalance is the headline characteristic of mule detection.
# MAGIC 3. **Fan-in distribution by class** — the first visual evidence that mules look different.

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Volume by day --------------------------------------------------------------
vol_by_day = (txns.withColumn("day_bucket", F.floor("day").cast("int"))
                  .groupBy("day_bucket")
                  .agg(F.count("*").alias("n_txns"),
                       F.sum("amount").alias("total_amount"))
                  .orderBy("day_bucket")
                  .toPandas())

fig = make_subplots(rows=1, cols=2,
                     subplot_titles=("Transactions per day", "Transaction value per day (millions)"))
fig.add_trace(go.Bar(x=vol_by_day["day_bucket"], y=vol_by_day["n_txns"],
                     marker_color="#1f77b4", name="count"), row=1, col=1)
fig.add_trace(go.Bar(x=vol_by_day["day_bucket"], y=vol_by_day["total_amount"] / 1e6,
                     marker_color="#2ca02c", name="THB millions"), row=1, col=2)
fig.update_layout(template="plotly_white", height=400, showlegend=False,
                   margin=dict(l=30, r=20, t=60, b=40))
fig.update_xaxes(title_text="day", row=1, col=1)
fig.update_xaxes(title_text="day", row=1, col=2)
plotly_show(fig)

# COMMAND ----------

# Mule prevalence — donut ----------------------------------------------------
prev = accounts.groupBy("is_mule").count().toPandas()
prev["label"] = np.where(prev["is_mule"], "mule", "legit")
fig = px.pie(prev, names="label", values="count", hole=0.55,
              color="label",
              color_discrete_map={"legit": "#7f7f7f", "mule": "#d62728"},
              title=f"Class balance — mules are {mule_rate:.2%} of accounts")
fig.update_traces(textinfo="label+percent+value", textposition="outside")
fig.update_layout(template="plotly_white", height=440,
                   margin=dict(l=20, r=20, t=60, b=20))
plotly_show(fig)

# COMMAND ----------

# Fan-in distribution by class ----------------------------------------------
fanin = (txns.groupBy("dst").agg(F.countDistinct("src").alias("in_distinct_src"))
              .join(accounts.select(F.col("account_id").alias("dst"), "is_mule"), "dst")
              .toPandas())
fanin["class"] = np.where(fanin["is_mule"], "mule", "legit")
upper = fanin["in_distinct_src"].quantile(0.995)

fig = px.histogram(
    fanin, x="in_distinct_src", color="class",
    color_discrete_map={"legit": "#9aa0a6", "mule": "#d62728"},
    barmode="overlay", opacity=0.65, histnorm="probability density", nbins=60,
    range_x=[0, upper], height=440,
    title="Fan-in by class — mules collect from many more counterparties",
    labels={"in_distinct_src": "# distinct senders into this account (lifetime)"},
)
fig.update_layout(template="plotly_white",
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Interactive demo visualisations 🎬
# MAGIC
# MAGIC The matplotlib charts above are utilitarian. The plotly visuals below are the
# MAGIC ones built for **walking a non-technical audience through what a money-mule attack
# MAGIC actually looks like**. Hover, zoom, click and drag.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Anatomy of a mule ring — interactive PyVis network
# MAGIC
# MAGIC This is **one real mule ring picked from the data we just generated**. We look up
# MAGIC the mule collector with the highest number of inbound senders, then expand its
# MAGIC neighbourhood one hop out in both directions.
# MAGIC
# MAGIC **How to read the graph**
# MAGIC
# MAGIC | What you see | What it means |
# MAGIC |---|---|
# MAGIC | 🔴 large red node | the **collector** — the mule receiving scam transfers from many victims |
# MAGIC | 🟠 orange nodes | other **mule-ring members** the collector forwards funds to |
# MAGIC | 🟢 green nodes | **scam-victim accounts** sending money in (legit accounts that were defrauded) |
# MAGIC | arrows | direction of money flow |
# MAGIC | edge thickness | proportional to the amount transferred |
# MAGIC
# MAGIC The whole laundering pattern is visible in one picture: a fan of green victims
# MAGIC pointing into the red collector, then a few thick orange arrows leaving toward
# MAGIC the ring members.
# MAGIC
# MAGIC **Click & drag** any node, **scroll** to zoom, **hover** for the account_id and amount.

# COMMAND ----------

from pyvis.network import Network as PyVisNetwork

# Pick the collector of the first planted ring (highest-inbound mule)
collector_row = (txns.groupBy(F.col("dst").alias("account_id"))
                       .agg(F.countDistinct("src").alias("in_distinct"))
                       .join(accounts, "account_id")
                       .where("is_mule = true")
                       .orderBy(F.col("in_distinct").desc())
                       .limit(1)
                       .first())
COLLECTOR_ID = int(collector_row["account_id"])

# 1-hop in (senders) and 1-hop out (recipients) of this collector
in_neighbors  = (txns.where(F.col("dst") == COLLECTOR_ID)
                       .groupBy("src").agg(F.sum("amount").alias("amount"))
                       .toPandas())
out_neighbors = (txns.where(F.col("src") == COLLECTOR_ID)
                       .groupBy("dst").agg(F.sum("amount").alias("amount"))
                       .toPandas())

# Look up actual mule labels for each node touched
node_ids = (set([COLLECTOR_ID])
            | set(int(x) for x in in_neighbors["src"])
            | set(int(x) for x in out_neighbors["dst"]))
labels_pd = (accounts.where(F.col("account_id").isin(list(node_ids)))
                       .toPandas()
                       .set_index("account_id"))

max_amount = max(
    float(in_neighbors["amount"].max() if len(in_neighbors) else 1.0),
    float(out_neighbors["amount"].max() if len(out_neighbors) else 1.0),
    1.0,
)

# Build PyVis network ------------------------------------------------------
net = PyVisNetwork(height="640px", width="100%", directed=True, notebook=False,
                    bgcolor="#0f172a", font_color="white")

def add_node(n):
    is_mule = bool(labels_pd.loc[n, "is_mule"])
    if n == COLLECTOR_ID:
        color, size, emoji, role = "#ef4444", 42, "🔴", "Mule collector"
    elif is_mule:
        color, size, emoji, role = "#f97316", 30, "🟠", "Mule ring member"
    else:
        color, size, emoji, role = "#22c55e", 22, "🟢", "Scam-victim sender"
    net.add_node(n,
                  label=f"{emoji} {n}",
                  title=f"<b>account_id={n}</b><br>Role: {role}",
                  color=color, size=size,
                  borderWidth=2, borderWidthSelected=4,
                  font={"size": 18, "color": "white", "face": "sans-serif"})

for n in node_ids:
    add_node(n)

for _, row in in_neighbors.iterrows():
    w = float(row["amount"])
    net.add_edge(int(row["src"]), COLLECTOR_ID,
                  value=w / max_amount * 12,
                  title=f"THB {w:,.0f}",
                  color="rgba(239,68,68,0.55)",
                  arrows="to")

for _, row in out_neighbors.iterrows():
    w = float(row["amount"])
    net.add_edge(COLLECTOR_ID, int(row["dst"]),
                  value=w / max_amount * 12,
                  title=f"THB {w:,.0f}",
                  color="rgba(249,115,22,0.8)",
                  arrows="to")

# Tuned physics, hidden config panel.
net.set_options("""
{
  "physics": {
    "barnesHut": {"gravitationalConstant": -5000, "springLength": 220, "springConstant": 0.02},
    "stabilization": {"iterations": 120}
  },
  "edges": {"smooth": {"type": "continuous"},
             "arrows": {"to": {"enabled": true, "scaleFactor": 0.9}},
             "width": 2},
  "interaction": {"hover": true, "tooltipDelay": 100}
}
""")

html_str = net.generate_html(notebook=False)
header_html = f"""
<div style='font-family:sans-serif; padding:12px 16px;
            background:#0f172a; color:white; border-radius:8px 8px 0 0;'>
  <h3 style='margin:0'>Anatomy of mule ring — collector account #{COLLECTOR_ID}</h3>
  <small style='color:#94a3b8'>🔴 collector · 🟠 ring member · 🟢 scam-victim sender</small>
</div>
"""
displayHTML(header_html + html_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 When do the mules attack? — daily transaction volume by class
# MAGIC
# MAGIC A simple picture: how many transactions happen each day, split into the legit
# MAGIC population and the mule population.
# MAGIC
# MAGIC - The **grey legit line** is a flat baseline — everyday banking, evenly spread.
# MAGIC - The **red mule line** shows visible spikes on the days the rings activate.
# MAGIC
# MAGIC Mule activity is bursty by design: many small inbound transfers from victims
# MAGIC followed by an immediate pass-through to ring members. The spikes are exactly
# MAGIC those bursts.

# COMMAND ----------

# Tag each transaction with the *destination* class (where the money lands)
# then count transactions per (day, class).
class_per_day = (
    txns.withColumn("d", F.floor("day").cast("int"))
        .join(accounts.select(F.col("account_id").alias("dst"), "is_mule"), "dst")
        .groupBy("d", "is_mule")
        .agg(F.count("*").alias("tx_count"))
        .toPandas()
)
class_per_day["class"] = np.where(class_per_day["is_mule"], "mule", "legit")
class_per_day = class_per_day.sort_values(["class", "d"])

fig = px.line(
    class_per_day, x="d", y="tx_count", color="class",
    color_discrete_map={"legit": "#9aa0a6", "mule": "#d62728"},
    markers=True, height=440,
    title="Daily transactions hitting legit accounts vs mule accounts",
    labels={"d": "day", "tx_count": "transactions arriving that day"},
)
fig.update_traces(line=dict(width=3), marker=dict(size=8))
fig.update_layout(template="plotly_white",
                   margin=dict(l=20, r=20, t=60, b=40),
                   legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
plotly_show(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Done
# MAGIC
# MAGIC `accounts` and `transactions` are now in Unity Catalog. Every modeling notebook
# MAGIC reads from these two tables — no notebook regenerates its own copy.

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
