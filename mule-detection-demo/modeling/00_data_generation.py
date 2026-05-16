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

import matplotlib.pyplot as plt

# Volume by day --------------------------------------------------------------
vol_by_day = (txns.withColumn("day_bucket", F.floor("day").cast("int"))
                  .groupBy("day_bucket")
                  .agg(F.count("*").alias("n_txns"),
                       F.sum("amount").alias("total_amount"))
                  .orderBy("day_bucket")
                  .toPandas())

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(vol_by_day["day_bucket"], vol_by_day["n_txns"], color="#1f77b4")
ax[0].set_title("Transactions per day"); ax[0].set_xlabel("day"); ax[0].set_ylabel("count")
ax[1].bar(vol_by_day["day_bucket"], vol_by_day["total_amount"] / 1e6, color="#2ca02c")
ax[1].set_title("Transaction value per day (millions)"); ax[1].set_xlabel("day")
fig.tight_layout()

# COMMAND ----------

# Mule prevalence ------------------------------------------------------------
prev = accounts.groupBy("is_mule").count().toPandas()
fig, ax = plt.subplots(figsize=(5, 4))
colors = ["#7f7f7f" if not v else "#d62728" for v in prev["is_mule"]]
ax.bar(["legit", "mule"], prev["count"], color=colors)
for i, c in enumerate(prev["count"]):
    ax.text(i, c, f"{c:,}\n({c/n_accounts:.1%})", ha="center", va="bottom")
ax.set_title(f"Class balance — mules are {mule_rate:.1%} of accounts")
ax.set_ylabel("# accounts")
fig.tight_layout()

# COMMAND ----------

# Fan-in distribution by class ----------------------------------------------
fanin = (txns.groupBy("dst").agg(F.countDistinct("src").alias("in_distinct_src"))
              .join(accounts.select(F.col("account_id").alias("dst"), "is_mule"), "dst")
              .toPandas())

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(fanin.loc[~fanin["is_mule"], "in_distinct_src"],
        bins=50, alpha=0.7, label="legit", color="#7f7f7f", density=True)
ax.hist(fanin.loc[ fanin["is_mule"], "in_distinct_src"],
        bins=50, alpha=0.7, label="mule",  color="#d62728", density=True)
ax.set_xlabel("# distinct senders into this account (lifetime)")
ax.set_ylabel("density")
ax.set_title("Fan-in by class — mules collect from many more counterparties")
ax.set_xlim(0, fanin["in_distinct_src"].quantile(0.995))
ax.legend()
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Interactive demo visualisations 🎬
# MAGIC
# MAGIC The matplotlib charts above are utilitarian. The plotly visuals below are the
# MAGIC ones built for **walking a non-technical audience through what a money-mule attack
# MAGIC actually looks like**. All three are interactive — hover, zoom, rotate, scrub the
# MAGIC time slider.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 The accounts wake up — animated bubble plot
# MAGIC
# MAGIC Every dot is one account. The horizontal axis is *who they receive from* (distinct
# MAGIC inbound counterparties in the last 7 days), the vertical axis is *who they send to*
# MAGIC (distinct outbound counterparties), the size is total amount moved. The animation
# MAGIC scrubs through the 30-day period. Legitimate accounts barely move; **mule rings
# MAGIC explode toward the top-right corner the moment their attack window opens, pulling
# MAGIC a visible cluster of victim accounts with them**.
# MAGIC
# MAGIC Press ▶ to play. Drag the slider to jump to a day.

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go

# Per-(account, day) rolling-7d aggregates ---------------------------------
# Output-size constraints in Databricks notebooks cap HTML payloads at ~10 MB.
# To stay safely under that we limit (a) the number of animation frames, and
# (b) the number of accounts shown per frame.
WINDOW_DAYS  = 7
FRAME_STRIDE = 2       # ⟶ one frame every other day (≈ 12 frames over 30 days)
SAMPLE_LEGIT = 600     # show all mules + a small legit sample for context
SAMPLE_MULES = 2_000   # cap mule sample too — the band stays clearly visible

day_buckets = list(range(WINDOW_DAYS, N_DAYS + 1, FRAME_STRIDE))

inb = (txns.withColumn("d", F.floor("day").cast("int"))
            .groupBy("dst", "d")
            .agg(F.countDistinct("src").alias("in_distinct"),
                 F.sum("amount").alias("in_amt"))
            .withColumnRenamed("dst", "account_id"))
out = (txns.withColumn("d", F.floor("day").cast("int"))
            .groupBy("src", "d")
            .agg(F.countDistinct("dst").alias("out_distinct"),
                 F.sum("amount").alias("out_amt"))
            .withColumnRenamed("src", "account_id"))
per_day = (accounts.crossJoin(spark.range(N_DAYS).withColumnRenamed("id", "d"))
                    .join(inb, ["account_id", "d"], "left")
                    .join(out, ["account_id", "d"], "left")
                    .na.fill(0))

# Build the rolling-window data for each frame day
frames_sdf = None
for day in day_buckets:
    win = (per_day.where((F.col("d") >= day - WINDOW_DAYS) & (F.col("d") < day))
                   .groupBy("account_id", "is_mule")
                   .agg(F.sum("in_distinct").alias("in_distinct"),
                        F.sum("out_distinct").alias("out_distinct"),
                        F.sum("in_amt").alias("in_amt"),
                        F.sum("out_amt").alias("out_amt"))
                   .withColumn("day", F.lit(day))
                   .withColumn("total_amt", F.col("in_amt") + F.col("out_amt")))
    frames_sdf = win if frames_sdf is None else frames_sdf.unionByName(win)

frames_pd = frames_sdf.toPandas()

# Sample legit + mules so the chart stays compact (output-size limit)
rng_pick = np.random.default_rng(SEED)
legit_ids = frames_pd[~frames_pd["is_mule"]]["account_id"].unique()
mule_ids  = frames_pd[ frames_pd["is_mule"]]["account_id"].unique()
legit_keep = set(rng_pick.choice(legit_ids, size=min(SAMPLE_LEGIT, len(legit_ids)), replace=False).tolist())
mule_keep  = set(rng_pick.choice(mule_ids,  size=min(SAMPLE_MULES, len(mule_ids)),  replace=False).tolist())
frames_pd = frames_pd[frames_pd["account_id"].isin(legit_keep | mule_keep)]

# Filter out frames with zero activity (in_distinct + out_distinct == 0)
frames_pd = frames_pd[(frames_pd["in_distinct"] + frames_pd["out_distinct"]) > 0].copy()
frames_pd["class"] = np.where(frames_pd["is_mule"], "mule", "legit")
frames_pd["account_id"] = frames_pd["account_id"].astype(int)
print(f"animated bubble plot: {len(frames_pd):,} frame-rows over {len(day_buckets)} frames")

fig = px.scatter(
    frames_pd,
    x="in_distinct", y="out_distinct",
    size="total_amt", color="class",
    animation_frame="day", animation_group="account_id",
    color_discrete_map={"legit": "#9aa0a6", "mule": "#d62728"},
    hover_data={"account_id": True, "in_amt": ":.0f", "out_amt": ":.0f"},
    log_x=True, log_y=True,
    size_max=45,
    range_x=[0.7, max(80, frames_pd["in_distinct"].max() * 1.1)],
    range_y=[0.7, max(80, frames_pd["out_distinct"].max() * 1.1)],
    title=f"Money-mule activation over time   (7-day rolling window, day {WINDOW_DAYS}–{N_DAYS})",
    labels={"in_distinct": "distinct inbound counterparties (log)",
             "out_distinct": "distinct outbound counterparties (log)"},
    height=620,
)
fig.update_layout(template="plotly_white",
                    margin=dict(l=40, r=20, t=70, b=40))
# `full_html=False` outputs only the <div> + the embedded data, not the full
# <html><head>… wrapper — keeps Databricks notebook output under its size limit.
displayHTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Anatomy of a mule ring — interactive PyVis network
# MAGIC
# MAGIC We pick the **highest-inbound mule collector**, follow its inbound senders (scam
# MAGIC victims) and its outbound destinations (ring members), and render the whole
# MAGIC structure as a force-directed interactive network using PyVis.
# MAGIC
# MAGIC - 🟥 large red node = the collector
# MAGIC - 🟧 medium orange nodes = other ring members
# MAGIC - ⚪ small grey nodes = scam-victim senders
# MAGIC - edge thickness ∝ amount transferred
# MAGIC
# MAGIC Click and drag any node. Scroll to zoom. Hover for the account_id.

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
                    bgcolor="#ffffff", font_color="#222")
net.barnes_hut(spring_length=160, spring_strength=0.02, gravity=-3000)

def add_node(n):
    is_mule = bool(labels_pd.loc[n, "is_mule"])
    if n == COLLECTOR_ID:
        color, size, role = "#d62728", 32, "🟥 collector"
    elif is_mule:
        color, size, role = "#ff8c00", 20, "🟧 ring member"
    else:
        color, size, role = "#cccccc", 10, "⚪ scam-victim sender"
    net.add_node(n, label=str(n), title=f"account_id={n}<br>{role}",
                  color=color, size=size,
                  borderWidth=1.5, borderWidthSelected=3)

for n in node_ids:
    add_node(n)

for _, row in in_neighbors.iterrows():
    w = float(row["amount"])
    net.add_edge(int(row["src"]), COLLECTOR_ID,
                  value=w / max_amount * 10,
                  title=f"THB {w:,.0f}",
                  color="rgba(214,39,40,0.55)",
                  arrows="to")

for _, row in out_neighbors.iterrows():
    w = float(row["amount"])
    net.add_edge(COLLECTOR_ID, int(row["dst"]),
                  value=w / max_amount * 10,
                  title=f"THB {w:,.0f}",
                  color="rgba(255,140,0,0.7)",
                  arrows="to")

# Slight tweak so PyVis output isn't a full standalone HTML page
net.show_buttons(filter_=[])
html_str = net.generate_html(notebook=False)
displayHTML(f"<h4>Anatomy of mule ring — collector account #{COLLECTOR_ID}</h4>" + html_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 The mule band — calendar heatmap of activity
# MAGIC
# MAGIC One row per account, one column per day, cell colour = transaction count that day.
# MAGIC We show **all 8 000 mule accounts** plus a 2 000-account legit sample, with mules
# MAGIC sorted to the bottom. The legit population is a faint speckle; **the mules form a
# MAGIC sharp horizontal band of intense activity bursts** where the ring activations
# MAGIC happened. You can literally see when each ring lit up.
# MAGIC
# MAGIC Hover any cell for the exact count and date.

# COMMAND ----------

# Per-(account, day) transaction count
daily_count = (txns.withColumn("d", F.floor("day").cast("int"))
                     .select(F.col("src").alias("account_id"), "d")
                     .union(txns.withColumn("d", F.floor("day").cast("int"))
                                  .select(F.col("dst").alias("account_id"), "d"))
                     .groupBy("account_id", "d")
                     .agg(F.count("*").alias("tx_count")))

LEGIT_SAMPLE_HEATMAP = 2_000
rng_h = np.random.default_rng(SEED)
mule_ids  = accounts.where("is_mule = true").select("account_id").toPandas()["account_id"].values
legit_all = accounts.where("is_mule = false").select("account_id").toPandas()["account_id"].values
legit_sample = rng_h.choice(legit_all, size=LEGIT_SAMPLE_HEATMAP, replace=False)
shown_ids = np.concatenate([legit_sample, mule_ids])

dc_pd = (daily_count.where(F.col("account_id").isin(shown_ids.tolist()))
                      .toPandas())
# Pivot to 2-D matrix (rows: account_id sorted with mules at bottom, cols: day)
mat = (dc_pd.pivot_table(index="account_id", columns="d", values="tx_count",
                          fill_value=0)
              .reindex(columns=range(N_DAYS), fill_value=0))
# Sort by is_mule ascending (legit on top, mules at bottom = visually the "band")
sort_order = pd.Series(shown_ids).map(lambda a: a in set(mule_ids.tolist())).values
sort_order_df = pd.DataFrame({"account_id": shown_ids, "is_mule": sort_order})
ordering = sort_order_df.sort_values("is_mule")["account_id"].values
mat = mat.reindex(index=ordering, fill_value=0)

fig_hm = go.Figure(go.Heatmap(
    z=mat.values,
    x=[f"day {d}" for d in mat.columns],
    y=[f"acct {a}" for a in mat.index],
    colorscale=[[0, "rgba(255,255,255,0.0)"],
                  [0.001, "#eef2f7"],
                  [0.15, "#fbd2c2"],
                  [0.4, "#f08562"],
                  [1.0, "#7a0000"]],
    hoverongaps=False,
    colorbar=dict(title="tx / day"),
))
n_legit_rows = LEGIT_SAMPLE_HEATMAP
# Add a horizontal line where the mule band begins
fig_hm.add_shape(type="line",
                  x0=-0.5, x1=N_DAYS - 0.5,
                  y0=n_legit_rows - 0.5, y1=n_legit_rows - 0.5,
                  line=dict(color="#000000", width=1.5, dash="dot"))
fig_hm.add_annotation(x=N_DAYS - 1, y=n_legit_rows + 200,
                       text="↓ mule band ↓", showarrow=False,
                       font=dict(color="#d62728", size=14))
fig_hm.update_layout(
    title=f"Calendar heatmap — activity by account × day (top {LEGIT_SAMPLE_HEATMAP:,} legit, then all {len(mule_ids):,} mules)",
    template="plotly_white", height=720,
    xaxis=dict(showticklabels=True),
    yaxis=dict(showticklabels=False, title=f"accounts (sorted: legit ⟶ mule)"),
    margin=dict(l=20, r=20, t=70, b=40),
)
displayHTML(fig_hm.to_html(include_plotlyjs="cdn", full_html=False))

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
