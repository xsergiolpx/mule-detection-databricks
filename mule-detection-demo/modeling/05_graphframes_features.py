# Databricks notebook source
# MAGIC %md
# MAGIC # 🕸️ 05 — Tier 4: GraphFrames features + SparkXGBClassifier
# MAGIC
# MAGIC The cheapest way to move from Tier 3 to Tier 4: the model architecture stays the
# MAGIC same (`SparkXGBClassifier`), but the **features now include structural signals
# MAGIC computed from the transaction graph**: PageRank, in/out degree, two-hop reach,
# MAGIC community-propagated label.
# MAGIC
# MAGIC ### Why graph features matter
# MAGIC
# MAGIC Two accounts can look identical at the row level — same KYC, same monthly volume,
# MAGIC same device — yet one is the hub of a 50-account mule ring and the other is a
# MAGIC legitimate small business. The difference lives in *who they transact with*.
# MAGIC Tabular models can't see it; graph features can.
# MAGIC
# MAGIC Published lift over the same XGBoost on raw features: **+46% F1** (IBM AMLworld
# MAGIC benchmark, ACM 2024).

# COMMAND ----------

# MAGIC %pip install pyvis --quiet

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
from pyspark.ml.functions import vector_to_array
from xgboost.spark import SparkXGBClassifier
from graphframes import GraphFrame

from sklearn.metrics import precision_recall_curve, average_precision_score

mlflow.set_experiment(MLFLOW_EXPERIMENT)
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")
spark.sparkContext.setCheckpointDir("/tmp/graphframes_modeling_ckpt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data → GraphFrame

# COMMAND ----------

accounts = spark.table(ACCOUNTS_TABLE)
txns     = spark.table(TXNS_TABLE)

vertices = accounts.withColumnRenamed("account_id", "id")
edges    = txns.select(F.col("src").alias("src"),
                        F.col("dst").alias("dst"),
                        F.col("amount").alias("amount"))

g = GraphFrame(vertices, edges)
print(f"{vertices.count():,} vertices    {edges.count():,} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compute structural features
# MAGIC
# MAGIC Five vertex-level features, each computed as a distributed Spark job:
# MAGIC
# MAGIC | Feature | Captures |
# MAGIC |---|---|
# MAGIC | `pagerank` | influence in the money-flow graph |
# MAGIC | `in_degree` / `out_degree` | how broad each account's counterparty set is |
# MAGIC | `two_hop_reach` | distinct accounts reachable in 2 forward hops |
# MAGIC | `community_size` | size of the label-propagation community the account joined |

# COMMAND ----------

t0 = time.perf_counter()

# PageRank --------------------------------------------------------------------
pr = (g.pageRank(resetProbability=0.15, maxIter=10)
       .vertices.select("id", F.col("pagerank").alias("pagerank")))

# In / out degree -------------------------------------------------------------
in_deg  = g.inDegrees    # id, inDegree
out_deg = g.outDegrees   # id, outDegree

# Two-hop reach via self-join (much faster than motif find() at this scale) ---
two_hop = (edges.alias("e1")
    .join(edges.alias("e2"), F.col("e1.dst") == F.col("e2.src"))
    .where(F.col("e1.src") != F.col("e2.dst"))
    .groupBy(F.col("e1.src").alias("id"))
    .agg(F.countDistinct("e2.dst").alias("two_hop_reach")))

# Label propagation → community size -----------------------------------------
labels   = g.labelPropagation(maxIter=5)
sizes    = (labels.groupBy("label").agg(F.count("*").alias("community_size")))
comm     = (labels.select("id", "label")
                   .join(sizes, "label")
                   .select("id", "community_size"))

graph_features = (vertices
    .join(pr,      "id", "left")
    .join(in_deg,  "id", "left")
    .join(out_deg, "id", "left")
    .join(two_hop, "id", "left")
    .join(comm,    "id", "left")
    .na.fill(0.0)
).cache()

n_rows = graph_features.count()
runtime_graph = time.perf_counter() - t0
print(f"✓ Graph features ready: {n_rows:,} rows in {runtime_graph:.1f}s")
display(graph_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train SparkXGBClassifier on graph features

# COMMAND ----------

FEATURE_COLS = ["pagerank", "inDegree", "outDegree", "two_hop_reach", "community_size"]
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
ml_df     = (assembler.transform(graph_features)
                       .select("id", "features",
                               F.col("is_mule").cast("int").alias("label")))

train_df, test_df = ml_df.randomSplit([0.7, 0.3], seed=SEED)

clf = SparkXGBClassifier(
    features_col="features", label_col="label",
    num_workers=1,    # match Spark task-slot count on this cluster
    n_estimators=300, max_depth=4, learning_rate=0.1,
    tree_method="hist", eval_metric="aucpr",
)

t0 = time.perf_counter()
model = clf.fit(train_df)
runtime_train = time.perf_counter() - t0

predictions = model.transform(test_df)
auprc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
).evaluate(predictions)
print(f"Graph-feature SparkXGBoost AUPRC = {auprc:.3f}  (train {runtime_train:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate at multiple operating points

# COMMAND ----------

scored_pd = (predictions
    .withColumn("p1", vector_to_array("probability")[1])
    .select("id", "label", "p1")
    .toPandas())

y     = scored_pd["label"].astype(int).values
proba = scored_pd["p1"].values

p1, _    = precision_recall_at_k(y, proba, 0.01)
p5, r5   = precision_recall_at_k(y, proba, 0.05)
print(f"P@1%={p1:.1%}    P@5%={p5:.1%}    R@5%={r5:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualisations
# MAGIC
# MAGIC 1. **PageRank distribution by class** — mules should sit in the tail.
# MAGIC 2. **Feature importance**.
# MAGIC 3. **PR curve**.
# MAGIC 4. **Interactive PyVis ring viz** — a sampled mule ring + neighbours.

# COMMAND ----------

# 1. PageRank distribution ---------------------------------------------------
pr_pd = graph_features.select("pagerank", "is_mule").toPandas()
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(pr_pd.loc[~pr_pd["is_mule"], "pagerank"], bins=80, alpha=0.6,
        color="#7f7f7f", label="legit", density=True)
ax.hist(pr_pd.loc[ pr_pd["is_mule"], "pagerank"], bins=80, alpha=0.7,
        color="#d62728", label="mule",  density=True)
ax.set_xlim(0, pr_pd["pagerank"].quantile(0.999))
ax.set_xlabel("PageRank"); ax.set_ylabel("density")
ax.set_title("PageRank by class — mule collectors accumulate inbound mass")
ax.legend()
fig.tight_layout()

# COMMAND ----------

# 2. Feature importance ------------------------------------------------------
importances = model.get_booster().get_score(importance_type="gain")
imp_pd = (pd.DataFrame({"feature": list(importances.keys()),
                         "gain":    list(importances.values())})
            .sort_values("gain", ascending=True))
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(imp_pd["feature"], imp_pd["gain"], color="#9467bd")
ax.set_xlabel("xgboost gain")
ax.set_title("Graph feature importance")
fig.tight_layout()

# COMMAND ----------

# 3. PR curve ----------------------------------------------------------------
p, r, _ = precision_recall_curve(y, proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r, p, color="#9467bd", label=f"graph + SparkXGB  AUPRC={auprc:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–Recall — Tier 4 (graph features)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# 4. Interactive PyVis network viz of one mule ring + neighbours -------------
# Pick a confirmed mule with high PageRank, expand 2 hops out.
seed_mule = (graph_features
    .where("is_mule = true")
    .orderBy(F.col("pagerank").desc())
    .limit(1).first())

one_hop  = edges.where((F.col("src") == seed_mule.id) | (F.col("dst") == seed_mule.id))
adj_ids  = (one_hop.select(F.col("src").alias("node_id"))
                    .union(one_hop.select(F.col("dst").alias("node_id")))
                    .distinct())
# Edges that *leave* any of the 1-hop adjacency nodes (cap for viz clarity).
two_hop_edges = (edges.alias("e")
    .join(adj_ids.alias("a"), F.col("e.src") == F.col("a.node_id"), "inner")
    .select("e.src", "e.dst", "e.amount")
    .limit(120))

sub_nodes_df = (two_hop_edges.select(F.col("src").alias("id"))
    .union(two_hop_edges.select(F.col("dst").alias("id")))
    .distinct()
    .join(vertices, "id")
    .toPandas())
sub_edges_df = two_hop_edges.toPandas()

from pyvis.network import Network
net = Network(height="600px", width="100%", directed=True, notebook=False,
              bgcolor="#ffffff", font_color="#222")
net.barnes_hut(spring_length=120)
for _, row in sub_nodes_df.iterrows():
    is_seed = (row["id"] == seed_mule.id)
    color   = "#d62728" if row["is_mule"] else "#7f7f7f"
    if is_seed: color = "#ff8c00"
    net.add_node(int(row["id"]),
                 label=str(row["id"]),
                 color=color,
                 size=25 if is_seed else (15 if row["is_mule"] else 8))
for _, row in sub_edges_df.iterrows():
    net.add_edge(int(row["src"]), int(row["dst"]),
                 value=float(row["amount"]) / 10000.0)

html = net.generate_html(notebook=False)
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist scores + MLflow + benchmark

# COMMAND ----------

(spark.createDataFrame(scored_pd.rename(columns={"id": "account_id",
                                                   "label": "is_mule",
                                                   "p1": "score"}))
       .write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(scores_table("05_graphframes")))

with mlflow.start_run(run_name="05_graphframes_features") as run:
    mlflow.log_params({"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1,
                        "features": FEATURE_COLS})
    mlflow.log_metric("pr_auc",            auprc)
    mlflow.log_metric("precision_at_1pct", p1)
    mlflow.log_metric("precision_at_5pct", p5)
    mlflow.log_metric("recall_at_5pct",    r5)
    mlflow.log_metric("runtime_seconds",   runtime_graph + runtime_train)
    run_id = run.info.run_id

log_tier_metrics("05_graphframes", 4,
                 auprc, p1, p5, r5, runtime_graph + runtime_train, run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test gates

# COMMAND ----------

run_gate("smoke",  len(proba) > 0,    "no graph-feature predictions")
run_gate("pr_auc", auprc > 0.60,      f"graph-feature AUPRC {auprc:.3f} below 0.60 floor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC | Tier catches | Tier misses | Escalation trigger |
# MAGIC |---|---|---|
# MAGIC | Ring-level structural signal via pre-computed graph features | Learned interactions between neighbouring node embeddings | Want the model itself to learn from neighbour embeddings → Tier 4 GNN (GraphSAGE) |

# COMMAND ----------

try:
    dbutils.notebook.exit("OK")  # noqa: F821
except NameError:
    pass
