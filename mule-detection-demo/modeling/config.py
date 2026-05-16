# Databricks notebook source
# MAGIC %md
# MAGIC # ⚙️ Mule Modeling — Demo Configuration
# MAGIC
# MAGIC **This is the only file a re-user needs to edit** to host this demo on a different
# MAGIC workspace, catalog, cluster, or data scale. Every other notebook reads from here via
# MAGIC `%run ./config` (transitively, through `%run ./_shared`).
# MAGIC
# MAGIC The variables fall into five groups:
# MAGIC
# MAGIC | Group | What to edit |
# MAGIC |---|---|
# MAGIC | **Unity Catalog** | `CATALOG`, `SCHEMA` |
# MAGIC | **MLflow** | `MLFLOW_EXPERIMENT` |
# MAGIC | **Infrastructure** | `CLUSTER_ID`, `WORKSPACE_FOLDER` |
# MAGIC | **Synthetic data scale** | `N_LEGIT`, `N_MULES`, `N_DAYS`, `RING_SIZE`, `SEED` |
# MAGIC | **Library coordinates** | `ISOLATION_FOREST_MAVEN_COORD`, `TORCH_DISTRIBUTOR_KWARGS` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unity Catalog destination

# COMMAND ----------

CATALOG = "vn"
SCHEMA  = "mule_demo"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. MLflow

# COMMAND ----------

MLFLOW_EXPERIMENT = "/Users/sergio.ballesteros@databricks.com/mule-modeling"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Infrastructure
# MAGIC
# MAGIC `CLUSTER_ID` is the GPU cluster the deploy driver submits notebook runs against.
# MAGIC `WORKSPACE_FOLDER` is the folder where the notebooks live in the Databricks workspace —
# MAGIC resolved in Phase 0 from folder ID `2734578869074633`.

# COMMAND ----------

CLUSTER_ID       = "0515-114053-9qo5ptwy"
WORKSPACE_FOLDER = "/Users/sergio.ballesteros@databricks.com/mule-detection-demo/modeling"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Synthetic data scale
# MAGIC
# MAGIC The research-appendix toy size (2 000 accounts) is too small for distributed PyTorch to
# MAGIC outperform single-node. The modeling notebooks bump to 200 000 accounts so
# MAGIC `TorchDistributor` has meaningful work — large enough that GPU + multi-process wins,
# MAGIC small enough to finish in minutes on the demo cluster.

# COMMAND ----------

N_LEGIT    = 200_000
N_MULES    = 8_000
N_DAYS     = 30
RING_SIZE  = 6
SEED       = 42

# Used by test gate 3 (determinism). Resolved at data-generation time
# from the first planted mule ring under SEED=42.
KNOWN_MULE_ID = 200_000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library coordinates
# MAGIC
# MAGIC The LinkedIn isolation-forest Maven coord must match the cluster's Spark + Scala
# MAGIC version. Check <https://mvnrepository.com/artifact/com.linkedin.isolation-forest/isolation-forest>
# MAGIC for the matching artifact when changing runtime.

# COMMAND ----------

# Cluster spark_version = 18.2.x-scala2.13 → Scala 2.13. Picking the closest
# LinkedIn IF build (Spark 3.5.5 / Scala 2.13, latest 4.0.1). If your cluster is
# on a different Scala/Spark, look up the matching artifact at the link above.
ISOLATION_FOREST_MAVEN_COORD = "com.linkedin.isolation-forest:isolation-forest_3.5.5_2.13:4.0.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### TorchDistributor kwargs
# MAGIC
# MAGIC Resolved in Phase 0 from the cluster spec:
# MAGIC - **Single-node or 1 GPU total** → `local_mode=True, num_processes=<gpu_count>`
# MAGIC - **Multi-worker, >1 GPU total** → `local_mode=False, num_processes=<total_gpu_count>`

# COMMAND ----------

# Cluster has 1 driver + 1 worker × 1 T4 GPU each. Using local_mode=True so the
# train function runs in the same Python process as the notebook driver — avoids
# Spark barrier-task scheduling and DBFS-handoff complexity, and gets us the
# driver's GPU. For genuinely multi-GPU clusters, switch to local_mode=False and
# set num_processes = total worker-GPU count.
TORCH_DISTRIBUTOR_KWARGS = {
    "local_mode":    True,
    "num_processes": 1,
    "use_gpu":       True,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Derived helpers
# MAGIC
# MAGIC Convenience constants downstream notebooks rely on. Do not edit these — change the
# MAGIC base variables above and these will follow.

# COMMAND ----------

ACCOUNTS_TABLE   = f"{CATALOG}.{SCHEMA}.accounts"
TXNS_TABLE       = f"{CATALOG}.{SCHEMA}.transactions"
BENCHMARK_TABLE  = f"{CATALOG}.{SCHEMA}.tier_benchmark"

def scores_table(tier_name: str) -> str:
    """Per-account score table for a given tier (e.g. '01_rules')."""
    return f"{CATALOG}.{SCHEMA}.scores_{tier_name}"

def registered_model(tier_name: str) -> str:
    """UC model registry name for a given tier."""
    return f"{CATALOG}.{SCHEMA}.mule_{tier_name}"

print(f"Config loaded: {CATALOG}.{SCHEMA} | cluster={CLUSTER_ID} | n_legit={N_LEGIT:,}")
