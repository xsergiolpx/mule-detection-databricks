# Databricks notebook source

# MAGIC %md
# MAGIC # 🤖 Create Genie Space for Mule Detection
# MAGIC # สร้าง Genie Space สำหรับตรวจจับบัญชีม้า

# COMMAND ----------

import requests, json

WAREHOUSE_ID = "5b8213d5c7a1ef85"
SPACE_TITLE = "Mule Detection Investigator"

TABLES = [
    "vn.mule_demo.gold_account_risk_graph",
    "vn.mule_demo.gold_transaction_edges",
    "vn.mule_demo.silver_transactions",
    "vn.mule_demo.silver_bot_mule_list",
]

DESCRIPTION = """Fraud investigation assistant for Thai banks. Analyze mule account networks detected by graph analytics.

Key concepts:
- BOT = Bank of Thailand. bot_confirmed_mule = TRUE means officially confirmed mule.
- risk_score (0-1): composite risk. Higher = more suspicious.
- pagerank_score: centrality in money routing. High = likely hub/operator.
- two_hop_ratio (0-1): fraction of incoming money leaving within 24h. >0.8 = mule.
- behavior_profile: steady, pass_through, dormant_then_active, burst.
- community_id: sub-group within the network.
- Amounts in Thai Baht (THB).

When asked about this network or this cluster, ask which community_id unless context makes it obvious."""

SAMPLE_QUESTIONS = [
    "How many BOT confirmed mules are there?",
    "List accounts with risk score above 0.5 that are NOT on the BOT confirmed list ordered by PageRank",
    "What is the total value of transactions for accounts with risk score above 0.5?",
    "Show me the top 5 accounts by PageRank that are not on the BOT list",
    "How many unique accounts sent money to BOT confirmed mules?",
    "What is the average two-hop ratio for BOT confirmed mules versus normal accounts?",
    "What is the average account age for BOT confirmed mules versus all other accounts?",
]

# Get auth context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
token = ctx.apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print(f"Host: {host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Check if space already exists

# COMMAND ----------

r = requests.get(f"{host}/api/2.0/genie/spaces", headers=headers)
r.raise_for_status()
spaces = r.json().get("spaces", [])
existing = [s for s in spaces if s.get("title") == SPACE_TITLE]

if existing:
    space_id = existing[0]["space_id"]
    print(f"Space already exists: {space_id}")
    # Delete and recreate to get a clean state
    r = requests.delete(f"{host}/api/2.0/genie/spaces/{space_id}", headers=headers)
    print(f"Deleted old space: {r.status_code}")
    import time; time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the Genie Space

# COMMAND ----------

# Create space
payload = {
    "warehouse_id": WAREHOUSE_ID,
    "serialized_space": json.dumps({"version": 2, "dataSources": {}}),
    "title": SPACE_TITLE,
    "description": DESCRIPTION,
}
r = requests.post(f"{host}/api/2.0/genie/spaces", json=payload, headers=headers)
r.raise_for_status()
space_data = r.json()
space_id = space_data["space_id"]
print(f"Created space: {space_id}")
print(f"URL: {host}/genie/rooms/{space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Attach tables

# COMMAND ----------

# Method 1: Try PUT with table_identifiers
payload = {
    "title": SPACE_TITLE,
    "description": DESCRIPTION,
    "warehouse_id": WAREHOUSE_ID,
    "table_identifiers": TABLES,
}
r = requests.put(f"{host}/api/2.0/genie/spaces/{space_id}", json=payload, headers=headers)
print(f"PUT with table_identifiers: {r.status_code}")

if r.status_code == 200:
    resp = r.json()
    print(f"Response keys: {list(resp.keys())}")
    print(f"Serialized space: {resp.get('serialized_space', 'N/A')[:300]}")
else:
    print(f"Error: {r.text[:300]}")

# COMMAND ----------

# Verify tables by asking a question
import time
time.sleep(3)

r = requests.post(
    f"{host}/api/2.0/genie/spaces/{space_id}/start-conversation",
    json={"content": "How many BOT confirmed mules are there?"},
    headers=headers
)

if r.status_code == 200:
    conv = r.json()
    conv_id = conv.get("conversation_id")
    msg_id = conv.get("message_id")
    print(f"Conversation started: {conv_id}, message: {msg_id}")

    # Poll for result
    for i in range(20):
        time.sleep(3)
        r2 = requests.get(
            f"{host}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}",
            headers=headers
        )
        if r2.status_code == 200:
            msg = r2.json()
            status = msg.get("status", "UNKNOWN")
            print(f"  [{i}] Status: {status}")
            if status in ("COMPLETED", "FAILED"):
                attachments = msg.get("attachments", [])
                for att in attachments:
                    if "text" in att:
                        print(f"  Text: {att['text'].get('content', '')[:300]}")
                    if "query" in att:
                        print(f"  SQL: {att['query'].get('query', '')[:300]}")
                break
        else:
            print(f"  [{i}] Poll error: {r2.status_code}")
else:
    print(f"Conversation error: {r.status_code} - {r.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("GENIE SPACE CONFIGURATION")
print("=" * 60)
print(f"  Space ID:    {space_id}")
print(f"  Title:       {SPACE_TITLE}")
print(f"  Warehouse:   {WAREHOUSE_ID}")
print(f"  Tables:      {len(TABLES)}")
for t in TABLES:
    print(f"               - {t}")
print(f"  URL:         {host}/genie/rooms/{space_id}")
print()
print("If the test question returned 'no tables', open the URL and add tables manually.")
