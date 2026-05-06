# Databricks notebook source
# MAGIC %md
# MAGIC # 🕸️ Mule Network Intelligence — Graph Feature Engineering
# MAGIC # การวิเคราะห์เครือข่ายบัญชีม้า — วิศวกรรมคุณสมบัติกราฟ
# MAGIC
# MAGIC This notebook builds a **graph of bank accounts** and computes network intelligence
# MAGIC features to detect mule networks. Starting from raw transaction and device data,
# MAGIC we apply graph algorithms to uncover hidden relationships.
# MAGIC
# MAGIC โน้ตบุ๊คนี้สร้าง **กราฟของบัญชีธนาคาร** และคำนวณคุณสมบัติเครือข่าย
# MAGIC เพื่อตรวจจับเครือข่ายบัญชีม้า โดยเริ่มจากข้อมูลธุรกรรมและอุปกรณ์ดิบ
# MAGIC แล้วใช้อัลกอริทึมกราฟเพื่อค้นหาความสัมพันธ์ที่ซ่อนอยู่
# MAGIC
# MAGIC **Algorithms / อัลกอริทึม:**
# MAGIC | Algorithm | Purpose |
# MAGIC |---|---|
# MAGIC | Connected Components | Find isolated networks / ค้นหาเครือข่ายที่แยกจากกัน |
# MAGIC | PageRank | Identify money-routing hubs / ระบุศูนย์กลางการหมุนเวียนเงิน |
# MAGIC | Triangle Count | Detect money cycling loops / ตรวจจับวงจรการฟอกเงิน |
# MAGIC | Label Propagation | Discover sub-communities / ค้นพบกลุ่มย่อยในเครือข่าย |
# MAGIC | Two-Hop Ratio | Measure pass-through behavior / วัดพฤติกรรมบัญชีทางผ่าน |
# MAGIC | Behavior Profiles | Classify account activity patterns / จำแนกรูปแบบกิจกรรมบัญชี |
# MAGIC | Device Patterns | Cluster accounts sharing devices / จัดกลุ่มบัญชีที่ใช้อุปกรณ์ร่วมกัน |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup / การตั้งค่า

# COMMAND ----------

# MAGIC %pip install graphframes pyvis --quiet

# COMMAND ----------

CATALOG = "vn"
SCHEMA = "mule_demo"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# GraphFrames requires a checkpoint directory
spark.sparkContext.setCheckpointDir("/tmp/graphframes_checkpoint")

from graphframes import GraphFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import json

print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the Graph / สร้างกราฟ
# MAGIC
# MAGIC We construct a graph where **nodes are bank accounts** and **edges represent
# MAGIC money flows or shared attributes** (devices, phones, addresses).
# MAGIC
# MAGIC เราสร้างกราฟที่ **โหนดคือบัญชีธนาคาร** และ **เส้นเชื่อมแสดงถึง
# MAGIC การไหลของเงินหรือคุณสมบัติที่ใช้ร่วมกัน** (อุปกรณ์ โทรศัพท์ ที่อยู่)

# COMMAND ----------

# Load silver tables
customers = spark.table("silver_customers")
transactions = spark.table("silver_transactions")
device_logins = spark.table("silver_device_logins")
shared_contacts = spark.table("silver_shared_contacts")
bot_list = spark.table("silver_bot_mule_list")

# --- Nodes: all unique accounts ---
nodes = customers.select(F.col("account_id").alias("id"))
print(f"Nodes: {nodes.count()}")

# --- Edges: transaction links (deduplicated to unique pairs) ---
txn_edges = (
    transactions
    .select(F.col("from_account").alias("src"), F.col("to_account").alias("dst"))
    .distinct()
)

# --- Edges: shared device links ---
# Find accounts that share the same device fingerprint
device_pairs = (
    device_logins
    .select("account_id", "device_fingerprint")
    .distinct()
    .alias("a")
    .join(
        device_logins.select("account_id", "device_fingerprint").distinct().alias("b"),
        (F.col("a.device_fingerprint") == F.col("b.device_fingerprint")) &
        (F.col("a.account_id") < F.col("b.account_id"))
    )
    .select(F.col("a.account_id").alias("src"), F.col("b.account_id").alias("dst"))
    .distinct()
)

# --- Edges: shared phone links ---
phone_pairs = (
    shared_contacts.select("account_id", "phone_number").alias("a")
    .join(
        shared_contacts.select("account_id", "phone_number").alias("b"),
        (F.col("a.phone_number") == F.col("b.phone_number")) &
        (F.col("a.account_id") < F.col("b.account_id"))
    )
    .select(F.col("a.account_id").alias("src"), F.col("b.account_id").alias("dst"))
    .distinct()
)

# --- Combine all edges ---
all_edges = txn_edges.unionByName(device_pairs).unionByName(phone_pairs).distinct()

print(f"Transaction edges: {txn_edges.count()}")
print(f"Device-shared edges: {device_pairs.count()}")
print(f"Phone-shared edges: {phone_pairs.count()}")
print(f"Total unique edges: {all_edges.count()}")

# Build the GraphFrame
g = GraphFrame(nodes, all_edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📖 What is a Graph? / กราฟคืออะไร?
# MAGIC
# MAGIC Before we run the algorithms, let's understand the basics.
# MAGIC A **graph** is a way to represent relationships:
# MAGIC - Each **circle (node)** = one bank account / แต่ละ **วงกลม (โหนด)** = หนึ่งบัญชีธนาคาร
# MAGIC - Each **arrow (edge)** = money was sent between them / แต่ละ **ลูกศร (เส้นเชื่อม)** = มีการโอนเงินระหว่างกัน
# MAGIC
# MAGIC Below is a small example with just a few accounts from our data.
# MAGIC
# MAGIC ด้านล่างเป็นตัวอย่างเล็กๆ ที่มีเพียงไม่กี่บัญชีจากข้อมูลของเรา

# COMMAND ----------

from pyvis.network import Network as PyVisNetwork

# Pick real accounts: 1 BOT-confirmed mule, 1 suspected mule (not on BOT), 2 victims, 3 normal
bot_ids = [r.account_id for r in bot_list.limit(2).collect()]
# Suspected: mule-like account NOT on BOT list — pick an account with high incoming from many senders
suspected = (
    transactions
    .where(~F.col("to_account").isin(bot_ids))
    .groupBy(F.col("to_account").alias("aid"))
    .agg(F.countDistinct("from_account").alias("senders"), F.sum("amount_thb").alias("total"))
    .orderBy(F.desc("total"))
    .limit(1).collect()[0]
)
suspected_id = suspected["aid"]

# Victims: accounts that sent money to our BOT mule
victim_sample_ids = (
    transactions
    .where(F.col("to_account") == bot_ids[0])
    .select("from_account").distinct().limit(2)
    .collect()
)
vic_ids = [r["from_account"] for r in victim_sample_ids]

# Normal: accounts that only transact with other normals
normal_sample = (
    customers
    .where(~F.col("account_id").isin(bot_ids + [suspected_id] + vic_ids))
    .limit(3).collect()
)
norm_ids = [r["account_id"] for r in normal_sample]

# Collect customer info for labels
sample_all_ids = bot_ids[:1] + [suspected_id] + vic_ids + norm_ids
sample_info = {r["account_id"]: r for r in customers.where(F.col("account_id").isin(sample_all_ids)).collect()}

# Get transactions among these accounts
sample_txns = (
    transactions
    .where(F.col("from_account").isin(sample_all_ids) & F.col("to_account").isin(sample_all_ids))
    .groupBy("from_account", "to_account")
    .agg(F.round(F.sum("amount_thb"), 0).alias("total"), F.count("*").alias("cnt"))
    .collect()
)

# Build the simple educational graph
simple_net = PyVisNetwork(height="450px", width="100%", directed=True, notebook=True,
                          bgcolor="#0f172a", font_color="white")

def add_edu_node(net, aid, role, color, emoji):
    info = sample_info.get(aid, {})
    name = info.get("customer_name", aid) if hasattr(info, "get") else getattr(info, "customer_name", aid)
    occ = info.get("occupation", "") if hasattr(info, "get") else getattr(info, "occupation", "")
    prov = info.get("province", "") if hasattr(info, "get") else getattr(info, "province", "")
    net.add_node(aid, label=f"{emoji} {aid}", color=color, size=28,
        title=f"<b>{aid}</b> — {name}<br><b>Role:</b> {role}<br><b>Occupation:</b> {occ}<br><b>Province:</b> {prov}",
        font={"size": 14, "color": "white"})

add_edu_node(simple_net, bot_ids[0], "BOT Confirmed Mule / บัญชีม้ายืนยัน", "#ef4444", "🔴")
add_edu_node(simple_net, suspected_id, "Suspected Mule / ต้องสงสัย", "#f97316", "🟠")
for vid in vic_ids:
    add_edu_node(simple_net, vid, "Victim / เหยื่อ", "#22c55e", "🟢")
for nid in norm_ids:
    add_edu_node(simple_net, nid, "Normal Customer / ลูกค้าปกติ", "#3b82f6", "🔵")

# Add real transaction edges
for txn in sample_txns:
    simple_net.add_edge(txn["from_account"], txn["to_account"], color="#94a3b8", value=2,
        title=f"฿{txn['total']:,.0f} ({txn['cnt']} txns)", arrows="to")

# Ensure key story edges exist (victim → mule, mule → mule)
for vid in vic_ids:
    simple_net.add_edge(vid, bot_ids[0], color="#94a3b8", value=2,
        title="Scam transfer / โอนจากการหลอกลวง", arrows="to")
simple_net.add_edge(bot_ids[0], suspected_id, color="#f97316", value=2,
    title="Mule pass-through / ม้าส่งต่อ", arrows="to")

# Add a couple of normal-to-normal edges
if len(norm_ids) >= 2:
    simple_net.add_edge(norm_ids[0], norm_ids[1], color="#60a5fa", value=1.5,
        title="Normal transfer / โอนปกติ", arrows="to")
    simple_net.add_edge(norm_ids[1], norm_ids[2] if len(norm_ids) > 2 else norm_ids[0],
        color="#60a5fa", value=1.5, title="Normal transfer / โอนปกติ", arrows="to")

simple_net.set_options("""
{
  "physics": {
    "barnesHut": {"gravitationalConstant": -4000, "springLength": 200},
    "stabilization": {"iterations": 100}
  },
  "edges": {"smooth": {"type": "continuous"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}, "width": 2},
  "interaction": {"hover": true, "tooltipDelay": 100}
}
""")

simple_html = simple_net.generate_html()

edu_legend = """
<div style="position:absolute; top:10px; left:10px; background:rgba(15,23,42,0.95);
     padding:16px 20px; border-radius:10px; color:white; font-family:sans-serif; font-size:14px;
     border:1px solid rgba(255,255,255,0.15); max-width:360px; backdrop-filter:blur(8px);">
  <b style="font-size:16px">How to read this graph / วิธีอ่านกราฟ</b><br><br>
  Each <b>circle</b> = one bank account / แต่ละ <b>วงกลม</b> = หนึ่งบัญชี<br>
  Each <b>arrow</b> = money transferred / แต่ละ <b>ลูกศร</b> = การโอนเงิน<br><br>
  <span style="color:#22c55e">🟢 Victim / เหยื่อ</span> sends money to →<br>
  <span style="color:#ef4444">🔴 BOT Confirmed Mule / บัญชีม้ายืนยัน</span> passes to →<br>
  <span style="color:#f97316">🟠 Suspected Mule / ต้องสงสัย</span><br>
  <span style="color:#3b82f6">🔵 Normal / ปกติ</span> customers trade normally<br><br>
  <i>Hover over any circle for details / วางเมาส์เพื่อดูรายละเอียด</i>
</div>
"""
simple_html = simple_html.replace("</body>", edu_legend + "</body>")
displayHTML(simple_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 The Raw Graph — Before Any Analysis / กราฟดิบ — ก่อนการวิเคราะห์
# MAGIC
# MAGIC This is our **entire graph** of 1,000 bank accounts before running any algorithms.
# MAGIC The only thing we know is which accounts the **Bank of Thailand (BOT)** has confirmed as mules.
# MAGIC
# MAGIC นี่คือ **กราฟทั้งหมด** ของบัญชีธนาคาร 1,000 บัญชี ก่อนรันอัลกอริทึมใดๆ
# MAGIC สิ่งเดียวที่เรารู้คือบัญชีไหนที่ **ธนาคารแห่งประเทศไทย (ธปท.)** ยืนยันว่าเป็นบัญชีม้า
# MAGIC
# MAGIC - 🔴 **Red** = BOT confirmed mule / บัญชีม้ายืนยัน
# MAGIC - 🔵 **Blue** = Everything else — could be normal, could be mule, we don't know yet
# MAGIC
# MAGIC - 🔴 **แดง** = บัญชีม้ายืนยันโดย ธปท.
# MAGIC - 🔵 **น้ำเงิน** = ที่เหลือทั้งหมด — อาจปกติ อาจเป็นม้า ยังไม่รู้
# MAGIC
# MAGIC **Can you spot the hidden mules just by looking?** Let's see if the algorithms can.
# MAGIC
# MAGIC **คุณเห็นบัญชีม้าที่ซ่อนอยู่ไหม?** มาดูกันว่าอัลกอริทึมจะหาได้หรือไม่

# COMMAND ----------

from pyvis.network import Network as PyVisNetworkFull
import random as _rnd

# Get BOT-confirmed account IDs
bot_account_ids = set(r.account_id for r in bot_list.collect())

# Sample a manageable subset for visualization:
# All BOT mules + their direct transaction partners + a random sample of others
bot_ids_list = list(bot_account_ids)

# Get direct neighbors of BOT mules (1 hop)
bot_neighbors = set(
    transactions
    .where(F.col("to_account").isin(bot_ids_list) | F.col("from_account").isin(bot_ids_list))
    .select(
        F.explode(F.array("from_account", "to_account")).alias("aid")
    )
    .distinct()
    .toPandas()["aid"].tolist()
)

# Add a random sample of other accounts for context (~80 more)
all_account_ids = [r.account_id for r in customers.select("account_id").collect()]
remaining = [a for a in all_account_ids if a not in bot_neighbors]
_rnd.seed(42)
sample_others = set(_rnd.sample(remaining, min(80, len(remaining))))

# Final set of nodes to visualize
viz_nodes = bot_neighbors | sample_others
viz_nodes_list = list(viz_nodes)

# Get edges between these nodes
viz_edges = (
    transactions
    .where(F.col("from_account").isin(viz_nodes_list) & F.col("to_account").isin(viz_nodes_list))
    .select("from_account", "to_account")
    .distinct()
    .toPandas()
)

print(f"Visualizing {len(viz_nodes)} accounts ({len(bot_account_ids)} BOT confirmed, {len(viz_nodes) - len(bot_account_ids)} unknown)")
print(f"Edges: {len(viz_edges)}")

# Build PyVis graph — only 2 colors: red (BOT) and blue (everything else)
net_full = PyVisNetworkFull(height="650px", width="100%", directed=True, notebook=True,
                            bgcolor="#0f172a", font_color="white")

for aid in viz_nodes:
    is_bot = aid in bot_account_ids
    color = "#ef4444" if is_bot else "#334d6e"
    size = 14 if is_bot else 8
    border_color = "#fca5a5" if is_bot else "#1e293b"
    label = aid if is_bot else ""

    net_full.add_node(aid, label=label, color=color, size=size,
        border_width=2 if is_bot else 1, border_color=border_color,
        title=f"<b>{aid}</b><br>{'🔴 BOT Confirmed Mule' if is_bot else '🔵 Unknown — mule or normal?'}",
        font={"size": 10, "color": "white"})

for _, row in viz_edges.iterrows():
    net_full.add_edge(row["from_account"], row["to_account"],
        color="#1e3a5f", width=0.5, arrows="to")

net_full.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -60,
      "centralGravity": 0.01,
      "springLength": 80,
      "springConstant": 0.06
    },
    "solver": "forceAtlas2Based",
    "stabilization": {"iterations": 200}
  },
  "edges": {
    "smooth": {"type": "continuous"},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.3}}
  },
  "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true}
}
""")

full_html = net_full.generate_html()

full_legend = """
<div style="position:absolute; top:10px; right:10px; background:rgba(15,23,42,0.95);
     padding:14px 18px; border-radius:10px; color:white; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(8px);">
  <b style="font-size:15px">Before Analysis / ก่อนวิเคราะห์</b><br><br>
  <span style="color:#ef4444">● BOT Confirmed Mule</span> — we know these<br>
  <span style="color:#ef4444">● บัญชีม้ายืนยัน</span> — เรารู้แล้ว<br><br>
  <span style="color:#334d6e">● Everything else</span> — normal? mule? unknown<br>
  <span style="color:#334d6e">● ที่เหลือ</span> — ปกติ? ม้า? ไม่รู้<br><br>
  <i>The algorithms will reveal what's hidden<br>
  อัลกอริทึมจะเปิดเผยสิ่งที่ซ่อนอยู่</i>
</div>
"""
full_html = full_html.replace("</body>", full_legend + "</body>")
displayHTML(full_html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Connected Components / ส่วนประกอบที่เชื่อมต่อกัน
# MAGIC
# MAGIC Identifies which accounts belong to the same **network cluster**.
# MAGIC If a BOT-confirmed mule is in a component, every account in that component deserves scrutiny.
# MAGIC
# MAGIC ระบุว่าบัญชีใดอยู่ใน **กลุ่มเครือข่าย** เดียวกัน
# MAGIC หากบัญชีม้าที่ ธปท. ยืนยันอยู่ในกลุ่ม ทุกบัญชีในกลุ่มนั้นควรถูกตรวจสอบ

# COMMAND ----------

cc = g.connectedComponents()
cc = cc.select(F.col("id"), F.col("component").alias("connected_component_id"))

# Show component sizes
comp_sizes = (
    cc.groupBy("connected_component_id")
    .count()
    .orderBy("count", ascending=False)
)

print("Top 10 largest connected components:")
display(comp_sizes.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PageRank — Find the Hubs / ค้นหาศูนย์กลาง
# MAGIC
# MAGIC PageRank identifies the most **central accounts** in the money flow network.
# MAGIC High PageRank = this account routes money from many sources. Likely a **syndicate operator**.
# MAGIC
# MAGIC PageRank ระบุบัญชีที่เป็น **ศูนย์กลาง** ในเครือข่ายการไหลของเงิน
# MAGIC PageRank สูง = บัญชีนี้เป็นทางผ่านเงินจากหลายแหล่ง น่าจะเป็น **ผู้ดำเนินการเครือข่าย**

# COMMAND ----------

pr = g.pageRank(resetProbability=0.15, maxIter=10)
pr_scores = pr.vertices.select(F.col("id"), F.col("pagerank").alias("pagerank_score"))

# Normalize pagerank to 0-1 range
pr_max = pr_scores.agg(F.max("pagerank_score")).collect()[0][0]
pr_scores = pr_scores.withColumn("pagerank_score", F.round(F.col("pagerank_score") / pr_max, 6))

print("Top 15 accounts by PageRank (potential hubs):")
display(pr_scores.orderBy("pagerank_score", ascending=False).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Triangle Count — Detect Money Loops / ตรวจจับวงจรเงิน
# MAGIC
# MAGIC Triangles (A→B→C→A) indicate **money cycling** — a classic layering technique
# MAGIC used by mule networks to obscure the money trail.
# MAGIC
# MAGIC สามเหลี่ยม (A→B→C→A) บ่งบอกถึง **การหมุนเวียนเงิน** — เทคนิคการซ้อนชั้น
# MAGIC ที่เครือข่ายบัญชีม้าใช้เพื่อปกปิดเส้นทางเงิน

# COMMAND ----------

# Triangle count requires undirected graph — GraphFrames handles this
tc = g.triangleCount()
tc_scores = tc.select(F.col("id"), F.col("count").alias("triangle_count"))

print("Top 15 accounts by triangle count (money cycling):")
display(tc_scores.orderBy("triangle_count", ascending=False).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Community Detection / ตรวจจับกลุ่มย่อย
# MAGIC
# MAGIC Label Propagation discovers **sub-communities** within connected components.
# MAGIC Each community may represent a different **recruiter** or **operational cell** within a syndicate.
# MAGIC
# MAGIC Label Propagation ค้นพบ **กลุ่มย่อย** ภายในส่วนประกอบที่เชื่อมต่อกัน
# MAGIC แต่ละกลุ่มอาจแสดงถึง **ผู้คัดเลือก** หรือ **เซลล์ปฏิบัติการ** ที่แตกต่างกันในเครือข่าย

# COMMAND ----------

communities = g.labelPropagation(maxIter=5)
community_scores = communities.select(F.col("id"), F.col("label").alias("community_id"))

comm_sizes = (
    community_scores.groupBy("community_id")
    .count()
    .orderBy("count", ascending=False)
)

print("Top 10 communities by size:")
display(comm_sizes.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Two-Hop Ratio — Pass-Through Detection / ตรวจจับบัญชีทางผ่าน
# MAGIC
# MAGIC For each account, we calculate: **what percentage of incoming funds leave within 24 hours?**
# MAGIC A ratio > 0.8 means the account is a **pure pass-through** — money doesn't stay.
# MAGIC This is the signature of a mule account.
# MAGIC
# MAGIC สำหรับแต่ละบัญชี เราคำนวณ: **เงินที่เข้ามากี่เปอร์เซ็นต์ที่ออกภายใน 24 ชั่วโมง?**
# MAGIC อัตราส่วน > 0.8 หมายความว่าบัญชีนี้เป็น **ทางผ่านล้วนๆ** — เงินไม่อยู่ในบัญชี
# MAGIC นี่คือลายเซ็นของบัญชีม้า

# COMMAND ----------

# Total incoming per account
incoming = (
    transactions
    .groupBy(F.col("to_account").alias("account_id"))
    .agg(F.sum("amount_thb").alias("total_in"))
)

# For each incoming transaction, find outgoing within 24 hours
txn_in = transactions.select(
    F.col("to_account").alias("acct"),
    F.col("amount_thb").alias("in_amount"),
    F.col("txn_timestamp").alias("in_ts")
)
txn_out = transactions.select(
    F.col("from_account").alias("acct"),
    F.col("amount_thb").alias("out_amount"),
    F.col("txn_timestamp").alias("out_ts")
)

pass_through = (
    txn_in.join(txn_out, "acct")
    .where(
        (F.col("out_ts") >= F.col("in_ts")) &
        (F.col("out_ts") <= F.col("in_ts") + F.expr("INTERVAL 24 HOURS"))
    )
    .groupBy(F.col("acct").alias("account_id"))
    .agg(F.sum("out_amount").alias("pass_through_amount"))
)

two_hop = (
    incoming.join(pass_through, "account_id", "left")
    .fillna(0, subset=["pass_through_amount"])
    .withColumn("two_hop_ratio",
        F.least(
            F.round(F.col("pass_through_amount") / F.greatest(F.col("total_in"), F.lit(1.0)), 4),
            F.lit(1.0)
        )
    )
    .select("account_id", "two_hop_ratio")
)

print("Top 15 accounts by two-hop ratio (pass-through):")
display(two_hop.orderBy("two_hop_ratio", ascending=False).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Behavior Profiles / โปรไฟล์พฤติกรรม
# MAGIC
# MAGIC We classify each account into behavioral categories based on transaction patterns:
# MAGIC - **dormant_then_active:** inactive for months, then suddenly active (coordinated activation)
# MAGIC - **pass_through:** most money flows in and out quickly
# MAGIC - **burst:** concentrated activity in short periods
# MAGIC - **steady:** regular, consistent activity
# MAGIC
# MAGIC จำแนกแต่ละบัญชีเป็นหมวดหมู่ตามรูปแบบธุรกรรม:
# MAGIC - **dormant_then_active:** ไม่มีกิจกรรมเป็นเดือนๆ แล้วเปิดใช้งานทันที
# MAGIC - **pass_through:** เงินส่วนใหญ่ไหลเข้าและออกอย่างรวดเร็ว
# MAGIC - **burst:** กิจกรรมกระจุกตัวในช่วงเวลาสั้นๆ
# MAGIC - **steady:** กิจกรรมสม่ำเสมอ

# COMMAND ----------

# Compute activity metrics per account
all_activity = (
    transactions
    .select(F.col("from_account").alias("account_id"), "txn_timestamp", "amount_thb")
    .unionByName(
        transactions.select(F.col("to_account").alias("account_id"), "txn_timestamp", "amount_thb")
    )
)

# Weekly activity counts
weekly = (
    all_activity
    .withColumn("week", F.weekofyear("txn_timestamp"))
    .withColumn("year_week", F.concat(F.year("txn_timestamp"), F.lit("-"), F.lpad(F.col("week"), 2, "0")))
    .groupBy("account_id", "year_week")
    .agg(F.count("*").alias("weekly_txns"))
)

behavior_stats = (
    weekly.groupBy("account_id")
    .agg(
        F.count("year_week").alias("active_weeks"),
        F.avg("weekly_txns").alias("avg_weekly_txns"),
        F.stddev("weekly_txns").alias("std_weekly_txns"),
        F.max("weekly_txns").alias("max_weekly_txns")
    )
    .fillna(0.0, subset=["std_weekly_txns"])
)

# Join with account open date and two-hop ratio
behavior = (
    behavior_stats
    .join(customers.select("account_id", "account_open_date"), "account_id", "left")
    .join(two_hop, "account_id", "left")
    .fillna(0.0, subset=["two_hop_ratio"])
    .withColumn("behavior_profile",
        F.when(
            (F.col("two_hop_ratio") > 0.7) & (F.col("active_weeks") <= 6),
            F.lit("pass_through")
        ).when(
            (F.col("active_weeks") <= 4) & (F.col("max_weekly_txns") > 20),
            F.lit("burst")
        ).when(
            (F.col("active_weeks") <= 5) &
            (F.datediff(F.lit("2025-12-15"), F.col("account_open_date")) > 120),
            F.lit("dormant_then_active")
        ).otherwise(F.lit("steady"))
    )
    .select("account_id", "behavior_profile")
)

print("Behavior profile distribution:")
display(behavior.groupBy("behavior_profile").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Device Pattern Clusters / กลุ่มรูปแบบอุปกรณ์
# MAGIC
# MAGIC We group accounts that share device fingerprints into clusters.
# MAGIC If 12 accounts share the same 3 devices, they are likely controlled by **one person or syndicate**.
# MAGIC
# MAGIC เราจัดกลุ่มบัญชีที่ใช้ลายนิ้วมืออุปกรณ์ร่วมกัน
# MAGIC หาก 12 บัญชีใช้อุปกรณ์เดียวกัน 3 เครื่อง น่าจะถูกควบคุมโดย **คนเดียวหรือเครือข่ายเดียว**

# COMMAND ----------

# Build device clusters using connected components on device-sharing graph
device_nodes = device_logins.select(F.col("account_id").alias("id")).distinct()
device_graph = GraphFrame(device_nodes, device_pairs)

device_cc = device_graph.connectedComponents()
device_clusters = device_cc.select(
    F.col("id").alias("account_id"),
    F.col("component").alias("device_pattern_cluster")
)

# Show clusters with multiple accounts (these are suspicious)
device_cluster_sizes = (
    device_clusters
    .groupBy("device_pattern_cluster")
    .agg(F.count("*").alias("cluster_size"))
    .where("cluster_size > 1")
    .orderBy("cluster_size", ascending=False)
)

print("Device clusters with shared devices (suspicious):")
display(device_cluster_sizes.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Assemble Risk Table / รวมตารางความเสี่ยง
# MAGIC
# MAGIC We join **all computed features** with the BOT mule list and customer demographics
# MAGIC to produce the final `gold_account_risk_graph` table with a **composite risk score**.
# MAGIC
# MAGIC เรารวม **คุณสมบัติที่คำนวณทั้งหมด** กับรายชื่อบัญชีม้า ธปท. และข้อมูลประชากร
# MAGIC เพื่อสร้างตาราง `gold_account_risk_graph` พร้อม **คะแนนความเสี่ยงรวม**

# COMMAND ----------

# Transaction volume stats per account
inflow = (
    transactions
    .groupBy(F.col("to_account").alias("account_id"))
    .agg(
        F.round(F.sum("amount_thb"), 2).alias("total_inflow_thb"),
        F.countDistinct("from_account").alias("unique_senders")
    )
)

outflow = (
    transactions
    .groupBy(F.col("from_account").alias("account_id"))
    .agg(
        F.round(F.sum("amount_thb"), 2).alias("total_outflow_thb"),
        F.countDistinct("to_account").alias("unique_receivers")
    )
)

# Average hold time: time between incoming and next outgoing transaction
hold_time = (
    txn_in.join(txn_out, "acct")
    .where(F.col("out_ts") > F.col("in_ts"))
    .withColumn("hold_hours",
        (F.unix_timestamp("out_ts") - F.unix_timestamp("in_ts")) / 3600.0
    )
    .groupBy(F.col("acct").alias("account_id"))
    .agg(F.round(F.avg("hold_hours"), 2).alias("avg_hold_time_hours"))
)

# --- Assemble the gold table ---
gold = (
    customers.select(
        "account_id", "customer_name", "age", "occupation",
        "monthly_income", "income_band", "province", "account_open_date"
    )
    # BOT list
    .join(
        bot_list.select(
            F.col("account_id"),
            F.lit(True).alias("bot_confirmed_mule"),
            F.col("flagged_date").alias("bot_flagged_date")
        ),
        "account_id", "left"
    )
    .fillna(False, subset=["bot_confirmed_mule"])
    # Graph algorithms
    .join(cc, customers["account_id"] == cc["id"], "left").drop("id")
    .join(pr_scores, customers["account_id"] == pr_scores["id"], "left").drop("id")
    .join(tc_scores, customers["account_id"] == tc_scores["id"], "left").drop("id")
    .join(community_scores, customers["account_id"] == community_scores["id"], "left").drop("id")
    # Custom features
    .join(two_hop, "account_id", "left")
    .join(behavior, "account_id", "left")
    .join(device_clusters, "account_id", "left")
    # Volume stats
    .join(inflow, "account_id", "left")
    .join(outflow, "account_id", "left")
    .join(hold_time, "account_id", "left")
    # Fill nulls
    .fillna(0.0, subset=["pagerank_score", "triangle_count", "two_hop_ratio",
                          "total_inflow_thb", "total_outflow_thb",
                          "unique_senders", "unique_receivers", "avg_hold_time_hours"])
    .fillna("steady", subset=["behavior_profile"])
)

# --- Compute composite risk score ---
# Normalize pagerank (already 0-1), triangle_count, unique_senders for scoring
tc_max = gold.agg(F.max("triangle_count")).collect()[0][0] or 1
senders_max = gold.agg(F.max("unique_senders")).collect()[0][0] or 1

gold = gold.withColumn("risk_score",
    F.round(
        F.lit(0.20) * F.col("pagerank_score") +
        F.lit(0.25) * F.col("two_hop_ratio") +
        F.lit(0.15) * F.least(F.col("triangle_count") / F.lit(tc_max), F.lit(1.0)) +
        F.lit(0.10) * F.least(F.col("unique_senders") / F.lit(senders_max), F.lit(1.0)) +
        F.lit(0.10) * F.when(F.col("behavior_profile").isin("pass_through", "dormant_then_active", "burst"), 0.9).otherwise(0.1) +
        F.lit(0.10) * F.when(F.col("income_band") == "<15K", 0.6)
                       .when(F.col("income_band") == "15-30K", 0.3)
                       .otherwise(0.1) +
        F.lit(0.10) * F.when(F.col("bot_confirmed_mule"), 1.0).otherwise(0.0)
    , 4)
)

gold.write.mode("overwrite").saveAsTable("gold_account_risk_graph")

print(f"Gold table written: {gold.count()} rows")

# COMMAND ----------

# Show high-risk accounts (mix of BOT confirmed and undiscovered)
print("Top 20 accounts by risk score:")
display(
    gold.select(
        "account_id", "customer_name", "risk_score", "bot_confirmed_mule",
        "pagerank_score", "two_hop_ratio", "triangle_count",
        "behavior_profile", "income_band", "total_inflow_thb"
    ).orderBy("risk_score", ascending=False).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Transaction Edges / เส้นเชื่อมธุรกรรม
# MAGIC
# MAGIC Aggregate transactions between account pairs for graph visualization.
# MAGIC
# MAGIC รวมธุรกรรมระหว่างคู่บัญชีสำหรับการแสดงผลกราฟ

# COMMAND ----------

txn_edges_gold = (
    transactions
    .groupBy("from_account", "to_account")
    .agg(
        F.round(F.sum("amount_thb"), 2).alias("total_amount_thb"),
        F.count("*").alias("txn_count"),
        F.min("txn_timestamp").alias("first_txn"),
        F.max("txn_timestamp").alias("last_txn"),
        F.collect_set("channel").alias("channels")
    )
)

txn_edges_gold.write.mode("overwrite").saveAsTable("gold_transaction_edges")
print(f"Transaction edges written: {txn_edges_gold.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Shared Attribute Edges / เส้นเชื่อมคุณสมบัติร่วม
# MAGIC
# MAGIC Non-transactional links: accounts sharing devices, phones, emails, or addresses.
# MAGIC These are critical for uncovering syndicate control structures.
# MAGIC
# MAGIC เส้นเชื่อมที่ไม่ใช่ธุรกรรม: บัญชีที่ใช้อุปกรณ์ โทรศัพท์ อีเมล หรือที่อยู่ร่วมกัน
# MAGIC สิ่งเหล่านี้สำคัญสำหรับการเปิดโปงโครงสร้างการควบคุมของเครือข่าย

# COMMAND ----------

# Device-shared edges
device_shared = (
    device_pairs
    .select(
        F.col("src").alias("account_a"),
        F.col("dst").alias("account_b"),
        F.lit("same_device").alias("link_type"),
        F.lit("device_fingerprint").alias("shared_value")
    )
)

# Phone-shared edges
phone_shared = (
    shared_contacts.select("account_id", "phone_number").alias("a")
    .join(
        shared_contacts.select("account_id", "phone_number").alias("b"),
        (F.col("a.phone_number") == F.col("b.phone_number")) &
        (F.col("a.account_id") < F.col("b.account_id"))
    )
    .select(
        F.col("a.account_id").alias("account_a"),
        F.col("b.account_id").alias("account_b"),
        F.lit("same_phone").alias("link_type"),
        F.col("a.phone_number").alias("shared_value")
    )
    .distinct()
)

# Email-shared edges
email_shared = (
    shared_contacts.select("account_id", "email").alias("a")
    .join(
        shared_contacts.select("account_id", "email").alias("b"),
        (F.col("a.email") == F.col("b.email")) &
        (F.col("a.account_id") < F.col("b.account_id"))
    )
    .select(
        F.col("a.account_id").alias("account_a"),
        F.col("b.account_id").alias("account_b"),
        F.lit("same_email").alias("link_type"),
        F.col("a.email").alias("shared_value")
    )
    .distinct()
)

# Combine all shared attribute edges
shared_edges = device_shared.unionByName(phone_shared).unionByName(email_shared).distinct()
shared_edges.write.mode("overwrite").saveAsTable("gold_shared_attribute_edges")

print(f"Shared attribute edges written: {shared_edges.count()}")
display(shared_edges.groupBy("link_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Visualize the Full Network / แสดงผลเครือข่ายทั้งหมด 🕸️
# MAGIC
# MAGIC Now let's look at the **real suspicious network** — the largest connected component
# MAGIC containing BOT-confirmed mules. This is what an investigator would explore.
# MAGIC
# MAGIC ตอนนี้มาดู **เครือข่ายที่น่าสงสัยจริง** — กลุ่มที่เชื่อมต่อกันใหญ่ที่สุด
# MAGIC ที่มีบัญชีม้ายืนยันโดย ธปท. นี่คือสิ่งที่ผู้ตรวจสอบจะสำรวจ
# MAGIC
# MAGIC **Legend / คำอธิบาย:**
# MAGIC - 🔴 **Red** = BOT confirmed mule / บัญชีม้ายืนยันโดย ธปท.
# MAGIC - 🟠 **Orange** = High risk, NOT on BOT list / ความเสี่ยงสูง ยังไม่ถูกรายงาน
# MAGIC - 🟡 **Yellow** = Medium risk / ความเสี่ยงปานกลาง
# MAGIC - 🔵 **Blue** = Normal account / บัญชีปกติ
# MAGIC - 🟢 **Green** = Victim / เหยื่อ (low risk, sends to mules)

# COMMAND ----------

from pyvis.network import Network
import networkx as nx

# Find the largest component that contains BOT-confirmed mules
bot_components = (
    gold.where("bot_confirmed_mule = true")
    .groupBy("connected_component_id")
    .count()
    .orderBy("count", ascending=False)
)

target_component = bot_components.first()["connected_component_id"]
print(f"Visualizing component: {target_component}")

# Pull subgraph data
sub_nodes = gold.where(F.col("connected_component_id") == target_component).toPandas()
sub_txn_edges = (
    txn_edges_gold
    .where(
        F.col("from_account").isin(sub_nodes["account_id"].tolist()) &
        F.col("to_account").isin(sub_nodes["account_id"].tolist())
    )
    .toPandas()
)
sub_shared = (
    shared_edges
    .where(
        F.col("account_a").isin(sub_nodes["account_id"].tolist()) &
        F.col("account_b").isin(sub_nodes["account_id"].tolist())
    )
    .toPandas()
)

print(f"Nodes in component: {len(sub_nodes)}")
print(f"Transaction edges: {len(sub_txn_edges)}")
print(f"Shared attribute edges: {len(sub_shared)}")

# COMMAND ----------

# Determine which accounts are likely victims (sent money to this component but low risk)
victim_senders = set(
    transactions
    .where(
        F.col("to_account").isin(sub_nodes["account_id"].tolist()) &
        ~F.col("from_account").isin(sub_nodes["account_id"].tolist())
    )
    .select("from_account").distinct().toPandas()["from_account"].tolist()
)

# Build PyVis network
G = nx.DiGraph()

for _, row in sub_nodes.iterrows():
    aid = row["account_id"]
    risk = row["risk_score"]
    is_bot = row["bot_confirmed_mule"]

    # Color logic
    if is_bot:
        color = "#ef4444"  # Red — BOT confirmed
    elif risk > 0.5:
        color = "#f97316"  # Orange — high risk undiscovered
    elif risk > 0.3:
        color = "#eab308"  # Yellow — medium risk
    else:
        color = "#3b82f6"  # Blue — normal

    # Size by PageRank
    size = 10 + (float(row["pagerank_score"]) * 60) *0.5

    # Tooltip
    title = (
        f"<b>{aid}</b> — {row['customer_name']}<br>"
        f"{'🔴 BOT Confirmed / บัญชีม้ายืนยัน' if is_bot else ''}<br>"
        f"<b>Risk Score:</b> {risk}<br>"
        f"<b>PageRank:</b> {row['pagerank_score']}<br>"
        f"<b>Two-Hop Ratio:</b> {row['two_hop_ratio']}<br>"
        f"<b>Triangles:</b> {int(row['triangle_count'])}<br>"
        f"<b>Behavior:</b> {row['behavior_profile']}<br>"
        f"<b>Income:</b> {row['income_band']} (declared ฿{row['monthly_income']:,.0f}/mo)<br>"
        f"<b>Inflow:</b> ฿{row['total_inflow_thb']:,.0f}<br>"
        f"<b>Outflow:</b> ฿{row['total_outflow_thb']:,.0f}<br>"
        f"<b>Avg Hold Time:</b> {row['avg_hold_time_hours']:.1f} hours<br>"
        f"<b>Community:</b> {int(row['community_id'])}<br>"
        f"<b>Province:</b> {row['province']}"
    )

    G.add_node(aid, color=color, size=size, title=title, label=aid)

# Transaction edges (solid arrows)
for _, row in sub_txn_edges.iterrows():
    if row["from_account"] in G.nodes and row["to_account"] in G.nodes:
        width = max(0.5, min(row["total_amount_thb"] / 500000, 5))
        G.add_edge(
            row["from_account"], row["to_account"],
            value=width, color="#6b7280",
            title=f"฿{row['total_amount_thb']:,.0f} ({row['txn_count']} txns)"
        )

# Shared attribute edges (dashed, different color)
for _, row in sub_shared.iterrows():
    if row["account_a"] in G.nodes and row["account_b"] in G.nodes:
        edge_color = {"same_device": "#a855f7", "same_phone": "#06b6d4", "same_email": "#10b981"}
        G.add_edge(
            row["account_a"], row["account_b"],
            color=edge_color.get(row["link_type"], "#9ca3af"),
            dashes=True,
            title=f"{row['link_type']}: {row['shared_value']}",
            value=0.5
        )

# Render
net = Network(height="700px", width="100%", directed=True, notebook=True, bgcolor="#0f172a", font_color="white")
net.from_nx(G)
net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -60,
      "centralGravity": 0.01,
      "springLength": 80,
      "springConstant": 0.01
    },
    "solver": "forceAtlas2Based",
    "stabilization": {"iterations": 200}
  },
  "edges": {
    "smooth": {"type": "continuous"},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.3}}
  },
  "interaction": {"hover": true, "tooltipDelay": 100, "zoomView": true}
}
""")

html_content = net.generate_html()

# Add a legend to the HTML
legend_html = """
<div style="position:absolute; top:10px; right:10px; background:rgba(15,23,42,0.9);
     padding:12px 16px; border-radius:8px; color:white; font-family:sans-serif; font-size:13px;
     border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(8px);">
  <b>Legend / คำอธิบาย</b><br>
  <span style="color:#ef4444">●</span> BOT Confirmed / บัญชีม้ายืนยัน<br>
  <span style="color:#f97316">●</span> High Risk / ความเสี่ยงสูง<br>
  <span style="color:#eab308">●</span> Medium Risk / ความเสี่ยงปานกลาง<br>
  <span style="color:#3b82f6">●</span> Normal / ปกติ<br>
  <hr style="border-color:rgba(255,255,255,0.2)">
  <b>Edges / เส้นเชื่อม</b><br>
  <span style="color:#6b7280">━</span> Transaction / ธุรกรรม<br>
  <span style="color:#a855f7">╌</span> Shared Device / อุปกรณ์ร่วม<br>
  <span style="color:#06b6d4">╌</span> Shared Phone / โทรศัพท์ร่วม<br>
  <span style="color:#10b981">╌</span> Shared Email / อีเมลร่วม<br>
  <hr style="border-color:rgba(255,255,255,0.2)">
  <b>Node Size</b> = PageRank (centrality)<br>
  <b>Hover</b> for details / วางเมาส์เพื่อดูรายละเอียด
</div>
"""
html_content = html_content.replace("</body>", legend_html + "</body>")

displayHTML(html_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### ✅ Phase 2 Complete / เฟส 2 เสร็จสมบูรณ์
# MAGIC
# MAGIC All graph features computed and saved to gold tables:
# MAGIC - `gold_account_risk_graph` — 1,000 accounts with all features and risk scores
# MAGIC - `gold_transaction_edges` — aggregated money flows between account pairs
# MAGIC - `gold_shared_attribute_edges` — device, phone, and email links
# MAGIC
# MAGIC คุณสมบัติกราฟทั้งหมดคำนวณและบันทึกในตารางโกลด์แล้ว
# MAGIC
# MAGIC **Next:** Proceed to Phase 3 (Lakebase sync) and Phase 5 (Graph Explorer App)
# MAGIC
# MAGIC **ถัดไป:** ดำเนินการต่อเฟส 3 (ซิงค์ Lakebase) และเฟส 5 (แอป Graph Explorer)