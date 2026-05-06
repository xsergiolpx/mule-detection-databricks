# Databricks notebook source
# MAGIC %md
# MAGIC # 🕸️ Introduction to Graph Algorithms with GraphFrames
# MAGIC # แนะนำอัลกอริทึมกราฟด้วย GraphFrames
# MAGIC
# MAGIC A graph is a way to represent **relationships** between things.
# MAGIC
# MAGIC กราฟคือวิธีแสดง **ความสัมพันธ์** ระหว่างสิ่งต่างๆ
# MAGIC
# MAGIC - Each **circle** is a **node** (a person, an account, a city…)
# MAGIC - Each **line** is an **edge** (a friendship, a transaction, a road…)
# MAGIC
# MAGIC - แต่ละ **วงกลม** คือ **โหนด** (คน, บัญชี, เมือง…)
# MAGIC - แต่ละ **เส้น** คือ **เส้นเชื่อม** (มิตรภาพ, ธุรกรรม, ถนน…)
# MAGIC
# MAGIC In this notebook we'll build a small graph by hand and run **4 algorithms** to discover hidden structure.
# MAGIC
# MAGIC ในโน้ตบุ๊คนี้เราจะสร้างกราฟเล็กๆ ด้วยมือ แล้วรัน **4 อัลกอริทึม** เพื่อค้นหาโครงสร้างที่ซ่อนอยู่

# COMMAND ----------

# MAGIC %pip install graphframes pyvis --quiet

# COMMAND ----------

from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyvis.network import Network
import json

spark.sparkContext.setCheckpointDir("/tmp/gf_tutorial_checkpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a Small Graph / สร้างกราฟเล็กๆ
# MAGIC
# MAGIC We'll create a small social network: **14 people** with friendships between them.
# MAGIC
# MAGIC เราจะสร้างเครือข่ายสังคมเล็กๆ: **14 คน** พร้อมความสัมพันธ์ระหว่างกัน
# MAGIC
# MAGIC Notice the structure:
# MAGIC - A **left group** (Alice, Bob, Carol, Dave, Eve) — they all know each other
# MAGIC - A **right group** (Frank, Grace, Hank, Ivy, Jack) — they all know each other
# MAGIC - **Kim** connects the two groups (a "bridge")
# MAGIC - **Leo** and **Mia** are a separate pair, disconnected from everyone else
# MAGIC - **Nora** only connects to Kim
# MAGIC
# MAGIC สังเกตโครงสร้าง:
# MAGIC - **กลุ่มซ้าย** (Alice, Bob, Carol, Dave, Eve) — รู้จักกันทั้งหมด
# MAGIC - **กลุ่มขวา** (Frank, Grace, Hank, Ivy, Jack) — รู้จักกันทั้งหมด
# MAGIC - **Kim** เชื่อมสองกลุ่มเข้าด้วยกัน ("สะพาน")
# MAGIC - **Leo** และ **Mia** เป็นคู่แยก ไม่เชื่อมกับใคร
# MAGIC - **Nora** เชื่อมกับ Kim เท่านั้น

# COMMAND ----------

# Define nodes (people)
nodes = spark.createDataFrame([
    ("Alice",), ("Bob",), ("Carol",), ("Dave",), ("Eve",),       # Left group
    ("Frank",), ("Grace",), ("Hank",), ("Ivy",), ("Jack",),      # Right group
    ("Kim",),                                                       # Bridge
    ("Nora",),                                                      # Connected only to Kim
    ("Leo",), ("Mia",),                                            # Isolated pair
], ["id"])

# Define edges (friendships — bidirectional)
edge_list = [
    # Left group (dense connections + triangles)
    ("Alice","Bob"), ("Alice","Carol"), ("Alice","Dave"),
    ("Bob","Carol"), ("Bob","Dave"), ("Bob","Eve"),
    ("Carol","Dave"), ("Dave","Eve"), ("Carol","Eve"),

    # Right group (dense connections + triangles)
    ("Frank","Grace"), ("Frank","Hank"), ("Frank","Ivy"),
    ("Grace","Hank"), ("Grace","Jack"), ("Hank","Ivy"),
    ("Ivy","Jack"), ("Hank","Jack"), ("Grace","Ivy"),

    # Bridge: Kim connects to both groups
    ("Kim","Alice"), ("Kim","Bob"),
    ("Kim","Frank"), ("Kim","Grace"),

    # Nora only connects to Kim
    ("Kim","Nora"),

    # Isolated pair
    ("Leo","Mia"),
]

# Make edges bidirectional (undirected graph)
all_edges = [(a, b) for a, b in edge_list] + [(b, a) for a, b in edge_list]
edges = spark.createDataFrame(list(set(all_edges)), ["src", "dst"])

g = GraphFrame(nodes, edges)
print(f"Nodes: {g.vertices.count()}, Edges: {g.edges.count()}")

# COMMAND ----------

# Visualize the plain graph — all nodes look the same
def render_graph(nodes_data, edges_data, title="Graph", node_colors=None,
                 node_sizes=None, height="500px"):
    """Render a PyVis graph inside the notebook."""
    net = Network(height=height, width="100%", notebook=True, bgcolor="#0f172a",
                  font_color="white")

    default_color = "#60a5fa"
    default_size = 22

    for row in nodes_data:
        nid = row["id"]
        color = node_colors.get(nid, default_color) if node_colors else default_color
        size = node_sizes.get(nid, default_size) if node_sizes else default_size
        net.add_node(nid, label=nid, color=color, size=size,
                     font={"size": 14, "color": "white"},
                     title=f"<b>{nid}</b>")

    seen = set()
    for row in edges_data:
        pair = tuple(sorted([row["src"], row["dst"]]))
        if pair not in seen:
            seen.add(pair)
            net.add_edge(row["src"], row["dst"], color="#475569", width=1.5)

    net.set_options(json.dumps({
        "physics": {
            "barnesHut": {"gravitationalConstant": -3000, "springLength": 150},
            "stabilization": {"iterations": 150}
        },
        "edges": {"smooth": {"type": "continuous"}},
        "interaction": {"hover": True, "tooltipDelay": 100}
    }))

    html = net.generate_html()

    # Add title overlay
    overlay = f"""
    <div style="position:absolute; top:10px; left:10px; background:rgba(15,23,42,0.9);
         padding:10px 16px; border-radius:8px; color:white; font-family:sans-serif;
         font-size:15px; font-weight:600; border:1px solid rgba(255,255,255,0.1);">
      {title}
    </div>
    """
    html = html.replace("</body>", overlay + "</body>")
    displayHTML(html)

# Collect data for visualization
nodes_list = nodes.collect()
edges_list = edges.collect()

render_graph(nodes_list, edges_list, title="🕸️ Our Graph — all nodes look the same / กราฟของเรา — ทุกโหนดเหมือนกัน")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. PageRank — Who is the most important? / ใครสำคัญที่สุด?
# MAGIC
# MAGIC **PageRank** measures how **important** or **central** a node is.
# MAGIC
# MAGIC **PageRank** วัดว่าโหนด **สำคัญ** หรือ **เป็นศูนย์กลาง** แค่ไหน
# MAGIC
# MAGIC The idea: a node is important if **many other important nodes point to it**.
# MAGIC It was invented by Google to rank web pages.
# MAGIC
# MAGIC แนวคิด: โหนดสำคัญถ้า **โหนดสำคัญหลายตัวชี้ไปหามัน**
# MAGIC ถูกคิดค้นโดย Google เพื่อจัดอันดับหน้าเว็บ
# MAGIC
# MAGIC ![Image title](https://i.makeagif.com/media/10-02-2022/0O9Ctc.gif)
# MAGIC
# MAGIC **In our graph:** Kim connects both groups → highest PageRank.
# MAGIC Alice and Frank are hubs within their groups → also high.
# MAGIC Leo and Mia are isolated → lowest.
# MAGIC
# MAGIC **ในกราฟของเรา:** Kim เชื่อมทั้งสองกลุ่ม → PageRank สูงสุด
# MAGIC Alice และ Frank เป็นศูนย์กลางในกลุ่ม → สูงเช่นกัน
# MAGIC Leo และ Mia ถูกแยก → ต่ำสุด
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

pr = g.pageRank(resetProbability=0.15, maxIter=10)
pr_results = {row["id"]: round(row["pagerank"], 3) for row in pr.vertices.collect()}

# Display scores sorted
print("PageRank Scores (higher = more important):")
for name, score in sorted(pr_results.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 15)
    print(f"  {name:8s} {score:.3f}  {bar}")

# COMMAND ----------

# Visualize: node SIZE = PageRank (bigger = more important)
max_pr = max(pr_results.values())
node_sizes = {n: 12 + (pr_results[n] / max_pr) * 40 for n in pr_results}

# Color gradient: low=blue, high=red
def pr_color(score, max_score):
    ratio = score / max_score
    if ratio > 0.7: return "#ef4444"   # Red — very important
    if ratio > 0.4: return "#f97316"   # Orange — important
    if ratio > 0.2: return "#eab308"   # Yellow — moderate
    return "#60a5fa"                    # Blue — low

node_colors = {n: pr_color(pr_results[n], max_pr) for n in pr_results}

render_graph(nodes_list, edges_list,
    title="📊 PageRank — node size = importance / ขนาดโหนด = ความสำคัญ",
    node_colors=node_colors, node_sizes=node_sizes)

# COMMAND ----------

# MAGIC %md
# MAGIC **Result / ผลลัพธ์:** Kim is the largest node — she's the bridge between both groups, so information
# MAGIC (or money, or influence) must flow through her. This makes her the most central person in the network.
# MAGIC
# MAGIC Kim เป็นโหนดใหญ่สุด — เธอเป็นสะพานระหว่างสองกลุ่ม ข้อมูล (หรือเงิน หรืออิทธิพล) ต้องไหลผ่านเธอ
# MAGIC ทำให้เธอเป็นคนสำคัญที่สุดในเครือข่าย

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Connected Components — Who belongs together? / ใครอยู่กลุ่มเดียวกัน?
# MAGIC
# MAGIC **Connected Components** finds groups of nodes that can reach each other
# MAGIC through any path. If you can walk from A to B (through any number of steps),
# MAGIC they are in the same component.
# MAGIC
# MAGIC **Connected Components** ค้นหากลุ่มของโหนดที่สามารถเข้าถึงกันได้
# MAGIC ผ่านเส้นทางใดก็ได้ ถ้าเดินจาก A ไป B ได้ (ผ่านกี่ขั้นก็ได้)
# MAGIC ทั้งสองอยู่ในกลุ่มเดียวกัน
# MAGIC
# MAGIC **In our graph:** The main group (Alice through Nora) is one component.
# MAGIC Leo and Mia are a separate component — they can't reach anyone else.
# MAGIC
# MAGIC **ในกราฟของเรา:** กลุ่มหลัก (Alice ถึง Nora) เป็นหนึ่งกลุ่ม
# MAGIC Leo และ Mia เป็นกลุ่มแยก — เข้าถึงคนอื่นไม่ได้

# COMMAND ----------

cc = g.connectedComponents()
cc_results = {row["id"]: row["component"] for row in cc.collect()}

# Map component IDs to colors
unique_components = list(set(cc_results.values()))
comp_colors = ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#ef4444"]
comp_color_map = {comp: comp_colors[i % len(comp_colors)] for i, comp in enumerate(unique_components)}
node_colors = {n: comp_color_map[cc_results[n]] for n in cc_results}

print(f"Found {len(unique_components)} connected components:")
for comp in unique_components:
    members = [n for n, c in cc_results.items() if c == comp]
    print(f"  Component: {', '.join(sorted(members))}")

render_graph(nodes_list, edges_list,
    title="🔗 Connected Components — same color = same group / สีเดียวกัน = กลุ่มเดียวกัน",
    node_colors=node_colors)

# COMMAND ----------

# MAGIC %md
# MAGIC **Result / ผลลัพธ์:** Two components clearly visible — the main network (blue) and the isolated pair (green).
# MAGIC Even though Kim is the only bridge, the left and right groups are still in ONE component because
# MAGIC you can walk between them through Kim.
# MAGIC
# MAGIC สองกลุ่มชัดเจน — เครือข่ายหลัก (น้ำเงิน) และคู่แยก (เขียว)
# MAGIC แม้ Kim จะเป็นสะพานเดียว กลุ่มซ้ายและขวายังอยู่ในกลุ่มเดียวกัน
# MAGIC เพราะเดินถึงกันได้ผ่าน Kim

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Triangle Count — Who is in a tight-knit group? / ใครอยู่ในกลุ่มแน่นแฟ้น?
# MAGIC
# MAGIC A **triangle** means: A knows B, B knows C, **and** C knows A.
# MAGIC Three people who all know each other form a tight bond.
# MAGIC
# MAGIC **สามเหลี่ยม** หมายความว่า: A รู้จัก B, B รู้จัก C, **และ** C รู้จัก A
# MAGIC สามคนที่รู้จักกันทั้งหมดสร้างพันธะที่แน่นแฟ้น
# MAGIC
# MAGIC **High triangle count** = this person is part of many tight-knit groups.
# MAGIC **Low or zero** = this person has loose connections.
# MAGIC
# MAGIC **จำนวนสามเหลี่ยมสูง** = คนนี้อยู่ในกลุ่มแน่นแฟ้นหลายกลุ่ม
# MAGIC **ต่ำหรือศูนย์** = คนนี้มีความเชื่อมโยงหลวมๆ
# MAGIC
# MAGIC
# MAGIC ![image_1773661068314.png](./image_1773661068314.png "image_1773661068314.png")

# COMMAND ----------

tc = g.triangleCount()
tc_results = {row["id"]: row["count"] for row in tc.collect()}

# Display
print("Triangle Count (higher = more tight-knit groups):")
for name, count in sorted(tc_results.items(), key=lambda x: -x[1]):
    bar = "▲" * count
    print(f"  {name:8s} {count:2d}  {bar}")

# COMMAND ----------

# Visualize: color intensity by triangle count
max_tc = max(tc_results.values()) or 1

def tc_color(count, max_count):
    if count == 0: return "#334155"    # Dark gray — no triangles
    ratio = count / max_count
    if ratio > 0.7: return "#ef4444"   # Red — many triangles
    if ratio > 0.4: return "#f97316"   # Orange
    if ratio > 0.1: return "#eab308"   # Yellow
    return "#60a5fa"                    # Blue — few

node_colors = {n: tc_color(tc_results[n], max_tc) for n in tc_results}
node_sizes = {n: 12 + (tc_results[n] / max_tc) * 35 for n in tc_results}

render_graph(nodes_list, edges_list,
    title="▲ Triangle Count — bigger & redder = more triangles / ใหญ่ & แดง = สามเหลี่ยมมากกว่า",
    node_colors=node_colors, node_sizes=node_sizes)

# COMMAND ----------

# MAGIC %md
# MAGIC **Result / ผลลัพธ์:** The nodes inside each dense group have the most triangles (Alice, Bob, Carol, Dave
# MAGIC in the left group; Frank, Grace, Hank, Ivy in the right group). Kim has fewer triangles because
# MAGIC she connects to nodes in different groups who don't know each other. Leo and Mia have zero.
# MAGIC
# MAGIC โหนดในกลุ่มแน่นแฟ้นมีสามเหลี่ยมมากที่สุด Kim มีน้อยกว่าเพราะเชื่อมกับคนต่างกลุ่ม
# MAGIC ที่ไม่รู้จักกัน Leo และ Mia เป็นศูนย์

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Community Detection (Label Propagation) — Who clusters together naturally?
# MAGIC ## การตรวจจับชุมชน — ใครจัดกลุ่มร่วมกันตามธรรมชาติ?
# MAGIC
# MAGIC **Label Propagation** discovers natural **sub-communities** — groups of nodes
# MAGIC that are more densely connected to each other than to the rest.
# MAGIC
# MAGIC **Label Propagation** ค้นพบ **กลุ่มย่อยตามธรรมชาติ** — กลุ่มของโหนด
# MAGIC ที่เชื่อมต่อกันแน่นแฟ้นกว่ากับส่วนที่เหลือ
# MAGIC
# MAGIC Unlike Connected Components (which only finds fully disconnected groups),
# MAGIC Community Detection can split a connected network into sub-groups.
# MAGIC
# MAGIC ต่างจาก Connected Components (ที่พบเฉพาะกลุ่มแยกขาดจากกัน)
# MAGIC Community Detection สามารถแบ่งเครือข่ายที่เชื่อมต่อกันออกเป็นกลุ่มย่อยได้
# MAGIC
# MAGIC
# MAGIC ![Image title](https://i0.wp.com/crowintelligence.org/wp-content/uploads/2020/03/image00.gif?fit=553%2C387&ssl=1)
# MAGIC

# COMMAND ----------

lp = g.labelPropagation(maxIter=5)
lp_results = {row["id"]: row["label"] for row in lp.collect()}

# Map community labels to colors
unique_labels = list(set(lp_results.values()))
label_colors = ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#ef4444", "#06b6d4", "#eab308"]
label_color_map = {lab: label_colors[i % len(label_colors)] for i, lab in enumerate(unique_labels)}
node_colors = {n: label_color_map[lp_results[n]] for n in lp_results}

print(f"Found {len(unique_labels)} communities:")
for lab in unique_labels:
    members = sorted([n for n, l in lp_results.items() if l == lab])
    print(f"  Community: {', '.join(members)}")

render_graph(nodes_list, edges_list,
    title="🏘️ Communities — same color = same community / สีเดียวกัน = ชุมชนเดียวกัน",
    node_colors=node_colors)

# COMMAND ----------

# MAGIC %md
# MAGIC **Result / ผลลัพธ์:** The algorithm found the natural sub-groups!
# MAGIC The left group and right group are now different colors, even though they're connected through Kim.
# MAGIC The algorithm detected that connections within each group are much denser than across groups.
# MAGIC
# MAGIC อัลกอริทึมพบกลุ่มย่อยตามธรรมชาติ!
# MAGIC กลุ่มซ้ายและกลุ่มขวาตอนนี้เป็นคนละสี แม้จะเชื่อมกันผ่าน Kim
# MAGIC อัลกอริทึมตรวจพบว่าการเชื่อมต่อภายในแต่ละกลุ่มแน่นแฟ้นกว่าระหว่างกลุ่ม

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary / สรุป
# MAGIC
# MAGIC | Algorithm | Question it answers | คำถามที่ตอบ |
# MAGIC |---|---|---|
# MAGIC | **PageRank** | Who is the most important/central? | ใครสำคัญ/เป็นศูนย์กลางที่สุด? |
# MAGIC | **Connected Components** | Which groups are completely separate? | กลุ่มไหนแยกขาดจากกันโดยสิ้นเชิง? |
# MAGIC | **Triangle Count** | Who is in tight-knit clusters? | ใครอยู่ในกลุ่มแน่นแฟ้น? |
# MAGIC | **Community Detection** | What are the natural sub-groups? | กลุ่มย่อยตามธรรมชาติคืออะไร? |
# MAGIC
# MAGIC These algorithms work on **any** graph — social networks, financial transactions,
# MAGIC supply chains, telecommunications, biology, and more.
# MAGIC
# MAGIC อัลกอริทึมเหล่านี้ใช้ได้กับ **ทุก** กราฟ — เครือข่ายสังคม ธุรกรรมทางการเงิน
# MAGIC ห่วงโซ่อุปทาน โทรคมนาคม ชีววิทยา และอื่นๆ