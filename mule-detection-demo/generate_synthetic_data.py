# Databricks notebook source

# MAGIC %md
# MAGIC # 🏦 Mule Detection Demo — Synthetic Data Generation
# MAGIC # สร้างข้อมูลจำลองสำหรับการตรวจจับบัญชีม้า
# MAGIC
# MAGIC This notebook generates realistic synthetic data for a Thai bank mule detection demo.
# MAGIC
# MAGIC โน้ตบุ๊คนี้สร้างข้อมูลจำลองที่สมจริงสำหรับการสาธิตการตรวจจับบัญชีม้าของธนาคารไทย
# MAGIC
# MAGIC **Tables created / ตารางที่สร้าง:**
# MAGIC - `silver_customers` — 1,000 accounts (ลูกค้า 1,000 บัญชี)
# MAGIC - `silver_transactions` — ~20,000 transactions (ธุรกรรม ~20,000 รายการ)
# MAGIC - `silver_device_logins` — ~5,000 logins (การเข้าสู่ระบบ ~5,000 ครั้ง)
# MAGIC - `silver_bot_mule_list` — ~25 BOT-confirmed mules (บัญชีม้ายืนยันโดย ธปท. ~25 บัญชี)
# MAGIC - `silver_shared_contacts` — ~1,000 contact records (ข้อมูลติดต่อ ~1,000 รายการ)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup / การตั้งค่า

# COMMAND ----------

CATALOG = "vn"
SCHEMA = "mule_demo"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

import random
import uuid
from datetime import datetime, timedelta
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import *

random.seed(42)

# Date range for transactions: last 90 days
END_DATE = datetime(2025, 12, 15)
START_DATE = END_DATE - timedelta(days=90)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Customers / สร้างข้อมูลลูกค้า
# MAGIC
# MAGIC We create 1,000 bank accounts with Thai names and demographics.
# MAGIC Accounts are assigned roles: **normal**, **mule**, or **victim**.
# MAGIC
# MAGIC สร้างบัญชีธนาคาร 1,000 บัญชีพร้อมชื่อไทยและข้อมูลประชากร
# MAGIC บัญชีถูกกำหนดบทบาท: **ปกติ**, **บัญชีม้า**, หรือ **เหยื่อ**

# COMMAND ----------

# Thai first and last names
THAI_FIRST_NAMES_M = [
    "สมชาย", "สุรชัย", "วิชัย", "ประเสริฐ", "สมศักดิ์", "อนุชา", "ธนกฤต", "ภูมิพัฒน์",
    "กิตติพงศ์", "ณัฐพล", "พีรพัฒน์", "จิรายุ", "ศุภกร", "ธีรภัทร", "ปิยะ", "วรพล",
    "ชาญวิทย์", "เกียรติศักดิ์", "นพดล", "อภิสิทธิ์", "รัฐพล", "ชัยวัฒน์", "พงศ์พัฒน์",
    "สุทธิพงศ์", "อัครพล", "ธนวัฒน์", "กฤษณะ", "ปรัชญา", "วีรยุทธ", "สราวุธ"
]
THAI_FIRST_NAMES_F = [
    "สมหญิง", "สุภาพร", "วิภาวดี", "นภาพร", "กัญญา", "ปิยะดา", "ธนพร", "พิมพ์ชนก",
    "ณัฐธิดา", "ศิริพร", "อรุณี", "พรทิพย์", "จันทร์เพ็ญ", "สุกัญญา", "รัตนา", "มาลี",
    "วันดี", "นิภา", "อัจฉรา", "พัชรี", "ลัดดา", "สุวรรณา", "กมลวรรณ", "ดวงใจ",
    "ชุติมา", "เบญจมาศ", "อรทัย", "พรรณี", "จุฑามาศ", "ปวีณา"
]
THAI_LAST_NAMES = [
    "สุขสวัสดิ์", "ทองดี", "แก้วมณี", "พงษ์สวัสดิ์", "ศรีสุข", "วงศ์สกุล", "เจริญผล",
    "บุญมาก", "รัตนกุล", "สิทธิชัย", "ชัยสิทธิ์", "ประเสริฐศรี", "วิไลวรรณ", "มงคลชัย",
    "พิทักษ์", "อมรรัตน์", "สมบูรณ์", "ศักดิ์สิทธิ์", "กิจเจริญ", "พานิชย์",
    "ธนาคม", "วัฒนา", "ลิ้มประเสริฐ", "แซ่ตั้ง", "แซ่ลิ้ม", "จงรักษ์", "ไชยวงค์",
    "คำแก้ว", "แสงจันทร์", "ใจดี", "สว่างวงศ์", "ภูวนาท", "ตันติกุล", "รุ่งเรือง",
    "พลอยแก้ว", "นาคสุข", "ชูศรี", "อินทร์แก้ว", "ศรีวิชัย", "เทพพิทักษ์"
]

OCCUPATIONS_NORMAL = ["office_worker", "merchant", "teacher", "engineer", "nurse", "driver",
                       "chef", "accountant", "lawyer", "civil_servant"]
OCCUPATIONS_MULE = ["student", "unemployed", "freelance", "part_time_worker"]

PROVINCES = ["Bangkok", "Chiang Mai", "Chonburi", "Nakhon Ratchasima", "Khon Kaen",
             "Songkhla", "Phuket", "Udon Thani", "Surat Thani", "Chiang Rai",
             "Nonthaburi", "Pathum Thani", "Samut Prakan", "Rayong", "Lampang"]

# Syndicate definitions: 4 clusters
SYNDICATES = [
    {"id": 1, "name": "BKK Ring", "size": 18, "province": "Bangkok", "activation_date": END_DATE - timedelta(days=21)},
    {"id": 2, "name": "Chiang Mai Ring", "size": 16, "province": "Chiang Mai", "activation_date": END_DATE - timedelta(days=18)},
    {"id": 3, "name": "Eastern Ring", "size": 14, "province": "Chonburi", "activation_date": END_DATE - timedelta(days=25)},
    {"id": 4, "name": "Isan Ring", "size": 12, "province": "Khon Kaen", "activation_date": END_DATE - timedelta(days=15)},
]

TOTAL_MULES = sum(s["size"] for s in SYNDICATES)  # 60
NUM_VICTIMS = 100
NUM_NORMAL = 1000 - TOTAL_MULES - NUM_VICTIMS  # 840

print(f"Mules: {TOTAL_MULES}, Victims: {NUM_VICTIMS}, Normal: {NUM_NORMAL}")

# COMMAND ----------

def make_thai_name():
    if random.random() < 0.5:
        first = random.choice(THAI_FIRST_NAMES_M)
    else:
        first = random.choice(THAI_FIRST_NAMES_F)
    last = random.choice(THAI_LAST_NAMES)
    return f"{first} {last}"

def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 1)))

customers = []
account_counter = 1000

# --- Generate MULE accounts ---
mule_ids = []
mule_syndicate_map = {}  # account_id -> syndicate_id

mule_role_map = {}  # account_id -> "hub", "recruiter", "outer"

for synd in SYNDICATES:
    # Syndicate structure: 1 hub + 2 recruiters + rest are outer mules
    hub_id = None
    recruiter_count = 0
    for i in range(synd["size"]):
        aid = f"A-{account_counter}"
        account_counter += 1
        is_hub = (i == 0)

        if i == 0:
            mule_role = "hub"
        elif i <= 2:
            mule_role = "recruiter"
        else:
            mule_role = "outer"

        age = random.randint(18, 25) if random.random() < 0.7 else random.randint(26, 35)
        occ = random.choice(OCCUPATIONS_MULE)
        income = random.choice([8000, 10000, 12000, 15000])
        # Accounts opened recently (within last 6 months), clustered around same time
        open_date = random_date(END_DATE - timedelta(days=180), END_DATE - timedelta(days=30))

        customers.append(Row(
            account_id=aid, customer_name=make_thai_name(), age=age, occupation=occ,
            monthly_income=float(income), income_band="<15K",
            province=synd["province"] if random.random() < 0.6 else random.choice(PROVINCES),
            account_open_date=open_date.strftime("%Y-%m-%d"),
            account_type="savings", kyc_risk_level="low",
            role="mule", syndicate_id=synd["id"], is_hub=is_hub
        ))
        mule_ids.append(aid)
        mule_syndicate_map[aid] = synd["id"]
        mule_role_map[aid] = mule_role
        if is_hub:
            hub_id = aid

# --- Generate VICTIM accounts ---
victim_ids = []
for _ in range(NUM_VICTIMS):
    aid = f"A-{account_counter}"
    account_counter += 1
    age = random.randint(25, 65)
    occ = random.choice(OCCUPATIONS_NORMAL)
    income = random.choice([25000, 35000, 45000, 60000, 80000, 120000])
    if income < 15000: band = "<15K"
    elif income < 30000: band = "15-30K"
    elif income < 60000: band = "30-60K"
    else: band = "60K+"
    open_date = random_date(END_DATE - timedelta(days=1800), END_DATE - timedelta(days=365))

    customers.append(Row(
        account_id=aid, customer_name=make_thai_name(), age=age, occupation=occ,
        monthly_income=float(income), income_band=band,
        province=random.choice(PROVINCES),
        account_open_date=open_date.strftime("%Y-%m-%d"),
        account_type=random.choice(["savings", "current"]),
        kyc_risk_level="low",
        role="victim", syndicate_id=0, is_hub=False
    ))
    victim_ids.append(aid)

# --- Generate NORMAL accounts ---
normal_ids = []
for _ in range(NUM_NORMAL):
    aid = f"A-{account_counter}"
    account_counter += 1
    age = random.randint(20, 70)
    occ = random.choice(OCCUPATIONS_NORMAL + ["student", "retired"])
    income = random.choice([15000, 20000, 25000, 35000, 45000, 60000, 80000, 100000])
    if income < 15000: band = "<15K"
    elif income < 30000: band = "15-30K"
    elif income < 60000: band = "30-60K"
    else: band = "60K+"
    open_date = random_date(END_DATE - timedelta(days=2500), END_DATE - timedelta(days=60))

    customers.append(Row(
        account_id=aid, customer_name=make_thai_name(), age=age, occupation=occ,
        monthly_income=float(income), income_band=band,
        province=random.choice(PROVINCES),
        account_open_date=open_date.strftime("%Y-%m-%d"),
        account_type=random.choice(["savings", "current"]),
        kyc_risk_level=random.choice(["low", "low", "low", "medium"]),
        role="normal", syndicate_id=0, is_hub=False
    ))
    normal_ids.append(aid)

# Create DataFrame (keep role/syndicate_id/is_hub as internal columns for later use, drop before saving)
customers_df = spark.createDataFrame(customers)
print(f"Total customers: {customers_df.count()}")
customers_df.groupBy("role").count().show()
customers_df.where("role = 'mule'").groupBy("syndicate_id").count().orderBy("syndicate_id").show()

# COMMAND ----------

# Save silver_customers (drop internal columns)
silver_customers = customers_df.drop("role", "syndicate_id", "is_hub")
silver_customers.write.mode("overwrite").saveAsTable("silver_customers")

display(spark.sql("SELECT * FROM silver_customers LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Transactions / สร้างธุรกรรม
# MAGIC
# MAGIC We generate ~20,000 transactions with distinct patterns:
# MAGIC - **Normal:** regular peer-to-peer, salary, bills
# MAGIC - **Victim → Mule:** scam transfers (medium/large amounts to mule accounts)
# MAGIC - **Mule → Mule:** rapid pass-through within syndicates
# MAGIC - **Mule → External:** outflow to cash-out
# MAGIC
# MAGIC สร้างธุรกรรม ~20,000 รายการที่มีรูปแบบเฉพาะ:
# MAGIC - **ปกติ:** โอนระหว่างบุคคล เงินเดือน ค่าใช้จ่าย
# MAGIC - **เหยื่อ → บัญชีม้า:** โอนเงินจากการหลอกลวง
# MAGIC - **บัญชีม้า → บัญชีม้า:** เงินผ่านเร็วภายในเครือข่าย
# MAGIC - **บัญชีม้า → ภายนอก:** ถอนเงินออก

# COMMAND ----------

transactions = []

# Helper: get mules in a syndicate
def get_syndicate_mules(synd_id):
    return [c.account_id for c in customers if c.syndicate_id == synd_id]

def get_syndicate_hub(synd_id):
    return [c.account_id for c in customers if c.syndicate_id == synd_id and c.is_hub][0]

syndicate_activation = {s["id"]: s["activation_date"] for s in SYNDICATES}

# --- 1. Normal transactions (~14,000) ---
all_account_ids = [c.account_id for c in customers]

for _ in range(14000):
    sender = random.choice(normal_ids + victim_ids)
    receiver = random.choice(normal_ids)
    while receiver == sender:
        receiver = random.choice(normal_ids)
    amount = round(random.uniform(50, 15000), 2)
    ts = random_date(START_DATE, END_DATE)
    ts = ts.replace(hour=random.randint(6, 23), minute=random.randint(0, 59))
    channel = random.choices(["promptpay", "mobile", "atm", "branch"],
                              weights=[50, 30, 10, 10])[0]
    transactions.append(Row(
        txn_id=str(uuid.uuid4()), from_account=sender, to_account=receiver,
        amount_thb=amount, txn_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
        channel=channel
    ))

# --- 2. Victim → Mule transactions (~800, 1-5 per victim) ---
for vid in victim_ids:
    # Each victim sends to 1-3 mules in same syndicate
    target_synd = random.choice(SYNDICATES)
    target_mules = get_syndicate_mules(target_synd["id"])
    num_txns = random.randint(1, 5)
    for _ in range(num_txns):
        receiver = random.choice(target_mules)
        amount = round(random.uniform(5000, 200000), 2)
        activation = syndicate_activation[target_synd["id"]]
        ts = random_date(activation, END_DATE)
        ts = ts.replace(hour=random.randint(8, 22), minute=random.randint(0, 59))
        transactions.append(Row(
            txn_id=str(uuid.uuid4()), from_account=vid, to_account=receiver,
            amount_thb=amount, txn_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            channel=random.choice(["promptpay", "mobile"])
        ))

# --- 3. Mule → Mule (pass-through within syndicate) (~2,500) ---
for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    hub = get_syndicate_hub(synd["id"])
    activation = syndicate_activation[synd["id"]]

    for _ in range(600):
        # Pattern: regular mule → hub → another mule or out
        sender = random.choice(mules)
        if random.random() < 0.6:
            # Send to hub
            receiver = hub
        else:
            # Hub sends outward
            receiver = random.choice(mules)
            while receiver == sender:
                receiver = random.choice(mules)

        amount = round(random.uniform(10000, 500000), 2)
        ts = random_date(activation, END_DATE)
        # Rapid: transactions happen within hours of each other
        ts = ts.replace(hour=random.randint(0, 23), minute=random.randint(0, 59))
        transactions.append(Row(
            txn_id=str(uuid.uuid4()), from_account=sender, to_account=receiver,
            amount_thb=amount, txn_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            channel="promptpay"
        ))

# --- 4. Mule → External (cash-out to normal-looking accounts) (~1,200) ---
for synd in SYNDICATES:
    hub = get_syndicate_hub(synd["id"])
    mules = get_syndicate_mules(synd["id"])
    activation = syndicate_activation[synd["id"]]

    for _ in range(300):
        sender = random.choice([hub, hub, random.choice(mules)])  # hub does most outflows
        receiver = random.choice(normal_ids)  # goes to "clean" accounts (could be external)
        amount = round(random.uniform(50000, 800000), 2)
        ts = random_date(activation, END_DATE)
        ts = ts.replace(hour=random.randint(0, 5), minute=random.randint(0, 59))  # late night
        transactions.append(Row(
            txn_id=str(uuid.uuid4()), from_account=sender, to_account=receiver,
            amount_thb=amount, txn_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            channel=random.choice(["promptpay", "atm"])
        ))

# --- 5. Triangle patterns (A→B→C→A) within syndicates (~600) ---
for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    activation = syndicate_activation[synd["id"]]
    if len(mules) >= 3:
        for _ in range(150):
            trio = random.sample(mules, 3)
            base_ts = random_date(activation, END_DATE)
            amount = round(random.uniform(20000, 200000), 2)
            for i in range(3):
                sender = trio[i]
                receiver = trio[(i + 1) % 3]
                ts = base_ts + timedelta(hours=random.randint(1, 6) * (i + 1))
                transactions.append(Row(
                    txn_id=str(uuid.uuid4()), from_account=sender, to_account=receiver,
                    amount_thb=amount * (0.95 + random.uniform(0, 0.1)),
                    txn_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                    channel="promptpay"
                ))

print(f"Total transactions: {len(transactions)}")

# COMMAND ----------

txn_df = spark.createDataFrame(transactions)
txn_df = txn_df.withColumn("txn_timestamp", F.to_timestamp("txn_timestamp"))
txn_df = txn_df.withColumn("amount_thb", F.round("amount_thb", 2))

txn_df.write.mode("overwrite").saveAsTable("silver_transactions")

# Show summary
print(f"Transactions written: {txn_df.count()}")
display(txn_df.groupBy("channel").agg(
    F.count("*").alias("count"),
    F.round(F.sum("amount_thb"), 2).alias("total_thb"),
    F.round(F.avg("amount_thb"), 2).alias("avg_thb")
).orderBy("count", ascending=False))

# COMMAND ----------

# Show top receivers by volume (should be mule hubs)
display(
    txn_df.groupBy("to_account").agg(
        F.count("*").alias("incoming_txn_count"),
        F.round(F.sum("amount_thb"), 0).alias("total_incoming_thb"),
        F.countDistinct("from_account").alias("unique_senders")
    ).orderBy("total_incoming_thb", ascending=False).limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Device Logins / สร้างข้อมูลอุปกรณ์
# MAGIC
# MAGIC **Realistic layered device sharing** — criminals are smart:
# MAGIC - **Hub/operator:** dedicated device shared with only 2-3 inner-circle mules
# MAGIC - **Recruiters:** own device shared with 2-3 mules they recruited
# MAGIC - **Outer mules:** personal devices only — look completely normal
# MAGIC
# MAGIC Device links alone catch only the inner circle (~5-6 accounts).
# MAGIC Graph algorithms (PageRank, two-hop ratio) are needed to find the rest.
# MAGIC
# MAGIC **การแชร์อุปกรณ์แบบแบ่งชั้นที่สมจริง** — อาชญากรฉลาด:
# MAGIC - **ศูนย์กลาง:** อุปกรณ์เฉพาะแชร์กับบัญชีม้าวงในเพียง 2-3 บัญชี
# MAGIC - **ผู้คัดเลือก:** อุปกรณ์ของตัวเองแชร์กับ 2-3 บัญชีที่คัดเลือกมา
# MAGIC - **บัญชีม้าชั้นนอก:** อุปกรณ์ส่วนตัวเท่านั้น — ดูปกติอย่างสมบูรณ์

# COMMAND ----------

logins = []

# Build layered device structure per syndicate
# Hub device: shared with hub + 2 inner mules
# Recruiter devices: each shared with recruiter + 2-3 outer mules they recruited
syndicate_device_assignments = {}  # aid -> list of devices

for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    hub = get_syndicate_hub(synd["id"])
    recruiters = [m for m in mules if mule_role_map[m] == "recruiter"]
    outers = [m for m in mules if mule_role_map[m] == "outer"]

    # Hub operator's device — shared with hub + first 2 outers (inner circle)
    hub_device = f"DEV-OP-{synd['id']}"
    inner_circle = [hub] + outers[:2]
    for aid in inner_circle:
        syndicate_device_assignments[aid] = [hub_device]

    # Each recruiter has their own device, shared with 2-3 of their recruited outers
    remaining_outers = outers[2:]
    for j, rec in enumerate(recruiters):
        rec_device = f"DEV-REC-{synd['id']}-{j}"
        # Recruiter gets their device
        syndicate_device_assignments[rec] = [rec_device]
        # Assign 2-3 outer mules to this recruiter's device
        num_assigned = min(random.randint(2, 3), len(remaining_outers))
        assigned = remaining_outers[:num_assigned]
        remaining_outers = remaining_outers[num_assigned:]
        for aid in assigned:
            syndicate_device_assignments[aid] = [rec_device]

    # Remaining outer mules: personal device ONLY (no sharing — they look normal)
    for aid in remaining_outers:
        syndicate_device_assignments[aid] = [f"DEV-{aid}-personal"]

# Normal + victim devices (personal, 1-2 each)
personal_devices = {}
for aid in normal_ids + victim_ids:
    num_devices = random.choices([1, 2], weights=[70, 30])[0]
    personal_devices[aid] = [f"DEV-{aid}-{i}" for i in range(num_devices)]

# Generate logins for normal + victim accounts
for aid in normal_ids + victim_ids:
    num_logins = random.randint(2, 8)
    for _ in range(num_logins):
        dev = random.choice(personal_devices[aid])
        ts = random_date(START_DATE, END_DATE)
        ts = ts.replace(hour=random.randint(6, 23), minute=random.randint(0, 59))
        logins.append(Row(
            login_id=str(uuid.uuid4()), account_id=aid,
            device_fingerprint=dev,
            ip_address=f"203.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            login_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            os=random.choice(["Android", "Android", "iOS"])
        ))

# Generate logins for mule accounts (layered device sharing)
for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    activation = syndicate_activation[synd["id"]]

    for aid in mules:
        assigned_devices = syndicate_device_assignments.get(aid, [f"DEV-{aid}-personal"])
        num_logins = random.randint(3, 8)
        for _ in range(num_logins):
            # Mule mostly uses their assigned device
            dev = random.choice(assigned_devices)
            # Small chance (10%) of using a personal fallback device too
            if random.random() < 0.1:
                dev = f"DEV-{aid}-personal"

            ts = random_date(activation, END_DATE)
            ts = ts.replace(hour=random.randint(0, 23), minute=random.randint(0, 59))
            logins.append(Row(
                login_id=str(uuid.uuid4()), account_id=aid,
                device_fingerprint=dev,
                ip_address=f"10.{synd['id']}.{random.randint(1,255)}.{random.randint(1,255)}",
                login_timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                os=random.choice(["Android", "Android", "Android", "iOS"])
            ))

print(f"Total logins: {len(logins)}")
print(f"\nDevice sharing structure per syndicate:")
for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    hub = get_syndicate_hub(synd["id"])
    shared_count = sum(1 for m in mules if not syndicate_device_assignments.get(m, [""])[0].startswith(f"DEV-{m}"))
    personal_count = sum(1 for m in mules if syndicate_device_assignments.get(m, [""])[0].startswith(f"DEV-{m}"))
    print(f"  Syndicate {synd['id']}: {shared_count} with shared devices, {personal_count} with personal only")

# COMMAND ----------

logins_df = spark.createDataFrame(logins)
logins_df = logins_df.withColumn("login_timestamp", F.to_timestamp("login_timestamp"))
logins_df.write.mode("overwrite").saveAsTable("silver_device_logins")

# Show devices shared across multiple accounts (should show syndicate devices)
display(
    logins_df.groupBy("device_fingerprint").agg(
        F.countDistinct("account_id").alias("num_accounts")
    ).where("num_accounts > 1").orderBy("num_accounts", ascending=False).limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate BOT Mule List / รายชื่อบัญชีม้ายืนยันโดย ธปท.
# MAGIC
# MAGIC The Bank of Thailand (BOT) and AMLO have confirmed ~25 accounts as mules.
# MAGIC These are a **subset** of the actual 60 mules — the rest are undiscovered.
# MAGIC This is the "seed" for the graph analysis.
# MAGIC
# MAGIC ธนาคารแห่งประเทศไทย (ธปท.) และ ปปง. ยืนยัน ~25 บัญชีเป็นบัญชีม้า
# MAGIC เหล่านี้เป็น **ส่วนหนึ่ง** ของบัญชีม้าจริง 60 บัญชี — ที่เหลือยังไม่ถูกค้นพบ

# COMMAND ----------

# Select ~25 mules as BOT-confirmed (spread across syndicates, but NOT the hubs)
bot_list = []
for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    hub = get_syndicate_hub(synd["id"])
    # Exclude hubs — they should be "undiscovered" for the demo story
    non_hub_mules = [m for m in mules if m != hub]
    # Pick ~40% of non-hub mules
    num_to_flag = max(3, int(len(non_hub_mules) * 0.4))
    flagged = random.sample(non_hub_mules, num_to_flag)

    for aid in flagged:
        bot_list.append(Row(
            account_id=aid,
            flagged_date=random_date(END_DATE - timedelta(days=30), END_DATE).strftime("%Y-%m-%d"),
            reason=random.choice(["scam_proceeds", "money_laundering", "syndicate_link"])
        ))

bot_df = spark.createDataFrame(bot_list)
bot_df.write.mode("overwrite").saveAsTable("silver_bot_mule_list")

print(f"BOT confirmed mules: {bot_df.count()}")
display(bot_df)

# COMMAND ----------

# Verify: hubs should NOT be on the BOT list (key for demo story)
hub_ids = [get_syndicate_hub(s["id"]) for s in SYNDICATES]
bot_account_ids = [r.account_id for r in bot_list]
for h in hub_ids:
    status = "ON BOT LIST" if h in bot_account_ids else "NOT on BOT list (good - undiscovered)"
    print(f"  Hub {h}: {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Shared Contacts / สร้างข้อมูลติดต่อร่วม
# MAGIC
# MAGIC **Realistic contact sharing** — mirrors the layered device structure:
# MAGIC - **Recruiters** used their phone to register 2-3 mules (shared phone at onboarding)
# MAGIC - **Most outer mules** have their own unique phone, email, address
# MAGIC - **Hub** has a completely unique phone — never appears in other accounts
# MAGIC
# MAGIC Phone links only catch recruiter ↔ mule pairs. The rest need graph algorithms.
# MAGIC
# MAGIC **การแชร์ข้อมูลติดต่อที่สมจริง:**
# MAGIC - **ผู้คัดเลือก** ใช้โทรศัพท์ของตนในการลงทะเบียนบัญชีม้า 2-3 บัญชี
# MAGIC - **บัญชีม้าชั้นนอกส่วนใหญ่** มีโทรศัพท์ อีเมล ที่อยู่ของตนเอง
# MAGIC - **ศูนย์กลาง** มีโทรศัพท์เฉพาะตัว ไม่ปรากฏในบัญชีอื่น

# COMMAND ----------

contacts = []

for synd in SYNDICATES:
    mules = get_syndicate_mules(synd["id"])
    hub = get_syndicate_hub(synd["id"])
    recruiters = [m for m in mules if mule_role_map[m] == "recruiter"]
    outers = [m for m in mules if mule_role_map[m] == "outer"]

    # Each recruiter has a phone they used to register some mules
    recruiter_phones = {}
    remaining_outers = list(outers)
    for rec in recruiters:
        rec_phone = f"08{random.randint(1,9)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
        recruiter_phones[rec] = rec_phone
        # 2-3 mules registered with recruiter's phone
        num_assigned = min(random.randint(2, 3), len(remaining_outers))
        assigned = remaining_outers[:num_assigned]
        remaining_outers = remaining_outers[num_assigned:]
        # Recruiter's own contact
        contacts.append(Row(
            account_id=rec,
            phone_number=rec_phone,
            email=f"{rec.lower().replace('-','')}@{random.choice(['gmail.com', 'hotmail.com'])}",
            registered_address=f"{random.randint(1,999)} ซอย {random.randint(1,50)} {synd['province']}"
        ))
        # Assigned mules share the recruiter's phone
        for aid in assigned:
            contacts.append(Row(
                account_id=aid,
                phone_number=rec_phone,  # shared with recruiter
                email=f"{aid.lower().replace('-','')}@{random.choice(['gmail.com', 'hotmail.com', 'yahoo.com'])}",
                registered_address=f"{random.randint(1,999)} ถนน {random.choice(['สุขุมวิท', 'พหลโยธิน', 'รัชดาภิเษก', 'ลาดพร้าว'])} {random.choice(PROVINCES)}"
            ))

    # Hub — completely unique contact (operator is careful)
    contacts.append(Row(
        account_id=hub,
        phone_number=f"08{random.randint(1,9)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        email=f"{hub.lower().replace('-','')}@{random.choice(['gmail.com', 'protonmail.com'])}",
        registered_address=f"{random.randint(1,999)} ถนน {random.choice(['สุขุมวิท', 'พหลโยธิน'])} Bangkok"
    ))

    # Remaining outer mules — all unique contacts (look completely normal)
    for aid in remaining_outers:
        contacts.append(Row(
            account_id=aid,
            phone_number=f"08{random.randint(1,9)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            email=f"{aid.lower().replace('-','')}@{random.choice(['gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com'])}",
            registered_address=f"{random.randint(1,999)} ถนน {random.choice(['สุขุมวิท', 'พหลโยธิน', 'รัชดาภิเษก', 'ลาดพร้าว', 'เพชรบุรี', 'สีลม'])} {random.choice(PROVINCES)}"
        ))

# Normal + victim contacts (all unique)
for aid in normal_ids + victim_ids:
    contacts.append(Row(
        account_id=aid,
        phone_number=f"08{random.randint(1,9)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        email=f"{aid.lower().replace('-','')}@{random.choice(['gmail.com', 'hotmail.com', 'outlook.com', 'yahoo.com'])}",
        registered_address=f"{random.randint(1,999)} ถนน {random.choice(['สุขุมวิท', 'พหลโยธิน', 'รัชดาภิเษก', 'ลาดพร้าว', 'เพชรบุรี', 'สีลม'])} {random.choice(PROVINCES)}"
    ))

contacts_df = spark.createDataFrame(contacts)
contacts_df.write.mode("overwrite").saveAsTable("silver_shared_contacts")

print(f"Contacts written: {contacts_df.count()}")

# Show shared phone numbers — should be small clusters (recruiter + 2-3 mules only)
display(
    contacts_df.groupBy("phone_number").agg(
        F.countDistinct("account_id").alias("num_accounts")
    ).where("num_accounts > 1").orderBy("num_accounts", ascending=False).limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Quality Summary / สรุปคุณภาพข้อมูล
# MAGIC
# MAGIC Final verification of all generated tables.
# MAGIC
# MAGIC ตรวจสอบขั้นสุดท้ายของตารางทั้งหมดที่สร้างขึ้น

# COMMAND ----------

print("=" * 60)
print("DATA GENERATION SUMMARY / สรุปการสร้างข้อมูล")
print("=" * 60)

tables = [
    "silver_customers",
    "silver_transactions",
    "silver_device_logins",
    "silver_bot_mule_list",
    "silver_shared_contacts"
]

for t in tables:
    count = spark.table(t).count()
    print(f"  {CATALOG}.{SCHEMA}.{t}: {count:,} rows")

print()
print("Mule syndicate distribution:")
mule_df = customers_df.where("role = 'mule'")
for synd in SYNDICATES:
    count = mule_df.where(f"syndicate_id = {synd['id']}").count()
    hub = get_syndicate_hub(synd["id"])
    print(f"  Syndicate {synd['id']} ({synd['name']}): {count} mules, hub={hub}")

print()
print(f"BOT confirmed: {spark.table('silver_bot_mule_list').count()} / {TOTAL_MULES} total mules")
print(f"Undiscovered mules: {TOTAL_MULES - spark.table('silver_bot_mule_list').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top mule accounts by transaction volume / บัญชีม้าที่มีปริมาณธุรกรรมสูงสุด
# MAGIC
# MAGIC These should be the hub accounts — highest inflow and outflow.
# MAGIC
# MAGIC เหล่านี้ควรเป็นบัญชีศูนย์กลาง — มีเงินเข้าและออกสูงสุด

# COMMAND ----------

# Join transactions with customer roles to show mule activity
mule_activity = (
    spark.table("silver_transactions").alias("t")
    .join(customers_df.select("account_id", "role", "is_hub", "syndicate_id").alias("c"),
          F.col("t.to_account") == F.col("c.account_id"))
    .where("c.role = 'mule'")
    .groupBy("c.account_id", "c.is_hub", "c.syndicate_id")
    .agg(
        F.count("*").alias("incoming_txns"),
        F.round(F.sum("t.amount_thb"), 0).alias("total_incoming_thb"),
        F.countDistinct("t.from_account").alias("unique_senders")
    )
    .orderBy("total_incoming_thb", ascending=False)
)

display(mule_activity.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### ✅ Phase 1 Complete / เฟส 1 เสร็จสมบูรณ์
# MAGIC
# MAGIC All silver tables have been generated. Proceed to **Phase 2: Graph Feature Engineering**.
# MAGIC
# MAGIC ตารางซิลเวอร์ทั้งหมดถูกสร้างแล้ว ดำเนินการต่อใน **เฟส 2: การคำนวณคุณสมบัติกราฟ**
