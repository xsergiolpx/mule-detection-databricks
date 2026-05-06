CREATE TABLE vn.mule_demo.gold_account_risk_graph (
  account_id STRING COLLATE UTF8_BINARY COMMENT 'Unique bank account identifier (format A-NNNN)',
  customer_name STRING COLLATE UTF8_BINARY,
  age BIGINT,
  occupation STRING COLLATE UTF8_BINARY,
  monthly_income DOUBLE,
  income_band STRING COLLATE UTF8_BINARY COMMENT 'Declared income band at account opening: <15K, 15-30K, 30-60K, 60K+ (THB/month). Compare with actual flows to detect mismatch',
  province STRING COLLATE UTF8_BINARY,
  account_open_date STRING COLLATE UTF8_BINARY,
  bot_confirmed_mule BOOLEAN COMMENT 'True if this account is on the official Bank of Thailand (BOT) / AMLO confirmed mule list',
  bot_flagged_date STRING COLLATE UTF8_BINARY,
  connected_component_id BIGINT COMMENT 'Graph cluster ID - accounts in the same component are linked by transactions or shared attributes',
  pagerank_score DOUBLE COMMENT 'PageRank centrality score (0-1, normalized). Higher = more central in money routing. Hub accounts have the highest PageRank',
  triangle_count BIGINT COMMENT 'Number of closed triangles (A->B->C->A) this account participates in. Indicates money cycling/layering',
  community_id BIGINT COMMENT 'Sub-group within a component discovered by Label Propagation algorithm. May represent different recruiters or cells within a syndicate',
  two_hop_ratio DOUBLE COMMENT 'Fraction of incoming funds that leave within 24 hours (0-1). Values > 0.8 indicate pure pass-through mule behavior',
  behavior_profile STRING COLLATE UTF8_BINARY COMMENT 'Behavioral classification: steady (normal), pass_through (mule), dormant_then_active (coordinated activation), burst (concentrated activity)',
  device_pattern_cluster BIGINT,
  total_inflow_thb DOUBLE COMMENT 'Total incoming money in Thai Baht over the last 90 days',
  unique_senders BIGINT,
  total_outflow_thb DOUBLE COMMENT 'Total outgoing money in Thai Baht over the last 90 days',
  unique_receivers BIGINT,
  avg_hold_time_hours DOUBLE COMMENT 'Average hours funds stay in the account before being sent out. Low values (< 24h) indicate pass-through behavior',
  risk_score DOUBLE COMMENT 'Composite risk score (0-1) combining PageRank, two-hop ratio, triangles, behavior, income mismatch, and BOT status. Higher = more suspicious')
USING delta
COMMENT 'Account-level risk features computed by graph algorithms. One row per bank account with demographics, BOT mule status, graph metrics (PageRank, triangles, communities), behavioral profiles, and composite risk score. Used for mule network detection in Thai banking.'
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

