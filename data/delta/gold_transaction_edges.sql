CREATE TABLE vn.mule_demo.gold_transaction_edges (
  from_account STRING COLLATE UTF8_BINARY,
  to_account STRING COLLATE UTF8_BINARY,
  total_amount_thb DOUBLE,
  txn_count BIGINT,
  first_txn TIMESTAMP,
  last_txn TIMESTAMP,
  channels ARRAY<STRING COLLATE UTF8_BINARY>)
USING delta
COMMENT 'Aggregated transaction flows between account pairs. Shows total amount, count, and time range of transactions between each sender-receiver pair.'
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

