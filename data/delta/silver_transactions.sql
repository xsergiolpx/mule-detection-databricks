CREATE TABLE vn.mule_demo.silver_transactions (
  txn_id STRING COLLATE UTF8_BINARY,
  from_account STRING COLLATE UTF8_BINARY,
  to_account STRING COLLATE UTF8_BINARY,
  amount_thb DOUBLE,
  txn_timestamp TIMESTAMP,
  channel STRING COLLATE UTF8_BINARY)
USING delta
COMMENT 'Raw individual transactions with timestamp, amount in Thai Baht, channel (promptpay/mobile/atm/branch). Use for time-series analysis.'
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

