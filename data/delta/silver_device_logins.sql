CREATE TABLE vn.mule_demo.silver_device_logins (
  login_id STRING COLLATE UTF8_BINARY,
  account_id STRING COLLATE UTF8_BINARY,
  device_fingerprint STRING COLLATE UTF8_BINARY,
  ip_address STRING COLLATE UTF8_BINARY,
  login_timestamp TIMESTAMP,
  os STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

