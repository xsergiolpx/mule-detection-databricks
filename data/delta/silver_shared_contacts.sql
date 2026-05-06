CREATE TABLE vn.mule_demo.silver_shared_contacts (
  account_id STRING COLLATE UTF8_BINARY,
  phone_number STRING COLLATE UTF8_BINARY,
  email STRING COLLATE UTF8_BINARY,
  registered_address STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

