CREATE TABLE vn.mule_demo.gold_shared_attribute_edges (
  account_a STRING COLLATE UTF8_BINARY,
  account_b STRING COLLATE UTF8_BINARY,
  link_type STRING COLLATE UTF8_BINARY,
  shared_value STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

