CREATE TABLE vn.mule_demo.silver_customers (
  account_id STRING COLLATE UTF8_BINARY,
  customer_name STRING COLLATE UTF8_BINARY,
  age BIGINT,
  occupation STRING COLLATE UTF8_BINARY,
  monthly_income DOUBLE,
  income_band STRING COLLATE UTF8_BINARY,
  province STRING COLLATE UTF8_BINARY,
  account_open_date STRING COLLATE UTF8_BINARY,
  account_type STRING COLLATE UTF8_BINARY,
  kyc_risk_level STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

