CREATE TABLE vn.mule_demo.silver_bot_mule_list (
  account_id STRING COLLATE UTF8_BINARY,
  flagged_date STRING COLLATE UTF8_BINARY,
  reason STRING COLLATE UTF8_BINARY)
USING delta
COMMENT 'Official Bank of Thailand (BOT) and AMLO confirmed mule accounts. This is the seed list - a subset of actual mules. Reason categories: scam_proceeds, money_laundering, syndicate_link.'
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2',
  'delta.parquet.compression.codec' = 'zstd')

