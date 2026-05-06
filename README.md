# Mule Detection on Databricks

End-to-end reference implementation for detecting money-mule accounts in retail banking, built for the Bank of Thailand (BOT) / AMLO regulatory context but applicable to any market with a centralised mule registry (Cifas UK, RBI India MuleHunter, MAS Singapore COSMIC, AUSTRAC Australia).

The project covers the full mule-detection maturity ladder on a Databricks Lakehouse:

1. **Business-logic rules** — FATF and BOT typologies as guardrails.
2. **Unsupervised anomaly detection** — Isolation Forest / autoencoders.
3. **Supervised ML** — XGBoost trained against the BOT HR-03 confirmed-mule list.
4. **Graph analytics** — PageRank, community detection, two-hop pass-through ratio, and GNN ring scoring on GraphFrames.
5. **Sequence models** — LSTM / temporal patterns over transaction histories.

## Repo layout

| Path | What it is |
|---|---|
| `mule-detection-demo/` | Databricks notebooks: synthetic-data generation, graph feature engineering, GraphFrames tutorial, Genie space creation. |
| `mule-explorer/` | Databricks App (Plotly Dash + Lakebase) — investigator UI for exploring the mule network, account risk scores, and confirmed-vs-suspected accounts. |
| `data/` | Local Delta + Lakebase exports used by the explorer for offline development. |
| `mule_detection_research.md` | Standalone research document — proven mule-detection techniques worldwide, named bank deployments, and quantitative results with citations. |

## Quick links

- **Research and benchmarks:** [`mule_detection_research.md`](./mule_detection_research.md)
- **Demo notebooks:** [`mule-detection-demo/`](./mule-detection-demo)
- **Investigator app:** [`mule-explorer/`](./mule-explorer)

## Stack

Databricks Lakehouse · Delta · Unity Catalog · MLflow · GraphFrames · Mosaic AI · Lakebase · Databricks Apps (Plotly Dash) · Genie

## Status

Demo / reference implementation. Synthetic data only — no real customer data is included.
