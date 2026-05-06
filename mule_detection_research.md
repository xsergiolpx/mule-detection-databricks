# Money-Mule Detection in Banking: Proven Techniques, Maturity Ladder, and Real-World Outcomes

**Audience:** financial-crime, data-science, and architecture stakeholders evaluating where to invest along the mule-detection maturity curve.
**Scope:** rules → unsupervised anomaly → supervised ML (with central-list labels) → graph & GNN → behavioral biometrics, sequence and deep-learning models.
**Method:** synthesis of regulator publications, peer-reviewed and arXiv papers, and named-bank deployments. Vendor-reported figures are flagged as such; numbers are quoted with the source so the reader can verify before customer-facing use.

---

## 1. Why mule detection has become a Tier-1 problem

The mule account is the **bottleneck of every modern scam**: authorised push payments (APP), call-centre scams, romance fraud, business-email-compromise, investment scams, and crypto off-ramps all require a mule to receive and disperse funds before they can be cashed-out. Killing the mule node therefore disproportionately reduces criminal yield.

Regulators have responded with directives that change the economics of detection:

- **Bank of Thailand (BOT) and AMLO** maintain the **HR-03 high-risk register** — by mid-2025 it lists roughly **700,000 individuals**. Banks must restrict all incoming and outgoing transactions for HR-03-flagged corporates, refuse new accounts, and unfreeze innocent customers within 4 hours. The Thai Bankers' Association's **Central Fraud Registry (CFR)** has already led to the **suspension of more than 1.8 million mule accounts**.
- **UK** — Cifas' National Fraud Database recorded more than **34,000 suspected mule filings in 2024** and more than **22,000 in 2025** (after a category change); roughly **65% of UK mules are under 30** and **23% under 21**.
- **Europol's EMMA** (European Money Mule Action) operations identified **10,759 mules**, **474 herders**, and led to **1,013 arrests** in their most recent reported wave; EMMA 7 included Western Union, Microsoft and Fourthline as private-sector partners.
- **Singapore** launched the **COSMIC** information-sharing platform on 1 April 2024 with DBS, OCBC, UOB, SCB Singapore, Citibank and HSBC.
- **Netherlands** — five banks (ABN AMRO, ING, Rabobank, de Volksbank, Triodos) operate **TMNL** for cross-bank transaction-monitoring of human-trafficking, VAT fraud and drug typologies.
- **Australia** — AUSTRAC's **Fintel Alliance** (public-private hub since 2017) has issued specific mule guidance focused on foreign students and temporary residents; Operation Pegasus alone yielded 6 arrests, **$2M in tainted assets**, **8 kg of gold bullion (~$600k)**, **$600k in cash** and **$636,176 in crypto**.
- **United States** — FinCEN Section 314(b) is the inter-bank info-sharing safe harbour; only ~**7,000 of 14,000 institutions** were registered in 2020, with privacy and data-security cited by >50% of non-participants — a measure of how much consortium-based mule signal is left on the table.

Sources for this section are listed at the end (BOT, Cifas, Europol, MAS, TMNL, AUSTRAC, FinCEN).

---

## 2. The maturity ladder

The reference ASEAN deck frames mule detection as a five-tier ladder of cumulative detection power versus implementation difficulty:

```
                                         ▲ cumulative detection power
                                         │
             Behavioral biometrics  ─────┤
                  Autoencoders / DL ─────┤
        MuleTrack / LSTM sequences ─────┤
        Graph ML / GNN / typologies ─────┤
   Mule-type-specific ensembles    ─────┤
       Supervised ML (XGBoost)    ─────┤
        Unsupervised ML (IF/AE)   ─────┤
        Business-logic rules      ─────┤
                                         └────────────────► implementation difficulty
```

Each tier addresses a failure mode of the previous one:

| Tier | What it adds | What it can't do alone |
|---|---|---|
| 1. Rules | Cheap, explainable, regulator-defensible | High FP, miss networks, miss novel typologies |
| 2. Unsupervised | Detects unknowns and concept drift | Hard to threshold; weak on adversarial behaviour |
| 3. Supervised | Highest precision when labels exist (BOT list!) | Needs labels; covariate-shift fragile |
| 4. Graph / GNN | Captures rings, multi-hop flows, shared-device cohorts | Compute heavy; ring-level labels scarce |
| 5. Sequence + biometrics | Detects exploited / coerced mules where account behaviour looks normal | Vendor-dependent; data licensing |

The right answer is not "pick one"; it is to operate **all tiers simultaneously, calibrated and combined** — exactly the "weak-supervision + supervised + unsupervised + graph + biometrics" stack the deck depicts.

---

## 3. Tier 1 — Business-logic rules and typologies

### What is in production

Rules remain the regulator-defensible core. The published red-flag set is convergent across jurisdictions:

- **Pass-through ratio** (inflow ≈ outflow within 24–72 h, near-zero retained balance).
- **Velocity spikes** versus account history; **dormancy-to-activity** transitions.
- **Structuring** below CTR / PromptPay reporting thresholds.
- **Multi-account-per-device**, VPN, emulator, geolocation jumps.
- **KYC mismatch** (declared occupation vs flow size — the canonical "student moving ฿100M" pattern).
- **Counterparty fan-in / fan-out** asymmetry.

These rules are codified in:

- **FATF** *Professional Money Laundering* (July 2018) — the canonical typology document with a dedicated mule-network section.
- **AUSTRAC** student-mule guidance (2024) and red-flag indicator papers via the Fintel Alliance.
- **BOT** circulars on dark-brown / brown / orange / yellow account categories and HR-03 handling.

### Documented outcomes

- **Thailand** — BOT + Thai Bankers' Association CFR: **>1.8M mule accounts suspended** under rule-based sharing; more than **1,000 new scam cases per day** still reported.
- **Europol EMMA 7–10** — **10,759 mules**, **1,013 arrests** in the latest wave; cumulative across multi-year operations: **2,469 mules in one year**; another wave **228 arrests / 3,800 mules**; and **422 arrests / 4,031 mules** in another.
- **TSB UK** — when Mastercard's Consumer Fraud Risk score (rules + ML hybrid) went live, fraud detection improved **+20% in four months**, with industry-wide projected savings **~£100M/year** if all UK banks matched TSB.

### Limits — and why every mature programme moves up the stack

- **Static rules are blind to the network**. The reference deck is correct: mule networks are graph-shaped, not transactional.
- **High FP rate**. Danske Bank's pre-ML monitoring generated **~1,200 false positives per day, of which 99.5% were unrelated to fraud** — i.e. **6 true positives per 1,200 alerts**.
- **Brittle to adversarial drift** — when herders change PromptPay format, account-age threshold, or amount distribution, rules need to be re-authored.

---

## 4. Tier 2 — Unsupervised ML / anomaly detection

The point of unsupervised methods is to flag "this account doesn't look like the population" without needing labels — important because:

- Confirmed-mule lists (HR-03, Cifas, Europol) cover only what has *already* been caught.
- Newly recruited and exploited mules (see §7 archetypes) have no positive label yet.

### Methods used in banking

| Method | Original paper | Use in mule / AML |
|---|---|---|
| **Isolation Forest** | Liu et al., 2008 (ICDM) | Per-account outlier scoring on amount, velocity, ratio features |
| **One-Class SVM** | Schölkopf et al., 2001 | Stricter boundary; better than IF on highly imbalanced AML — see below |
| **Local Outlier Factor / DBSCAN** | Breunig et al., 2000 | Density-based outlier flagging |
| **Autoencoders / Variational AE** | Kingma & Welling, 2013 | Reconstruction-error scoring on transaction sequences |
| **Markov / behavioural-state models** | Various | Detects abrupt regime shifts (dormant → active) |

### Documented results

- **OCBC Singapore** + **ThetaRay** (the first ML-based AML deployment by a Singaporean bank) — analysed one year of corporate-banking transactions; reduced **non-actionable alerts by 35%** and increased **accuracy of suspicious-transaction identification by more than 4x**. ThetaRay is unsupervised by design.
- **East-African commercial bank** (research deployment, 54,258 cross-border records) — hybrid unsupervised deep-learning framework processed **1,000 transactions per second** with high-priority alert triage.
- **Variational autoencoders** in published AML evaluations — **halved the false-positive rate** versus prior baseline.
- **AutoEncoder + LightGBM** (PMC 11623290) — **AUC 96.83%, F1 80.27%** with SMOTE on imbalanced fraud dataset.
- **Comparative evaluation**: One-Class SVM achieved **99.63% precision in the top-5% prioritised alerts**, beating Isolation Forest and LOF on the same AML benchmark.

### Limits

- Threshold tuning is dataset-specific; without calibration the alert volume is unstable.
- Explainability is weaker than rules or trees — investigators ask "why did this score 0.97?" and reconstruction error is not a satisfying answer for SAR documentation.
- Concept drift requires periodic retraining; the IF / AE assumption that "normal = majority" breaks during scam surges.

**Why it matters for Thailand:** unsupervised models are the right second layer on top of HR-03 rules — they catch the "not-yet-on-the-list" mule before it crystallises into a confirmed case.

---

## 5. Tier 3 — Supervised ML, especially gradient boosting (XGBoost) and PU learning

Supervised learning is the single biggest accuracy lift available **whenever positive labels exist** — and the BOT HR-03 list, AMLO confirmed-mule register, and intra-bank confirmed-fraud cases give Thai banks exactly that. This is the tier where the reference deck's curve steepens.

### Why XGBoost dominates in deployed AML

- Tabular financial features (amounts, ratios, velocities, KYC fields) are exactly what GBT excels at.
- Built-in feature importance — defensible to compliance, regulators, and model-risk-management.
- Strong with class imbalance via `scale_pos_weight`, focal loss, or SMOTE/ADASYN sampling.
- Efficient on CPU, easy to MLOps, well-supported in Databricks.

### Published quantitative results

- **Springer / Information Systems Frontiers (Hajek et al., 2022)** — "Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework". The **45.2% feature-importance** number cited in your reference deck for "payment format / channel pattern" matches this family of mobile-payment XGBoost papers — the original peer-reviewed source.
- **Nature *Scientific Reports* (2022)** — "Feature generation and contribution comparison for electronic fraud detection": XGBoost with engineered features yields **F1 = 78.3%** with strong interpretability.
- **IJSRED 2025**, "Fighting Money Laundering with Statistics and Machine Learning" — XGBoost outperformed alternatives with **precision = 94%, AUC-ROC = 0.97**; SHAP showed *large frequent international transfers from a low-income profile* as the top driver — directly the BOT-style "student moving ฿100M" pattern.
- **The Journal of Supercomputing 2023** — ASXAML framework (XGBoost + RFECV + Optuna) automatically suppresses false-positive alerts.
- **Medium/practitioner summary (Candir, 2025)** of an industry deployment — **AUC 97.5%** processing >5M transactions, **0.1% true laundering rate**.
- **ACM 2024** *Graph Feature Preprocessor* — XGBoost on **graph-derived** features achieves **+46% F1** over the same XGBoost on basic features. Graph features as inputs to a gradient boosted tree is a high-leverage architectural choice.

### Real bank deployments

- **Stripe Radar** — gradient-boosted core, with neural ensembles. Stripe reports **+20% YoY ML performance**; **+1.3 percentage point payment-success-rate** with adaptive rules; **>30% fraud reduction** on eligible transactions for early adopters; **17% reduction in dispute rates** even as industry e-commerce fraud grew 15%; **42% SEPA / 20% ACH fraud reduction**.
- **Itaú Unibanco (Brazil) + FICO Falcon** — cloud migration of fraud management is reported by FICO to **avoid >US$20M/month in fraud losses**, with **15% lower per-account cost** and **+20% CNP fraud detection**. Itaú additionally cut ML deployment time **from up to 6 months to 3–5 days** on AWS SageMaker — relevant for any bank operationalising mule scoring.
- **UOB Singapore** — first SG bank to apply AI to both transaction-monitoring and name-screening simultaneously. Published metrics: **96% true-positive rate** in the high-priority queue, **+5% TP and −50% FP** in transaction-monitoring, **−70% FP for individual** and **−60% FP for corporate** name-screening, with **<1% misclassification**.
- **Danske Bank (Denmark) + Teradata Think Big** — ML ensembles cut **false positives 20–30%** in 12-week DevOps sprints; deep-learning models (TensorFlow) showed **double-digit further detection improvement** in pre-prod.

### Positive-Unlabeled (PU) learning — ideal when only confirmed mules are labelled

Banks rarely have a clean negative class — *unconfirmed* is not the same as *not a mule*. PU learning (Elkan & Noto, 2008) was made for this. Recent applications to mule / AML / fraud:

- **IJCAI 2021** — *Positive-Unlabeled Learning from Imbalanced Data* — handles the double burden of class imbalance + missing negatives.
- **arXiv 2412.06203 (2024)** — survey of PU and Negative-Unlabeled learning in cybersecurity, including financial-fraud sub-domain.
- **ACM 2025** — *Enhancing Anti-Money Laundering by Money Mules Detection on Transaction Graphs* — explicit mule-targeted PU + graph hybrid.

PU learning is the technically-correct training paradigm for any model trained on the BOT HR-03 list as the positive-only supervisor.

### Calibration

For Thai banks where investigators have a fixed daily review capacity, the practical trick is **probability calibration** (Platt scaling, isotonic regression) so that thresholds map directly to expected case loads — the *Precision@Capacity* objective the reference deck calls out.

---

## 6. Tier 4 — Graph and GNN methods

This is the tier the demo focuses on, and rightly so. Mule operations are **structurally graph-shaped**: rings of accounts on shared devices, multi-hop pass-through chains, herder-recruit-deposit subgraphs.

### Real production deployments with public numbers

- **HSBC + Quantexa** — the most widely cited public case study. After deploying Quantexa's Decision Intelligence (entity resolution + network analytics) globally from 2018, HSBC reports the platform **auto-closed 1 million false-positive alerts**, reducing alerts requiring investigation by **83%** — saving the time of **140–180 analysts** previously occupied by those FPs. Industry coverage frames it as "**4× more financial criminals identified while cutting false alarms by 60%**".
- **Danske Bank + Quantexa** — post-Estonian-scandal pivot to contextual decision intelligence. Documented **60% reduction in false positives** with a **60% increase in fraud detection** and **50% drop in false positives** on the payment-fraud workload (Best Practice AI summary).
- **Standard Chartered** — Quantexa case study on AML detection and investigation management (graph-driven prioritisation).
- **NatWest + Featurespace ARIC** — included in the broader graph/contextual stack: **+135% scam detection rate** and **−75% false positives for scams** within 24 hours of deployment. ARIC's check-fraud variant detects **>90% of check fraud at a 5:1 FP ratio**.
- **Mastercard TRACE** — network-wide AML/mule platform. **UK**: **21 financial institutions**, covering **~90% of the UK Faster Payments Service** network, since launch in 2018; **identified thousands of mule accounts** and **hundreds of new mule accounts every month**. **Asia Pacific**: launched **February 2025** in the Philippines via **BancNet (36 domestic banks)** — directly relevant comparator for any Thai national-level rollout.
- **TMNL Netherlands** — five-bank cross-bank graph monitoring, focused on human-trafficking, VAT fraud and drugs typologies. Confirms that *joint* monitoring surfaces signals invisible to any single bank.

### Published results on real bank graphs

- **DNB (Norway)** — heterogeneous GNN on Norway's largest bank, **5M nodes, ~10M edges**. **GraphSAGE > GAT > GCN**; first publication applying heterogeneous GNNs to AML on a large real-world bank graph (Fronzetti Colladon-style). Hybrid LSTM-GraphSAGE reaches **95.4% accuracy** on simulated data; standalone GraphSAGE achieves **acc 92.8% / prec 91.1% / recall 91.8% / F1 91.4%**.

### Benchmarks on synthetic data (IBM AMLworld)

- **NeurIPS 2023 Datasets & Benchmarks** — *Realistic Synthetic Financial Transactions for AML* (Altman et al., IBM). Publicly released **HI / LI** datasets, large variants of **175–180M transactions**. GNNs (PNA, GIN+EU) "**significantly enhance**" GNN performance and produce competitive results without feature engineering.
- **arXiv 2306.11586** — *Provably Powerful GNNs for Directed Multigraphs* (Multi-PNA / Multi-GIN). Improves the **minority-class F1 of standard message-passing GNNs by up to +30%** on AMLworld.
- **arXiv 2604.12241** — *BlazingAML*: high-throughput multi-stage graph mining pipeline on AMLworld.
- **ACM 2024** — *Graph Feature Preprocessor*: graph-feature pre-processor that boosts an XGBoost downstream by **+46% F1** vs raw features. **This is the architectural pattern most banks should default to**: graph-derived features + GBT, with an optional GNN stage above.

### Why graph beats trees alone for mule networks

- **Multi-hop money flow**: a 3-hop pass-through cannot be detected from per-account features.
- **Shared-device / shared-IP rings**: cohort-level signal that does not exist at row level.
- **Ring-level labelling**: Quantexa/HSBC's central insight — investigators can off-board entire networks rather than one account at a time, raising true-positive yield per investigator-hour by an order of magnitude.

### Limits

- **Compute** — 175 M-edge graphs need careful infrastructure (Spark/GraphFrames or specialised GNN frameworks). This is exactly where Lakeflow + GPU serverless on Databricks pay off.
- **Label scarcity at ring level** — most banks have account-level labels, not ring-level — semi-supervised propagation, weak supervision, and PU-learning matter.
- **Explainability** — investigators need *why this ring*, not *the embedding said so*. Quantexa's productisation around entity-resolution + visual graph context is what made HSBC's investigators trust it.

---

## 7. Tier 5 — Behavioral biometrics, sequence (LSTM/TGN), and deep learning

This is the tier that matters for the **third mule archetype** the reference deck identifies: **Exploited (account-takeover or coerced)**. The account looks legitimate at the row and graph levels — the only signal is *how the human interacts with the device during the session*.

### Three mule archetypes and which tier catches them

| Archetype | Behaviour | Best-fit detection |
|---|---|---|
| **Complicit** (knowing accomplice) | Short-lived account, rapid activation, atypical spikes, no banking history | Tier 1 + Tier 3 (rules + XGBoost) |
| **Recruited** (social-media lured) | Normal baseline, dormancy, low-value testers, then sudden spike | Tier 2 + Tier 4 (anomaly + graph) |
| **Exploited** (ATO / coerced) | Existing genuine account, uncharacteristic velocity, new device | **Tier 5 (behavioral biometrics, sequence)** |

### Behavioral biometrics — what is deployed

- **BioCatch** (the most-cited mule deployment). Customer base of **257 financial institutions** in 2024; reports those customers identified and acted on **~2 million (some sources: 2.3 million) mule accounts in 2024**. Detects behavioural shifts: typing rhythm, swipe geometry, navigation flow, hesitation, copy-paste patterns, dwell times — signals that survive even when the device, IP, and account are all "trusted".
- **Feedzai**, **NICE Actimize**, **LexisNexis ThreatMetrix**, **Mastercard NuData**, **Callsign**, **Revelock (Buguroo)** — comparable behavioural-biometrics stacks.
- **RBI MuleHunter.AI (India)** — Reserve Bank Innovation Hub built an AI/ML mule-detection engine codifying **19 distinct mule behaviour patterns**. Pilot with two large public-sector banks reported "encouraging results"; by August 2025 **at least 15 banks had implemented** it; an RTI in December 2025 confirmed **23 banks**. This is a flagship example of a regulator-led, ML-based replacement for static rules.

### Sequence and temporal models

- **LSTM / GRU on transaction sequences** — captures temporal dependencies that XGBoost cannot. Hybrid **LSTM + GraphSAGE** on simulated data: **95.4% accuracy**.
- **Temporal Graph Networks (TGN)** — Rossi et al.; in production-style fraud benchmarks, TGN-based methods raise **Precision@20 Recall** from **68.5% → 86.2% (+17.7 points)** on Taobao and from **47.2% → 56.5%** on offline-merchant fraud, vs MLP with hand-engineered real-time features.
- **Causal Temporal GNN (CaT-GNN, arXiv 2402.14708)** — preserves recall while gaining precision; an architectural answer to drift.
- **Spatio-temporal attention GNN** — published metrics **96.4% accuracy / 97.8% precision / 93.5% recall / 95.6% F1** on credit-card fraud.

### Deep-learning anomaly detection

- **Mastercard Consumer Fraud Risk** (the deck-relevant example): an ensemble blending GBT, neural networks, and graph signals over Faster Payments. UK pilot at **TSB**: **+20% fraud detection in 4 months**; potential **£100M/year saved** if all UK banks matched TSB performance.
- **DBS Singapore** — flags high-risk transactions in **<10 ms**, claiming **15% of customers' money saved from scams**.

### Why this tier closes the loop

The exploited-account problem is the residual after rules + anomaly + supervised + graph. Without biometrics, banks either (a) freeze legitimate customers (the BOT 4-hour-unfreeze problem) or (b) miss coerced-mule activity entirely. With biometrics + step-up auth, the tail is resolvable.

---

## 8. Cross-cutting design choices that the literature is unanimous on

1. **Rolling-window features** (1-day, 7-day, 30-day) are universally used; mule activation timescales are bursty.
2. **Ratio features** (in/out, balance/volume) outperform raw amounts.
3. **Entropy of counterparties** is a strong tier-2/3 signal — low entropy = repetitive structuring.
4. **Calibrated probabilities** + capacity-tuned thresholds beat hard-coded score cutoffs (Precision@Capacity).
5. **Ensembling rules + ML + graph** beats any single model — every named-bank deployment in this document does this.
6. **Investigator-in-the-loop / active learning** is now standard. The deck's "active learning queue" is consistent with the BioCatch Link, Quantexa case, and Featurespace ARIC investigator workflows.
7. **Cross-bank consortium signal** (TMNL, COSMIC, TRACE, 314(b), Fintel Alliance, BOT CFR) is what unlocks ring-level detection; any single-bank model is structurally blind.

---

## 9. Comparator-bank summary table

| Bank / regulator | Tech stack | Headline number | Source |
|---|---|---|---|
| HSBC + Quantexa | Entity resolution + graph | 1M FP auto-closed; 83% alert reduction; 4× criminals; -60% FP | Quantexa case study; ChiefAIOfficer summary |
| Danske Bank + Quantexa | Graph + ML ensembles | -50–60% FP, +60% fraud detection | Quantexa; Best Practice AI |
| Danske Bank + Teradata Think Big | ML ensembles + DL (TensorFlow) | 20–30% FP cut; double-digit DL gains in test | Teradata case |
| NatWest + Featurespace ARIC | Adaptive Behavioural Analytics + graph | +135% scam detection; -75% scam FP within 24 h | Featurespace |
| Mastercard CFR (TSB, NatWest, Lloyds, Halifax, Monzo, BoS, AIB) | Graph + ML over Faster Payments | TSB +20% fraud detection in 4 months; ~£100M/yr UK projected | Mastercard, FCA, TSB |
| Mastercard TRACE UK | Network-level AML | 21 banks; ~90% of UK FPS; thousands of mules; hundreds new monthly | Mastercard |
| Mastercard TRACE APAC (Philippines / BancNet) | Network-level AML | 36 banks onboarded Feb 2025 | Mastercard, PRNewswire |
| RBI MuleHunter.AI (India) | ML, 19 behaviour patterns | 23 banks adopted by Dec 2025 | RBIH; medianama RTI |
| OCBC + ThetaRay | Unsupervised anomaly | -35% non-actionable alerts; +4× detection accuracy | OCBC, ThetaRay |
| UOB Singapore | AI on TM + name screening | 96% TP high-priority; -50% FP TM; -70% FP individual screening | UOB press release |
| DBS Singapore | Real-time scoring | <10 ms decision; 15% of customer scam funds saved | DBS |
| BioCatch (vendor, 257 FIs) | Behavioural biometrics | ~2M – 2.3M mule accounts identified in 2024 | BioCatch |
| Stripe Radar | GBT + neural ensembles | +20% YoY ML; -30% fraud (eligible); -42% SEPA / -20% ACH fraud | Stripe |
| Itaú + FICO Falcon | Supervised + cloud | -US$20M/month fraud loss avoided; +20% CNP detection | FICO |
| TMNL Netherlands | Cross-bank graph TM | Joint monitoring surfaces signals invisible to single banks | TMNL |
| BOT + Thai Bankers Assoc CFR | Rules + cross-bank registry | >1.8M mule accounts suspended | Bangkok Post; Thailand Business News |

(All vendor-reported figures should be verified against the original primary source before customer-facing use; this table is a curated reading list, not an audit.)

---

## 10. Recommended target architecture for a Thai bank

The literature converges on a single design — and it is the same one the reference ASEAN deck advocates:

1. **Data foundation** — unify KYC + payments + sessions + device + BOT HR-03 / AMLO labels in a single Lakehouse with Unity Catalog governance.
2. **Tier 1 rules** — codify FATF, BOT and internal typologies as guardrails, not as the detector. Use them to generate **weak labels** for tier 3.
3. **Tier 2 unsupervised** — Isolation Forest / autoencoder per-account scoring for the "not on any list yet" tail.
4. **Tier 3 supervised** — XGBoost / LightGBM trained with PU-learning on HR-03 + confirmed-fraud as positives. Calibrated to Precision@Capacity.
5. **Tier 4 graph** — graph-feature preprocessor (community ID, PageRank, two-hop ratio, k-core, triangle counts) feeding tier 3, plus a GNN (PNA / Multi-GIN / TGN) for ring-level scoring.
6. **Tier 5 sequence + biometrics** — LSTM / TGN over transaction sequences, plus a behavioural-biometrics partner (BioCatch / NuData / Feedzai BB) for the exploited-mule tail.
7. **Calibration + decisioning** — Platt / isotonic calibration; thresholds tuned to investigator review capacity; hybrid scoring of rules + ML; explainable case packages for SAR.
8. **Active learning loop** — investigator confirmations feed back into PU-learning labels and graph supervision.
9. **Cross-bank signal** — feed into / consume from the BOT CFR; track Mastercard TRACE-APAC as it expands beyond Philippines.

---

## 11. Sources

### Regulators and national programmes

- Bank of Thailand, FAQ on financial threats: <https://www.bot.or.th/en/faqs/faqs-03.html>
- Thailand Business News, "Thailand to Prohibit Financial Transactions Using Mule Accounts from March 2025": <https://www.thailand-business-news.com/banking/197770-thailand-to-prohibit-financial-transactions-using-mule-accounts-from-march-2025>
- Bangkok Post, "Banks share data with Bank of Thailand to combat fraudulent mule accounts": <https://www.bangkokpost.com/business/general/2858127/banks-share-data-with-bank-of-thailand-to-combat-fraudulent-mule-accounts>
- Bangkok Post, "Bid to halt financial fraud intensifies": <https://www.bangkokpost.com/business/general/2980261/bid-to-halt-financial-fraud-intensifies>
- Bangkok Post, "AI boosts detection of bank fraud": <https://www.bangkokpost.com/opinion/opinion/2977176/ai-boosts-detection-of-bank-fraud>
- Tilleke & Gibbins, "Thailand Issues Mandatory Guidelines Enhancing Digital Fraud Controls": <https://www.tilleke.com/insights/thailand-issues-mandatory-guidelines-enhancing-digital-fraud-controls/57/>
- BioCatch, "How Thailand's Royal Decree is changing financial fraud accountability": <https://www.biocatch.com/blog/thailand-royal-decree-financial-fraud>
- Cifas, *Fraudscape 2025*: <https://www.cifas.org.uk/newsroom/fraudscape-2025-record-fraud-levels>
- Cifas, *Fraudscape 2026*: <https://www.cifas.org.uk/newsroom/fraudscape2026>
- Cifas, "65% of UK Money Mules Are Under 30": <https://ffnews.com/newsarticle/fintech/new-research-reveals-65-of-uk-money-mules-are-under-30/>
- UK Government, "Money mule and financial exploitation action plan 2024": <https://assets.publishing.service.gov.uk/media/65e0a18f2f2b3b001c7cd7b7/Money+Mule+and+Financial+Exploitation+Action+Plan+.pdf>
- Europol, EMMA results pages: <https://www.europol.europa.eu/media-press/newsroom/news/european-money-mule-action-leads-to-1-803-arrests>, <https://www.europol.europa.eu/media-press/newsroom/news/paper-trail-ends-in-jail-time-for-1-013-money-mules>, <https://www.europol.europa.eu/media-press/newsroom/news/422-arrested-and-4%20031-money-mules-identified-in-global-crackdown-money-laundering>, <https://www.europol.europa.eu/media-press/newsroom/news/228-arrests-and-over-3800-money-mules-identified-in-global-action-against-money-laundering>, <https://www.europol.europa.eu/media-press/newsroom/news/178-arrests-in-successful-hit-against-money-muling>
- MAS, "COSMIC Platform launch": <https://www.mas.gov.sg/news/media-releases/2024/mas-launches-cosmic-platform>; COSMIC overview: <https://www.mas.gov.sg/regulation/anti-money-laundering/cosmic>
- AUSTRAC, Fintel Alliance: <https://www.austrac.gov.au/partners/fintel-alliance>; Fintel Alliance Annual Report 2023–24: <https://www.austrac.gov.au/sites/default/files/2024-11/Fintel%20Alliance%20Annual%20Report%20Extract%202023-24.pdf>; Foreign-student mule guidance: <https://www.austrac.gov.au/news-and-media/media-release/new-guidance-released-help-combat-use-foreign-students-money-mules>
- TMNL: <https://tmnl.nl/en/article/transaction-monitoring-netherlands-adapts-its-working-method-to-new-european-legislation/>
- FinCEN Section 314(b): <https://www.fincen.gov/section-314b>; Fact Sheet (PDF): <https://www.fincen.gov/sites/default/files/shared/314bfactsheet.pdf>
- FATF, *Professional Money Laundering* (July 2018): <https://www.fatf-gafi.org/content/dam/fatf/documents/Professional-Money-Laundering.pdf>

### Bank deployments and vendor case studies

- Quantexa, HSBC case study: <https://www.quantexa.com/resources/hsbc-contextual-decision-intelligence-raising-the-bar-of-aml-technology/>; "How Decision Intelligence is Helping HSBC Combat Financial Crime and Fraud" (PDF): <https://www.quantexa.com/assets/x/75b2f9dc6f/hsbc-case-study-jennifer-calvery-financial-crime.pdf>
- ChiefAIOfficer, "How HSBC's AI Catches 4× More Financial Criminals While Cutting False Alarms by 60%": <https://chiefaiofficer.com/blog/how-hsbcs-ai-catches-4x-more-financial-criminals-while-cutting-false-alarms-by-60/>
- Quantexa, Danske Bank case study: <https://www.quantexa.com/resources/danske-bank/>
- Best Practice AI, "Danske Bank increases payment fraud detection by 60% and reduces false positives by 50%": <https://www.bestpractice.ai/ai-case-study-best-practice/danish_danske_bank_increases_payment_fraud_detection_by_60%25_and_reduces_false_positives_by_50%25_with_machine_learning>
- Teradata, "Danske Bank Saves Millions Fighting Fraud With Deep Learning and AI" (PDF): <https://assets.teradata.com/resourceCenter/downloads/CaseStudies/CaseStudy_EB9821_Danske_Bank_Saves_Millions_Fighting_Fraud_With_Deep_Learning_and_AI.pdf>
- Featurespace, NatWest scam-detection +135% / FP −75%: <https://www.featurespace.com/newsroom/natwest-improves-scam-detection-rate-by-135-using-featurespaces-technology>; ARIC overview: <https://www.featurespace.com/aric-risk-hub>
- Mastercard, Consumer Fraud Risk launch (UK, 2023): <https://newsroom.mastercard.com/news/press/2023/july/mastercard-leverages-its-ai-capabilities-to-fight-real-time-payment-scams/>; "Mastercard transforms the fight against scams" (2024): <https://www.mastercard.com/news/press/2024/april/mastercard-transforms-the-fight-against-scams-with-latest-ai-tech/>
- Mastercard, TRACE Asia Pacific launch (Feb 2025): <https://www.mastercard.com/us/en/news-and-trends/press/2025/february/mastercard-launches-anti-money-laundering-service-trace-to-combat-financial-crime-in-asia-pacific.html>; PRNewswire: <https://www.prnewswire.com/apac/news-releases/mastercard-launches-anti-money-laundering-service-trace-to-combat-financial-crime-in-asia-pacific-302375041.html>
- The Financial Technology Report, "UK Banks Save Millions with Mastercard's AI-Powered Fraud Detection Solution": <https://thefinancialtechnologyreport.com/uk-banks-save-millions-with-mastercards-ai-powered-fraud-detection-solution/>
- BioCatch, mule detection report 2024: <https://www.biometricupdate.com/202501/biocatch-puts-spotlight-on-money-mule-problem-biometric-solution>; Mule Account Detection product page: <https://www.biocatch.com/mule-account-detection>
- RBI / RBIH MuleHunter.AI: <https://www.fintechfutures.com/ai-in-fintech/reserve-bank-of-india-pilots-new-mulehunter-ai-solution-to-help-identify-mule-accounts>; Banking Frontiers: <https://bankingfrontiers.com/rbi-introduces-mulehunter-ai-ai-driven-solution-to-detect-mule-accounts-developed-by-rbih/>; medianama RTI on 23 banks: <https://www.medianama.com/2025/12/223-rti-23-banks-mulehunter-mule-accounts/>
- OCBC + ThetaRay: <https://fintechnews.sg/14047/fintech/ocbc-bank-first-singapore-bank-tap-artificial-intelligence-machine-learning-combat-financial-crime/>
- UOB AI AML deployment: <https://www.uobgroup.com/uobgroup/newsroom/2020/new-money-laundering-solution.page?path=data/uobgroup/2020/133&cr=segment>
- Stripe Radar: <https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection>; <https://stripe.com/blog/using-ai-dynamic-radar-rules>; <https://stripe.com/blog/using-ai-optimize-payments-performance-payments-intelligence-suite>
- Itaú + FICO: <https://www.fico.com/en/newsroom/itau-unibanco-avoids-over-usd-20m-month-fraud-losses-using-fico-s-cloud-based-fraud>; Itaú + AWS SageMaker: <https://aws.amazon.com/solutions/case-studies/itau-ml-case-study/>

### Peer-reviewed and arXiv references

- Hajek et al., "Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework", *Information Systems Frontiers* / PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9560719/>; <https://link.springer.com/article/10.1007/s10796-022-10346-6>
- "Feature generation and contribution comparison for electronic fraud detection", *Scientific Reports*, Nature 2022: <https://www.nature.com/articles/s41598-022-22130-2>
- "Automatic suppression of false positive alerts in AML systems using machine learning", *J. Supercomputing* 2023: <https://link.springer.com/article/10.1007/s11227-023-05708-z>
- "Fighting Money Laundering with Statistics and Machine Learning", arXiv 2201.04207 / IJSRED: <https://arxiv.org/pdf/2201.04207>; <https://ijsred.com/volume8/issue3/IJSRED-V8I3P249.pdf>
- Altman et al., IBM, "Realistic Synthetic Financial Transactions for AML Models", NeurIPS 2023: <https://proceedings.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf>; arXiv 2306.16424: <https://arxiv.org/pdf/2306.16424>; AMLworld data on GitHub: <https://github.com/IBM/AML-Data>
- "Provably Powerful GNNs for Directed Multigraphs" (Multi-PNA / Multi-GIN), arXiv 2306.11586: <https://arxiv.org/html/2306.11586v3>
- "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction" (ACM 2024): <https://dl.acm.org/doi/pdf/10.1145/3677052.3698674>
- "BlazingAML: High-Throughput AML via Multi-Stage Graph Mining", arXiv 2604.12241: <https://arxiv.org/abs/2604.12241>
- "Finding Money Launderers Using Heterogeneous GNNs" (DNB Norway), arXiv 2307.13499 / ScienceDirect: <https://arxiv.org/pdf/2307.13499>; <https://www.sciencedirect.com/science/article/pii/S2405918825000273>
- "Temporal Graph Networks for Graph Anomaly Detection in Financial Networks", arXiv 2404.00060: <https://arxiv.org/html/2404.00060v1>
- "CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal GNNs", arXiv 2402.14708: <https://arxiv.org/html/2402.14708v2>
- "AutoEncoder enhanced LightGBM for credit card fraud detection", PMC 11623290: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11623290/>
- "Hybrid deep learning for AML: unsupervised detection of emerging schemes via feature fusion and XAI" (East African bank), ScienceDirect 2026: <https://www.sciencedirect.com/science/article/pii/S2666827026000216>
- "Comparative analysis of ML algorithms for money laundering detection", *Discover AI* 2025: <https://link.springer.com/article/10.1007/s44163-025-00397-4>
- "Positive-Unlabeled Learning from Imbalanced Data", IJCAI 2021: <https://www.ijcai.org/proceedings/2021/0412.pdf>
- "Applications of Positive Unlabeled (PU) and Negative Unlabeled (NU) Learning in Cybersecurity", arXiv 2412.06203: <https://arxiv.org/abs/2412.06203>
- "Enhancing Anti-Money Laundering by Money Mules Detection on Transaction Graphs" (ACM 2025): <https://dl.acm.org/doi/10.1145/3766918.3766933>
- Feedzai, "Behavioral Biometrics for Money Mule Account Detection": <https://feedzai.com/blog/behavioral-biometrics-for-money-mule-account-detection/>
- Feedzai, "Inbound Payment Fraud Detection and Mule Risk Modelling" (Bank Negara Malaysia, PDF): <https://www.feedzai.com/wp-content/uploads/2024/11/Feedzai-Inbound-Payment-Fraud-Detection-and-Mule-Risk-Modeling-Complying-with-Bank-Negara-Malaysia-Recommendations.pdf>

---

*Document generated 2026-05-04. Vendor-reported figures should be re-verified against primary sources before being relied on for customer-facing claims.*
