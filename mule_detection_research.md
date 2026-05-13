# Money-Mule Detection in Banking: Proven Techniques, Maturity Ladder, and Real-World Outcomes

**Audience:** financial-crime, data-science and architecture leaders evaluating where to invest along the mule-detection maturity curve.
**Scope:** rules → unsupervised anomaly detection → supervised ML (with central-list labels) → graph and graph neural networks → behavioural biometrics, sequence and deep-learning models.
**Method:** synthesis of regulator publications, peer-reviewed and arXiv papers, and publicly disclosed bank deployments. Every numerical claim is sourced; vendor-published figures are noted as such so they can be cross-checked against the primary source.

---

## Table of contents

1. [Why mule detection has become a Tier-1 problem](#1-why-mule-detection-has-become-a-tier-1-problem)
2. [The maturity ladder](#2-the-maturity-ladder)
3. [Tier 1 — Business-logic rules and typologies](#3-tier-1--business-logic-rules-and-typologies)
4. [Tier 2 — Unsupervised machine learning and anomaly detection](#4-tier-2--unsupervised-machine-learning-and-anomaly-detection)
5. [Tier 3 — Supervised ML, especially gradient boosting (XGBoost) and PU-learning](#5-tier-3--supervised-ml-especially-gradient-boosting-xgboost-and-pu-learning)
6. [Tier 4 — Graph and graph-neural-network methods](#6-tier-4--graph-and-graph-neural-network-methods)
7. [Tier 5 — Behavioural biometrics, sequence and deep-learning models](#7-tier-5--behavioural-biometrics-sequence-and-deep-learning-models)
8. [Technical deep-dives](#8-technical-deep-dives)
   - 8.1 [How PU-learning works (and why HR-03 calls for it)](#81-how-pu-learning-works-and-why-hr-03-calls-for-it)
   - 8.2 [What a graph neural network is (and why graphs outperform trees on rings)](#82-what-a-graph-neural-network-is-and-why-graphs-outperform-trees-on-rings)
   - 8.3 [What MuleTrack is (and how it fits the maturity ladder)](#83-what-muletrack-is-and-how-it-fits-the-maturity-ladder)
   - 8.4 [The incomplete-graph problem: how graph detection works when most mule activity crosses institutional boundaries](#84-the-incomplete-graph-problem-how-graph-detection-works-when-most-mule-activity-crosses-institutional-boundaries)
   - 8.5 [Data sources and where they typically reside in a Thai bank](#85-data-sources-and-where-they-typically-reside-in-a-thai-bank)
   - 8.6 [The investigator experience — current state and a Lakehouse-native target state](#86-the-investigator-experience--current-state-and-a-lakehouse-native-target-state)
   - 8.7 [Evolving from a point-solution stack toward a unified platform](#87-evolving-from-a-point-solution-stack-toward-a-unified-platform)
   - 8.8 [Scaling graph-feature computation at production volumes (the "500M-node" question)](#88-scaling-graph-feature-computation-at-production-volumes-the-500m-node-question)
9. [Cross-cutting design choices on which the literature is unanimous](#9-cross-cutting-design-choices-on-which-the-literature-is-unanimous)
10. [Comparator-bank summary table](#10-comparator-bank-summary-table)
11. [Recommended target architecture for a Thai bank](#11-recommended-target-architecture-for-a-thai-bank)
12. [Sources](#12-sources)

---

## 1. Why mule detection has become a Tier-1 problem

The mule account is the bottleneck of every modern scam. Authorised push payments (APP), call-centre scams, romance fraud, business-email-compromise, investment scams and crypto off-ramps all require a mule to receive and disperse funds before they can be cashed out. Disrupting the mule node therefore disproportionately reduces criminal yield, which is why regulators worldwide have shifted from prosecuting individual scams to systematically dismantling the mule layer.

Recent regulatory and operational signals across the regions most relevant to ASEAN financial institutions:

- **Thailand** — the **Bank of Thailand (BOT)** and **AMLO** maintain the **HR-03 high-risk register**, which by mid-2025 listed roughly **700,000 individuals**. Banks must restrict all incoming and outgoing transactions for HR-03-flagged corporates, refuse new accounts, and unfreeze innocent customers within **4 hours**. The Thai Bankers' Association's **Central Fraud Registry (CFR)** has driven the **suspension of more than 1.8 million mule accounts** through cross-bank data sharing.
- **United Kingdom** — Cifas' National Fraud Database recorded more than **34,000 suspected mule filings in 2024** and more than **22,000 in 2025** (after a category change). Approximately **65% of UK mules are under 30** and **23% under 21**.
- **European Union** — Europol's **EMMA** (European Money Mule Action) operations identified **10,759 mules** and **474 herders**, and led to **1,013 arrests** in their most recent reported wave. EMMA 7 onboarded Western Union, Microsoft and Fourthline as private-sector partners.
- **Singapore** — the **COSMIC** information-sharing platform launched on 1 April 2024 with DBS, OCBC, UOB, Standard Chartered Singapore, Citibank and HSBC.
- **Netherlands** — five banks (ABN AMRO, ING, Rabobank, de Volksbank, Triodos) operate **TMNL** for cross-bank transaction-monitoring of human-trafficking, VAT-fraud and drug typologies.
- **Australia** — AUSTRAC's **Fintel Alliance** (public-private partnership since 2017) has issued specific mule guidance focused on foreign students and temporary residents. Operation Pegasus alone yielded **6 arrests**, **AUD 2M in tainted assets**, **8 kg of gold bullion (~AUD 600k)**, **AUD 600k in cash** and **AUD 636,176 in crypto**.
- **United States** — FinCEN Section 314(b) provides an inter-bank information-sharing safe harbour. As of 2020 only ~**7,000 of 14,000 institutions** were registered, with privacy and data-security concerns cited by more than 50% of non-participants — an indicator of how much consortium-based mule signal remains untapped.

All primary sources are listed in §12.

---

## 2. The maturity ladder

Mule-detection capability tends to evolve through a recognisable sequence of analytical tiers. Each tier adds detection power but also requires more data, more engineering and stronger governance. The relationship is shown below.

| Level | Maturity tier | Implementation effort | Cumulative detection power |
|---|---|---|---|
| 1 | Business-logic rules | `█░░░░░░░░░` | `█░░░░░░░░░` |
| 2 | Unsupervised ML (Isolation Forest, autoencoders) | `██░░░░░░░░` | `███░░░░░░░` |
| 3 | Supervised ML (XGBoost + PU-learning) | `███░░░░░░░` | `█████░░░░░` |
| 4 | Mule-type-specific ensembles | `████░░░░░░` | `██████░░░░` |
| 5 | Graph ML / GNN / typologies | `██████░░░░` | `███████░░░` |
| 6 | MuleTrack / sequence models | `███████░░░` | `████████░░` |
| 7 | Autoencoders / deep learning | `████████░░` | `█████████░` |
| 8 | Behavioural biometrics | `██████████` | `██████████` |

Each tier addresses a residual failure mode of the previous one:

| Tier | What it adds | What it cannot do alone |
|---|---|---|
| 1. Rules | Cheap, explainable, regulator-defensible | High false-positive rate; cannot see networks; brittle to novel typologies |
| 2. Unsupervised | Detects unknowns and concept drift | Harder to threshold; weaker against adversarial drift |
| 3. Supervised | Highest precision when positive labels exist (e.g. BOT HR-03) | Requires labels; sensitive to covariate shift |
| 4. Graph / GNN | Captures rings, multi-hop flows, shared-device cohorts | Compute-intensive; ring-level labels scarce |
| 5. Sequence + biometrics | Detects exploited or coerced mules where account behaviour looks normal | Often vendor-dependent; data-licensing considerations |

Mature programmes do not pick a tier — they operate all of them in parallel, calibrate them against one another, and combine them via a final decisioning layer. This ensemble approach is the design point of every named-bank deployment cited in §3–§7.

---

## 3. Tier 1 — Business-logic rules and typologies

### What is in production today

Rules remain the regulator-defensible core of any mule-detection programme. The published red-flag set is highly convergent across jurisdictions:

- **Pass-through ratio** (inflow ≈ outflow within 24–72 h, near-zero retained balance).
- **Velocity spikes** versus account history; **dormancy-to-activity** transitions.
- **Structuring** below CTR / PromptPay reporting thresholds.
- **Multi-account-per-device**, VPN usage, emulator detection, geolocation jumps.
- **KYC mismatch** — declared occupation vs flow size (the canonical "student moving ฿100M" pattern).
- **Counterparty fan-in / fan-out** asymmetry.

These rules are codified across several authoritative sources:

- **FATF** — *Professional Money Laundering* (July 2018) is the canonical typology document, with a dedicated mule-network section.
- **AUSTRAC** — student-mule guidance (2024) and red-flag indicator papers via the Fintel Alliance.
- **BOT** — circulars on the dark-brown / brown / orange / yellow account categories and HR-03 handling.

### Documented outcomes

- **Thailand** — the BOT and Thai Bankers' Association CFR have driven **>1.8M mule-account suspensions** through rule-based data sharing; more than **1,000 new scam cases per day** continue to be reported.
- **Europol EMMA 7–10** — most recent wave identified **10,759 mules** and led to **1,013 arrests**. Earlier waves: **2,469 mules** in one year; **228 arrests / 3,800 mules** in another; **422 arrests / 4,031 mules** in another.
- **TSB (UK)** — after deploying Mastercard's Consumer Fraud Risk score (a rules + ML hybrid), TSB reported a **+20% increase in fraud detection within four months**. If extrapolated across all UK banks, the indicative system-wide saving is **~£100M/year**.

### Limits of rules alone — and why mature programmes move up the stack

- **Network blindness.** Static rules see one account at a time. Mule networks are graph-shaped — multi-hop pass-throughs, shared devices, and herder rings are invisible to threshold-based detection.
- **False-positive load.** Pre-ML transaction monitoring at one major European bank (Danske Bank) generated approximately **1,200 false positives per day, of which 99.5% were unrelated to fraud** — roughly **6 true positives per 1,200 alerts**.
- **Adversarial drift.** When criminal syndicates change PromptPay format, account-age behaviour, or amount distributions, rules must be re-authored manually. Detection lags the typology by weeks.

---

## 4. Tier 2 — Unsupervised machine learning and anomaly detection

Unsupervised methods identify accounts that deviate from the population without requiring labels. This matters for two structural reasons:

- Confirmed-mule lists (HR-03, Cifas, Europol registers) cover only what has *already* been caught.
- Newly recruited and exploited mules (see §7) have no positive label at the point detection is needed.

### Methods used in banking

| Method | Original paper | Use in mule / AML |
|---|---|---|
| **Isolation Forest** | Liu et al., 2008 (ICDM) | Per-account outlier scoring on amount, velocity and ratio features |
| **One-Class SVM** | Schölkopf et al., 2001 | Stricter boundary; in published benchmarks outperforms Isolation Forest on highly imbalanced AML data |
| **Local Outlier Factor / DBSCAN** | Breunig et al., 2000 | Density-based outlier identification |
| **Autoencoders / Variational AE** | Kingma & Welling, 2013 | Reconstruction-error scoring on transaction sequences |
| **Markov / behavioural-state models** | Various | Detects abrupt regime shifts (dormant → active) |

### Documented results

- **OCBC Singapore + ThetaRay** — the first ML-based AML deployment by a Singaporean bank. Analysing one year of corporate-banking transactions, OCBC reported a **35% reduction in non-actionable alerts** and a **4× improvement in suspicious-transaction identification accuracy**. ThetaRay is unsupervised by design.
- **East-African commercial bank** (research deployment on 54,258 cross-border records) — a hybrid unsupervised deep-learning framework processed **1,000 transactions per second** with high-priority alert triage.
- **Variational autoencoders** in published AML evaluations have **halved the false-positive rate** versus prior baselines.
- **AutoEncoder + LightGBM** (PMC 11623290) — **AUC 96.83%, F1 80.27%** with SMOTE on an imbalanced fraud dataset.
- **Comparative evaluation**: One-Class SVM achieved **99.63% precision in the top-5% prioritised alerts**, outperforming Isolation Forest and LOF on the same AML benchmark.

### Limits of unsupervised methods

- Threshold tuning is dataset-specific; without calibration, alert volume is unstable.
- Explainability is weaker than rules or trees — investigators benefit from a clear narrative for SAR documentation, and reconstruction error alone is rarely sufficient.
- Concept drift requires periodic retraining; the assumption that "normal = majority" weakens during scam surges.

### Implication for the BOT / Thailand context

Unsupervised models are the natural second layer above HR-03 rules — they surface accounts that exhibit mule-like behaviour but have not yet been added to any confirmed-mule list.

---

## 5. Tier 3 — Supervised ML, especially gradient boosting (XGBoost) and PU-learning

Supervised learning typically provides the single largest accuracy improvement available **whenever positive labels exist**. The combination of the BOT HR-03 list, the AMLO confirmed-mule register and intra-bank confirmed-fraud cases provides exactly that label foundation for institutions in Thailand. This is the tier at which detection power begins to step-change.

### Why gradient-boosted trees are the dominant choice in deployed AML

- Tabular financial features (amounts, ratios, velocities, KYC fields) are well-suited to gradient-boosted trees.
- Built-in feature importance is defensible to compliance, regulators and internal model-risk management.
- Class imbalance is handled through `scale_pos_weight`, focal loss, or sampling techniques (SMOTE / ADASYN).
- Models train efficiently on CPU and integrate well with MLOps tooling.

### Published quantitative results

- **Hajek et al., 2022 (Information Systems Frontiers)** — *Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework*. This is the peer-reviewed source for the often-cited **~45% feature importance** for the "payment format / channel pattern" feature.
- **Nature *Scientific Reports* (2022)** — *Feature generation and contribution comparison for electronic fraud detection*: XGBoost with engineered features achieves **F1 = 78.3%** with strong interpretability.
- **IJSRED 2025**, *Fighting Money Laundering with Statistics and Machine Learning* — XGBoost outperforms alternatives with **precision = 94%, AUC-ROC = 0.97**. SHAP analysis identifies *large, frequent international transfers from a low-income profile* as the top driver — directly the BOT-style pattern.
- **Journal of Supercomputing 2023** — the ASXAML framework (XGBoost + RFECV + Optuna) automatically suppresses false-positive alerts.
- **Industry summary (Candir, 2025)** — production-scale deployment reports **AUC 97.5%** while processing more than 5M transactions at a 0.1% true laundering rate.
- **ACM 2024** — *Graph Feature Preprocessor*: XGBoost on **graph-derived** features achieves **+46% F1** over the same XGBoost on basic features. Graph features feeding a gradient-boosted tree is a high-leverage architectural pattern that most institutions can adopt before training a full GNN.

### Publicly disclosed bank deployments

- **Stripe Radar** — gradient-boosted core with neural-network ensembles. Stripe reports **+20% YoY ML performance**, a **+1.3 percentage-point payment-success-rate** lift from adaptive rules, **>30% fraud reduction** on eligible transactions for early adopters, a **17% reduction in dispute rates** as industry e-commerce fraud grew 15%, and **42% SEPA / 20% ACH** fraud reduction.
- **Itaú Unibanco (Brazil) + FICO Falcon** — cloud migration of fraud management is reported by FICO to **avoid >US$20M/month in fraud losses**, with **15% lower per-account cost** and **+20% CNP fraud detection**. Itaú additionally reduced ML deployment cycles **from up to 6 months to 3–5 days** on AWS SageMaker — directly relevant to operationalising mule scoring.
- **UOB Singapore** — first Singaporean bank to apply AI simultaneously to transaction monitoring and name screening. Published metrics: **96% true-positive rate** in the high-priority queue, **+5% TP and −50% FP** in transaction monitoring, **−70% FP for individual** and **−60% FP for corporate** name screening, with **<1% misclassification**.
- **Danske Bank (Denmark)** with Teradata Think Big — ML ensembles cut **false positives by 20–30%** in 12-week sprints; deep-learning models (TensorFlow) showed **double-digit additional detection improvements** in pre-production testing.

### Positive-Unlabeled (PU) learning — the technically-correct training paradigm for HR-03 labels

Banks rarely have a clean negative class — *unconfirmed* is not the same as *not a mule*. PU learning (Elkan & Noto, 2008) addresses exactly this case. Recent work applying it to mule / AML / fraud detection:

- **IJCAI 2021** — *Positive-Unlabeled Learning from Imbalanced Data* handles the combination of class imbalance and missing negatives.
- **arXiv 2412.06203 (2024)** — survey of PU and Negative-Unlabeled learning in cybersecurity, including financial-fraud applications.
- **ACM 2025** — *Enhancing Anti-Money Laundering by Money Mules Detection on Transaction Graphs* — explicit mule-targeted PU + graph hybrid.

PU-learning is the right paradigm when the BOT HR-03 list is used as the positive-only supervisor.

### Probability calibration

When investigators have a fixed daily review capacity, **probability calibration** (Platt scaling, isotonic regression) ensures that score thresholds map directly to expected case loads — the *Precision@Capacity* operating point.

---

## 6. Tier 4 — Graph and graph-neural-network methods

Mule operations are structurally graph-shaped: rings of accounts on shared devices, multi-hop pass-through chains, herder-recruit-deposit subgraphs. Graph analytics, and particularly graph neural networks (GNNs), have become the dominant technique for surfacing this structure.

### Publicly disclosed production deployments

- **HSBC + Quantexa** — the most widely cited public case study. After deploying Quantexa's Decision Intelligence (entity resolution + network analytics) from 2018, HSBC reports that the platform **auto-closed 1 million false-positive alerts**, reducing alerts requiring investigation by **83%** — saving the time of **140–180 analysts**. Industry summaries describe the outcome as "**4× more financial criminals identified while cutting false alarms by 60%**".
- **Danske Bank + Quantexa** — post-Estonian-scandal modernisation. Documented **60% reduction in false positives** with a **60% increase in fraud detection** and **50% reduction in false positives** on the payment-fraud workload.
- **Standard Chartered** — Quantexa case study on graph-driven AML detection and investigation prioritisation.
- **NatWest + Featurespace ARIC** — graph-augmented adaptive behavioural analytics: **+135% scam-detection rate** and **−75% false positives for scams** within 24 hours of deployment. The check-fraud variant detects **>90% of check fraud at a 5:1 FP ratio**.
- **Mastercard TRACE** — a network-level AML and mule platform. **United Kingdom**: **21 financial institutions** participating, covering **~90% of UK Faster Payments**, since launch in 2018; **thousands of mule accounts identified**, with **hundreds of new mule accounts identified every month**. **Asia Pacific**: launched February 2025 in the Philippines via **BancNet (36 domestic banks)** — a directly relevant comparator for any national-scale ASEAN rollout.
- **TMNL Netherlands** — five-bank cross-bank graph-monitoring consortium focused on human-trafficking, VAT-fraud and drug typologies. The published programme confirms that joint cross-bank monitoring surfaces signals invisible to any single bank.

### Published results on real bank graphs

- **DNB (Norway)** — heterogeneous GNN on Norway's largest bank, with **5M nodes and ~10M edges**. Among tested architectures, **GraphSAGE outperformed GAT and GCN**. A hybrid LSTM-GraphSAGE configuration achieved **95.4% accuracy** on simulated data; standalone GraphSAGE achieved **92.8% accuracy / 91.1% precision / 91.8% recall / 91.4% F1**.

### Benchmark performance on the IBM AMLworld synthetic dataset

- **NeurIPS 2023 Datasets & Benchmarks** — *Realistic Synthetic Financial Transactions for AML* (Altman et al., IBM). Publicly released **HI / LI** datasets, with large variants of **175–180M transactions**. Advanced GNN architectures (PNA, GIN+EU) significantly enhance performance and produce competitive results without manual feature engineering.
- **arXiv 2306.11586** — *Provably Powerful GNNs for Directed Multigraphs* (Multi-PNA / Multi-GIN). Improves the **minority-class F1 of standard message-passing GNNs by up to +30%** on AMLworld.
- **arXiv 2604.12241** — *BlazingAML*: high-throughput multi-stage graph-mining pipeline on AMLworld.
- **ACM 2024** — *Graph Feature Preprocessor*: a pre-processor that boosts a downstream XGBoost by **+46% F1** versus raw features. Graph-derived features fed into a gradient-boosted tree is a robust default for most institutions, with an optional GNN stage above.

### Why graph methods materially outperform tabular methods on mule networks

- **Multi-hop money flow** — a 3-hop pass-through cannot be detected from per-account features.
- **Shared-device / shared-IP rings** — cohort-level signal that does not exist at the row level.
- **Ring-level off-boarding** — the operational insight from HSBC + Quantexa: investigators can off-board entire networks rather than one account at a time, raising true-positive yield per investigator-hour substantially.

### Limits of graph methods

- **Compute intensity** — 100M+ edge graphs require infrastructure designed for distributed graph computation (Spark/GraphFrames or specialised GNN frameworks). §8.8 covers scaling techniques.
- **Label scarcity at ring level** — most institutions have account-level labels, not ring-level. Semi-supervised propagation, weak supervision and PU-learning all help.
- **Explainability** — investigators need to understand *why this ring*, not simply that the embedding scored high. Entity-resolution and visual graph context are what made HSBC's investigators able to trust the model output.

---

## 7. Tier 5 — Behavioural biometrics, sequence and deep-learning models

This tier exists to address the third mule archetype that the other tiers struggle to catch: the **exploited** account (account-takeover or coerced user). The account, its history, and its graph structure all look legitimate — the only available signal is *how the human interacts with the device during the session*.

### Three mule archetypes and the detection tier that fits each

| Archetype | Behaviour | Best-fit detection tier |
|---|---|---|
| **Complicit** (knowing accomplice) | Short-lived account, rapid activation, atypical spikes, no banking history | Tier 1 + Tier 3 (rules + XGBoost) |
| **Recruited** (lured via social media) | Normal baseline → dormancy → low-value testers → sudden spike | Tier 2 + Tier 4 (anomaly + graph) |
| **Exploited** (account-takeover or coerced) | Genuine active account, uncharacteristic velocity, new device usage | **Tier 5 (behavioural biometrics, sequence models)** |

### Behavioural biometrics — deployments and vendors

- **BioCatch** — the most-cited mule deployment. With a customer base of **257 financial institutions** in 2024, BioCatch reports that those institutions identified and acted on approximately **2–2.3 million mule accounts in 2024**. Detection signals include typing rhythm, swipe geometry, navigation flow, hesitation, copy-paste patterns and dwell times — signals that persist even when device, IP and account are otherwise "trusted".
- **Feedzai**, **NICE Actimize**, **LexisNexis ThreatMetrix**, **Mastercard NuData**, **Callsign**, **Revelock** — comparable behavioural-biometrics stacks with overlapping capability.
- **RBI MuleHunter.AI (India)** — the Reserve Bank Innovation Hub developed an in-house AI/ML mule-detection engine that codifies **19 distinct mule behaviour patterns**. The pilot with two large public-sector banks reported "encouraging results"; by August 2025 at least 15 banks had adopted it, and a December 2025 RTI confirmed **23 banks**. This is a flagship example of a regulator-led, ML-based modernisation programme replacing static rules.

### Sequence and temporal models

- **LSTM / GRU on transaction sequences** — captures temporal dependencies that pointwise XGBoost cannot. A hybrid LSTM + GraphSAGE configuration achieves **95.4% accuracy** on simulated data.
- **Temporal Graph Networks (TGN)** — Rossi et al. (2020). In production-style fraud benchmarks, TGN-based methods raise **Precision@20 Recall** from **68.5% → 86.2% (+17.7 points)** on Taobao and from **47.2% → 56.5%** on offline-merchant fraud, versus MLPs with hand-engineered real-time features.
- **Causal Temporal GNN (CaT-GNN, arXiv 2402.14708)** — preserves recall while gaining precision; an architectural response to concept drift.
- **Spatio-temporal attention GNN** — published metrics of **96.4% accuracy / 97.8% precision / 93.5% recall / 95.6% F1** on credit-card fraud.

### Deep-learning anomaly detection

- **Mastercard Consumer Fraud Risk** — an ensemble blending gradient-boosted trees, neural networks and graph signals over Faster Payments. UK pilot at TSB: **+20% fraud detection in four months**; indicative system-wide saving of **~£100M/year** if all UK banks matched TSB performance.
- **DBS Singapore** — real-time scoring flags high-risk transactions in **<10 ms**, with the bank reporting that **15% of customers' money is saved from scams**.

### Why this tier closes the loop

The exploited-account problem is the residual after rules, anomaly, supervised and graph detection have been applied. Without behavioural-biometrics signal, institutions either freeze legitimate customers (creating the BOT 4-hour unfreeze problem) or miss coerced-mule activity entirely. With biometrics and step-up authentication, this residual becomes addressable.

---

## 8. Technical deep-dives

This section addresses the technical questions that typically arise once the maturity ladder has been discussed. Each subsection is self-contained.

### 8.1 How PU-learning works (and why HR-03 calls for it)

**The problem.** When training a supervised mule classifier, a bank typically has:

- A set of confirmed mules (BOT HR-03, internal SAR confirmations, CFR feedback) — these are reliable positives.
- A much larger set of accounts *not* on any list. These are **unlabelled**, not negative. Many are legitimate customers, but some are mules that have not yet been caught.

Treating "not on the list" as `y = 0` and training a standard classifier (XGBoost, logistic regression) is a documented failure mode: the model learns to predict accounts that resemble *recorded* HR-03 cases, rather than accounts that exhibit mule behaviour. It systematically under-predicts on novel typologies because they were silently labelled negative during training.

**The theory.** *Elkan & Noto (2008), "Learning Classifiers from Only Positive and Unlabeled Data"*. Under the **Selected Completely At Random (SCAR)** assumption (the labelled positives are a uniform random sample of all true positives), the classifier trained on positive-versus-unlabelled data is **proportional** to the true positive-versus-negative classifier by a constant `c = P(s=1 | y=1)`, which can be estimated from a held-out positive set:

```
P(y = 1 | x) = P(s = 1 | x) / c
```

In plain terms: train as if unlabelled means negative, then divide the score by `c` to obtain a calibrated probability. The SCAR assumption is rarely strictly true in banking (HR-03 over-represents certain typologies), so practitioners use extensions — **nnPU** (non-negative PU, Kiryo et al. 2017) is the modern default because it stabilises training under high class imbalance.

**Python — the `pulearn` library.** Scikit-learn-compatible wrappers around an XGBoost base classifier:

```python
import numpy as np
from pulearn import ElkanotoPuClassifier
from xgboost import XGBClassifier

# y_pu convention for pulearn:
#   +1 = confirmed mule (HR-03 / SAR)
#   -1 = unlabelled (everyone else, NOT "confirmed clean")
y_pu = np.where(df["bot_confirmed_mule"], 1, -1)

base = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    tree_method="hist", eval_metric="aucpr",
)

pu_clf = ElkanotoPuClassifier(estimator=base, hold_out_ratio=0.2)
pu_clf.fit(X_train.values, y_pu_train)

# Calibrated mule probability — corrected for the c factor.
proba = pu_clf.predict_proba(X_test.values)
```

When more confirmed mules are available and the proportion of mules in the unlabelled set is expected to be larger:

```python
from pulearn import BaggingPuClassifier  # Mordelet & Vert (2014)

pu_clf = BaggingPuClassifier(
    base_estimator=base,
    n_estimators=15,           # bootstrap a balanced negative subsample per tree
    max_samples=sum(y_pu_train == 1),
    n_jobs=-1,
)
pu_clf.fit(X_train.values, y_pu_train)
```

The non-negative PU variant, when a calibrated estimate of class prior `π = P(y = 1)` is available:

```python
from pulearn import NonNegativePUClassifier
pu_clf = NonNegativePUClassifier(estimator=base, prior=0.003)  # 0.3% mule prior
pu_clf.fit(X_train.values, y_pu_train)
```

**Related libraries.** For research and benchmarking: `pu-learning` (Bekker & Davis lab, <https://github.com/aldro61/pu-learning>), TensorFlow / PyTorch implementations of nnPU for neural networks, and `LibPU` for graph-PU. All of these run unchanged in a Databricks Mosaic AI cluster; the trained classifier is registered in MLflow and served through Model Serving like any other model.

**What this provides.** Two concrete benefits:

1. **Higher recall on unseen typologies** — the model is no longer trained to treat unlabelled data as clean.
2. **Calibrated probabilities** — *Precision@Capacity* becomes a meaningful operating point because the score reflects `P(mule)` rather than an arbitrary rank.

---

### 8.2 What a graph neural network is (and why graphs outperform trees on rings)

**The motivation.** Two accounts can look identical at the row level — same KYC age, same monthly volume, same device — yet one is the hub of a 50-account mule ring and the other is an entirely legitimate small business. The difference is *who they transact with*. Tree-based models and rules see one row at a time; a graph neural network sees an account *and the learned embeddings of its neighbours*.

**The core idea in one paragraph.** A GNN stacks layers of **message passing**. At each layer, every node `v` constructs a new embedding by:

1. Collecting messages from its neighbours `N(v)` — each message is a function of the neighbour's current embedding and edge attributes (transaction amount, timestamp, channel).
2. Aggregating those messages — mean, sum, max, or attention-weighted aggregation. The choice differentiates GCN, GraphSAGE, GAT and PNA.
3. Combining the aggregated message with its own current embedding via a small MLP.

After `k` layers, every node's embedding incorporates information from accounts up to `k` hops away. For mule detection, `k = 2` or `k = 3` is generally sufficient — the receive-and-forward pattern is the primary signal, not the entire downstream chain.

**Popular architectures, ordered by maturity in real bank deployments:**

| Architecture | Aggregation | When to use |
|---|---|---|
| **GraphSAGE** (Hamilton et al. 2017) | Mean or LSTM aggregator over a sampled neighbourhood | Default for very large graphs (used in DNB Norway production at 5M nodes / 10M edges); efficient and scales well |
| **GAT** (Veličković et al. 2018) | Attention-weighted | When some counterparties matter substantially more than others |
| **PNA / Multi-PNA** (Corso et al. 2020 / Egressy et al. 2023) | Multiple aggregators concatenated | State of the art on IBM AMLworld — **+30% minority-class F1** over standard message-passing GNNs |
| **TGN** (Rossi et al. 2020) | Temporal — uses memory state per node | When transaction *order* matters (e.g., short-window pass-through); +17.7 P@20R in published fraud benchmarks |
| **R-GCN** (Schlichtkrull et al. 2018) | Per-edge-type weights | When edges carry semantics — payment vs shared-device vs shared-IP |

**Python — minimal GraphSAGE for mule scoring with PyTorch Geometric:**

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# Build the graph once:
#   x          : float tensor [num_accounts, num_features]  (rolling-window features)
#   edge_index : long tensor  [2, num_edges]                (transaction edges A -> B)
#   y          : long tensor  [num_accounts]                (1 if HR-03, 0 if unlabelled)
data = Data(x=x, edge_index=edge_index, y=y)

class MuleSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head  = torch.nn.Linear(hidden, out)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)  # logits per node

model = MuleSAGE(in_dim=data.x.size(1)).to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(50):
    model.train()
    opt.zero_grad()
    out = model(data.x.cuda(), data.edge_index.cuda())
    loss = F.cross_entropy(out[train_mask], data.y[train_mask].cuda())
    loss.backward(); opt.step()
```

Inference is `model(data.x, data.edge_index)`, taking `softmax(...)[:, 1]` as the per-account mule probability. On Databricks, a graph of up to ~50M accounts / ~500M edges runs on a single GPU node; larger graphs use neighbour sampling (`NeighborLoader`) or distributed DGL — see §8.8.

**Consider the lighter alternative first.** Before training a full GNN, the **Graph Feature Preprocessor** pattern (ACM 2024) is often sufficient: compute classical graph features (PageRank, community ID, k-core, two-hop ratio, triangle count, shared-device count) in GraphFrames and feed them as columns into XGBoost. Published lift is **+46% F1** over the same XGBoost on raw features. For many institutions this captures most of the value at a fraction of the implementation cost; a GNN is then introduced to address the residual.

---

### 8.3 What MuleTrack is (and how it fits the maturity ladder)

**MuleTrack** is the *Lightweight Temporal Learning Framework for Money Mule Detection in Digital Payments* by Jambhrunkar, Sharma, Singla and Kailasam, published in Springer LNCS as part of IWANN 2025 proceedings (2026). It is the most recent named, peer-reviewed mule-specific paper and is therefore frequently referenced as a maturity-tier example.

**What it actually is.** While MuleTrack is often grouped under "LSTM sequences" in summary materials, the paper's headline model is a **Markov chain over discretised account states**, not an LSTM:

1. Each account's daily / weekly activity is bucketed into a small set of behavioural states (e.g., `dormant`, `low-activity`, `burst-inbound`, `pass-through`, `cash-out`).
2. The framework learns a per-account state-transition matrix from the historical sequence.
3. Steady-state distributions and transition probabilities are compared against the population-level Markov chain of normal accounts; deviations score as mule-likelihood.
4. Domain heuristics (UPI-specific: per-day transfer count, fan-out, beneficiary diversity) are combined with the Markov-derived features.

**Why the authors chose Markov over LSTM.** Three factors are explicit in the paper and apply equally to PromptPay:

- **Inference latency** — 28-minute batch over the entire UPI population, no GPU dependency, no deep-learning retraining cycle.
- **Interpretability** — a state-transition matrix is auditable; an LSTM internal state is not.
- **Cold-start performance** — works on accounts with very short histories, which matters for the *recruited* mule archetype where syndicates activate the account after weeks of dormancy.

**Where MuleTrack fits in a Thai-bank stack.** It sits between Tier 3 (XGBoost) and Tier 5 (deep sequence models). It captures the temporal regime change (dormant → burst → pass-through) that a pointwise XGBoost record will miss, without requiring the GPU and training pipeline of an LSTM or TGN. The pragmatic deployment pattern is to treat MuleTrack as a **complementary feature generator**: compute Markov state-transition deviations per account on a daily Lakeflow job, materialise them as columns in the Feature Store, and let the supervised tier consume them. This delivers temporal signal at a fraction of the cost of a sequence neural network.

**Reference:** Jambhrunkar et al., *MuleTrack*, Springer LNCS, IWANN 2025 (2026). <https://link.springer.com/chapter/10.1007/978-3-032-02725-2_30>

---

### 8.4 The incomplete-graph problem: how graph detection works when most mule activity crosses institutional boundaries

Every bank's transaction graph is **structurally incomplete**:

- A mule receives funds from a customer of Bank A, holds them briefly, and sends them onward to Bank B, then Bank C, then a crypto exchange, then offshore. Bank A sees only the first outbound edge; the remainder is invisible.
- Cifas UK data indicates that approximately **half of mule pass-through hops cross bank boundaries** within 24 hours.
- Once funds leave the country (USDT-on-Tron is a common Thai off-ramp), no domestic bank sees them again.

A naïvely trained GNN inside a single bank's silo therefore learns the leaves of mule networks but not the hubs. The literature and the deployed-vendor practice have converged on **five complementary strategies** to address this:

**1. Use the partial signal effectively.** Most mule rings have *some* intra-bank density — recruits often onboard at the same bank, herders test with internal accounts first. Even a 10% intra-bank density is sufficient for community detection and PageRank to surface meaningful structure.

**2. Use HR-03 (and equivalents) as 1-hop counterparty features.** Direct visibility into another bank's mule transactions is unnecessary if it is known that a customer sent THB 200k to a HR-03-listed account at another bank. Joining the BOT CFR / HR-03 list onto the counterparty side of every outbound transaction yields one of the highest-signal features available to a single institution, and requires no new consortium agreement.

**3. Subscribe to a network-operator view of the cross-bank graph.** **Mastercard TRACE** is the operative example: 21 UK institutions covering ~90% of UK Faster Payments; APAC launched February 2025 in the Philippines via BancNet (36 banks). TRACE consumes the inter-bank rail directly and emits per-account risk scores that participating banks ingest as features. A Thai analogue would be a BOT-CFR feed enriched with rail-level signal — a natural use case for a Lakehouse-native consortium platform.

**4. Privacy-preserving cross-bank collaboration.** Under PDPA-class regimes the following can be shared legally:

- Hashed counterparty identifiers and risk scores.
- Aggregated subgraph fingerprints (ring size, age, fan-out distribution) without individual identities.
- Federated GNN gradients — train one model on partitioned graphs without moving the raw edges (see *Privacy-Preserving Graph-Based ML with Fully Homomorphic Encryption*, arXiv 2411.02926, 2024).
- Bloom-filter or PSI-based "is this account on your watchlist?" queries.

**Databricks Clean Rooms** productises this pattern with cryptographic isolation and audit. TMNL Netherlands, Singapore COSMIC and FinCEN 314(b) provide established regulatory precedent that supervisors already understand.

**5. Treat the cross-border / crypto tail as a separate problem.** Once funds reach a VASP or leave the country, on-chain analytics specialists (Chainalysis, Elliptic, TRM Labs) take over. The pragmatic pattern is to **ingest outbound risk scores from these specialists** as feature columns, rather than attempt to extend the bank graph onto chain.

**Recommended sequencing.** A bank starting from intra-bank-only graph features can adopt these strategies independently and incrementally: (1) intra-bank graph with HR-03 as a counterparty feature → (2) BOT-CFR ingestion as a global label feed → (3) Mastercard-TRACE-style operator data when available → (4) bilateral or consortium Clean Room with one or two peer banks for the highest-density cross-bank corridors → (5) on-chain feed for the off-ramp tail. Each step is independently shippable and delivers measurable signal lift; none of them require a regulator-driven utility to begin working.

---

### 8.5 Data sources and where they typically reside in a Thai bank

The table below maps the canonical mule-detection data domains to the systems on which Thai banks typically host them, and the recommended ingestion pattern into the Lakehouse.

| Data domain | What it contains | Where it typically lives in a Thai major bank | Ingestion pattern into the Lakehouse |
|---|---|---|---|
| **Core banking — accounts** | Account master, open/close date, type, branch, status, daily balance snapshots | IBM mainframe (`z/OS` + DB2), occasionally Oracle Exadata for newer institutions; the major Thai banks predominantly run DB2 | Lakeflow Connect (DB2 CDC) → Bronze Delta; daily reconciliation for balance snapshots |
| **Payments — domestic transfers** | PromptPay (NITMX rail), BAHTNET (high-value RTGS), ATM withdrawals, intra-bank transfers | Switch systems (BCMS / EPS / proprietary) writing to messaging queues (typically IBM MQ); ITMX provides near-real-time event streams; ATM via switch journals | Lakeflow + Kafka / Auto Loader → Bronze; <1 minute freshness is achievable |
| **Payments — international and cards** | SWIFT, Visa / Mastercard authorisations and clearing | SWIFT Alliance, card switch (Tieto / TSYS / FIS depending on the bank), settlement systems | Batch (clearing) plus streaming (auth); typical SLA 5–15 minutes |
| **KYC / onboarding** | Declared income, occupation, nationality, address, ID documents, FATCA, eKYC selfie + liveness | Dedicated KYC platform (Fenergo, NICE Actimize KYC, in-house Java) backed by Oracle or DB2; documents in object storage or a content-management system (FileNet, Documentum) | Lakeflow CDC from KYC DB; documents land via volume mount; OCR / embeddings computed on demand |
| **Digital channels / sessions** | Login events, device fingerprint, IP, geolocation, session duration, in-app navigation, failed-auth events | Mobile / internet-banking platform — typically Hadoop / Cloudera, Elastic or a modern stream platform; device intelligence from a partner (BioCatch, ThreatMetrix, NuData) | Auto Loader from log buckets; partner APIs as Silver tables |
| **Cards** | Card-level activity, MCC, chargeback history, dispute outcomes | Card-processor systems (Tieto, FIS, in-house); often siloed from current-account data | Batch nightly + streaming auth via Kafka |
| **CRM / contact** | Phone, email, address history, customer-service notes, complaint records | Salesforce or in-house CRM; complaint cases in ServiceNow | API connectors / Lakeflow |
| **Credit bureau / external** | NCB (National Credit Bureau Thailand) reports, credit score, total outstanding debt, query history | Procured per-customer via NCB API; cached internally | Lakeflow / DLT with explicit consent logging |
| **BOT / regulator feeds** | **HR-03 high-risk register**, dark-brown / brown / orange / yellow account categories, Central Fraud Registry (CFR) entries | Distributed by the Thai Bankers' Association to member banks; typically delivered as encrypted batch files or via a secure-portal API | Daily Lakeflow ingestion into a Unity-Catalog-governed Silver table; lineage tracked back to BOT for audit |
| **Investigator / case management** | SAR drafts, fraud confirmations, freezing actions, customer escalations | ServiceNow or an in-house case-management system; SAR submissions via DataPro / BOT eForm | API or DB sync into a Silver case table; closes the loop for label generation |
| **Alternative data** | Telco signals, geolocation, behavioural-biometrics scores | Per-vendor APIs (AIS / dtac / Truemove via licensed brokers; BioCatch / Feedzai BB for biometrics) | API → Silver |

**Three observations that typically emerge in early architecture discussions:**

1. **The mainframe is real and will remain in scope for the foreseeable horizon.** Workable patterns rely on DB2 CDC or batch unloads, not on prior modernisation of the core banking system.
2. **The largest source of data fragmentation is usually the digital-channel platform.** Mobile-banking session events — which carry the strongest signal for the *exploited* mule archetype — tend to live on a separate Hadoop or stream platform from payments data. Unifying these two domains under one governed Lakehouse is the highest-ROI integration in most engagements.
3. **The BOT CFR / HR-03 feed should be treated as a first-class data product.** It should be versioned, lineage-tracked and ingested with explicit point-in-time semantics for model training — models must be trained on labels that were *knowable as of t*, not on labels that arrived later. Temporal leakage is the most common pitfall in supervised AML modelling.

---

### 8.6 The investigator experience — current state and a Lakehouse-native target state

Investigator productivity is where operational ROI is realised. A model with an AUC of 0.99 cannot deliver its full value if the analyst requires 45 minutes to clear each alert.

#### Current state in a typical major Thai bank

1. An alert lands in a transaction-monitoring queue (commonly NICE Actimize, SAS AML, or an in-house rules engine).
2. The analyst opens 4–8 separate applications to assemble context:
   - Core-banking screen (DB2)
   - KYC system for declared occupation and income
   - CRM for contact and complaint history
   - Mobile-banking session log (sometimes a separate Splunk dashboard)
   - Call-centre notes (ServiceNow)
   - Manual lookup of counterparty HR-03 status
   - Locally maintained spreadsheet of previously seen accounts
3. The analyst stitches a working narrative in Excel or Word — typically **30 minutes to several hours per case** depending on complexity.
4. A decision is recorded — file SAR, freeze, escalate or close — and submitted via DataPro / BOT eForm.
5. A typical analyst workload is **15–40 alerts per day**, of which more than **99% are false positives** at pre-ML baselines.

The structural consequences of this workflow are:

- The **BOT 4-hour unfreeze SLA** is operationally challenging during scam surges, because the manual context-assembly process does not parallelise.
- Time-pressured analysts default to the cautious decision, contributing to wrongful freezes.
- The same true-positive mule ring is often investigated several times by different analysts, each seeing only a single account in isolation — there is no mechanism for ring-level off-boarding.

#### Lakehouse-native target state

A unified investigator workspace, delivered as a Databricks App backed by Lakebase (managed PostgreSQL) for sub-50 ms reads of the 360 view, with Genie for natural-language drill-down. The illustrative layout below corresponds to the `mule-explorer` reference implementation in this repository:

```
┌──────────────────────────────────────────────────────────────────┐
│ Alert #4192 — account THB-12...8821 — risk 0.94 — ring R-302    │
├──────────────────────────────────────────────────────────────────┤
│ 360 view (Lakebase, <50ms):                                      │
│   KYC: student, 22yo, declared income 15k THB/mo, onboarded 11d │
│   Recent: 87 inbound (THB 4.2M) and 84 outbound (THB 4.1M) /3d  │
│   Devices: 3 new devices in 24h, all from same IP /24            │
│   Top SHAP drivers: pass_through_ratio (0.98), age_vs_volume     │
├──────────────────────────────────────────────────────────────────┤
│ Ring view (GraphFrames + cached layout):                         │
│   [interactive graph — 14 accounts, 2 HR-03 hits, 1 cross-bank] │
│   Hover any node for its 360. Click to expand 1 more hop.        │
├──────────────────────────────────────────────────────────────────┤
│ Ask Genie:                                                       │
│   "Show me every counterparty in this ring that received >50k    │
│    from a new device in the last 7 days"                         │
│   [executes SQL against gold tables, renders inline]             │
├──────────────────────────────────────────────────────────────────┤
│ Actions:                                                         │
│   [ Freeze account ]  [ Off-board entire ring ]  [ Draft SAR ]   │
│   [ Escalate to L2 ]  [ Mark false positive — feeds AL queue ]   │
└──────────────────────────────────────────────────────────────────┘
```

**The architecture supporting this experience:**

- **Lakehouse Bronze → Gold** pipelines unify every data source from §8.5 into a governed 360 table per account.
- **Lakebase** (managed PostgreSQL on Databricks) holds the hot serving copy of the 360 table, ring memberships and score history, synced from the Lakehouse via reverse-ETL synced tables. Read latency for the app is **<50 ms**.
- **The Databricks App** (Plotly Dash in this reference implementation) renders the layout, queries Lakebase for facts and calls MLflow Model Serving for live re-scoring.
- **Genie** sits on top of the same Gold tables, providing natural-language-to-SQL access for the long tail of ad-hoc analyst questions.
- **An active-learning loop** captures every "mark false positive" and "confirm mule" decision back into a Silver table that feeds the next PU-learning retrain.

#### Operational implications

| Metric | Current state | Lakehouse-native target |
|---|---|---|
| Cases / analyst / day | 15–40 | 80–150 (consistent with the published Quantexa / HSBC band) |
| Time per case | 20–60 min | 3–10 min |
| Off-boarding granularity | Per account | Per ring |
| Compliance with BOT 4-hour unfreeze | Best-effort | Default — alerts arrive pre-explained |
| Label feedback to model | Spreadsheet → quarterly retrain | Real-time → weekly retrain |

The decisive operational advantage of this pattern is that **the full 360 view, the ring graph and the model explanation arrive in one screen at sub-50 ms read latency**, because the investigator app sits directly on top of the same Lakehouse that produced the scores. No point-vendor that does not own the underlying data foundation can deliver this single-screen experience.

---

### 8.7 Evolving from a point-solution stack toward a unified platform

A common starting position is a portfolio of specialist vendors — a rules engine, an anomaly-detection product, a graph platform, a behavioural-biometrics product — each with its own data store, its own model lifecycle and its own governance. The Lakehouse pattern does not require replacing all of these. The recommended approach is to **re-layer** the architecture: the Lakehouse becomes the data platform; the specialist products that provide genuinely differentiated capability become *features* into it.

The table below summarises what is typically modernised on the Lakehouse versus what is retained as a feature provider.

| Layer | Common current state | Lakehouse pattern |
|---|---|---|
| Rules engine on legacy infrastructure | Vendor rules engine on Hadoop / proprietary store | **Modernised on the Lakehouse.** Rules run in DLT / SQL against governed Delta tables, with full lineage. |
| Anomaly detection | Vendor anomaly engine (e.g., ThetaRay) | **Either retained as a feature provider or rebuilt on Mosaic AI.** Many institutions find that rebuilding is faster than the next renewal cycle once data is unified. |
| Supervised ML scoring | Vendor-hosted gradient-boosted models | **Modernised on the Lakehouse.** XGBoost / LightGBM with PU-learning, MLflow versioning and Unity Catalog lineage — with full model transparency for MRM. |
| Graph analytics and entity resolution | A specialist graph platform (e.g., Quantexa) | **Complementary.** Specialist entity-resolution and visual investigator UX remain best-in-class. Ingest entity-resolution edges into Delta and consume risk scores as features; the Lakehouse owns the data plane and the ensemble. |
| Behavioural biometrics | Vendor biometrics platform (e.g., BioCatch, NuData, Feedzai BB) | **Complementary.** Ingest per-session scores via API into Silver and feature-engineer on top. |
| Cross-bank network signal | Network operator (e.g., Mastercard TRACE) where available | **Complementary.** Ingest TRACE risk scores per account as features. |
| Case management and SAR submission | ServiceNow or vendor case management | **Either retained or replaced.** Most institutions retain ServiceNow for workflow and audit, and add a Databricks App on top for the actual investigation work. |

#### Why a re-layering approach is generally preferred over a full replacement

1. **Lower risk.** No big-bang migration. Existing vendors continue to operate while the Lakehouse runs in parallel and progressively takes over tiers as detection metrics warrant.
2. **Predictable cost.** The vendor contracts that are easiest to retire are the rules engines and supervised-ML services — the tiers the Lakehouse handles natively. The vendor contracts that are most painful to retire (specialist graph platforms, behavioural biometrics) are also the ones that retain a clear differentiated value when used as feature providers.
3. **Lower lock-in.** Today a typical stack has five vendors, five data copies, and five governance regimes. In the target state there is one platform (Unity Catalog) and N feature providers — any individual provider can be swapped without re-platforming.
4. **Faster iteration.** Creating a new feature inside the Lakehouse is a notebook plus a Feature Store entry. Creating the equivalent inside a closed vendor product typically requires a statement of work.
5. **Stronger regulatory posture.** Regulator expectations for model-risk management map cleanly onto MLflow + Unity Catalog lineage. A vendor-hosted scoring service that cannot be inspected end-to-end is a more difficult MRM conversation.

#### TCO crossover heuristic

Per-event vendor pricing in AML / fraud is typically in the range of **USD 0.001–0.005 per transaction scored**. On a serverless Lakehouse the equivalent compute cost crosses below this at roughly **10–30 million transactions per day**. A Thai major bank typically exceeds that threshold on PromptPay volume alone; mid-tier banks reach it within 18 months at current PromptPay growth.

**The architectural conclusion.** The Lakehouse becomes the system of record for data, features, models and lineage. Specialist vendors are integrated as feature providers where they retain a clear capability advantage. The bank gains a single governed data plane while preserving investments in specialist capability that genuinely contribute to detection quality.

---

### 8.8 Scaling graph-feature computation at production volumes (the "500M-node" question)

Most published graph-AML studies run on graphs that fit on a single machine. DNB Norway's heterogeneous-GNN deployment used **5M nodes and ~10M edges**; even the *large* IBM AMLworld benchmark tops out at **~50M nodes and 180M transactions**. A Thai major bank's PromptPay-plus-corporate transaction graph routinely sits at **~500M nodes and several billion edges per quarter** (30–50M individual customers, plus all counterparties at other banks, plus merchants, plus historical edges). A 12-month rolling window pushes past 1B edges.

At that scale, naïve PageRank on a single driver does not complete, naïve Louvain runs for days, and a three-layer GraphSAGE forward pass over the full neighbourhood is computationally infeasible because the receptive field grows combinatorially. The literature and deployed-vendor practice have converged on **seven proven scaling strategies**; most production deployments combine three or four of them simultaneously.

#### Strategy 1 — Prune before computing

The cheapest order-of-magnitude reduction is to remove nodes that cannot contribute meaningful mule signal. The literature on graph reduction and k-core decomposition (Batagelj & Zaveršnik, 2003) provides the formal basis. In practice the following heuristics typically drop **60–85% of nodes** with negligible signal loss:

- **Degree pruning** — drop accounts with `degree < 2` over the rolling window. These are terminal leaves that cannot route money and therefore cannot be mules in the structural sense.
- **k-core extraction** — compute the **2-core** or **3-core** of the graph. A k-core is the maximal subgraph in which every node has degree ≥ k. Mule rings exhibit reinforcing connectivity and survive k-core filtering; legitimate retail customers with one occasional bill-pay relationship do not. Empirically the 2-core of a retail-banking transaction graph is **~10–25% of the original node count**.
- **Dormancy filtering** — remove accounts with zero activity in the last `T` days from the graph for graph-feature purposes (they remain in the rules and supervised-ML pipelines). Mule activation is bursty; dormant nodes do not require real-time graph scoring.
- **Amount-thresholded edges** — collapse very-low-value edges (e.g., `< THB 100`) into aggregated weights or drop them. Mule pass-throughs are by design above reporting thresholds.
- **Supernode masking** — large merchants and government agencies act as graph supernodes with millions of incoming edges, which distort PageRank. Mask them out or compute a masked PageRank where supernode contributions are zeroed. This is the standard treatment in published AML PageRank work.

A typical reduction for a Thai major bank is from a 500M-node working graph to a **75–150M-node analytically-relevant graph** — the point at which the next strategies become tractable.

#### Strategy 2 — Partition the graph

Once distribution is required, the question is how the graph is split across workers.

- **Edge-cut partitioning** (METIS — Karypis & Kumar, 1998; Spinner — Martella et al., 2017) splits vertices across machines and replicates edges that cross. Effective for bounded-degree graphs.
- **Vertex-cut partitioning** (PowerGraph — Gonzalez et al., OSDI 2012; PowerLyra — Chen et al. 2015) splits edges across machines and replicates high-degree vertices. **This is generally the preferred default for banking transaction graphs**, where degree distributions are heavy-tailed (a small number of hub accounts — payment aggregators, payroll accounts, e-wallet bridges — have millions of edges, while most accounts have <10). PowerGraph's empirical result on power-law graphs is **>10× communication reduction** versus edge-cut.

In Databricks the practical realisations are:

- **GraphFrames** on Spark — edge-cut by default; for very skewed graphs the `aggregateMessages` Pregel API can be used with custom partitioning on `srcId % N` after explicit degree-aware skew handling (broadcasting the top supernodes and processing them separately).
- **NVIDIA cuGraph** for vertex-cut on a single multi-GPU node — handles 1B-edge graphs on a single 4×GPU machine and integrates with Databricks GPU runtimes.

**Streaming partitioners** such as **LDG** (Stanton & Kliot, KDD 2012) and **HDRF** (Petroni et al., CIKM 2015) are appropriate when edges arrive from Lakeflow as a stream — partition assignments are computed online with no global view and locality only **~5% worse than offline METIS**.

#### Strategy 3 — Sample rather than enumerate

For GNN training and inference on large graphs, **neighbourhood sampling** is the dominant technique. The four canonical methods, ordered by maturity in production deployments:

| Method | Paper | What it does | Trade-off |
|---|---|---|---|
| **GraphSAGE neighbour sampling** | Hamilton et al., NeurIPS 2017 | At each layer, sample a fixed `k` neighbours per node | Simple, scales linearly with number of nodes; used in DNB Norway production |
| **FastGCN** | Chen et al., ICLR 2018 | Importance-sample nodes per layer rather than per node | Faster training; small variance penalty |
| **ClusterGCN** | Chiang et al., KDD 2019 | METIS-partition the graph and train on one cluster per mini-batch | Memory-efficient; loses some inter-cluster edges; strong empirical performance |
| **GraphSAINT** | Zeng et al., ICLR 2020 | Subgraph-sample whole subgraphs each mini-batch, with bias correction | Best published accuracy / speed trade-off at the time |

For **classical graph features** (non-GNN), the equivalent insight is **Personalised PageRank from a seed set**: rather than computing global PageRank, compute Personalised PageRank seeded at the HR-03 confirmed mules. The result is a *proximity-to-known-mule* score per account, produced via the **local push algorithm** (Andersen, Chung & Lang, FOCS 2006) in `O(1/ε)` work *independent of graph size*. This is the highest-impact approximate-algorithm idea for mule detection: only the part of the graph reachable from known mules ever needs to be touched.

#### Strategy 4 — Use approximate algorithms with provable error bounds

Exact graph algorithms are wasteful when a 1–5% error is operationally invisible:

- **Approximate PageRank** — push algorithm (Andersen et al. 2006); Monte Carlo PageRank (Bahmani et al., VLDB 2011) — `O(n log n)` work for ε-approximate ranks.
- **HyperLogLog** for cardinality of neighbours and unique counterparties — `O(1)` memory per node for ≤2% error; available as a first-class function in Spark SQL (`approx_count_distinct`).
- **MinHash / LSH** for Jaccard similarity between account neighbourhoods — surfaces "two accounts transacting with nearly the same set of counterparties" (a strong shared-controller signal) without `O(n²)` pairwise comparisons.
- **HyperBall** (Boldi, Rosa & Vigna, 2011) for approximate closeness and harmonic centrality — orders of magnitude faster than exact computation.
- **Local clustering / Nibble** (Spielman & Teng, STOC 2004) — find the community around a seed node in time proportional to the *community* size rather than the graph size. The cluster around an HR-03 node is, by construction, the mule ring.

The recommended pattern is a **two-tier compute approach**: approximate everywhere by default, with exact computation reserved for the candidate set surfaced by the approximate pass.

#### Strategy 5 — Compute incrementally, not from scratch

A common operational inefficiency is recomputing all graph features daily. Mule networks change slowly; the graph is largely stable day-to-day. Three incremental patterns are well established:

- **Snapshot plus delta** — compute the full feature set weekly and maintain it incrementally with edge insertions and deletions. **Dynamic PageRank** algorithms (Bahmani, Chowdhury, Goel, VLDB 2010) update ranks in time proportional to the changed subgraph rather than the whole graph.
- **Streaming community detection** — incremental Louvain (Cordeiro et al., 2016) keeps community labels stable across updates, which matters because investigators rely on consistent ring identifiers across days.
- **Event-driven recompute** — subscribe to graph events of interest (a new edge to or from an HR-03 node, a new shared-device link) and recompute features only for the affected ego-network (~2 hops). This is the Lakeflow + Streaming Tables pattern.

In practice, the daily fresh-compute target shrinks from "500M nodes" to "the few percent of nodes that have changed materially since yesterday" — typically fewer than 5M nodes.

#### Strategy 6 — Use the right hardware

Compute density is decisive at this scale:

- **GPU acceleration via NVIDIA cuGraph / RAPIDS** — Louvain on **1B-edge graphs in under 60 seconds** on a single 8×GPU node (published cuGraph benchmarks). PageRank, BFS and connected components are similarly accelerated by **50–500×** versus CPU-only Spark.
- **Photon / serverless Spark** for the wide-scan operations (edge counting, degree calculation, group-bys). Photon's vectorised execution is 3–10× faster than vanilla Spark on the same compute budget.
- **DGL-distributed / PyTorch-Geometric distributed** for multi-GPU GNN training. The DGL distributed trainer scales near-linearly to ≥8 nodes.

The deployable Databricks recipe is: **Spark / Photon for ingestion, pruning and classical aggregations; a GPU cluster with cuGraph for the heavy global algorithms (PageRank, Louvain, k-core, connected components); GPU nodes for GNN training and inference.**

#### Strategy 7 — Cache aggressively in a feature store

Once a graph feature is computed, it should not be recomputed for read traffic. Two layers:

1. **Databricks Feature Store** (Delta-backed) — the system of record for per-account graph features (PageRank, community ID, two-hop ratio, k-core number, shared-device count). Versioned, lineage-tracked, joinable in MLflow training.
2. **Lakebase** (managed PostgreSQL) — the hot serving copy synced from the Feature Store via reverse-ETL synced tables. Read latency **<50 ms** for the Databricks App and investigator workflow.

This decouples **producer cadence** (heavy graph compute on a daily or hourly GPU schedule) from **consumer latency** (sub-second reads from Lakebase during investigator drill-down). It is the same OLAP/OLTP pattern implemented internally by leading graph-analytics and fraud-detection vendors; the Lakehouse exposes it as a first-class platform capability rather than as a proprietary implementation.

#### Putting it together for a 500M-node bank graph

The deployable recipe, in roughly the order it would be adopted:

1. **Ingest the full edge stream** into Bronze Delta with Lakeflow. Do not filter at ingest — retain the raw record for audit.
2. **Build the analytical graph in Silver** with the Strategy-1 pruning stack (degree, k-core, dormancy, supernode masking). Typical reduction: **500M → ~100M nodes**.
3. **Compute classical features** (degree, in/out ratio, two-hop ratio, HyperLogLog distinct counterparties, MinHash sketches) in Photon SQL. These are the cheapest features and provide most of the lift via the Graph Feature Preprocessor pattern (**+46% F1** on downstream XGBoost).
4. **Run global algorithms on GPU** — Louvain via cuGraph for community assignment; approximate PageRank for centrality; connected components for ring discovery. Weekly snapshot, daily incremental updates.
5. **Run Personalised PageRank seeded at HR-03 nodes** via the local push algorithm — a proximity-to-known-mule score per account.
6. **Train a GNN** (GraphSAGE with neighbour sampling, or ClusterGCN where cluster structure is informative) for the residual signal that classical features do not capture. Use the pruned analytical graph rather than the raw 500M-node version.
7. **Materialise the per-account feature vector** into the Feature Store and sync to Lakebase. The investigator app reads from Lakebase at <50 ms latency; the training pipeline reads from the Feature Store with point-in-time correctness.
8. **Drive recomputation incrementally** — Lakeflow streaming tables propagate new edges into affected ego-networks only; nightly batch reconciles drift.

This recipe is consistent with the published practice at HSBC + Quantexa (graph entity resolution + classical features dominate, GNN as residual), with the DNB Norway heterogeneous-GNN paper (GraphSAGE neighbour-sampling), and with cuGraph reference deployments at NVIDIA fraud-detection customers. With this recipe in place, the "500M nodes" question becomes a routine engineering exercise rather than a research problem.

#### Algorithm and platform references

- Karypis & Kumar, *METIS: A Software Package for Partitioning Unstructured Graphs*, 1998.
- Gonzalez et al., *PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs*, OSDI 2012.
- Andersen, Chung & Lang, *Local Graph Partitioning using PageRank Vectors*, FOCS 2006.
- Bahmani, Chakrabarti & Xin, *Fast Personalized PageRank on MapReduce*, SIGMOD 2011.
- Bahmani, Chowdhury & Goel, *Fast Incremental and Personalized PageRank*, VLDB 2010.
- Spielman & Teng, *Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear systems*, STOC 2004.
- Hamilton, Ying & Leskovec, *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS 2017.
- Chiang et al., *Cluster-GCN: An Efficient Algorithm for Training Deep and Large GCNs*, KDD 2019.
- Zeng et al., *GraphSAINT: Graph Sampling Based Inductive Learning Method*, ICLR 2020.
- Batagelj & Zaveršnik, *An O(m) Algorithm for Cores Decomposition of Networks*, 2003.
- Boldi, Rosa & Vigna, *HyperANF: Approximating the Neighbourhood Function of Very Large Graphs*, 2011.
- NVIDIA cuGraph documentation and benchmarks: <https://docs.rapids.ai/api/cugraph/stable/>.
- GraphFrames: <https://graphframes.io/>.

---

## 9. Cross-cutting design choices on which the literature is unanimous

1. **Rolling-window features** (1-day, 7-day, 30-day) are universally used because mule activation timescales are bursty.
2. **Ratio features** (in/out, balance/volume) outperform raw amounts.
3. **Counterparty entropy** is a strong Tier-2/Tier-3 signal — low entropy indicates repetitive structuring.
4. **Calibrated probabilities** combined with capacity-tuned thresholds outperform hard-coded score cut-offs (the *Precision@Capacity* objective).
5. **Ensembling rules, ML and graph signals** outperforms any single model. Every named-bank deployment cited in this document follows this principle.
6. **Investigator-in-the-loop / active learning** is now standard practice. Feedback loops from confirmed and rejected alerts feed the next training round.
7. **Cross-bank consortium signal** (TMNL, COSMIC, TRACE, FinCEN 314(b), AUSTRAC Fintel Alliance, BOT CFR) is what unlocks ring-level detection; any single-bank model is structurally blind to cross-institutional patterns.

---

## 10. Comparator-bank summary table

| Bank / regulator | Tech stack | Headline number | Source |
|---|---|---|---|
| HSBC + Quantexa | Entity resolution + graph | 1M FP auto-closed; 83% alert reduction; 4× criminals identified; -60% FP | Quantexa case study; industry summaries |
| Danske Bank + Quantexa | Graph + ML ensembles | -50–60% FP; +60% fraud detection | Quantexa; Best Practice AI |
| Danske Bank + Teradata Think Big | ML ensembles + DL (TensorFlow) | 20–30% FP reduction; double-digit DL gains in pre-prod | Teradata case |
| NatWest + Featurespace ARIC | Adaptive behavioural analytics + graph | +135% scam detection; -75% scam FP within 24 h | Featurespace |
| Mastercard CFR (TSB, NatWest, Lloyds, Halifax, Monzo, BoS, AIB) | Graph + ML over Faster Payments | TSB +20% fraud detection in 4 months; ~£100M/yr UK projection | Mastercard; FCA; TSB |
| Mastercard TRACE UK | Network-level AML | 21 banks; ~90% of UK FPS; thousands of mules; hundreds new monthly | Mastercard |
| Mastercard TRACE APAC (Philippines / BancNet) | Network-level AML | 36 banks onboarded Feb 2025 | Mastercard; PRNewswire |
| RBI MuleHunter.AI (India) | ML, 19 behaviour patterns | 23 banks adopted by Dec 2025 | RBIH; medianama RTI |
| OCBC + ThetaRay | Unsupervised anomaly | -35% non-actionable alerts; +4× detection accuracy | OCBC; ThetaRay |
| UOB Singapore | AI on transaction monitoring + name screening | 96% TP high-priority; -50% FP TM; -70% FP individual screening | UOB press release |
| DBS Singapore | Real-time scoring | <10 ms decision; 15% of customer scam funds saved | DBS |
| BioCatch (vendor, 257 FIs) | Behavioural biometrics | ~2M–2.3M mule accounts identified in 2024 | BioCatch |
| Stripe Radar | GBT + neural ensembles | +20% YoY ML; -30% fraud (eligible); -42% SEPA / -20% ACH fraud | Stripe |
| Itaú + FICO Falcon | Supervised + cloud | -US$20M/month fraud loss; +20% CNP detection | FICO |
| TMNL Netherlands | Cross-bank graph TM | Joint monitoring surfaces signals invisible to single banks | TMNL |
| BOT + Thai Bankers Assoc CFR | Rules + cross-bank registry | >1.8M mule accounts suspended | Bangkok Post; Thailand Business News |

Vendor-reported figures should be cross-checked against the primary source listed in §12 before being relied on in formal documentation.

---

## 11. Recommended target architecture for a Thai bank

The literature, the regulator guidance and the publicly disclosed bank deployments converge on a single architectural pattern:

1. **Unified data foundation** — KYC, payments, sessions, device telemetry and BOT HR-03 / AMLO labels in a single Lakehouse governed by Unity Catalog.
2. **Tier 1 rules** — FATF, BOT and internal typologies codified in DLT / SQL as guardrails (not as the detector); also used to generate weak labels for Tier 3.
3. **Tier 2 unsupervised** — Isolation Forest / autoencoder scoring for the accounts not yet on any confirmed list.
4. **Tier 3 supervised** — XGBoost or LightGBM trained with PU-learning on HR-03 plus confirmed-fraud positives; outputs calibrated to *Precision@Capacity*.
5. **Tier 4 graph** — graph-feature pre-processor (community ID, PageRank, two-hop ratio, k-core, triangle counts) feeding Tier 3, plus a GNN (PNA / Multi-GIN / TGN) for ring-level scoring.
6. **Tier 5 sequence and biometrics** — LSTM / TGN over transaction sequences plus a behavioural-biometrics partner (e.g., BioCatch, NuData, Feedzai BB) for the exploited-mule tail.
7. **Calibration and decisioning** — Platt / isotonic calibration; thresholds tuned to investigator review capacity; hybrid scoring across rules + ML + graph; explainable case packages for SAR.
8. **Active-learning loop** — investigator confirmations feed back into PU-learning labels and graph supervision.
9. **Cross-bank signal** — bidirectional integration with the BOT CFR; track Mastercard TRACE APAC as it expands beyond the Philippines.

---

## 12. Sources

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
- AUSTRAC, Fintel Alliance: <https://www.austrac.gov.au/partners/fintel-alliance>; Fintel Alliance Annual Report 2023–24: <https://www.austrac.gov.au/sites/default/files/2024-11/Fintel%20Alliance%20Annual%20Report%20Extract%202023-24.pdf>; foreign-student mule guidance: <https://www.austrac.gov.au/news-and-media/media-release/new-guidance-released-help-combat-use-foreign-students-money-mules>
- TMNL: <https://tmnl.nl/en/article/transaction-monitoring-netherlands-adapts-its-working-method-to-new-european-legislation/>
- FinCEN Section 314(b): <https://www.fincen.gov/section-314b>; Fact Sheet (PDF): <https://www.fincen.gov/sites/default/files/shared/314bfactsheet.pdf>
- FATF, *Professional Money Laundering* (July 2018): <https://www.fatf-gafi.org/content/dam/fatf/documents/Professional-Money-Laundering.pdf>

### Bank deployments and vendor case studies

- Quantexa, HSBC case study: <https://www.quantexa.com/resources/hsbc-contextual-decision-intelligence-raising-the-bar-of-aml-technology/>; "How Decision Intelligence is Helping HSBC Combat Financial Crime and Fraud" (PDF): <https://www.quantexa.com/assets/x/75b2f9dc6f/hsbc-case-study-jennifer-calvery-financial-crime.pdf>
- ChiefAIOfficer, "How HSBC's AI Catches 4× More Financial Criminals While Cutting False Alarms by 60%": <https://chiefaiofficer.com/blog/how-hsbcs-ai-catches-4x-more-financial-criminals-while-cutting-false-alarms-by-60/>
- Quantexa, Danske Bank case study: <https://www.quantexa.com/resources/danske-bank/>
- Best Practice AI, "Danske Bank increases payment fraud detection by 60% and reduces false positives by 50%": <https://www.bestpractice.ai/ai-case-study-best-practice/danish_danske_bank_increases_payment_fraud_detection_by_60%25_and_reduces_false_positives_by_50%25_with_machine_learning>
- Teradata, "Danske Bank Saves Millions Fighting Fraud With Deep Learning and AI" (PDF): <https://assets.teradata.com/resourceCenter/downloads/CaseStudies/CaseStudy_EB9821_Danske_Bank_Saves_Millions_Fighting_Fraud_With_Deep_Learning_and_AI.pdf>
- Featurespace, NatWest scam detection +135% / FP -75%: <https://www.featurespace.com/newsroom/natwest-improves-scam-detection-rate-by-135-using-featurespaces-technology>; ARIC overview: <https://www.featurespace.com/aric-risk-hub>
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
- "Automatic suppression of false positive alerts in AML systems using machine learning", *Journal of Supercomputing*, 2023: <https://link.springer.com/article/10.1007/s11227-023-05708-z>
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
- Jambhrunkar, Sharma, Singla, Kailasam, "MuleTrack: A Lightweight Temporal Learning Framework for Money Mule Detection in Digital Payments", Springer LNCS / IWANN 2025 (2026): <https://link.springer.com/chapter/10.1007/978-3-032-02725-2_30>
- "Privacy-Preserving Graph-Based ML with Fully Homomorphic Encryption for Collaborative AML", arXiv 2411.02926 (2024): <https://arxiv.org/html/2411.02926v2>
- `pulearn` Python library — Elkan-Noto, nnPU, Bagging-PU wrappers for scikit-learn: <https://pulearn.github.io/pulearn/>; GitHub: <https://github.com/pulearn/pulearn>
- `pu-learning` (Bekker & Davis lab): <https://github.com/aldro61/pu-learning>
- PyTorch Geometric documentation: <https://pytorch-geometric.readthedocs.io/>; DGL: <https://www.dgl.ai/>
- Feedzai, "Behavioral Biometrics for Money Mule Account Detection": <https://feedzai.com/blog/behavioral-biometrics-for-money-mule-account-detection/>
- Feedzai, "Inbound Payment Fraud Detection and Mule Risk Modelling" (Bank Negara Malaysia, PDF): <https://www.feedzai.com/wp-content/uploads/2024/11/Feedzai-Inbound-Payment-Fraud-Detection-and-Mule-Risk-Modeling-Complying-with-Bank-Negara-Malaysia-Recommendations.pdf>

---

*Document last revised May 2026. Vendor-published figures should be cross-checked against the primary source before being cited in formal materials.*
