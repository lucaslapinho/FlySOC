# Executed FlySOC v0.1 results

Training run: `run_20260914T174213877086Z_eeac25`.
Initial evaluation: `eval_20260914T174359077810Z_240cca`.
The evaluation was repeated after improving plot labeling and handling exports
without labels; the repeated run preserves the earlier output directory.

Public aggregate evidence: [benchmark snapshot](benchmarks/README.md),
[metrics](benchmarks/summary.json), [training measurements](benchmarks/training.json).
Full prediction/model artifacts remain local and can be regenerated.

## Data and executed configuration

- Python 3.12.14; CPU-only execution in an isolated virtual environment.
- Seed 42; 5,000 synthetic alerts, 15 families.
- Verdict distribution: 1,965 TP, 1,675 FP, 1,360 benign.
- Chronological split: 3,500 training / 750 validation / 750 test.
- Credential-dumping family: absent from training and validation; 225 test alerts.
- Test: 525 known-family and 225 unseen-family alerts.
- PN dimensions: 4,096. KC dimensions: 8,192. Fan-in: 8. Top-k: 64.
- Every encoded alert in training, validation and test had exactly 64 active KCs.
- Regenerating the dataset after the timestamp compatibility fix produced the
  identical CSV SHA-256:
  `10158db6dd4637d2d61a40f4a39dc95c1249bcb7c4c5b28d0ec77215014e3dc5`.

## Retrieval: both representations recover recurrent families

| Representation | Precision@5 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| Fly, Jaccard | 0.9916 | 0.0198 | 0.9954 |
| Original PN, cosine | 0.9935 | 0.0198 | 0.9965 |

The conventional feature-space baseline is slightly better on this run. Recall
has all same-family historical alerts as its denominator, so a short result list
can have high precision and low recall. These scores use the 525 eligible known
queries; the 225 unseen queries are counted separately, not treated as successes.

The inspect command executed successfully for `ALERT004251`, a backup alert.
Its first three historical matches had Jaccard 1.0 and benign labels. Its novelty
score was 0, and the separate Fly Logistic Regression classifier predicted benign.
The query was a test alert and was not present in the historical reference bank.

## Classification: results depend on the downstream classifier

| Representation | Classifier | Full-test macro F1 | Known-only macro F1 |
|---|---|---:|---:|
| PN | Logistic Regression | 0.7011 | 0.9367 |
| PN | Random Forest | 0.8052 | 0.9367 |
| PN | kNN | 0.6902 | 0.9387 |
| Fly | Logistic Regression | 0.6967 | 0.9367 |
| Fly | Random Forest | 0.8750 | 0.9350 |
| Fly | kNN | 0.7345 | 0.9290 |

The largest full-test F1 was produced by Random Forest on Fly fingerprints. This
is one observed comparison, not model selection validated on an independent
dataset. Known-family scores are similar, and the withheld-family shift accounts
for much of the full-test degradation.

Per-class metrics for the two Random Forest models on all 750 test alerts:

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| PN RF | BENIGN | 0.9000 | 0.9184 | 0.9091 | 147 |
| PN RF | FALSE_POSITIVE | 0.5748 | 0.9301 | 0.7105 | 186 |
| PN RF | TRUE_POSITIVE | 0.9532 | 0.6835 | 0.7961 | 417 |
| Fly RF | BENIGN | 0.6850 | 0.9320 | 0.7896 | 147 |
| Fly RF | FALSE_POSITIVE | 0.9553 | 0.9194 | 0.9370 | 186 |
| Fly RF | TRUE_POSITIVE | 0.9542 | 0.8489 | 0.8985 | 417 |

Higher macro F1 does not mean every class improved: benign precision was worse
for Fly RF. The full confusion matrices and all six models' per-class results
are saved in the metrics and prediction artifacts.

## Novelty: strong distance separation, failed Isolation Forest baseline

| Method | AUROC | AUPRC (AP) | TPR | FPR | Precision |
|---|---:|---:|---:|---:|---:|
| Fly nearest neighbor | 1.0000 | 1.0000 | 1.0000 | 0.0019 | 0.9956 |
| PN nearest neighbor | 1.0000 | 1.0000 | 1.0000 | 0.0057 | 0.9868 |
| PN Isolation Forest | 0.0151 | 0.1693 | 0.0000 | 0.0038 | 0.0000 |

Fly NN flagged all 225 unseen alerts and one of 525 known alerts. PN NN flagged
all unseen alerts and three known alerts. Fly's threshold was 0.117647; its mean
known and unseen scores were 0.011069 and 0.844779 respectively.

Isolation Forest missed all 225 unseen alerts. Its correctly oriented
`-score_samples` were lower on the unseen family (mean 0.315261) than on known
families (0.334142). The poor result is retained; the score direction was not
reversed after observing test labels. The cause requires further investigation.

Both NN validation flag rates were 1.33%, illustrating that a 99th-percentile
training calibration does not guarantee exactly 1% future flags. Perfect AUROC
on a deliberately distinct synthetic family does not establish real SOC novelty
performance. AUPRC also depends on the artificial 30% unseen prevalence.

## Deduplication: reduction includes harmful merges

| Representation | Alerts | Clusters | Reduction | Purity | False merge pairs / all collapsed pairs |
|---|---:|---:|---:|---:|---:|
| Fly Jaccard >= 0.75 | 750 | 131 | 82.53% | 96.80% | 195 / 5,267 = 3.7023% |
| PN cosine >= 0.75 | 750 | 25 | 96.67% | 99.73% | 3 / 22,843 = 0.0131% |

Eight Fly clusters mixed families. These results do not support deploying this
configuration as automatic suppression. Jaccard and cosine thresholds must be
selected independently on validation to compare at the same false-merge budget.
The families are coarse synthetic truth; even same-family merges may combine
distinct real incidents.

## Runtime and storage from the first evaluation

- Fly pipeline fitting, including NN calibration: 4.36 seconds.
- Random Forest training: PN 1.92 seconds; Fly 16.76 seconds.
- End-to-end Fly test encoding: 0.485 ms per alert in a batch.
- Individual warmed Top-5 median query latency: Fly 5.93 ms; PN 2.97 ms.
- Training PN CSR storage: 983,300 bytes.
- Training fingerprint CSR storage: 1,134,004 bytes.
- Sparse projection: 540,676 bytes; compressed model bundle: 5,846,184 bytes.

Fly fingerprints did not reduce storage in this CSR implementation, and Fly
queries were slower in this measurement. Storage measurements exclude other
process allocations and are not peak RAM. Encoding is an extra computation on
top of PN extraction. Timings can vary between evaluations.

## Validation and next research step

The suite reached 37 passing tests, including a complete train/evaluate test with
an export lacking verdicts and family annotations. Independent output checks
parsed JSON/CSV/PNG files, recomputed 12 confusion matrices and novelty AUC/flags,
checked probability sums, verified temporal separation and model/split hashes,
and confirmed retrieval references belong to training. Plots were also inspected
visually. The synthetic dataset reproduced byte-for-byte from seed 42.

The next experiment should compare multiple seeds and held-out families, then
select per-representation deduplication thresholds on validation under a fixed
false-merge budget. Real-world conclusions require de-identified exports and
analyst-reviewed similarity/incident labels. MBON-style learning and Cortex API
integration remain future work.
