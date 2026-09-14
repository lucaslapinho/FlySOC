# Experimental methodology

## Shared protocol

The default dataset contains 5,000 alerts with seed 42. The oldest 70% train the
feature extractor, projection, memory, classifiers and novelty estimators. The
next 15% form validation, and the final 15% form test. Identical timestamps are
never split across boundaries; ratios become approximate when boundary ties
occur. Missing or invalid timestamps fail explicitly. Random train/test splitting
is not used.

`family`, `is_unseen`, verdicts, alert IDs, rule IDs and MITRE annotations are
excluded from features. Operational category and human-readable alert names are
retained: these may still be strong proxies for family and should be ablated in
future experiments. Training and test share recurring templates by design.

Defaults are fixed before testing. Validation results are saved as a separate
diagnostic; v0.1 performs no hyperparameter search. The historical retrieval bank
contains training rows only. Validation is not folded into the final model.

## Experiment 1: similarity retrieval

Test queries search training history using Jaccard on Fly fingerprints and
cosine on original PN features (which combine TF-IDF, categorical and numeric
features). The latter is the feature-space baseline, not pure TF-IDF alone.

Relevance is exact equality of the synthetic operational family. P@K counts
relevant results among K returns. Recall@K divides retrieved relevant rows by
**all** relevant training rows; with many repeated patterns it can be small even
when P@K is high. AP@K uses `min(K, relevant training rows)` as denominator.
NDCG uses binary relevance. MRR is truncated at the stated K. K values are 1, 5
and 10; if history is smaller, effective K is recorded.

Only queries with a same-family historical reference enter ranking metrics.
Held-out queries have no relevant history and are counted as excluded, then
evaluated explicitly in novelty. Retrieval predictions still contain all test
queries, including held-out ones.

## Experiment 2: recurrent verdict recognition

Logistic Regression, Random Forest and distance-weighted cosine kNN are fitted
to each representation using labeled training rows. Report validation, full
chronological test, known-family test and unseen-family test separately.
Metrics include per-class precision, recall, F1, support, confusion matrices,
one-versus-rest AUROC and average precision when both positive and negative
examples exist. Undefined AUCs are omitted; missing numeric values serialize as
JSON null. The representation itself is not called a classifier.

Synthetic verdict noise and overlapping rule names prevent perfectly
deterministic label lookup. However, templates still contain strong operational
cues and do not model real analyst disagreement, delayed labels or concept drift.

## Experiment 3: unseen-family novelty

All credential-dumping/LSASS patterns are withheld from both training and
validation. Exactly 225 of the 750 default test alerts belong to that family;
525 belong to families observed in training. No T1003-prefixed annotation is
present in training. This is a family-withheld experiment, not a demonstration
that a novel event is malicious.

Fly novelty is one minus nearest Jaccard similarity. PN novelty is one minus
nearest cosine similarity. Each threshold is the 99th percentile of training
leave-one-out nearest-neighbor distances. Self matches are excluded, while
other duplicate records remain valid historical examples. Flags require
`score > threshold`; ties at the threshold are not flagged.

Isolation Forest on PN features is the additional novelty baseline. Higher
novelty means negative `score_samples`. Its threshold is the 99th percentile of
**in-sample training** scores; this calibration differs from NN leave-one-out
and may be optimistic. Historical data mixes TP, FP and benign alerts: the
objective is family familiarity, not one-class benign training.

Report AUROC, AUPRC computed as average precision, TPR, FPR, precision, recall,
threshold and known/unseen mean scores. AUPRC depends on the deliberately chosen
30% unseen prevalence. No predictive value at real SOC prevalence is implied.
Histograms use common bin edges per estimator, density normalization and the
training threshold. Validation NN flag rates help reveal threshold drift.

LOF and One-Class SVM are deferred: the minimum implementation already includes
two distance baselines and Isolation Forest. If LOF is added, its
[novelty-mode scoring must use unseen rows](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html).

## Experiment 4: deduplication

Process test alerts in chronological order. Match each alert to the most similar
existing representative at similarity >= 0.75, or create a new cluster. This
exemplar method is order-dependent and does not guarantee pairwise similarity
among all members. It does not implement production incident correlation,
asset/time-window restrictions, alert suppression or response actions.

Report original alert count, cluster count and percentage reduction. Weighted
purity is the sum of each cluster's largest family count divided by all alerts.
False merge rate is unrelated-family pairs divided by all within-cluster pairs.
Also report unrelated pair counts and the fraction of non-singleton clusters
that mix families. With no collapsed pairs, false merge rate is recorded as zero
alongside a zero denominator. Reduction alone is not a success criterion.

The same 0.75 threshold is applied to Jaccard and cosine for a descriptive first
run. These are different similarity scales. A fair operating-point comparison
requires selecting each threshold on validation under the same false-merge
constraint, then freezing it before test evaluation.

## Runtime and memory

Training time is recorded per pipeline, classifier and novelty baseline.
Encoding reports PN latency and end-to-end Fly latency, including PN extraction.
Retrieval reports batched throughput and median/p95 of 30 warmed individual Top-5
queries. Timings are workstation observations, not controlled benchmarks.

Storage metrics measure CSR arrays for PN, fingerprints and projection plus the
serialized model file. They are not peak process RAM. The sparse index keeps
its own float32 copy; fingerprints are not bit packed. Query computation still
scales linearly with historical bank size. Dense allocations are limited to KC
and similarity batches, but larger datasets need resource measurement and
possibly inverted or approximate retrieval.

## Artifacts and reproducibility

Each dataset, training run and evaluation gets a unique timestamp/UUID suffix.
Only `latest.json` pointers are updated. Existing dataset/model/evaluation files
are preserved. Every run contains configuration, dataset and model hashes,
source hashes, exact dependency versions, Python/platform details, split counts
and time ranges, and CSV snapshots of the three partitions. Model/split hashes
are checked before evaluation. `requirements-lock.txt` pins the installed
environment; `requirements.txt` describes supported version ranges.

`joblib` artifacts are trusted local files. Do not load an untrusted model file:
pickle-based deserialization can execute code. Hashes detect accidental changes
relative to the manifest; they are not signatures or an authenticity guarantee.
CSV outputs preserve telemetry: spreadsheet software may interpret formula-like
cells, so import real exports as text when reviewing them outside FlySOC.

## Limits and next experiments

This initial run is a functional research baseline. It has one seed, one small
synthetic generator, a single held-out family and no confidence intervals.
Repeated templates and coarse family truth limit conclusions about semantics,
incident identity and operational safety. Time separation alone does not remove
campaign/template dependence.

Next: multiple seeds, fan-in/top-k sweeps, feature ablations, more held-out
families, drift/missingness stress tests, and human-labeled relevance pairs from
de-identified SOC exports. Use validation for threshold selection and reserve a
future temporal test window. No production alert suppression is justified by
these experiments.
