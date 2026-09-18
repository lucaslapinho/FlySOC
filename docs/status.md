# Research status and roadmap

## Implemented and measured

- Deterministic generation of 5,000 synthetic SOC alerts and chronological splits.
- Sparse preprocessing, random KC expansion, Top-K fingerprints and memory feedback.
- Retrieval, verdict classification, nearest-neighbor novelty and deduplication.
- Feature-space baselines, six classifiers and Isolation Forest comparison.
- Unique run artifacts, explicit metric definitions and independent output validation.
- FlyWire download with provenance, full-graph abstract diffusion and 3D exploration.
- English live-trace interface, manual JSON input, playback and local launchers.
- Isolated experimental KC-to-MBON readout with configurable reward, bounded
  incremental updates and per-event audit history.
- Optional ConnectomeHash backend, verified MaleCNS artifact builder/downloader,
  degree-preserving rewired control and deterministic engineered feature bridge.
- Validation-calibrated LOF/One-Class SVM modules, per-representation deduplication
  calibration, multiple-seed configuration and configurable feature ablation.
- 43 passing local tests before publication, plus full-data numerical validation.

The new architecture has received only import, compilation and small synthetic
smoke checks. No MaleCNS dataset, benchmark, model training or scientific result
was produced during this integration.

## Research priorities

1. Repeat experiments across seeds and held-out families; report uncertainty.
2. Calibrate deduplication thresholds on validation with a fixed false-merge budget.
3. Compare a connectome-derived ALPN/KC projection to random connectivity under
   matched representation size, sparsity and classifier settings.
4. Evaluate authorized de-identified SOC exports and analyst-reviewed relevance.
5. Study temporal drift, campaign separation, feature ablations and retrieval cost.
6. Evaluate the MBON-inspired layer prequentially across reward policies, seeds
   and held-out families; report calibration, stability and clipping frequency.

## Psychology and SOC operations

A longer-term learning objective is to understand how analysts recognize patterns,
allocate attention, use experience and make decisions under alert volume. Questions
about fatigue, biases and feedback should be studied through an explicit human-study
protocol with consent and privacy protections. No current experiment contains human
participants, measures psychological outcomes or models human cognition.

Learning about fly circuitry and studying analyst behavior are complementary
research interests; neither establishes that a fly brain explains human psychology.

## Not implemented or claimed

- A complete physiological simulation of a fly brain.
- A measured result using MaleCNS topology to classify SOC alerts.
- Production Cortex API integration or automatic alert suppression.
- General superiority, validated real-world detection rates or a peer-reviewed finding.
- Integration of experimental MBON updates into the default pipeline or
  `AssociativeMemory`.

See [measured results](initial_results.md), [methods](experiments.md) and
[Brain Lab scope](brain_lab.md) for evidence and limitations.
