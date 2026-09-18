# Changelog

All notable user-facing and research changes to FlySOC are documented here.

## 2026-09-18 — Integrated research platform

### Added

- A new Home page that connects the research status, Brain Lab, and Alert Analyzer.
- A dedicated Brain Lab at `/lab.html` for live sparse-activation and connectome views.
- A dedicated Alert Analyzer at `/analyze.html` with manual entry and JSON, CSV, or TSV import.
- A visual alert journey from feature extraction through PN/KC activation, sparse fingerprinting,
  associative-memory retrieval, novelty scoring, and downstream verdict prediction.
- Research modules for connectome projection, projection controls, novelty baselines,
  MBON-inspired feedback experiments, and threshold calibration.
- A sample import file at `docs/examples/alert_import_sample.csv`.

### Changed

- The original Brain Lab interface now lives on its own route while preserving shared navigation.
- Alert IDs and model prediction metadata are preserved through the analysis API.
- Preprocessing, experiment configuration, and evaluation code now support the expanded research pack.
- Documentation now distinguishes the Mushroom Body-inspired engineering abstraction from a literal
  biological simulation and separates the SOC pipeline from connectome visualization experiments.

### Validation

- `47` automated tests passed on Python 3.12.
- The Brain Lab validation completed `50` deterministic traces and artifact integrity checks.
- Home, Brain Lab, and Alert Analyzer were checked in desktop and mobile layouts without browser
  console errors.

### Current limits

- Reported SOC measurements are based on synthetic alerts and do not establish production efficacy.
- The public connectome is visualized and explored separately; it is not used as the alert classifier.
- Verdict prediction is produced by an explicit downstream classifier, not by FlyHash alone.
