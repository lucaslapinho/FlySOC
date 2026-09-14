# Contributing to FlySOC

FlySOC welcomes reproducible research, bug fixes, documentation and careful
negative results. Contributions to original project code and documentation
are made under the MIT license. Third-party data retain their own licenses.

## Development

1. Fork the repository and create a branch for a focused change.
2. Create a Python 3.12 virtual environment and install the project in editable
   mode using the README instructions.
3. Run `python -m pytest -q`. Run the relevant experiment when changing an
   algorithm or metric. Browser changes should be checked in both Brain Lab modes.
4. Open a pull request describing the problem, resulting behavior, validation
   and any limitations. Include the configuration and seeds for research changes.

Keep modules small, preserve sparse operations and use type hints where helpful.
Do not execute telemetry strings. Do not silently change metric denominators,
drop failed baselines, or select thresholds on the test set.

## Research proposals

Describe the hypothesis, baseline, data source, chronological split, leakage
controls, metrics and planned decision criteria before reporting conclusions.
Distinguish code correctness from scientific validation. Report uncertainty
and failures, including results that do not favor FlySOC.

Do not submit real SOC telemetry, credentials, private infrastructure identifiers,
untrusted serialized models, or datasets without redistribution rights. Synthetic
minimal reproductions are preferred for issues. Human analyst studies require
appropriate consent, privacy protections and institutional review where applicable.

Psychology and analyst decision-making are future research interests; current
results do not validate a psychological model or describe human participants.
