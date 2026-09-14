# Reproduction environment

Initial measurements used Windows build 26200 (25H2), Python 3.12.14 x64 and
an isolated virtual environment. Package versions are recorded in
`requirements-lock.txt`; supported ranges are in `requirements.txt`.

Use an editable installation from a Git checkout. The current paths for data,
models and results are resolved relative to that source checkout. A virtual
environment is machine-specific and is intentionally not distributed.

The automated workflow targets Python 3.12 on Windows and Linux. Its badge
reports actual remote status; a configured workflow alone is not a passing run.
No dedicated GPU or system-wide Python dependency installation is required.
The 3D browser viewer requires WebGL2.

For cache permission problems, use `PIP_NO_CACHE_DIR=1` or configure a writable
pip cache. Matplotlib uses a local results cache. Generated manifests record
machine paths and environment details for local reproducibility; inspect and
redact those before sharing full experiment directories publicly.
