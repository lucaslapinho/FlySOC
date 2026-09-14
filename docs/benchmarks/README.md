# Initial numerical evidence

These aggregate metrics were copied from the executed synthetic experiment:

- Training: `run_20260914T174213877086Z_eeac25`
- Evaluation: `eval_20260914T175152547064Z_2a1583`
- Seed: 42; 5,000 alerts; chronological 3,500 / 750 / 750 split.
- Synthetic CSV SHA-256: `10158db6dd4637d2d61a40f4a39dc95c1249bcb7c4c5b28d0ec77215014e3dc5`.

`summary.json` preserves the aggregate classification, retrieval, novelty and
deduplication results. CSV files provide comparison tables. `training.json`
and `performance.json` record observed timing/storage, not portable guarantees.
`config.yaml` records the parameter set used. No real SOC telemetry, serialized
models, private machine manifest or raw connectome data are distributed here.

See [interpretation and caveats](../initial_results.md). Recreate detailed local
predictions and plots with the generation, training and evaluation commands in
the README. A benchmark snapshot is evidence from one run, not a model ranking.
