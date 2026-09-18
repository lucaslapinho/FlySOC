# Brain Lab: connectome exploration and live FlySOC traces

The English interface contains two independent experiments. **Live FlySOC**
traces the trained random-projection model on saved test alerts or a manual JSON
object. **Real connectome** applies abstract diffusion to FlyWire connectivity.
The connectome is not yet used as the alert representation.

## Setup and controls

Follow the README to create a virtual environment, generate synthetic alerts,
train the model, download/prepare the connectome and vendor Three.js. A fresh
Git clone contains source code, not prepared datasets or model binaries.

Run `python scripts/brain_lab.py --open-browser` with the project environment.
The server listens on http://127.0.0.1:8765; `--port 8766` selects another port.
The root address opens the shared Home, `/lab.html` opens Brain Lab and
`/analyze.html` opens Alert Analyzer. All three screens use the same loopback
Python process and provide navigation to each other.
Keep its terminal open. Ctrl+C stops it. The optional Windows PowerShell launcher
reuses an existing Brain Lab; the CMD launcher starts a new server directly.

1. Select **Real connectome**, choose ALPN and click **Apply pulse**.
2. Click **Next step**. In the measured snapshot, 685 initially stimulated ALPN
   neurons yield 11,604 neurons above the display threshold after one step.
3. Use **Start**, **Pause**, **Auto-rotate**, zoom and **Show active signal only**.
   The expand button enlarges the viewer; Escape collapses it.
4. Select **Live FlySOC** and process a saved test alert. Inspect the 64 winning
   KCs, input connections, five nearest historical alerts and novelty score.
5. Choose **Unseen family** to inspect credential-dumping examples excluded from
   training. A classifier can predict a known verdict incorrectly while novelty
   correctly flags an unfamiliar input.
6. Use **Test a JSON alert** for a flat object. Telemetry is never executed.

During replay, the selector advances to the next alert; the result panel shows
what has just been computed. Live means per-request Python computation, not
live SOC ingestion. The pulse state is shared between browser tabs on one server.

## Interface design and interpretation

The console uses a three-part layout: experiment controls, the interactive 3D
scene and the live readout. This keeps configuration separate from computed
results. The statistic strip summarizes the active mode, while the processing
trace names each computational stage without implying an anatomical pathway.

Teal identifies the FlyWire diffusion experiment. Violet identifies Live
FlySOC and its engineered PN/KC representation. These colors are interface
semantics only; they do not encode neurotransmitter identity, excitation,
inhibition or biological cell state. Amber scope cards mark interpretation
limits that should remain visible when screenshots are shared.

The layout collapses to a single column on small screens, preserves keyboard
focus indicators and honors reduced-motion preferences. The 3D canvas still
requires WebGL2. The textual controls, metrics and scientific-scope notes remain
available when the viewport is narrow.

## Alert Decision Journey

Live FlySOC presents each completed alert computation as a six-stage replay:

1. **Alert** — displays selected telemetry as inert data.
2. **PN** — highlights nonzero engineered feature coordinates.
3. **KC** — shows the strongest positive expansion activations.
4. **Top-K** — isolates the sparse binary FlyFingerprint winners.
5. **Memory** — reveals nearest historical alerts, similarity and novelty.
6. **Hypothesis** — shows class probabilities from the separately trained
   Logistic Regression classifier on the fingerprint.

Previous, replay and next controls allow the researcher to inspect any stage.
The browser does not recompute or fabricate intermediate values: Python first
returns one complete trace, then the viewer reveals fields from that trace in
sequence. “Computed trace replay” describes the timing accurately; it is not a
claim about biological firing time or streaming SOC latency.

The final hypothesis is analyst support. It is not an automatic incident
verdict, and FlyHash remains a representation rather than a classifier. The
novelty label only measures distance from historical fingerprints; unfamiliar
does not mean malicious. Historical verdicts shown beside matches are recorded
context and are not ground truth for the current alert. The current interface
does not feed analyst actions back into the saved classifier.

## Alert Analyzer

The separate `/analyze.html` screen is the intake path for user-supplied alerts.
It offers a common-field form, pasted JSON and local JSON/CSV/TSV import. JSON
may be one flat object, an array of objects, or an object containing an `alerts`
array. Delimited files require a unique, nonempty header row and support quoted
values. The included `docs/examples/alert_import_sample.csv` is a safe template.

The browser applies these limits before analysis:

- maximum file size: 5 MB;
- maximum imported records: 1,000;
- maximum fields per record: 100;
- flat scalar values only; nested objects and arrays are rejected;
- maximum selected-alert JSON size: approximately 60 KB.

The complete file is not uploaded to the Python service. It is parsed locally,
shown in a one-row preview and kept in browser memory. Only the selected alert
is posted to the loopback `/api/fly/trace` endpoint. Imported alert IDs are
preserved; a missing ID receives `MANUAL`.

After computation, the canvas replays the same six stages as Live FlySOC using
actual PN indices, KC activations, winning indices, contributing edges, memory
matches and prediction probabilities. Its brain-like layout is schematic and
does not use anatomical coordinates. The detailed result intentionally omits
the full 8,192-value activation array while retaining the active indices,
summary counts, novelty, matches, prediction and timing.

## Real data and provenance

The four compressed CSV files come from the public URL pattern in the
[official loader](https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py).
URLs, sizes, Last-Modified and SHA-256 values are preserved in the local
`data/connectome/fafb783/download_manifest.json`. Existing downloads are verified
and reused. Changed files are not silently substituted for a cached snapshot.

| Measured property | Value |
|---|---:|
| Neurons | 139,255 |
| Annotated coordinate points | 238,909 |
| Neurons with coordinates | 139,255 |
| Connection rows by region | 3,869,878 |
| Aggregated directed neuron pairs | 2,700,513 |
| Sum of synapse counts | 34,153,566 |
| Real edges rendered, seed 42 | 18,000 |
| Compressed raw download bytes | 58,218,136 |

These counts describe the downloaded `connections.csv.gz`, not other advertised
versions of the connectome. Classifications may be refreshed under the same
materialization version. Use hashes to identify an exact source snapshot.

Positions are community-marked points in nanometers, not claims about soma
centers or complete neurite skeletons. Display edges are straight chords between
representative points, not reconstructed axon paths. Original coordinates remain
in raw files; display transforms are recorded in `prepared/summary.json`.

Credit: FlyWire Consortium;
[Dorkenwald et al., 2024](https://doi.org/10.1038/s41586-024-07558-y) and
[Schlegel et al., 2024](https://doi.org/10.1038/s41586-024-07686-5).
Public data are CC BY-NC 4.0 according to the
[official guidance](https://home.flywire.ai/guidelines).
See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Exact exploratory dynamics

Let A[i,j] be the sum of synapses from i to j. Normalize nonempty rows to obtain
P; rows without outgoing connections remain zero. A selected population starts
with amplitude 1. The recurrence is:

```text
a[t+1] = 0.90 * (0.20 * a[t] + 0.80 * P.T @ a[t])
```

All neurons and aggregated pairs participate. The `1e-8` threshold affects only
display counts/selection; it does not prune computation. At most the strongest
2,000 nodes are highlighted. Brightness is normalized per frame, so consult
signal mass to distinguish absolute decay from relative brightness.

Steps are abstract iterations, not biological time. The model has no spikes,
voltages, synaptic delays, inhibitory neurotransmitter signs or plasticity.
Predicted transmitter labels are displayed as annotations only. Without new
input, total nonnegative activity mass must decay.

## Verification

Run `python -m pytest -q` and, after preparing all data,
`python scripts/validate_brain_lab.py`. The unit/integration suite reached 47
passing tests. Data validation checked all raw hashes, dimensions, finite
coordinates, synapse sums and rendered-edge membership. Thirty propagation
steps stayed finite with decreasing mass; resetting repeated the first step
exactly. Fifty alert traces matched the core fingerprints, each with 64 winners.

Two local validations measured median diffusion times around 5–9 ms and median
instrumented alert times around 29–38 ms. These are machine-specific measurements,
not interchangeable benchmarks. The transition CSR arrays occupied 22,161,128
bytes, excluding other process allocations. Saved validation directories contain
CSV timings, JSON traces and a plot; a local latest pointer identifies the run.

## Next research questions

Compare a carefully defined ALPN/KC-derived projection with random connectivity
under matched dimensionality and sparsity. Define the SOC-feature-to-neural-input
mapping explicitly before evaluating held-out families and multiple seeds.
Full morphologies, signed neural dynamics and physiological parameters are
separate research extensions, not properties of the current viewer.
