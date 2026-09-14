# Third-party data and software

## FlyWire FAFB version 783

Credit: FlyWire Consortium, Dorkenwald et al. (2024), Schlegel et al. (2024),
and the original data/annotation contributors identified by FlyWire.

- Project: https://home.flywire.ai/
- Public release license and citation guidance: https://home.flywire.ai/guidelines
- License: Creative Commons Attribution-NonCommercial 4.0 International,
  https://creativecommons.org/licenses/by-nc/4.0/
- Terms: https://home.flywire.ai/tos
- Download source: https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py
- Raw local files: `data/connectome/fafb783/raw/`; individual URLs and SHA-256
  values are preserved in `data/connectome/fafb783/download_manifest.json`.

Raw compressed CSVs are unchanged. Derived files in `prepared/` aggregate
directed synapse counts, normalize outgoing weights, map IDs to array indices,
transform coordinates for display, and sample visualization edges. Details
and measured counts appear in `docs/brain_lab.md` and `prepared/summary.json`.
Derived data retain the attribution and noncommercial terms of the source.

## Three.js 0.180.0 and OrbitControls

Copyright belongs to the Three.js authors. Distributed under the MIT license;
the complete notice is included as `src/flysoc/web/vendor/THREE-LICENSE.txt`.
The registry archive was integrity-checked against its published SHA-512 value.
The selected vendor files and hashes are in `src/flysoc/web/vendor/manifest.json`.

## Python packages

Python dependencies are installed separately in the virtual environment and
are not bundled as binaries in the FlySOC ZIP. Their versions are recorded in
`requirements-lock.txt`; each package retains its own license.
