"""Download official FlyWire FAFB v783 CSV snapshots, without executing them."""

import gzip
import json
from pathlib import Path
import urllib.request

from flysoc.artifacts import sha256, write_json
from flysoc.config import ROOT

BASE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"
FILES = ["neurons.csv.gz", "classification.csv.gz", "coordinates.csv.gz", "connections.csv.gz"]
DESTINATION = ROOT / "data/connectome/fafb783/raw"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION.parent / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    manifest = {"dataset": "FlyWire FAFB", "version": "783",
                "source_reference": "https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py",
                "terms": "https://home.flywire.ai/tos",
                "citation_guidelines": "https://home.flywire.ai/guidelines", "files": {}}
    for name in FILES:
        path = DESTINATION / name
        if path.exists():
            old = previous["files"].get(name)
            if old is None or sha256(path) != old["sha256"]:
                raise ValueError(f"Existing file has no matching manifest; preserving it: {path}")
            manifest["files"][name] = old
            print(f"Verified cached {name}", flush=True)
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"Incomplete earlier download preserved: {temporary}")
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(BASE + name, timeout=60) as response, temporary.open("xb") as destination:
            length = int(response.headers.get("Content-Length", 0))
            if length > 300 * 1024 * 1024:
                raise ValueError("Unexpectedly large connectome file")
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > 300 * 1024 * 1024:
                    raise ValueError("Download exceeded the bounded size limit")
                destination.write(block)
            if length and count != length:
                raise ValueError("Truncated download")
            modified = response.headers.get("Last-Modified")
        with gzip.open(temporary, "rb") as compressed:
            while compressed.read(1024 * 1024):
                pass  # verify gzip checksum without retaining decompressed data
        temporary.rename(path)
        manifest["files"][name] = {"url": BASE + name, "bytes": path.stat().st_size,
                                    "sha256": sha256(path), "last_modified": modified}
        # Save after each file so interrupted downloads can resume safely.
        write_json(manifest_path, {**manifest, "files": {**previous["files"], **manifest["files"]}})
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
