"""List or explicitly download the bounded MaleCNS v1.0 bulk source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
OFFICIAL_PROJECT = "https://male-cns.janelia.org/"
FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
MANIFEST_NAME = "flysoc_source_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the fixed official MaleCNS URLs by default. Network and disk "
            "writes require the explicit --download flag."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/male-cns-v1.0"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Hash and record complete existing files instead of refusing them.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = [BASE_URL + name for name in FILES]
    print(json.dumps({"download": bool(args.download), "files": urls}, indent=2))
    if not args.download:
        return
    if not 1 <= args.timeout <= 600:
        raise ValueError("--timeout must be between 1 and 600 seconds")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing source provenance: {manifest_path}"
        )

    records: dict[str, dict[str, object]] = {}
    for name, url in zip(FILES, urls):
        target = args.output / name
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            raise FileExistsError(f"Preserved partial download requires review: {partial}")
        if target.exists():
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Refusing unverified existing source without --reuse-existing: {target}"
                )
            if not target.is_file() or target.stat().st_size == 0:
                raise ValueError(f"Existing source is not a non-empty regular file: {target}")
            records[name] = {
                "url": url,
                "path": str(target.resolve()),
                "status": "reused_existing",
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
            continue

        request = urllib.request.Request(
            url, headers={"User-Agent": "FlySOC-MaleCNS-source-fetch/1"}
        )
        print(f"Downloading {name} ...", flush=True)
        response_headers: dict[str, str | None]
        final_url: str
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            final_url = response.geturl()
            if urlparse(final_url).scheme != "https":
                raise ValueError(f"Refusing non-HTTPS download redirect: {final_url}")
            if urlparse(final_url).hostname != "storage.googleapis.com":
                raise ValueError(f"Refusing download redirect to unexpected host: {final_url}")
            response_headers = {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_length": response.headers.get("Content-Length"),
                "content_type": response.headers.get("Content-Type"),
            }
            with partial.open("xb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        if partial.stat().st_size == 0:
            raise ValueError(f"Downloaded source is empty; partial retained: {partial}")
        expected_size = response_headers["content_length"]
        if expected_size is not None and partial.stat().st_size != int(expected_size):
            raise ValueError(
                f"Content-Length mismatch; partial retained for review: {partial}"
            )
        partial.replace(target)
        records[name] = {
            "url": url,
            "final_url": final_url,
            "path": str(target.resolve()),
            "status": "downloaded",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "response_headers": response_headers,
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "MaleCNS",
            "version": "v1.0",
            "official_project": OFFICIAL_PROJECT,
            "source_base_url": BASE_URL,
            "license_note": "CC BY per the official MaleCNS project page; verify current terms at the official project URL.",
        },
        "request": {
            "explicit_download": True,
            "reuse_existing": bool(args.reuse_existing),
            "output": str(args.output.resolve()),
        },
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
