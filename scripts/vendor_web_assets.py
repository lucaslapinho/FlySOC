"""Vendor pinned Three.js modules from the official npm registry, with integrity."""
import base64
import hashlib
import io
import json
import tarfile
import urllib.request

from flysoc.artifacts import write_json
from flysoc.config import ROOT

VERSION = "0.180.0"


def main():
    with urllib.request.urlopen(f"https://registry.npmjs.org/three/{VERSION}", timeout=30) as response:
        metadata = json.load(response)
    with urllib.request.urlopen(metadata["dist"]["tarball"], timeout=60) as response:
        archive = response.read(25 * 1024 * 1024)
    expected = metadata["dist"]["integrity"]
    actual = "sha512-" + base64.b64encode(hashlib.sha512(archive).digest()).decode()
    if actual != expected:
        raise ValueError("npm package integrity verification failed")
    destination = ROOT / "src/flysoc/web/vendor"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {"package/build/three.module.js": "three.module.js", "package/build/three.core.js": "three.core.js",
               "package/examples/jsm/controls/OrbitControls.js": "OrbitControls.js", "package/LICENSE": "THREE-LICENSE.txt"}
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        for source, name in sources.items():
            content = package.extractfile(source).read()
            path = destination / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Preserving different existing vendor asset: {path}")
            path.write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    write_json(destination / "manifest.json", {"package": "three", "version": VERSION,
                                               "registry_integrity": expected, "files_sha256": hashes})
    print(f"Vendored Three.js {VERSION}; npm integrity verified")


if __name__ == "__main__":
    main()
