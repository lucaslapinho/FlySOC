"""Prepare real graph and 3D geometry from downloaded FlyWire tables."""
import json
from flysoc.connectome import prepare_connectome

if __name__ == "__main__":
    print(json.dumps(prepare_connectome(), indent=2))
