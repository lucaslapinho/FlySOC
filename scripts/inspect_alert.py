"""Display fingerprint, nearest history, novelty and inferred verdict context."""
import sys
from flysoc.cli import main

if __name__ == "__main__":
    main(["inspect", *sys.argv[1:]])
