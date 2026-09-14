"""Evaluate a saved experiment without replacing previous evaluations."""
import sys
from flysoc.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
