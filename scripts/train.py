"""Train and save a unique experiment."""
import sys
from flysoc.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
