"""Generate a uniquely named dataset using the package CLI."""
import sys
from flysoc.cli import main

if __name__ == "__main__":
    main(["generate", *sys.argv[1:]])
