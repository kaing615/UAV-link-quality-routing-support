"""v10t tuning-package entry point."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(tuning=True))
