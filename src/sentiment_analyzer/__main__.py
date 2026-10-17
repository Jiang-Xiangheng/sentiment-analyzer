"""CLI entrypoint: python -m sentiment_analyzer"""

from __future__ import annotations

import sys


def main() -> int:
    print("Sentiment Analyzer - use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
