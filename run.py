#!/usr/bin/env python3
"""Entry point: `python run.py [run|scan|doctor|config]`.

Defaults to `run` so `python run.py` starts paper trading immediately.
"""

from __future__ import annotations

import sys

from polymarket_bot.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["run"]
    sys.exit(main(argv))
