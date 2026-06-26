#!/usr/bin/env python3
"""Compatibility wrapper for scripts.offline.run_offline_pipeline."""

from scripts.offline.run_offline_pipeline import *  # noqa: F401,F403
from scripts.offline.run_offline_pipeline import main


if __name__ == "__main__":
    main()

