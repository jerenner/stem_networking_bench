#!/usr/bin/env python3
"""Compatibility wrapper for scripts.dark.make_dark_frame."""

from scripts.dark.make_dark_frame import *  # noqa: F401,F403
from scripts.dark.make_dark_frame import main


if __name__ == "__main__":
    main()

