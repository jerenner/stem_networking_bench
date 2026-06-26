#!/usr/bin/env python3
"""Compatibility wrapper for scripts.diagnostics.verify_output."""

from scripts.diagnostics.verify_output import *  # noqa: F401,F403
from scripts.diagnostics.verify_output import verify_and_plot


if __name__ == "__main__":
    verify_and_plot()

