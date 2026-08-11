#!/usr/bin/env python3
"""Compatibility wrapper for the dead-ADC spectrum correction study."""

from scripts.studies.apply_dead_adc_spectrum_correction import *  # noqa: F401,F403
from scripts.studies.apply_dead_adc_spectrum_correction import main


if __name__ == "__main__":
    main()
