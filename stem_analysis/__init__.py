"""Offline STEM detector analysis utilities.

The package contains CPU/NumPy implementations of the detector corrections used
by the Holoscan `PyTorchProcessorOp`. Scripts under `scripts/` use these helpers
for reproducible offline studies and diagnostics.
"""

from .config import ProcessorConfig
from .processing import process_tensor_block

__all__ = ["ProcessorConfig", "process_tensor_block"]

