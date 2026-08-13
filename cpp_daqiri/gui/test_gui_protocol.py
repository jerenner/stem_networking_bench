#!/usr/bin/env python3
"""Hardware-free protocol tests for the STEM DAQ console."""

from __future__ import annotations

import json
import unittest

import numpy as np

from mock_stem_daq import handle, initial_state
from stem_daq_gui import decode_product


class ControlProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = initial_state("tcp://*:5556", "tcp://*:5557")

    def test_runtime_controls_are_independent(self) -> None:
        response = handle(
            self.state,
            {
                "command": "set_runtime",
                "thinned_stream": {
                    "publishing": False,
                    "processing_stage": "dark_blr",
                    "total_refresh_hz": 7.5,
                },
                "burst_writer": {"action": "arm"},
            },
        )
        self.assertFalse(response["thinned_stream"]["publishing"])
        self.assertEqual(response["thinned_stream"]["processing_stage"], "dark_blr")
        self.assertEqual(response["thinned_stream"]["total_refresh_hz"], 7.5)
        self.assertTrue(response["burst_writer"]["armed"])

    def test_restart_staging_can_be_discarded(self) -> None:
        handle(
            self.state,
            {
                "command": "stage_restart",
                "updates": {
                    "processor": {"apply_blr_correction": False},
                    "stem_rx": {"payload_size": 8192},
                },
            },
        )
        self.assertTrue(self.state["acquisition"]["restart_pending"])
        self.assertFalse(
            self.state["pending_config"]["processor"]["apply_blr_correction"]
        )
        handle(self.state, {"command": "discard_restart"})
        self.assertFalse(self.state["acquisition"]["restart_pending"])
        self.assertTrue(
            self.state["pending_config"]["processor"]["apply_blr_correction"]
        )


class StreamProtocolTest(unittest.TestCase):
    def test_optional_payload_decoding(self) -> None:
        image = np.arange(12, dtype="<f4").reshape(3, 4)
        metadata = {
            "schema": "stem.thinned.v1",
            "height": 3,
            "width": 4,
            "parts": ["metadata", "sum"],
        }
        topic, decoded, arrays = decode_product(
            [b"stem/rx/0/corrected", json.dumps(metadata).encode(), image.tobytes()]
        )
        self.assertEqual(topic, "stem/rx/0/corrected")
        self.assertEqual(decoded["height"], 3)
        np.testing.assert_array_equal(arrays["sum"], image)
        self.assertFalse(arrays["sum"].flags.owndata)
        self.assertFalse(arrays["sum"].flags.writeable)


if __name__ == "__main__":
    unittest.main()
