#!/usr/bin/env python3
"""Hardware-free protocol tests for the STEM DAQ console."""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
from mock_stem_daq import handle, initial_state
from stem_daq_gui import MainWindow, decode_product


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

    def test_persistent_supervisor_start_stop(self) -> None:
        response = handle(self.state, {"command": "stop_acquisition"})
        self.assertFalse(response["acquisition"]["running"])
        self.assertEqual(response["supervisor"]["state"], "stopped")
        response = handle(self.state, {"command": "start_acquisition"})
        self.assertTrue(response["acquisition"]["running"])
        self.assertEqual(response["supervisor"]["state"], "running")

    def test_mock_instrument_controls_and_scan_configuration(self) -> None:
        response = handle(self.state, {"command": "camera.power_up"})
        self.assertEqual(response["instrument"]["camera"]["power_state"], "ready")
        response = handle(self.state, {"command": "detector.resync"})
        self.assertTrue(response["instrument"]["detector"]["synchronized"])
        response = handle(
            self.state,
            {
                "command": "scan.configure",
                "scan": {"read_count": 2, "positions_x": 4, "rows": 8},
            },
        )
        self.assertEqual(response["instrument"]["scan"]["expected_frames"], 64)
        response = handle(self.state, {"command": "scan.start"})
        self.assertTrue(response["instrument"]["scan"]["running"])


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


class InstrumentPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_reads_work_without_acquisition_and_show_units_errors_and_capabilities(self):
        host = QtWidgets.QWidget()
        requests = []
        host.send_control = lambda tag, request: requests.append(request)
        panel = MainWindow._instrument_tab(host)
        payload = {
            "enabled": True, "service_online": True, "mode": "hardware_readonly",
            "capabilities": {"read_only": True, "commands": ["camera.read_temperature", "camera.read_biases"]},
            "camera": {"temperatures": {"Q3": {"value": 198.86, "unit": "C", "valid": False}},
                       "biases": {"BIASN": {"value": .4, "unit": "mA"}}},
            "reads": {"temperature": {"status": "error", "stale": True, "error": "SSH timed out"}},
            "operation": {"state": "completed"},
        }
        try:
            MainWindow._update_instrument(host, payload)
            self.assertTrue(host.instrument_read_buttons["camera.read_temperature"].isEnabled())
            self.assertFalse(host.instrument_read_buttons["detector.read_link_status"].isEnabled())
            self.assertIn("suspect", host.instrument_temperature_table.item(0, 1).text())
            self.assertEqual(host.instrument_biases_table.item(0, 1).text(), "0.4 mA")
            self.assertIn("SSH timed out", host.instrument_read_status["temperature"].text())
            host.instrument_read_buttons["camera.read_temperature"].click()
            self.assertEqual(requests, [{"command": "camera.read_temperature"}])
            payload["operation"]["state"] = "running"
            MainWindow._update_instrument(host, payload)
            self.assertFalse(host.instrument_read_buttons["camera.read_biases"].isEnabled())
        finally:
            panel.close()
            host.close()


if __name__ == "__main__":
    unittest.main()
