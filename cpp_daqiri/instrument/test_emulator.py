"""Persistent instrument emulation tests; all listeners use ephemeral loopback ports."""

import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from stem_instrument_emulator import InstrumentEmulator, REGISTER_REQUEST
from stem_instrument_readonly import ReadOnlyInstrumentService, create_service


def config():
    return {
        "enabled": True, "mode": "emulated_readonly",
        "readonly": {
            "camera": {"host": "camera.must.not.be.contacted"},
            "detector": {"transport": "helper", "reader_argv": ["must-not-execute"],
                         "modules": [{"id": 0, "host": "fpga.must.not.be.contacted"}]},
        },
        "emulator": {"module_count": 2, "camera_delay_ms": 0, "fpga_delay_ms": 0,
                     "temperature_drift_c": 0, "read_timeout_seconds": .05},
    }


class EmulatorTest(unittest.TestCase):
    def setUp(self):
        self.service = create_service(config())

    def tearDown(self):
        self.service.close()

    def read(self, command):
        result = self.service.handle({"command": command})
        self.assertTrue(result["accepted"], result)
        self.service.tasks.join()
        return self.service.snapshot()

    def scenario(self, name, module_id=0):
        result = self.service.handle({"command": "emulator.set_scenario",
                                      "scenario": name, "module_id": module_id})
        self.assertTrue(result["ok"], result)
        return result["instrument"]

    def test_full_read_path_and_no_hardware_fallback(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("no SSH or helper should execute")):
            state = self.service.snapshot()
            self.assertEqual(state["emulator"]["camera_requests"]["requests"], 0)
            self.assertTrue(state["simulated"])
            self.assertTrue(state["reads"]["temperature"]["simulated"])
            self.assertTrue(all(item["host"] == "127.0.0.1" for item in self.service.reader.modules))
            state = self.read("camera.read_temperature")
            self.assertEqual(len(state["camera"]["temperatures"]), 8)
            self.assertEqual(state["camera"]["temperatures"]["Q1"]["value"], 44.63)
            self.assertIn("Sensor Temperatures", state["reads"]["temperature"]["raw"])
            state = self.read("camera.read_biases")
            self.assertEqual(len(state["camera"]["biases"]), 24)
            self.assertEqual(state["camera"]["biases"]["BIASN"]["unit"], "mA")
            self.assertLess(state["camera"]["biases"]["M2V5"]["value"], 0)
            state = self.read("detector.read_link_status")
            self.assertTrue(state["detector"]["synchronized"])
            for module in state["detector"]["modules"]:
                self.assertEqual(module["register420"], 63)
                self.assertFalse(module["status_reset_performed"])
            self.assertTrue(all(value["requests"] == 1 for value in state["emulator"]["fpga_requests"].values()))

    def test_suspect_probe_and_cached_value_invalidation(self):
        before = self.read("camera.read_temperature")
        state = self.scenario("suspect_temperature")
        self.assertTrue(state["reads"]["temperature"]["stale"])
        self.assertEqual(state["reads"]["temperature"]["sampled_at"], before["reads"]["temperature"]["sampled_at"])
        state = self.read("camera.read_temperature")
        self.assertFalse(state["camera"]["temperatures"]["Q3"]["valid"])
        self.assertEqual(state["camera"]["temperatures"]["Q3"]["value"], 198.86)
        self.assertLess(state["camera"]["temperature_c"], 150)

    def test_error_timeout_malformed_and_recovery_scenarios(self):
        before = self.read("camera.read_temperature")
        for scenario in ("camera_error", "camera_timeout", "malformed_temperature"):
            with self.subTest(scenario=scenario):
                self.scenario(scenario)
                state = self.read("camera.read_temperature")
                self.assertEqual(state["operation"]["state"], "failed")
                self.assertTrue(state["reads"]["temperature"]["stale"])
                self.assertEqual(state["reads"]["temperature"]["sampled_at"], before["reads"]["temperature"]["sampled_at"])
        for scenario in ("fpga_timeout", "malformed_fpga"):
            with self.subTest(scenario=scenario):
                self.scenario(scenario, 1)
                state = self.read("detector.read_link_status")
                self.assertEqual(state["operation"]["state"], "failed")
                self.assertIsNone(state["detector"]["synchronized"])
                self.assertEqual(state["detector"]["modules"][0]["status"], "ready")
                self.assertEqual(state["detector"]["modules"][1]["status"], "error")
        self.scenario("healthy")
        self.assertEqual(self.read("camera.read_temperature")["operation"]["state"], "completed")
        self.assertTrue(self.read("detector.read_link_status")["detector"]["synchronized"])

    def test_link_error_masks_and_missing_sync(self):
        self.scenario("link_errors", 1)
        module = self.read("detector.read_link_status")["detector"]["modules"][1]
        self.assertEqual(module["status"], "errors_latched")
        self.assertEqual(module["masks"], {"jesd_sync": 63, "not_in_table": 2,
                                         "disparity_error": 4, "characters_were_ks": 8})
        self.scenario("unsynchronized", 1)
        state = self.read("detector.read_link_status")
        self.assertFalse(state["detector"]["synchronized"])
        self.assertFalse(state["detector"]["modules"][1]["lanes"][2]["jesd_sync"])

    def test_udp_responder_rejects_writes_and_camera_rejects_other_commands(self):
        module = self.service.reader.modules[0]
        words = list(struct.unpack("<6I", REGISTER_REQUEST))
        words[2] |= 0x80000000
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(.05)
            client.sendto(struct.pack("<6I", *words), (module["host"], module["port"]))
            with self.assertRaises(TimeoutError):
                client.recv(4096)
        self.assertEqual(self.service.emulator.snapshot()["fpga_requests"]["0"]["rejected"], 1)
        with self.assertRaises(ValueError):
            self.service.emulator.camera_runner(["ssh", "root@127.0.0.1"], .1, "dsh sensor dac avdd 2\n")
        for command in ("camera.power_up", "scan.start", "detector.resync"):
            self.assertFalse(self.service.handle({"command": command})["ok"])

    def test_scenario_validation_and_live_mode_guard(self):
        for fields in ({"scenario": "bogus"}, {"scenario": "healthy", "host": "elsewhere"},
                       {"scenario": "healthy", "module_id": 9}):
            with self.subTest(fields=fields):
                self.assertFalse(self.service.handle(dict(command="emulator.set_scenario", **fields))["ok"])
        live = ReadOnlyInstrumentService({"mode": "hardware_readonly"})
        try:
            self.assertFalse(live.handle({"command": "emulator.set_scenario", "scenario": "healthy"})["ok"])
        finally:
            live.close()
        for bad in (0, 9, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                InstrumentEmulator({"module_count": bad})

    def test_busy_read_stays_responsive_and_prevents_scenario_change(self):
        self.service.emulator.camera_delay = .03
        self.service.handle({"command": "camera.read_temperature"})
        before = time.monotonic()
        result = self.service.handle({"command": "get_state"})
        self.assertLess(time.monotonic() - before, .02)
        self.assertEqual(result["instrument"]["operation"]["state"], "running")
        self.assertFalse(self.service.handle({"command": "emulator.set_scenario", "scenario": "healthy"})["ok"])
        self.service.tasks.join()

    def test_shutdown_releases_all_listeners(self):
        emulator = self.service.emulator
        threads = list(emulator.threads)
        address = emulator.camera.server_address
        self.service.close()
        self.assertFalse(any(thread.is_alive() for thread in threads))
        with self.assertRaises(OSError):
            socket.create_connection(address, timeout=.1)


@unittest.skipUnless(os.getenv("STEM_TEST_SUPERVISOR_BIN"), "native supervisor binary not provided")
class SupervisorEmulatorTest(unittest.TestCase):
    def test_persistent_backend_and_validation_client(self):
        import yaml
        import zmq

        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="stem-emulator-", dir="/tmp") as temporary:
            directory = Path(temporary)
            cfg = {
                "control": {"enabled": True, "start_acquisition": False,
                            "endpoint": "ipc://" + str(directory / "public.ipc")},
                "instrument": config(),
            }
            cfg["instrument"].update(launch_service=True,
                                     service_binary=str(here / "stem_instrument_readonly.py"),
                                     endpoint="ipc://" + str(directory / "instrument.ipc"))
            cfg["instrument"]["emulator"]["read_timeout_seconds"] = .15
            path = directory / "config.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                STEM_DAQIRI_CHILD_CONFIG=str(directory / "child.yaml"),
                STEM_DAQIRI_CHILD_CONTROL_ENDPOINT="ipc://" + str(directory / "rx.ipc"),
            )
            process = subprocess.Popen([os.environ["STEM_TEST_SUPERVISOR_BIN"], str(path)], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            context = zmq.Context()
            try:
                result = subprocess.run(
                    [sys.executable, str(here.parent / "scripts/validate_emulated_instrument.py"),
                     "--endpoint", cfg["control"]["endpoint"], "--exercise-failures"],
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("emulated instrument validation OK", result.stdout)
                with context.socket(zmq.REQ) as connection:
                    connection.setsockopt(zmq.LINGER, 0)
                    connection.setsockopt(zmq.RCVTIMEO, 3000)
                    connection.connect(cfg["control"]["endpoint"])
                    connection.send_json({"command": "get_state"})
                    state = connection.recv_json()
                    self.assertFalse(state["acquisition"]["running"])
                    self.assertEqual(state["instrument"]["mode"], "emulated_readonly")
                    connection.send_json({"command": "shutdown_supervisor"})
                    self.assertTrue(connection.recv_json()["ok"])
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    output, _ = process.communicate(timeout=3)
                    print(output, file=sys.stderr)
                context.term()


if __name__ == "__main__":
    unittest.main()
