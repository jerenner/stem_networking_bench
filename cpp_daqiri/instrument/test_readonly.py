"""Read-only instrument tests; hardware access is replaced at the transport boundary."""

import json
import os
import signal
import socket
import struct
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from stem_instrument_readonly import (
    HardwareReader, ReadOnlyInstrumentService, decode_link_register,
    execute, parse_biases, parse_temperatures, read_link_register,
)


TEMPERATURE_OUTPUT = """Sensor Temperatures:
      Q1  : 44.63
      Q2  : 41.91
      Q3  : 54.62
      Q4  : 46.05
      B1  : 36.06
      B2  : 35.53
      HS1 : 32.46
      HS2 : 32.76
"""
BIAS_OUTPUT = """Digital Supply
DVDD18 : 2.500 V
LVDSBIAS : 0.100 mA
Analog Supply
AVDD : 2.500 V
VDDR : 1.802 V
M2V5 : -2.477 V
Analog Bias
BIASN : 0.400 mA
N3BUF : 4.020 mA
"""


def config():
    return {
        "camera": {"host": "camera.example", "user": "root",
                   "identity_file": "/keys/camera", "known_hosts_file": "/keys/known_hosts"},
        "detector": {"transport": "helper", "reader_argv": ["/test/register_reader"],
                     "modules": [{"id": 0, "host": "10.20.20.32"},
                                 {"id": 1, "host": "10.20.21.32"}]},
    }


class ParsingTest(unittest.TestCase):
    def test_named_probes_and_suspect_reading_preserved(self):
        values = parse_temperatures(TEMPERATURE_OUTPUT.replace("54.62", "198.86"))
        self.assertEqual(len(values["temperatures"]), 8)
        self.assertFalse(values["temperatures"]["Q3"]["valid"])
        self.assertEqual(values["temperatures"]["Q3"]["value"], 198.86)
        self.assertEqual(values["temperature_c"], 46.05)
        self.assertIn("Q3", values["temperature_warning"])

    def test_biases_preserve_current_and_voltage_units(self):
        values = parse_biases(BIAS_OUTPUT)["biases"]
        self.assertEqual(values["AVDD"], {"value": 2.5, "unit": "V"})
        self.assertEqual(values["BIASN"], {"value": .4, "unit": "mA"})
        self.assertEqual(values["M2V5"]["value"], -2.477)

    def test_invalid_or_empty_readbacks_fail(self):
        for parser, text in ((parse_temperatures, "permission denied"),
                             (parse_temperatures, "Q1 : nan"),
                             (parse_biases, "AVDD : inf V"),
                             (parse_biases, "prompt only")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parser(text)

    def test_each_link_flag_and_lane(self):
        item = decode_link_register(3, "board", 0x0820043B)
        self.assertEqual(item["masks"], {"jesd_sync": 0x3b, "not_in_table": 4,
                                        "disparity_error": 0x20, "characters_were_ks": 8})
        self.assertFalse(item["lanes"][2]["jesd_sync"])
        self.assertTrue(item["lanes"][2]["not_in_table"])
        self.assertFalse(item["synchronized"])
        self.assertFalse(item["status_reset_performed"])
        self.assertEqual(decode_link_register(0, "b", "0x0000003f")["status"], "ready")
        for bad in (True, -1, 2**32, 1.2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decode_link_register(0, "b", bad)


class HardwareReaderTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def runner(self, argv, timeout, stdin=None):
        self.calls.append((argv, stdin))
        if stdin == "dsh sensor temp\n":
            return TEMPERATURE_OUTPUT
        if stdin == "dsh bias adc volts\n":
            return BIAS_OUTPUT
        if argv[-2:] == ["--register", "420"]:
            return '{"register420": 63}'
        raise AssertionError("unexpected hardware command")

    def test_fixed_read_allowlist_and_no_automatic_probe(self):
        reader = HardwareReader(config(), self.runner)
        self.assertEqual(self.calls, [])
        reader.read("temperature")
        reader.read("biases")
        reader.read("links")
        self.assertEqual(len(self.calls), 4)
        self.assertIn("BatchMode=yes", self.calls[0][0])
        self.assertIn("StrictHostKeyChecking=yes", self.calls[0][0])
        self.assertEqual(self.calls[0][0][-1], "root@camera.example")
        with self.assertRaises(ValueError):
            reader.camera_read("sensor dac avdd 2.5")
        self.assertEqual(len(self.calls), 4)

    def test_partial_link_failure_is_not_healthy(self):
        def runner(argv, timeout, stdin=None):
            if "10.20.21.32" in argv:
                raise RuntimeError("board timed out")
            return '{"register420": 63}'
        values, _ = HardwareReader(config(), runner).read("links")
        self.assertIsNone(values["synchronized"])
        self.assertEqual(values["modules"][1]["status"], "error")
        self.assertIn("board timed out", values["read_error"])

    def test_missing_transport_reported_explicitly(self):
        reader = HardwareReader({"camera": {"host": "camera"}}, self.runner)
        self.assertNotIn("detector.read_link_status", reader.supported_commands)
        with self.assertRaisesRegex(RuntimeError, "no FPGA modules configured"):
            reader.read("links")

    def test_process_failure_and_timeout_are_checked(self):
        with self.assertRaisesRegex(RuntimeError, "exited 9"):
            execute([sys.executable, "-c", "raise SystemExit(9)"], 2)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            execute([sys.executable, "-c", "import time; time.sleep(2)"], .05)


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.release = threading.Event()
        self.fail = False

        def runner(argv, timeout, stdin=None):
            self.release.wait(2)
            if self.fail:
                raise RuntimeError("SSH connection lost")
            return TEMPERATURE_OUTPUT

        self.service = ReadOnlyInstrumentService(
            {"readonly": {"stale_after_seconds": 10}},
            HardwareReader(config(), runner), clock=lambda: self.now[0],
        )

    def tearDown(self):
        self.release.set()
        self.service.close()

    def finish(self):
        self.release.set()
        self.service.tasks.join()
        return self.service.snapshot()

    def test_slow_read_does_not_block_state_or_other_controls(self):
        started = self.service.handle({"command": "camera.read_temperature"})
        self.assertTrue(started["accepted"])
        before = time.monotonic()
        state = self.service.handle({"command": "get_state"})
        self.assertLess(time.monotonic() - before, .1)
        self.assertEqual(state["instrument"]["operation"]["state"], "running")
        self.assertFalse(self.service.handle({"command": "camera.read_biases"})["ok"])
        state = self.finish()
        self.assertEqual(state["operation"]["state"], "completed")
        self.assertFalse(state["reads"]["temperature"]["stale"])

    def test_cached_values_expire_and_failed_read_keeps_timestamp(self):
        self.service.handle({"command": "camera.read_temperature"})
        self.finish()
        self.now[0] += 11
        self.assertTrue(self.service.snapshot()["reads"]["temperature"]["stale"])
        self.fail = True
        self.service.handle({"command": "camera.read_temperature"})
        state = self.finish()
        self.assertEqual(state["reads"]["temperature"]["sampled_at"], 1000)
        self.assertEqual(state["reads"]["temperature"]["status"], "error")
        self.assertEqual(state["camera"]["temperatures"]["Q3"]["value"], 54.62)
        self.assertEqual(state["operation"]["state"], "failed")

    def test_mutating_and_parameterized_requests_rejected(self):
        for command in ("camera.power_up", "camera.power_down", "camera.insert",
                        "detector.resync", "detector.auto_align", "scan.configure",
                        "scan.start", "scan.abort", "mock.set_failure"):
            with self.subTest(command=command):
                self.assertFalse(self.service.handle({"command": command})["ok"])
        self.assertFalse(self.service.handle({"command": "camera.read_temperature", "host": "other"})["ok"])
        self.assertEqual(self.service.snapshot()["camera"]["power_state"], "unknown")


class RegisterTransportTest(unittest.TestCase):
    def test_udp_protocol_matches_legacy_and_retries_invalid_reply(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(("127.0.0.1", 0))
            server.settimeout(2)
            calls = []

            def serve():
                for attempt in range(2):
                    packet, peer = server.recvfrom(4096)
                    calls.append(packet)
                    response = struct.pack("<6I", 0x203B2D29, 0,
                                           (0x70000000 | 420) if attempt else 420,
                                           0x0000013F, 0x70000000, 0)
                    server.sendto(response, peer)

            worker = threading.Thread(target=serve)
            worker.start()
            try:
                result = read_link_register("127.0.0.1", server.getsockname()[1], .3, 2)
            finally:
                worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, 0x13F)
            self.assertEqual(calls, [struct.pack("<6I", 0x203B2D29, 0, 420, 0, 0, 0)] * 2)

    def test_udp_timeout_is_bounded(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(("127.0.0.1", 0))
            with self.assertRaisesRegex(RuntimeError, "failed after 1 attempts"):
                read_link_register("127.0.0.1", server.getsockname()[1], .03, 1)


@unittest.skipUnless(os.getenv("STEM_TEST_SUPERVISOR_BIN"), "native supervisor binary not provided")
class SupervisorIntegrationTest(unittest.TestCase):
    def test_supervisor_routes_real_service_and_remains_responsive(self):
        import yaml
        import zmq

        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="stem-reads-", dir="/tmp") as temporary:
            directory = Path(temporary)
            (directory / "ssh").symlink_to(here / "test_hardware_process.py")
            cfg = {"control": {"enabled": True, "start_acquisition": False,
                                "endpoint": "ipc://" + str(directory / "control.ipc")},
                   "instrument": {"enabled": True, "mode": "hardware_readonly", "launch_service": True,
                                  "service_binary": str(here / "stem_instrument_readonly.py"),
                                  "endpoint": "ipc://" + str(directory / "reads.ipc"),
                                  "readonly": config()}}
            cfg["instrument"]["readonly"]["detector"]["reader_argv"] = [sys.executable, str(here / "test_hardware_process.py")]
            path = directory / "config.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                PATH=str(directory) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                STEM_DAQIRI_CHILD_CONFIG=str(directory / "child.yaml"),
                STEM_DAQIRI_CHILD_CONTROL_ENDPOINT="ipc://" + str(directory / "rx.ipc"),
                STEM_TEST_HARDWARE_AUDIT=str(directory / "calls.jsonl"),
            )
            context = zmq.Context()

            def request(command):
                with context.socket(zmq.REQ) as socket:
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.RCVTIMEO, 3000)
                    socket.connect(cfg["control"]["endpoint"])
                    socket.send_json({"command": command})
                    return socket.recv_json()

            process = subprocess.Popen([os.environ["STEM_TEST_SUPERVISOR_BIN"], str(path)], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            try:
                state = request("get_state")
                self.assertFalse(state["acquisition"]["running"])
                self.assertFalse((directory / "calls.jsonl").exists())
                for command in ("camera.read_temperature", "camera.read_biases", "detector.read_link_status"):
                    response = request(command)
                    self.assertTrue(response["ok"], response)
                    self.assertTrue(response["accepted"])
                    before = time.monotonic()
                    request("get_state")
                    self.assertLess(time.monotonic() - before, .5)
                    deadline = time.monotonic() + 6
                    while time.monotonic() < deadline:
                        state = request("get_state")
                        if state["instrument"]["operation"]["state"] != "running":
                            break
                        time.sleep(.03)
                    self.assertEqual(state["instrument"]["operation"]["state"], "completed", state)
                self.assertEqual(state["instrument"]["camera"]["temperatures"]["Q3"]["value"], 54.62)
                self.assertEqual(state["instrument"]["camera"]["biases"]["BIASN"]["unit"], "mA")
                self.assertTrue(state["instrument"]["detector"]["synchronized"])
                self.assertFalse(request("camera.power_up")["ok"])
                self.assertFalse(request("emulator.set_scenario")["ok"])
                calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(calls, ["dsh sensor temp", "dsh bias adc volts", "register 420", "register 420"])
                self.assertTrue(request("shutdown_supervisor")["ok"])
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        output, _ = process.communicate(timeout=3)
                        print("supervisor test cleanup:\n" + output, file=sys.stderr)
                context.term()


if __name__ == "__main__":
    unittest.main()
