"""Persistent loopback camera-text and FPGA-register responders, not an SSH server."""

from __future__ import annotations

import copy
import json
import math
import socket
import socketserver
import struct
import threading
import time


SCENARIOS = (
    "healthy", "suspect_temperature", "camera_error", "camera_timeout",
    "malformed_temperature", "fpga_timeout", "malformed_fpga",
    "link_errors", "unsynchronized",
)
TEMPERATURES = {"Q1": 44.63, "Q2": 41.91, "Q3": 54.62, "Q4": 46.05,
                "B1": 36.06, "B2": 35.53, "HS1": 32.46, "HS2": 32.76}
BIASES = (
    ("DVDD18", 2.500, "V"), ("DVDD33", 3.311, "V"), ("DVDD2", 2.500, "V"),
    ("LVDSBIAS", .100, "mA"), ("LVDSB_SUM", .395, "mA"), ("3V3DIG", 3.316, "V"),
    ("3V3_ANA", 3.314, "V"), ("3V3_EPR", 3.308, "V"), ("AVDD", 2.500, "V"),
    ("VDDP", 1.805, "V"), ("VDDR", 1.802, "V"), ("VGUARD", -.000, "V"),
    ("4V3", 4.155, "V"), ("2V8", 2.667, "V"), ("2V0", 1.839, "V"),
    ("M2V5", -2.477, "V"), ("BIASN", .400, "mA"), ("BIASP", .200, "mA"),
    ("N3BUF", 4.020, "mA"), ("OSBUF", .403, "mA"), ("BIASN_SUM", 1.616, "mA"),
    ("BIASP_SUM", .802, "mA"), ("N3BUF_SUM", 16.146, "mA"), ("OSBUF_SUM", 1.509, "mA"),
)
REGISTER_REQUEST = struct.pack("<6I", 0x203B2D29, 0, 420, 0, 0, 0)


def bounded_number(config, name, default, minimum, maximum):
    value = float(config.get(name, default))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("emulator.{} must be in [{}, {}]".format(name, minimum, maximum))
    return value


class CameraHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        emulator = self.server.emulator
        try:
            packet = self.rfile.readline(4097)
            if len(packet) > 4096:
                return
            command = json.loads(packet)["command"]
            response = emulator.camera_response(command)
            if response is not None:
                self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
        except (OSError, ValueError, KeyError, TypeError):
            return


class CameraServer(socketserver.ThreadingTCPServer):
    daemon_threads = True


class FpgaHandler(socketserver.BaseRequestHandler):
    def handle(self):
        packet, connection = self.request
        response = self.server.emulator.fpga_response(self.server.module_id, packet)
        if response is not None:
            try:
                connection.sendto(response, self.client_address)
            except OSError:
                pass


class InstrumentEmulator:
    """Own ephemeral IPv4 loopback listeners for the lifetime of one service.

    No configurable target address, external process, or real-hardware fallback
    is available here. Only the read-only adapter consumes these responders.
    """

    def __init__(self, config=None):
        config = config or {}
        self.module_count = config.get("module_count", 4)
        if type(self.module_count) is not int or not 1 <= self.module_count <= 8:
            raise ValueError("emulator.module_count must be an integer in [1,8]")
        self.camera_delay = bounded_number(config, "camera_delay_ms", 50, 0, 2000) / 1000
        self.fpga_delay = bounded_number(config, "fpga_delay_ms", 2, 0, 1000) / 1000
        self.drift = bounded_number(config, "temperature_drift_c", .2, 0, 10)
        self.read_timeout = bounded_number(config, "read_timeout_seconds", 1, .05, 30)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.started = time.monotonic()
        self.servers = []
        self.threads = []
        self.camera_counts = {"requests": 0, "rejected": 0}
        self.fpga_counts = {str(index): {"requests": 0, "replies": 0, "rejected": 0,
                                        "dropped": 0} for index in range(self.module_count)}
        self.set_scenario(config.get("scenario", "healthy"), config.get("fault_module", 0))
        try:
            self.camera = self._start(CameraServer, CameraHandler)
            self.modules = []
            for module_id in range(self.module_count):
                server = self._start(socketserver.UDPServer, FpgaHandler, module_id)
                self.modules.append({"id": module_id, "host": "127.0.0.1",
                                     "port": server.server_address[1]})
        except BaseException:
            self.close()
            raise

    def _start(self, server_type, handler, module_id=None):
        server = server_type(("127.0.0.1", 0), handler)
        server.emulator = self
        server.module_id = module_id
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05},
                                  name="instrument-emulator", daemon=True)
        self.servers.append(server)
        thread.start()
        self.threads.append(thread)
        return server

    def set_scenario(self, scenario, module_id=0):
        if scenario not in SCENARIOS:
            raise ValueError("unknown emulator scenario; choose " + ", ".join(SCENARIOS))
        if type(module_id) is not int or not 0 <= module_id < self.module_count:
            raise ValueError("emulator fault module must be a configured module ID")
        with self.lock:
            self.scenario = scenario
            self.fault_module = module_id

    def snapshot(self):
        with self.lock:
            return {
                "simulated": True, "no_hardware_access": True,
                "camera_transport": "loopback TCP text (SSH authentication not emulated)",
                "camera_port": self.camera.server_address[1],
                "fpga_transport": "legacy UDP register protocol on loopback",
                "modules": copy.deepcopy(self.modules), "scenario": self.scenario,
                "fault_module": self.fault_module, "scenarios": list(SCENARIOS),
                "camera_requests": dict(self.camera_counts),
                "fpga_requests": copy.deepcopy(self.fpga_counts),
            }

    def reader_config(self):
        # Do not merge hardware addresses or helper executables from readonly.
        return {
            "timeout_seconds": self.read_timeout,
            "camera": {"host": "127.0.0.1", "user": "root"},
            "detector": {"transport": "udp", "timeout_seconds": min(.2, self.read_timeout),
                         "retries": 2, "modules": copy.deepcopy(self.modules)},
        }

    def camera_runner(self, argv, timeout, stdin=None):
        if (not argv or argv[0] != "ssh" or argv[-1] != "root@127.0.0.1"
                or stdin not in ("dsh sensor temp\n", "dsh bias adc volts\n")):
            raise ValueError("emulated camera accepts only the fixed camera read commands")
        try:
            with socket.create_connection(self.camera.server_address, timeout=timeout) as connection:
                connection.settimeout(timeout)
                connection.sendall(json.dumps({"command": stdin.strip()}).encode("utf-8") + b"\n")
                with connection.makefile("rb") as stream:
                    packet = stream.readline(1024 * 1024 + 1)
            if not packet or len(packet) > 1024 * 1024:
                raise RuntimeError("invalid emulated camera response length")
            response = json.loads(packet)
            if response["returncode"]:
                raise RuntimeError("hardware read exited {}: {}".format(
                    response["returncode"], response.get("stderr", "")))
            return response["stdout"]
        except TimeoutError as error:
            raise RuntimeError("emulated camera read timed out after {} s".format(timeout)) from error

    def camera_response(self, command):
        with self.lock:
            self.camera_counts["requests"] += 1
            scenario = self.scenario
            if command not in ("dsh sensor temp", "dsh bias adc volts"):
                self.camera_counts["rejected"] += 1
                return {"returncode": 17, "stdout": "", "stderr": "read-only emulator"}
        if self.stopping.wait(self.camera_delay):
            return None
        if scenario == "camera_timeout":
            self.stopping.wait(self.read_timeout * 2)
            return None
        if scenario == "camera_error":
            return {"returncode": 17, "stdout": "", "stderr": "simulated camera command failure"}
        if command == "dsh sensor temp":
            if scenario == "malformed_temperature":
                output = "Sensor Temperatures:\nno numeric channels\n"
            else:
                drift = self.drift * math.sin((time.monotonic() - self.started) / 10)
                values = {name: value + drift for name, value in TEMPERATURES.items()}
                if scenario == "suspect_temperature":
                    values["Q3"] = 198.86
                output = "Sensor Temperatures:\n" + "".join(
                    "  {} : {:.3f}\n".format(name, value) for name, value in values.items())
        else:
            output = "Measured supplies and biases:\n" + "".join(
                "{} : {:.3f} {}\n".format(name, value, unit) for name, value, unit in BIASES)
        return {"returncode": 0, "stdout": output}

    def fpga_response(self, module_id, packet):
        with self.lock:
            counts = self.fpga_counts[str(module_id)]
            counts["requests"] += 1
            if packet != REGISTER_REQUEST:
                counts["rejected"] += 1
                return None
            scenario = self.scenario if module_id == self.fault_module else "healthy"
            if scenario == "fpga_timeout":
                counts["dropped"] += 1
                return None
        if self.stopping.wait(self.fpga_delay):
            return None
        value = 0x0000003F
        if scenario == "unsynchronized":
            value = 0x0000003B
        elif scenario == "link_errors":
            value = 0x0804023F
        response = struct.pack("<6I", 0x203B2D29, 0,
                               420 if scenario == "malformed_fpga" else 0x70000000 | 420,
                               value, 0x70000000, 0)
        with self.lock:
            counts["replies"] += 1
        return response

    def close(self):
        self.stopping.set()
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)
        self.servers.clear()
        self.threads.clear()
