#!/usr/bin/env python3
"""Read camera telemetry through SSH and FPGA status through read-only UDP.

Hardware reads run on a worker; get_state returns cached observations immediately.
The only camera commands are the fixed `dsh sensor temp` and `dsh bias adc volts`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import re
import signal
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path


READS = {
    "camera.read_temperature": "temperature",
    "camera.read_biases": "biases",
    "detector.read_link_status": "links",
}
TEMPERATURE = re.compile(r"^\s*(Q\d+|B\d+|HS\d+)\s*:\s*(\S+)", re.M)
BIAS = re.compile(
    r"^\s*([A-Z0-9][A-Z0-9_]*)\s*:\s*(\S+)\s+(mV|V|uA|mA|A)\s*$", re.M
)


def finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite hardware readback")
    return number


def parse_temperatures(text, minimum=-50.0, maximum=150.0):
    channels = {}
    for name, token in TEMPERATURE.findall(text):
        value = finite_number(token)
        valid = minimum <= value <= maximum
        channels[name] = {"value": value, "unit": "C", "valid": valid}
    if not channels:
        raise ValueError("no Q/B/HS temperature channels found in camera output")
    # Preserve suspect channels rather than silently hiding a bad thermistor.
    valid_values = [item["value"] for item in channels.values() if item["valid"]]
    return {
        "temperatures": channels,
        "temperature_c": max(valid_values) if valid_values else None,
        "temperature_warning": ", ".join(
            name + " outside configured range"
            for name, item in channels.items() if not item["valid"]
        ),
    }


def parse_biases(text):
    measurements = {
        name: {"value": finite_number(token), "unit": unit}
        for name, token, unit in BIAS.findall(text)
    }
    if not measurements:
        raise ValueError("no bias/supply measurements with units found in camera output")
    return {"biases": measurements}


def decode_link_register(module_id, host, value):
    if isinstance(value, bool):
        raise ValueError("register420 must be a 32-bit unsigned integer")
    if isinstance(value, str):
        value = int(value, 16 if value.lower().startswith("0x") else 10)
    if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("register420 must be a 32-bit unsigned integer")
    masks = {
        "jesd_sync": value & 0x3F,
        "not_in_table": (value >> 8) & 0x3F,
        "disparity_error": (value >> 16) & 0x3F,
        "characters_were_ks": (value >> 24) & 0x3F,
    }
    synchronized = masks["jesd_sync"] == 0x3F
    errors = any(masks[name] for name in masks if name != "jesd_sync")
    return {
        "module_id": module_id, "host": host, "register420": value,
        "status": "unsynchronized" if not synchronized else (
            "errors_latched" if errors else "ready"
        ),
        "synchronized": synchronized, "masks": masks,
        "latched": True, "status_reset_performed": False,
        "lanes": [
            {name: bool(mask & (1 << lane)) for name, mask in masks.items()}
            for lane in range(6)
        ],
    }


def read_link_register(host, port=42000, timeout=.2, retries=7):
    """Match NetworkInterface::ReadRegister(420), including its padding read.

    The recovered Linux implementation sends native little-endian uint32 words.
    Bit 31 would request a write; it is never set in either address here.
    """
    packet = struct.pack("<6I", 0x203B2D29, 0, 420, 0, 0, 0)
    last_error = "no response"
    for _ in range(retries):
        try:
            # A new socket per attempt prevents an earlier response being
            # mistaken for a later sample; connect also filters the sender.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
                connection.settimeout(timeout)
                connection.connect((host, port))
                connection.send(packet)
                response = connection.recv(65535)
            if len(response) != len(packet):
                raise ValueError("register response length {} != 24".format(len(response)))
            key, sequence, address, value, _, _ = struct.unpack("<6I", response)
            if (key, sequence, address) != (0x203B2D29, 0, 0x70000000 | 420):
                raise ValueError("register response key, sequence, or address mismatch")
            return value
        except (OSError, ValueError) as error:
            last_error = str(error) or type(error).__name__
    raise RuntimeError("register 420 read failed after {} attempts: {}".format(retries, last_error))


def execute(argv, timeout, stdin=None):
    """Check process status and limit how long a read can occupy the worker."""
    try:
        result = subprocess.run(
            argv, input=stdin, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("hardware read timed out after {} s".format(timeout)) from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RuntimeError("hardware read exited {}: {}".format(result.returncode, detail))
    if len(result.stdout) > 1024 * 1024:
        raise RuntimeError("hardware response exceeds 1 MiB")
    return result.stdout


class HardwareReader:
    def __init__(self, config, runner=execute):
        self.camera = config.get("camera", {})
        self.detector = config.get("detector", {})
        self.runner = runner
        self.timeout = finite_number(config.get("timeout_seconds", 8.0))
        if not 0 < self.timeout <= 60:
            raise ValueError("instrument.readonly.timeout_seconds must be in (0, 60]")
        self.modules = self.detector.get("modules", [])
        for module in self.modules:
            if (not isinstance(module.get("id"), int) or isinstance(module["id"], bool)
                    or not 0 <= module["id"] < 8):
                raise ValueError("detector module IDs must be integers from 0 to 7")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", str(module.get("host", ""))):
                raise ValueError("invalid detector module host")
            if "port" in module and (type(module["port"]) is not int or not 1 <= module["port"] <= 65535):
                raise ValueError("detector module port must be an integer in [1,65535]")
        identifiers = [item["id"] for item in self.modules]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("detector module IDs must be unique")
        self.reader_argv = self.detector.get("reader_argv", [])
        if not isinstance(self.reader_argv, list) or any(
            not isinstance(arg, str) or not arg for arg in self.reader_argv
        ):
            raise ValueError("detector.reader_argv must be a list of executable arguments")
        self.transport = self.detector.get("transport", "udp")
        if self.transport not in ("udp", "helper"):
            raise ValueError("detector.transport must be udp or helper")
        self.port = int(self.detector.get("port", 42000))
        self.register_timeout = finite_number(self.detector.get("timeout_seconds", .2))
        self.retries = int(self.detector.get("retries", 7))
        if not 1 <= self.port <= 65535 or not 0 < self.register_timeout <= 2 or not 1 <= self.retries <= 10:
            raise ValueError("invalid detector port, timeout_seconds (0,2], or retries [1,10]")

    @property
    def supported_commands(self):
        commands = ["camera.read_temperature", "camera.read_biases"]
        if self.modules and (self.transport == "udp" or self.reader_argv):
            commands.append("detector.read_link_status")
        return commands

    def camera_read(self, command):
        if command not in ("sensor temp", "bias adc volts"):
            raise ValueError("camera command is not in the read-only allowlist")
        host = self.camera.get("host", "")
        user = self.camera.get("user", "root")
        for name, value in (("host", host), ("user", user)):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
                raise ValueError("invalid camera SSH " + name)
        port = int(self.camera.get("port", 22))
        if not 1 <= port <= 65535:
            raise ValueError("invalid camera SSH port")
        argv = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout={}".format(max(1, int(self.timeout))),
            "-p", str(port),
        ]
        if self.camera.get("identity_file"):
            argv.extend(["-i", self.camera["identity_file"], "-o", "IdentitiesOnly=yes"])
        if self.camera.get("known_hosts_file"):
            argv.extend(["-o", "UserKnownHostsFile=" + self.camera["known_hosts_file"]])
        argv.append(user + "@" + host)
        return self.runner(argv, self.timeout, "dsh " + command + "\n")

    def read(self, name):
        if name == "temperature":
            raw = self.camera_read("sensor temp")
            minimum = finite_number(self.camera.get("temperature_min_c", -50))
            maximum = finite_number(self.camera.get("temperature_max_c", 150))
            if minimum >= maximum:
                raise ValueError("temperature_min_c must be below temperature_max_c")
            return parse_temperatures(raw, minimum, maximum), raw
        if name == "biases":
            raw = self.camera_read("bias adc volts")
            return parse_biases(raw), raw
        if name != "links":
            raise ValueError("unknown read " + name)
        if not self.modules:
            raise RuntimeError("no FPGA modules configured in instrument.readonly.detector.modules")
        if self.transport == "helper" and not self.reader_argv:
            raise RuntimeError("helper transport requires detector.reader_argv")
        modules = []
        for module in self.modules:
            try:
                if self.transport == "udp":
                    value = read_link_register(module["host"], module.get("port", self.port),
                                               self.register_timeout, self.retries)
                else:
                    raw = self.runner(
                        self.reader_argv + ["--host", module["host"], "--register", "420"], self.timeout,
                    )
                    value = json.loads(raw)["register420"]
                modules.append(decode_link_register(module["id"], module["host"], value))
            except Exception as error:
                modules.append({
                    "module_id": module["id"], "host": module["host"],
                    "status": "error", "error": str(error), "synchronized": None,
                })
        failures = [item for item in modules if item["status"] == "error"]
        result = {
            "modules": modules, "links": [item["status"] for item in modules],
            "synchronized": None if failures else all(item["synchronized"] for item in modules),
        }
        if failures:
            result["read_error"] = "; ".join(
                "module {}: {}".format(item["module_id"], item["error"])
                for item in failures
            )
        return result, json.dumps(modules)


class ReadOnlyInstrumentService:
    def __init__(self, config, reader=None, clock=time.time, emulator=None):
        self.config = config
        self.emulator = emulator
        self.mode = config.get("mode", "hardware_readonly")
        if self.mode not in ("hardware_readonly", "emulated_readonly"):
            raise ValueError("unsupported read-only instrument mode")
        if (self.mode == "emulated_readonly") != (emulator is not None):
            raise ValueError("emulated_readonly mode requires its loopback emulator")
        self.reader = reader if reader is not None else HardwareReader(config.get("readonly", {}))
        self.clock = clock
        self.stale_seconds = finite_number(config.get("readonly", {}).get("stale_after_seconds", 120))
        if self.stale_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self.audit_path = config.get("readonly", {}).get("audit_log_path", "")
        self.lock = threading.Lock()
        self.tasks = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.next_id = 1
        self.state = {
            "enabled": True, "service_online": True, "mode": self.mode,
            "simulated": emulator is not None,
            "capabilities": {
                "read_only": True, "commands": self.reader.supported_commands,
                "camera_control": False, "detector_control": False, "scan_control": False,
            },
            "camera": {"power_state": "unknown", "insertion_state": "unknown",
                       "temperature_c": None, "target_temperature_c": None,
                       "temperatures": {}, "biases": {}},
            "detector": {"synchronized": None, "aligned": None, "links": [], "modules": []},
            "scan": {"state": "unknown", "running": False},
            "operation": {"state": "idle", "name": "", "id": "", "error": "",
                          "step": "", "completed_steps": 0, "total_steps": 0},
            "reads": {name: {"status": "not_read", "sampled_at": None, "attempted_at": None,
                             "error": "", "raw": "", "simulated": emulator is not None}
                      for name in READS.values()},
        }
        self.worker = threading.Thread(target=self._work, name="instrument-reader", daemon=True)
        self.worker.start()

    def _audit(self, event, command, detail=""):
        if not self.audit_path:
            return
        try:
            with open(self.audit_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch_ms": int(self.clock() * 1000), "event": event,
                                         "command": command, "detail": detail}) + "\n")
        except OSError:
            # A telemetry log failure must not kill the operation worker.
            pass

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        for record in state["reads"].values():
            timestamp = record["sampled_at"]
            record["age_seconds"] = None if timestamp is None else max(0, self.clock() - timestamp)
            record["stale"] = timestamp is None or record["status"] != "ok" or (
                record["age_seconds"] > self.stale_seconds
            )
        if self.emulator is not None:
            state["emulator"] = self.emulator.snapshot()
        return state

    def handle(self, request):
        command = request.get("command", "instrument.get_state")
        if command in ("instrument.get_state", "get_state", "operation.get"):
            return {"ok": True, "schema": "stem.instrument.v1", "instrument": self.snapshot()}
        if command == "shutdown":
            self.stopping.set()
            return {"ok": True}
        if command.startswith("emulator."):
            if self.emulator is None:
                return {"ok": False, "error": "emulator commands require emulated_readonly mode"}
            if command == "emulator.get_state":
                return {"ok": True, "instrument": self.snapshot()}
            if command != "emulator.set_scenario" or set(request) - {
                "command", "request_id", "scenario", "module_id"
            }:
                return {"ok": False, "error": "unsupported emulator command or parameters"}
            with self.lock:
                if self.state["operation"]["state"] == "running":
                    return {"ok": False, "error": "an instrument read is already running"}
                try:
                    self.emulator.set_scenario(request.get("scenario"), request.get("module_id", 0))
                except ValueError as error:
                    return {"ok": False, "error": str(error)}
                for record in self.state["reads"].values():
                    record.update(status="scenario_changed", error="")
            return {"ok": True, "message": "emulator scenario updated; request fresh readbacks",
                    "instrument": self.snapshot()}
        if command not in READS:
            return {"ok": False, "error": "command unavailable in " + self.mode + " mode: " + str(command)}
        if set(request) - {"command", "request_id"}:
            return {"ok": False, "error": "read commands accept no hardware parameters"}
        with self.lock:
            if self.state["operation"]["state"] == "running":
                return {"ok": False, "error": "an instrument read is already running"}
            identifier = "read-{:05d}".format(self.next_id)
            self.next_id += 1
            self.state["operation"] = {
                "id": identifier, "name": command, "state": "running", "step": "reading",
                "completed_steps": 0, "total_steps": 1, "error": "",
            }
            self.state["reads"][READS[command]].update(
                status="reading", attempted_at=self.clock(), error=""
            )
            self.tasks.put_nowait(command)
        return {"ok": True, "accepted": True, "operation_id": identifier,
                "message": "instrument read accepted", "instrument": self.snapshot()}

    def _work(self):
        while not self.stopping.is_set():
            try:
                command = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            name = READS[command]
            self._audit("read_started", command)
            try:
                values, raw = self.reader.read(name)
                error = values.pop("read_error", "")
                with self.lock:
                    target = "detector" if name == "links" else "camera"
                    self.state[target].update(values)
                    record = self.state["reads"][name]
                    record.update(status="error" if error else "ok", sampled_at=self.clock(),
                                  error=error, raw=raw[:16384])
                    self.state["operation"].update(
                        state="failed" if error else "completed", step="finished",
                        completed_steps=1, error=error,
                    )
                self._audit("read_failed" if error else "read_completed", command, error)
            except Exception as error:
                with self.lock:
                    self.state["reads"][name].update(status="error", error=str(error))
                    self.state["operation"].update(state="failed", step="failed", error=str(error))
                self._audit("read_failed", command, str(error))
            finally:
                self.tasks.task_done()

    def close(self):
        self.stopping.set()
        if self.emulator is not None:
            self.emulator.close()
        self.worker.join()


def create_service(config):
    if config.get("mode") != "emulated_readonly":
        return ReadOnlyInstrumentService(config)
    from stem_instrument_emulator import InstrumentEmulator

    emulator = InstrumentEmulator(config.get("emulator", {}))
    try:
        reader = HardwareReader(emulator.reader_config(), runner=emulator.camera_runner)
        return ReadOnlyInstrumentService(config, reader=reader, emulator=emulator)
    except BaseException:
        emulator.close()
        raise


def main():
    import yaml
    import zmq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["instrument"]
    if not config.get("enabled") or config.get("mode") not in ("hardware_readonly", "emulated_readonly"):
        raise ValueError("requires enabled hardware_readonly or emulated_readonly instrument mode")
    endpoint = config.get("endpoint", "ipc:///tmp/stem_daqiri_instrument.ipc")
    if not (endpoint.startswith("ipc://") or endpoint.startswith("tcp://127.0.0.1:")):
        raise ValueError("instrument service endpoint must be local IPC or loopback TCP")
    service = create_service(config)
    signal.signal(signal.SIGINT, lambda *_: service.stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: service.stopping.set())
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    try:
        socket.bind(endpoint)
        print(service.mode + " instrument service listening on " + endpoint, flush=True)
        while not service.stopping.is_set():
            try:
                request = socket.recv_json()
            except zmq.Again:
                continue
            except ValueError as error:
                socket.send_json({"ok": False, "error": "invalid JSON: " + str(error)})
                continue
            try:
                response = service.handle(request)
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            socket.send_json(response, allow_nan=False)
    finally:
        service.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
