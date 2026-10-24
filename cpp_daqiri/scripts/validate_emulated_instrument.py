#!/usr/bin/env python3
"""Read the persistent camera/FPGA emulator through the public supervisor port."""

from __future__ import annotations

import argparse
import time

import zmq


SCENARIOS = (
    "healthy", "suspect_temperature", "camera_error", "camera_timeout",
    "malformed_temperature", "fpga_timeout", "malformed_fpga",
    "link_errors", "unsynchronized",
)
READS = ("camera.read_temperature", "camera.read_biases", "detector.read_link_status")


def request(endpoint, command, **fields):
    with zmq.Context.instance().socket(zmq.REQ) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 1500)
        socket.setsockopt(zmq.RCVTIMEO, 3000)
        socket.connect(endpoint)
        socket.send_json(dict(command=command, **fields))
        response = socket.recv_json()
    if not response.get("ok"):
        raise RuntimeError("{}: {}".format(command, response.get("error")))
    return response


def read(endpoint, command, timeout):
    accepted = request(endpoint, command)
    if not accepted.get("accepted"):
        raise RuntimeError("read was not accepted asynchronously")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(endpoint, "get_state")
        operation = state["instrument"]["operation"]
        if operation["id"] == accepted["operation_id"] and operation["state"] != "running":
            return state["instrument"]
        time.sleep(.03)
    raise RuntimeError("read did not finish before the validation timeout")


def validate(endpoint, timeout, scenario=None, module_id=0):
    instrument = request(endpoint, "get_state").get("instrument", {})
    if instrument.get("mode") != "emulated_readonly" or not instrument.get("service_online"):
        raise RuntimeError("requires an online emulated_readonly service; no live hardware mode is allowed")
    if scenario is not None:
        instrument = request(endpoint, "emulator.set_scenario", scenario=scenario,
                             module_id=module_id)["instrument"]
    scenario = instrument["emulator"]["scenario"]
    for command in READS:
        state = read(endpoint, command, timeout)
        camera_failure = scenario in ("camera_error", "camera_timeout")
        expected_failure = (camera_failure and command.startswith("camera.")) or (
            scenario == "malformed_temperature" and command == "camera.read_temperature"
        ) or (scenario in ("fpga_timeout", "malformed_fpga") and command == "detector.read_link_status")
        expected = "failed" if expected_failure else "completed"
        if state["operation"]["state"] != expected:
            raise RuntimeError("{}: expected {}, got {}".format(command, expected, state["operation"]))
        if not expected_failure:
            if command == "camera.read_temperature":
                values = state["camera"]["temperatures"]
                if len(values) != 8 or (scenario == "suspect_temperature" and values["Q3"]["valid"]):
                    raise RuntimeError("incorrect temperature probes or suspect flag")
            elif command == "camera.read_biases":
                values = state["camera"]["biases"]
                if len(values) != 24 or values["BIASN"]["unit"] != "mA" or values["M2V5"]["value"] >= 0:
                    raise RuntimeError("bias names, current units, or negative rail not preserved")
            else:
                modules = state["detector"]["modules"]
                if len(modules) != len(state["emulator"]["modules"]):
                    raise RuntimeError("incomplete emulated module list")
                fault_module = state["emulator"]["fault_module"]
                affected = next(item for item in modules if item["module_id"] == fault_module)
                status = {"link_errors": "errors_latched", "unsynchronized": "unsynchronized"}.get(scenario, "ready")
                if affected["status"] != status:
                    raise RuntimeError("unexpected emulated link flags")
        print("PASS {} / {}: {}".format(scenario, command, expected), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--module-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--exercise-failures", action="store_true",
                        help="test all scenarios, then restore and read healthy")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # Check the backend before attempting even simulated failure injection.
    state = request(args.endpoint, "get_state").get("instrument", {})
    if state.get("mode") != "emulated_readonly":
        raise RuntimeError("refusing to test any backend except emulated_readonly")
    if args.exercise_failures:
        try:
            for scenario in SCENARIOS:
                validate(args.endpoint, args.timeout, scenario, args.module_id)
        finally:
            request(args.endpoint, "emulator.set_scenario", scenario="healthy")
        validate(args.endpoint, args.timeout)
    else:
        validate(args.endpoint, args.timeout, args.scenario, args.module_id)
    print("emulated instrument validation OK: " + args.endpoint)


if __name__ == "__main__":
    main()
