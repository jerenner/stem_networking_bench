#!/usr/bin/env python3
"""Exercise the mock instrument service through the supervisor REP endpoint."""

from __future__ import annotations

import argparse
import time

import zmq


def request(socket, command: str, **fields) -> dict:
    payload = {"command": command}
    payload.update(fields)
    socket.send_json(payload)
    response = socket.recv_json()
    if not response.get("ok"):
        raise RuntimeError("{} failed: {}".format(command, response.get("error")))
    return response


def wait_operation(socket, expected_state: str = "completed", timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(socket, "get_state")
        operation = state.get("instrument", {}).get("operation", {})
        if operation.get("state") != "running":
            if operation.get("state") != expected_state:
                raise RuntimeError(
                    "operation ended in {!r}: {}".format(
                        operation.get("state"), operation.get("error", "")
                    )
                )
            return state
        time.sleep(0.05)
    raise RuntimeError("mock operation did not complete before timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    args = parser.parse_args()

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 1500)
    socket.setsockopt(zmq.RCVTIMEO, 1500)
    socket.connect(args.endpoint)

    state = request(socket, "get_state")
    instrument = state.get("instrument", {})
    if instrument.get("mode") != "mock" or not instrument.get("service_online"):
        raise RuntimeError("online mock instrument service was not reported")

    request(socket, "camera.power_up")
    state = wait_operation(socket)
    assert state["instrument"]["camera"]["power_state"] == "ready"

    request(socket, "detector.resync")
    state = wait_operation(socket)
    assert state["instrument"]["detector"]["synchronized"]

    request(
        socket,
        "scan.configure",
        scan={
            "pause_count": 200,
            "read_count": 2,
            "positions_x": 4,
            "rows": 8,
            "flyback": 100,
            "flush_memory": True,
        },
    )
    request(socket, "scan.start")
    state = wait_operation(socket)
    assert state["instrument"]["scan"]["scan_number"] == 1

    request(
        socket,
        "mock.set_failure",
        operation="detector.resync",
        step=3,
        once=True,
    )
    request(socket, "detector.resync")
    state = wait_operation(socket, expected_state="failed")
    assert "injected mock failure" in state["instrument"]["operation"]["error"]

    print("mock instrument validation OK: {}".format(args.endpoint))


if __name__ == "__main__":
    main()
