"""Display the live STEM thinned stream inside Gatan DigitalMicrograph.

Open this file in DigitalMicrograph's Python editor, edit the configuration
constants below, enable "Execute in background", and execute it. Stop a
background script with Ctrl+Shift+Q.
"""

import os
import sys
import time

import numpy as np


# Use the forwarded endpoint here (normally localhost) or the directly
# reachable IGX management address. This viewer only needs the PUB endpoint.
STREAM_ENDPOINT = "tcp://127.0.0.1:15556"
TOPIC_PREFIX = "stem/"

# Phase-one DAQ control integration. The Python loop owns the REQ socket and
# services modeless palette commands between image polls.
ENABLE_CONTROL = True
CONTROL_ENDPOINT = "tcp://127.0.0.1:15557"
CONTROL_STATE_POLL_SECONDS = 1.0
AUTO_LAUNCH_CONTROL_PALETTE = True

# DM updates can be more expensive than receipt. The socket is always drained
# to the newest product so reducing this limit does not backpressure the DAQ.
MAX_DISPLAY_HZ = 2.0
DISPLAY_PRODUCTS = ("representative", "sum")
RECEIVE_TIMEOUT_MS = 250
STATUS_EVERY_PRODUCTS = 10

# Zero runs until Ctrl+Shift+Q. Set a small positive value for initial tests.
MAX_DISPLAYED_PRODUCTS = 0


def _add_module_directory():
    try:
        directory = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        directory = os.getcwd()
    if directory not in sys.path:
        sys.path.insert(0, directory)


_add_module_directory()
try:
    from stem_stream_protocol import decode_product  # noqa: E402
    from stem_dm_control import (  # noqa: E402
        DMControlBridge,
        DMPersistentTagStore,
        PersistentTagMailbox,
        VIEWER_STOP_REQUESTED,
        ZmqControlClient,
        initialize_defaults,
        launch_control_palette,
    )
except ImportError as error:
    raise ImportError(
        "Cannot import the STEM stream/control modules. Run "
        "install_dm_module_path.py with the GMS Python environment, then "
        "restart DigitalMicrograph."
    ) from error


class DMImageWindow(object):
    """One persistent DigitalMicrograph image and its mapped NumPy storage."""

    def __init__(self, dm_module, title, array):
        # Decoded ZeroMQ arrays are views over immutable message bytes. DM's
        # CreateImage requires an owning NumPy array, not merely a contiguous
        # view, so an explicit copy is required at window creation.
        initial = np.array(array, dtype=np.float32, order="C", copy=True)
        self._image = dm_module.CreateImage(initial)
        if self._image is None:
            raise RuntimeError("DigitalMicrograph CreateImage returned no image")
        self._image.SetName(title)
        self._image.ShowImage()
        self._array = self._image.GetNumArray()
        self._title = title

    def update(self, title, array):
        if array.shape != self._array.shape:
            raise ValueError(
                "image shape changed from {} to {}".format(
                    self._array.shape, array.shape
                )
            )
        if title != self._title:
            self._image.SetName(title)
            self._title = title
        # The DM array maps image memory. Replace values in place rather than
        # rebinding the variable, then explicitly refresh the image display.
        np.copyto(self._array, array, casting="unsafe")
        self._image.UpdateImage()

    def release(self):
        """Release Python references without closing the displayed DM image."""
        self._array = None
        image = self._image
        self._image = None
        del image


class DigitalMicrographViewer(object):
    def __init__(self, dm_module, display_products=DISPLAY_PRODUCTS):
        self._dm = dm_module
        self._display_products = tuple(display_products)
        self._windows = {}

    @staticmethod
    def _title(metadata, product_name):
        receiver = int(metadata.get("receiver_id", -1))
        stage = metadata.get("processing_stage", "unknown")
        label = "Single frame" if product_name == "representative" else "128-frame sum"
        return "STEM RX{} - {} - {}".format(receiver, stage, label)

    def display(self, metadata, arrays):
        receiver = int(metadata.get("receiver_id", -1))
        updated = 0
        for product_name in self._display_products:
            if product_name not in arrays:
                continue
            key = (receiver, product_name)
            title = self._title(metadata, product_name)
            if key not in self._windows:
                self._windows[key] = DMImageWindow(
                    self._dm, title, arrays[product_name]
                )
            else:
                self._windows[key].update(title, arrays[product_name])
            updated += 1
        return updated

    def release(self):
        for window in self._windows.values():
            window.release()
        self._windows.clear()


def _receive_newest(socket, zmq_module):
    newest = socket.recv_multipart()
    while True:
        try:
            newest = socket.recv_multipart(zmq_module.NOBLOCK)
        except zmq_module.Again:
            return newest


def run(dm_module):
    try:
        import zmq
    except ImportError as error:
        raise RuntimeError(
            "pyzmq is not installed in the DigitalMicrograph Python environment"
        ) from error

    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC_PREFIX.encode("utf-8"))
    subscriber.setsockopt(zmq.RCVHWM, 2)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVTIMEO, RECEIVE_TIMEOUT_MS)
    subscriber.connect(STREAM_ENDPOINT)

    control_bridge = None
    control_mailbox = None
    control_tags = None
    if ENABLE_CONTROL:
        control_tags = DMPersistentTagStore(dm_module)
        initialize_defaults(control_tags)
        control_tags.set_bool(VIEWER_STOP_REQUESTED, False)
        control_mailbox = PersistentTagMailbox(control_tags)
        control_client = ZmqControlClient(zmq, CONTROL_ENDPOINT)
        control_bridge = DMControlBridge(
            control_mailbox,
            control_client,
            state_poll_seconds=CONTROL_STATE_POLL_SECONDS,
        )
        control_mailbox.set_engine_online(True)
        # Fetch initial values before constructing the palette so fields match
        # the currently effective DAQ configuration where possible.
        control_bridge.tick()
        if AUTO_LAUNCH_CONTROL_PALETTE:
            try:
                launch_control_palette(dm_module)
            except Exception as error:
                print(
                    "Could not auto-launch STEM DAQ controls: {}. Open and "
                    "execute stem_dm_controls.s manually.".format(error)
                )

    viewer = DigitalMicrographViewer(dm_module)
    displayed = 0
    last_display = 0.0
    minimum_interval = 0.0 if MAX_DISPLAY_HZ <= 0 else 1.0 / MAX_DISPLAY_HZ
    print(
        "STEM DM viewer connected to {}; subscribed to {!r}".format(
            STREAM_ENDPOINT, TOPIC_PREFIX
        )
    )

    try:
        while MAX_DISPLAYED_PRODUCTS <= 0 or displayed < MAX_DISPLAYED_PRODUCTS:
            if (
                control_mailbox is not None
                and control_tags.get_bool(VIEWER_STOP_REQUESTED, False)
            ):
                print("STEM DM viewer stop requested from control palette")
                break
            if control_bridge is not None:
                control_bridge.tick()
            try:
                parts = _receive_newest(subscriber, zmq)
            except zmq.Again:
                continue

            now = time.monotonic()
            if now - last_display < minimum_interval:
                continue
            topic, metadata, arrays = decode_product(parts)
            if viewer.display(metadata, arrays) == 0:
                continue
            displayed += 1
            last_display = now
            if STATUS_EVERY_PRODUCTS > 0 and displayed % STATUS_EVERY_PRODUCTS == 0:
                print(
                    "Displayed {} products; latest {} batch {} ({})".format(
                        displayed,
                        topic,
                        metadata.get("batch_index", "?"),
                        "complete" if metadata.get("complete") else "incomplete",
                    )
                )
    except KeyboardInterrupt:
        print("STEM DM viewer interrupted after {} products".format(displayed))
    finally:
        subscriber.close()
        if control_mailbox is not None:
            control_mailbox.set_engine_online(False)
            control_tags.set_bool(VIEWER_STOP_REQUESTED, False)
        viewer.release()
        print("STEM DM viewer stopped")


if __name__ == "__main__":
    try:
        import DigitalMicrograph as DM
    except ImportError as error:
        raise SystemExit(
            "Run this script inside DigitalMicrograph; DigitalMicrograph could not be imported"
        ) from error
    run(DM)
