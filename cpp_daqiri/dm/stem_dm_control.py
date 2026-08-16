"""DigitalMicrograph persistent-tag bridge for STEM DAQ controls.

The modeless DM-script palette writes typed settings and a monotonically
increasing request sequence into DigitalMicrograph persistent tags. The Python
image loop consumes each request, translates it to the DAQIRI JSON protocol,
and publishes the response and a compact status snapshot back to the tags.
"""

import json
import os
import time


TAG_ROOT = "STEM DAQ"
REQUEST_SEQUENCE = TAG_ROOT + ":Control:Request:Sequence"
REQUEST_COMMAND = TAG_ROOT + ":Control:Request:Command"
RESPONSE_SEQUENCE = TAG_ROOT + ":Control:Response:Sequence"
RESPONSE_JSON = TAG_ROOT + ":Control:Response:JSON"
STATE_SEQUENCE = TAG_ROOT + ":State:Sequence"
STATE_JSON = TAG_ROOT + ":State:JSON"
ENGINE_ONLINE = TAG_ROOT + ":State:EngineOnline"
ENGINE_HEARTBEAT = TAG_ROOT + ":State:EngineHeartbeat"

VISUALIZATION_ROOT = TAG_ROOT + ":Control:Visualization"
BURST_ROOT = TAG_ROOT + ":Control:Burst"

STAGES = (
    "raw",
    "dark_subtracted",
    "dark_blr",
    "corrected",
    "thresholded",
)


class DMPersistentTagStore(object):
    """Scalar-only adapter around DigitalMicrograph's persistent TagGroup."""

    def __init__(self, dm_module):
        self._dm = dm_module

    def get_long(self, path, default=0):
        success, value = self._dm.GetPersistentTagGroup().GetTagAsLong(path)
        return int(value) if success else int(default)

    def get_float(self, path, default=0.0):
        success, value = self._dm.GetPersistentTagGroup().GetTagAsDouble(path)
        return float(value) if success else float(default)

    def get_bool(self, path, default=False):
        success, value = self._dm.GetPersistentTagGroup().GetTagAsBoolean(path)
        return bool(value) if success else bool(default)

    def get_string(self, path, default=""):
        success, value = self._dm.GetPersistentTagGroup().GetTagAsString(path)
        return str(value) if success else str(default)

    def set_long(self, path, value):
        self._dm.GetPersistentTagGroup().SetTagAsLong(path, int(value))

    def set_float(self, path, value):
        self._dm.GetPersistentTagGroup().SetTagAsDouble(path, float(value))

    def set_bool(self, path, value):
        self._dm.GetPersistentTagGroup().SetTagAsBoolean(path, bool(value))

    def set_string(self, path, value):
        self._dm.GetPersistentTagGroup().SetTagAsString(path, str(value))


def _field(root, name):
    return root + ":" + name


def _stage(value, field_name):
    value = str(value).strip()
    if value not in STAGES:
        raise ValueError(
            "{} must be one of {} (received {!r})".format(
                field_name, ", ".join(STAGES), value
            )
        )
    return value


def build_request(command, tags):
    """Build one control-protocol request from typed palette settings."""

    if command in ("start_acquisition", "stop_acquisition", "get_state"):
        return {"command": command}

    if command == "apply_visualization":
        return {
            "command": "set_runtime",
            "thinned_stream": {
                "publishing": tags.get_bool(
                    _field(VISUALIZATION_ROOT, "Publishing"), True
                ),
                "processing_stage": _stage(
                    tags.get_string(
                        _field(VISUALIZATION_ROOT, "ProcessingStage"),
                        "corrected",
                    ),
                    "visualization processing stage",
                ),
                "total_refresh_hz": tags.get_float(
                    _field(VISUALIZATION_ROOT, "RefreshHz"), 1.0
                ),
                "representative_frame_index": tags.get_long(
                    _field(VISUALIZATION_ROOT, "RepresentativeFrame"), 64
                ),
                "include_representative_frame": tags.get_bool(
                    _field(VISUALIZATION_ROOT, "IncludeRepresentative"), True
                ),
                "include_bucket_sum": tags.get_bool(
                    _field(VISUALIZATION_ROOT, "IncludeSum"), True
                ),
                "threshold": {
                    "zlp": tags.get_float(
                        _field(VISUALIZATION_ROOT, "ZLPThreshold"), 0.0
                    ),
                    "core_loss": tags.get_float(
                        _field(VISUALIZATION_ROOT, "CoreLossThreshold"), 0.0
                    ),
                },
            },
        }

    if command in ("configure_burst", "arm_burst"):
        action = "arm" if command == "arm_burst" else ""
        return {
            "command": "set_runtime",
            "burst_writer": {
                "action": action,
                "processing_stage": _stage(
                    tags.get_string(
                        _field(BURST_ROOT, "ProcessingStage"), "corrected"
                    ),
                    "burst processing stage",
                ),
                "filepath_template": tags.get_string(
                    _field(BURST_ROOT, "FilepathTemplate"),
                    "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5",
                ),
                "dataset_name": tags.get_string(
                    _field(BURST_ROOT, "DatasetName"), "/frames"
                ),
                "buckets_per_capture": tags.get_long(
                    _field(BURST_ROOT, "BucketsPerCapture"), 1
                ),
                "capture_count": tags.get_long(
                    _field(BURST_ROOT, "CaptureCount"), 1
                ),
                "rearm_after_write": tags.get_bool(
                    _field(BURST_ROOT, "RearmAfterWrite"), True
                ),
                "strict_complete": tags.get_bool(
                    _field(BURST_ROOT, "StrictComplete"), False
                ),
                "threshold": {
                    "zlp": tags.get_float(
                        _field(BURST_ROOT, "ZLPThreshold"), 0.0
                    ),
                    "core_loss": tags.get_float(
                        _field(BURST_ROOT, "CoreLossThreshold"), 0.0
                    ),
                },
            },
        }

    if command in ("disarm_burst", "abort_burst"):
        action = "disarm" if command == "disarm_burst" else "abort"
        return {
            "command": "set_runtime",
            "burst_writer": {"action": action},
        }

    raise ValueError("unknown DM control command {!r}".format(command))


def initialize_defaults(tags):
    """Populate palette fields only when they have not been set previously."""

    defaults = (
        ("bool", _field(VISUALIZATION_ROOT, "Publishing"), True),
        ("string", _field(VISUALIZATION_ROOT, "ProcessingStage"), "corrected"),
        ("float", _field(VISUALIZATION_ROOT, "RefreshHz"), 1.0),
        ("long", _field(VISUALIZATION_ROOT, "RepresentativeFrame"), 64),
        ("bool", _field(VISUALIZATION_ROOT, "IncludeRepresentative"), True),
        ("bool", _field(VISUALIZATION_ROOT, "IncludeSum"), True),
        ("float", _field(VISUALIZATION_ROOT, "ZLPThreshold"), 0.0),
        ("float", _field(VISUALIZATION_ROOT, "CoreLossThreshold"), 0.0),
        ("string", _field(BURST_ROOT, "ProcessingStage"), "corrected"),
        (
            "string",
            _field(BURST_ROOT, "FilepathTemplate"),
            "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5",
        ),
        ("string", _field(BURST_ROOT, "DatasetName"), "/frames"),
        ("long", _field(BURST_ROOT, "BucketsPerCapture"), 1),
        ("long", _field(BURST_ROOT, "CaptureCount"), 1),
        ("bool", _field(BURST_ROOT, "RearmAfterWrite"), True),
        ("bool", _field(BURST_ROOT, "StrictComplete"), False),
        ("float", _field(BURST_ROOT, "ZLPThreshold"), 0.0),
        ("float", _field(BURST_ROOT, "CoreLossThreshold"), 0.0),
    )
    getters = {
        "bool": tags.get_bool,
        "string": tags.get_string,
        "float": tags.get_float,
        "long": tags.get_long,
    }
    setters = {
        "bool": tags.set_bool,
        "string": tags.set_string,
        "float": tags.set_float,
        "long": tags.set_long,
    }
    for value_type, path, default in defaults:
        # Scalar Py_TagGroup access has no generic existence query. Strings use
        # a private sentinel; numeric defaults are harmless to refresh because
        # the palette writes them back before queuing an apply command.
        if value_type == "string":
            current = getters[value_type](path, "__STEM_DAQ_MISSING__")
            if current == "__STEM_DAQ_MISSING__":
                setters[value_type](path, default)
        else:
            current = getters[value_type](path, default)
            setters[value_type](path, current)


def compact_status(state):
    """Extract stable, human-readable palette status fields from DAQ state."""

    supervisor = state.get("supervisor", {})
    acquisition = state.get("acquisition", {})
    running = bool(
        supervisor.get("acquisition_running", acquisition.get("running", False))
    )
    lifecycle = supervisor.get("state", "running" if running else "stopped")

    thinned = state.get("thinned_stream", {})
    if not thinned.get("capability_enabled", False):
        visualization = "not allocated"
    elif thinned.get("publishing", False):
        visualization = "{} at {:.3g} Hz".format(
            thinned.get("processing_stage", "unknown"),
            float(thinned.get("total_refresh_hz", 0.0)),
        )
    else:
        visualization = "publishing disabled"

    burst = state.get("burst_writer", {})
    if not burst.get("capability_enabled", False):
        burst_status = "not allocated"
    elif burst.get("busy", False):
        burst_status = "capturing / writing"
    elif burst.get("armed", False):
        burst_status = "armed / waiting"
    else:
        burst_status = "idle"

    return {
        "control": "online" if state.get("ok", False) else "error",
        "acquisition": str(lifecycle),
        "visualization": visualization,
        "burst": burst_status,
        "message": str(state.get("message", "")),
    }


class PersistentTagMailbox(object):
    def __init__(self, tags):
        self.tags = tags
        self.last_request_sequence = tags.get_long(REQUEST_SEQUENCE, 0)
        self.state_sequence = tags.get_long(STATE_SEQUENCE, 0)

    def next_request(self):
        sequence = self.tags.get_long(REQUEST_SEQUENCE, 0)
        if sequence <= self.last_request_sequence:
            return None
        self.last_request_sequence = sequence
        command = self.tags.get_string(REQUEST_COMMAND, "")
        return sequence, build_request(command, self.tags)

    def publish_response(self, sequence, response):
        self.tags.set_string(RESPONSE_JSON, json.dumps(response, sort_keys=True))
        self.tags.set_long(RESPONSE_SEQUENCE, sequence)

    def publish_state(self, state):
        status = compact_status(state)
        thinned = state.get("thinned_stream", {})
        burst = state.get("burst_writer", {})
        visualization_fields = (
            ("bool", "Publishing", thinned.get("publishing")),
            ("string", "ProcessingStage", thinned.get("processing_stage")),
            ("float", "RefreshHz", thinned.get("total_refresh_hz")),
            ("long", "RepresentativeFrame", thinned.get("representative_frame_index")),
            (
                "bool",
                "IncludeRepresentative",
                thinned.get("include_representative_frame"),
            ),
            ("bool", "IncludeSum", thinned.get("include_bucket_sum")),
        )
        burst_fields = (
            ("string", "ProcessingStage", burst.get("processing_stage")),
            ("string", "FilepathTemplate", burst.get("filepath_template")),
            ("string", "DatasetName", burst.get("dataset_name")),
            ("long", "BucketsPerCapture", burst.get("buckets_per_capture")),
            ("long", "CaptureCount", burst.get("capture_count")),
            ("bool", "RearmAfterWrite", burst.get("rearm_after_write")),
            ("bool", "StrictComplete", burst.get("strict_complete")),
        )
        setters = {
            "bool": self.tags.set_bool,
            "string": self.tags.set_string,
            "float": self.tags.set_float,
            "long": self.tags.set_long,
        }
        for value_type, name, value in visualization_fields:
            if value is not None:
                setters[value_type](_field(VISUALIZATION_ROOT, name), value)
        for value_type, name, value in burst_fields:
            if value is not None:
                setters[value_type](_field(BURST_ROOT, name), value)
        for root, threshold in (
            (VISUALIZATION_ROOT, thinned.get("threshold", {})),
            (BURST_ROOT, burst.get("threshold", {})),
        ):
            if "zlp" in threshold:
                self.tags.set_float(_field(root, "ZLPThreshold"), threshold["zlp"])
            if "core_loss" in threshold:
                self.tags.set_float(
                    _field(root, "CoreLossThreshold"), threshold["core_loss"]
                )
        self.state_sequence += 1
        self.tags.set_string(STATE_JSON, json.dumps(state, sort_keys=True))
        for name, value in status.items():
            self.tags.set_string(_field(TAG_ROOT + ":State", name.title()), value)
        self.tags.set_long(STATE_SEQUENCE, self.state_sequence)

    def publish_error(self, message):
        self.state_sequence += 1
        self.tags.set_string(_field(TAG_ROOT + ":State", "Control"), "offline")
        self.tags.set_string(_field(TAG_ROOT + ":State", "Message"), str(message))
        self.tags.set_long(STATE_SEQUENCE, self.state_sequence)

    def set_engine_online(self, online):
        self.tags.set_bool(ENGINE_ONLINE, online)
        self.tags.set_float(ENGINE_HEARTBEAT, time.time())


class ZmqControlClient(object):
    def __init__(self, zmq_module, endpoint, receive_timeout_ms=1500):
        self._zmq = zmq_module
        self.endpoint = endpoint
        self.receive_timeout_ms = int(receive_timeout_ms)

    def request(self, request):
        socket = self._zmq.Context.instance().socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDTIMEO, 1000)
        socket.setsockopt(self._zmq.RCVTIMEO, self.receive_timeout_ms)
        try:
            socket.connect(self.endpoint)
            socket.send_json(request)
            return socket.recv_json()
        finally:
            socket.close()


class DMControlBridge(object):
    """Service palette commands and periodic state polls from the image loop."""

    def __init__(self, mailbox, client, state_poll_seconds=1.0):
        self.mailbox = mailbox
        self.client = client
        self.state_poll_seconds = float(state_poll_seconds)
        self.next_state_poll = 0.0

    def tick(self, now=None):
        now = time.monotonic() if now is None else float(now)
        try:
            pending = self.mailbox.next_request()
        except Exception as error:
            response = {"ok": False, "error": str(error)}
            sequence = self.mailbox.last_request_sequence
            self.mailbox.publish_response(sequence, response)
            self.mailbox.publish_error(response["error"])
            self.next_state_poll = now + self.state_poll_seconds
            return response
        if pending is not None:
            sequence, request = pending
            try:
                response = self.client.request(request)
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            self.mailbox.publish_response(sequence, response)
            if response.get("ok"):
                self.mailbox.publish_state(response)
            else:
                self.mailbox.publish_error(
                    response.get("error", "control rejected")
                )
            self.next_state_poll = now + self.state_poll_seconds
            return response

        if now < self.next_state_poll:
            return None
        try:
            response = self.client.request({"command": "get_state"})
        except Exception as error:
            response = {"ok": False, "error": str(error)}
            self.mailbox.publish_error(response["error"])
            self.next_state_poll = now + max(5.0, self.state_poll_seconds)
        else:
            self.next_state_poll = now + self.state_poll_seconds
            if response.get("ok"):
                self.mailbox.publish_state(response)
            else:
                self.mailbox.publish_error(
                    response.get("error", "control rejected")
                )
        return response


def launch_control_palette(dm_module):
    """Launch the bundled modeless DM-script palette from embedded Python."""

    palette_path = os.path.join(os.path.dirname(__file__), "stem_dm_controls.s")
    with open(palette_path, "r") as stream:
        source = stream.read()
    dm_module.ExecuteScriptString(source)
