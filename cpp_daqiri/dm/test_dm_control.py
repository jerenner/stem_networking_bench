#!/usr/bin/env python3
"""Hardware-free tests for the DigitalMicrograph DAQ control bridge."""

import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(__file__))
from stem_dm_control import (  # noqa: E402
    BURST_ROOT,
    REQUEST_COMMAND,
    REQUEST_SEQUENCE,
    RESPONSE_SEQUENCE,
    STATE_SEQUENCE,
    VISUALIZATION_ROOT,
    DMControlBridge,
    PersistentTagMailbox,
    build_request,
    compact_status,
    initialize_defaults,
    launch_control_palette,
)


class FakeTags(object):
    def __init__(self):
        self.values = {}

    def get_long(self, path, default=0):
        return int(self.values.get(path, default))

    def get_float(self, path, default=0.0):
        return float(self.values.get(path, default))

    def get_bool(self, path, default=False):
        return bool(self.values.get(path, default))

    def get_string(self, path, default=""):
        return str(self.values.get(path, default))

    def set_long(self, path, value):
        self.values[path] = int(value)

    def set_float(self, path, value):
        self.values[path] = float(value)

    def set_bool(self, path, value):
        self.values[path] = bool(value)

    def set_string(self, path, value):
        self.values[path] = str(value)


class FakeClient(object):
    def __init__(self, response):
        self.response = response
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeScriptDM(object):
    def __init__(self):
        self.source = None

    def ExecuteScriptString(self, source):
        self.source = source


def path(root, name):
    return root + ":" + name


def state():
    return {
        "ok": True,
        "message": "ready",
        "supervisor": {"state": "running", "acquisition_running": True},
        "acquisition": {"running": True},
        "thinned_stream": {
            "capability_enabled": True,
            "publishing": True,
            "processing_stage": "dark_blr",
            "total_refresh_hz": 2.5,
            "representative_frame_index": 17,
            "include_representative_frame": True,
            "include_bucket_sum": False,
            "threshold": {"zlp": 12.0, "core_loss": 34.0},
        },
        "burst_writer": {
            "capability_enabled": True,
            "armed": True,
            "busy": False,
            "processing_stage": "corrected",
            "filepath_template": "/data/capture_{capture}.h5",
            "dataset_name": "/frames",
            "buckets_per_capture": 3,
            "capture_count": 8,
            "rearm_after_write": True,
            "strict_complete": True,
            "threshold": {"zlp": 56.0, "core_loss": 78.0},
        },
    }


class RequestBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tags = FakeTags()
        initialize_defaults(self.tags)

    def test_visualization_request(self):
        self.tags.set_bool(path(VISUALIZATION_ROOT, "Publishing"), False)
        self.tags.set_string(path(VISUALIZATION_ROOT, "ProcessingStage"), "dark_blr")
        self.tags.set_float(path(VISUALIZATION_ROOT, "RefreshHz"), 7.5)
        self.tags.set_long(path(VISUALIZATION_ROOT, "RepresentativeFrame"), 31)
        self.tags.set_bool(path(VISUALIZATION_ROOT, "IncludeRepresentative"), True)
        self.tags.set_bool(path(VISUALIZATION_ROOT, "IncludeSum"), False)
        self.tags.set_float(path(VISUALIZATION_ROOT, "ZLPThreshold"), 11.0)
        self.tags.set_float(path(VISUALIZATION_ROOT, "CoreLossThreshold"), 22.0)

        request = build_request("apply_visualization", self.tags)
        self.assertEqual(request["command"], "set_runtime")
        update = request["thinned_stream"]
        self.assertFalse(update["publishing"])
        self.assertEqual(update["processing_stage"], "dark_blr")
        self.assertEqual(update["total_refresh_hz"], 7.5)
        self.assertEqual(update["representative_frame_index"], 31)
        self.assertFalse(update["include_bucket_sum"])
        self.assertEqual(update["threshold"], {"zlp": 11.0, "core_loss": 22.0})

    def test_burst_arm_request(self):
        self.tags.set_string(path(BURST_ROOT, "ProcessingStage"), "thresholded")
        self.tags.set_string(path(BURST_ROOT, "FilepathTemplate"), "/data/test.h5")
        self.tags.set_long(path(BURST_ROOT, "BucketsPerCapture"), 4)
        self.tags.set_long(path(BURST_ROOT, "CaptureCount"), 9)
        self.tags.set_bool(path(BURST_ROOT, "StrictComplete"), True)

        request = build_request("arm_burst", self.tags)
        update = request["burst_writer"]
        self.assertEqual(update["action"], "arm")
        self.assertEqual(update["processing_stage"], "thresholded")
        self.assertEqual(update["filepath_template"], "/data/test.h5")
        self.assertEqual(update["buckets_per_capture"], 4)
        self.assertEqual(update["capture_count"], 9)
        self.assertTrue(update["strict_complete"])

    def test_simple_actions(self):
        self.assertEqual(
            build_request("start_acquisition", self.tags),
            {"command": "start_acquisition"},
        )
        self.assertEqual(
            build_request("disarm_burst", self.tags),
            {
                "command": "set_runtime",
                "burst_writer": {"action": "disarm"},
            },
        )


class MailboxTest(unittest.TestCase):
    def setUp(self):
        self.tags = FakeTags()
        initialize_defaults(self.tags)
        self.mailbox = PersistentTagMailbox(self.tags)

    def queue(self, command):
        self.tags.set_string(REQUEST_COMMAND, command)
        self.tags.set_long(
            REQUEST_SEQUENCE, self.tags.get_long(REQUEST_SEQUENCE, 0) + 1
        )

    def test_request_processed_once_and_response_committed(self):
        self.queue("start_acquisition")
        sequence, request = self.mailbox.next_request()
        self.assertEqual(request, {"command": "start_acquisition"})
        self.assertIsNone(self.mailbox.next_request())
        self.mailbox.publish_response(sequence, {"ok": True})
        self.assertEqual(self.tags.get_long(RESPONSE_SEQUENCE), sequence)

    def test_state_updates_status_and_palette_values(self):
        self.mailbox.publish_state(state())
        self.assertEqual(self.tags.get_long(STATE_SEQUENCE), 1)
        self.assertEqual(self.tags.get_string("STEM DAQ:State:Acquisition"), "running")
        self.assertEqual(self.tags.get_string("STEM DAQ:State:Burst"), "armed / waiting")
        self.assertEqual(
            self.tags.get_string(path(VISUALIZATION_ROOT, "ProcessingStage")),
            "dark_blr",
        )
        self.assertEqual(self.tags.get_long(path(BURST_ROOT, "BucketsPerCapture")), 3)

    def test_compact_status(self):
        status = compact_status(state())
        self.assertEqual(status["control"], "online")
        self.assertEqual(status["visualization"], "dark_blr at 2.5 Hz")
        self.assertEqual(status["burst"], "armed / waiting")


class BridgeTest(unittest.TestCase):
    def test_pending_command_has_priority_over_state_poll(self):
        tags = FakeTags()
        initialize_defaults(tags)
        mailbox = PersistentTagMailbox(tags)
        tags.set_string(REQUEST_COMMAND, "abort_burst")
        tags.set_long(REQUEST_SEQUENCE, 1)
        client = FakeClient(state())
        bridge = DMControlBridge(mailbox, client, state_poll_seconds=1.0)

        bridge.tick(now=10.0)
        self.assertEqual(
            client.requests,
            [{"command": "set_runtime", "burst_writer": {"action": "abort"}}],
        )
        self.assertEqual(tags.get_long(RESPONSE_SEQUENCE), 1)

    def test_bundled_palette_can_be_loaded_for_hybrid_execution(self):
        dm = FakeScriptDM()
        launch_control_palette(dm)
        self.assertIn("class STEMDAQControlPalette : UIFrame", dm.source)
        self.assertIn('QueueCommand("start_acquisition")', dm.source)
        self.assertIn('tabs.DLGAddTab("Status")', dm.source)
        self.assertIn('tabs.DLGAddTab("Visualization")', dm.source)
        self.assertIn('tabs.DLGAddTab("Burst")', dm.source)
        self.assertIn(":Viewer:StopRequested", dm.source)
        self.assertIn("self.Close()", dm.source)

    def test_palette_does_not_split_dm_expressions_across_lines(self):
        dm = FakeScriptDM()
        launch_control_palette(dm)
        for line_number, line in enumerate(dm.source.splitlines(), 1):
            stripped = line.rstrip()
            self.assertFalse(
                stripped.endswith(("(", ",")),
                "DM expression continues after line {}".format(line_number),
            )

    def test_control_failure_is_published(self):
        tags = FakeTags()
        initialize_defaults(tags)
        mailbox = PersistentTagMailbox(tags)
        client = FakeClient(RuntimeError("connection refused"))
        bridge = DMControlBridge(mailbox, client)

        response = bridge.tick(now=1.0)
        self.assertFalse(response["ok"])
        self.assertEqual(tags.get_string("STEM DAQ:State:Control"), "offline")
        self.assertIn("connection refused", tags.get_string("STEM DAQ:State:Message"))

    def test_invalid_palette_command_does_not_escape_image_loop(self):
        tags = FakeTags()
        initialize_defaults(tags)
        mailbox = PersistentTagMailbox(tags)
        tags.set_string(REQUEST_COMMAND, "not_a_command")
        tags.set_long(REQUEST_SEQUENCE, 1)
        bridge = DMControlBridge(mailbox, FakeClient(state()))

        response = bridge.tick(now=1.0)
        self.assertFalse(response["ok"])
        self.assertIn("unknown DM control command", response["error"])
        self.assertEqual(tags.get_long(RESPONSE_SEQUENCE), 1)


if __name__ == "__main__":
    unittest.main()
