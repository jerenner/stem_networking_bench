#!/usr/bin/env python3
"""Live STEM DAQ viewer and ZeroMQ control client."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
import zmq
from PySide6 import QtCore, QtGui, QtWidgets


STAGES = [
    "raw",
    "dark_subtracted",
    "dark_blr",
    "corrected",
    "thresholded",
]


def decode_product(parts: list[bytes]) -> tuple[str, dict, dict[str, np.ndarray]]:
    if len(parts) < 3:
        raise ValueError(f"expected at least 3 message parts, received {len(parts)}")
    topic = parts[0].decode("utf-8")
    metadata = json.loads(parts[1].decode("utf-8"))
    if metadata.get("schema") != "stem.thinned.v1":
        raise ValueError(f"unsupported stream schema {metadata.get('schema')!r}")
    height, width = int(metadata["height"]), int(metadata["width"])
    names = [name for name in metadata["parts"] if name != "metadata"]
    if len(names) != len(parts) - 2:
        raise ValueError("metadata parts do not match multipart payload")
    expected_bytes = height * width * np.dtype("<f4").itemsize
    arrays: dict[str, np.ndarray] = {}
    for name, payload in zip(names, parts[2:], strict=True):
        if len(payload) != expected_bytes:
            raise ValueError(
                f"{name} payload has {len(payload)} bytes; expected {expected_bytes}"
            )
        # The immutable ZeroMQ bytes object remains the ndarray's backing store.
        # Avoid copying each multi-megabyte image before it reaches the viewer.
        arrays[name] = np.frombuffer(payload, dtype="<f4").reshape(height, width)
    return topic, metadata, arrays


class StreamWorker(QtCore.QObject):
    product = QtCore.Signal(str, object, object)
    status = QtCore.Signal(str, bool)

    def __init__(self, endpoint: str, topic: str) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.topic = topic
        self._timer: QtCore.QTimer | None = None
        self._socket: zmq.Socket | None = None

    @QtCore.Slot()
    def start(self) -> None:
        try:
            self._socket = zmq.Context.instance().socket(zmq.SUB)
            self._socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
            self._socket.setsockopt(zmq.RCVHWM, 2)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self.endpoint)
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.poll)
            self._timer.start(20)
            # ZeroMQ connect is asynchronous; receipt of a product establishes
            # that the data endpoint is actually online.
            self.status.emit(f"Waiting for data on {self.endpoint}", False)
        except Exception as error:  # pragma: no cover - environment dependent
            self.status.emit(str(error), False)

    @QtCore.Slot()
    def poll(self) -> None:
        if self._socket is None:
            return
        try:
            newest = None
            while True:
                try:
                    newest = self._socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
            if newest is not None:
                topic, metadata, arrays = decode_product(newest)
                self.product.emit(topic, metadata, arrays)
        except Exception as error:
            self.status.emit(f"Stream error: {error}", False)

    @QtCore.Slot()
    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class ControlWorker(QtCore.QObject):
    response = QtCore.Signal(str, object)
    status = QtCore.Signal(str, bool)

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.requests: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._timer: QtCore.QTimer | None = None

    @QtCore.Slot()
    def start(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.process_one)
        self._timer.start(30)

    @QtCore.Slot(str, object)
    def enqueue(self, tag: str, request: dict) -> None:
        self.requests.put((tag, request))

    @QtCore.Slot()
    def process_one(self) -> None:
        try:
            tag, request = self.requests.get_nowait()
        except queue.Empty:
            return
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1500)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        try:
            socket.connect(self.endpoint)
            socket.send_json(request)
            response = socket.recv_json()
            self.status.emit(f"Control connected: {self.endpoint}", True)
        except Exception as error:
            response = {"ok": False, "error": str(error)}
            self.status.emit(f"Control unavailable: {error}", False)
        finally:
            socket.close()
        self.response.emit(tag, response)

    @QtCore.Slot()
    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()


class Badge(QtWidgets.QLabel):
    def set_state(self, text: str, good: bool) -> None:
        self.setText(text)
        color = "#17735b" if good else "#a13c2f"
        background = "#dff4ec" if good else "#fae5df"
        self.setStyleSheet(
            f"color:{color}; background:{background}; border-radius:9px;"
            "padding:4px 9px; font-weight:600;"
        )


class DetectorImageView(pg.PlotWidget):
    """Lean detector image canvas without ImageView histogram/LUT recomputation."""

    def __init__(self) -> None:
        super().__init__()
        self.image_item = pg.ImageItem()
        self.addItem(self.image_item)
        self.setLabel("bottom", "Detector column")
        self.setLabel("left", "Detector row")
        self.showGrid(x=True, y=True, alpha=0.12)
        self.getViewBox().invertY(True)

    def set_image(
        self,
        image: np.ndarray,
        levels: tuple[float, float] | None,
        auto_range: bool,
    ) -> None:
        self.image_item.setImage(image, autoLevels=False, levels=levels)
        if auto_range:
            self.getViewBox().autoRange()


def combo(items: list[str]) -> QtWidgets.QComboBox:
    widget = QtWidgets.QComboBox()
    widget.addItems(items)
    return widget


def spin(minimum: int, maximum: int, value: int = 0) -> QtWidgets.QSpinBox:
    widget = QtWidgets.QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def double_spin(
    minimum: float, maximum: float, value: float = 0.0, decimals: int = 3
) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def add_row(layout: QtWidgets.QFormLayout, label: str, widget) -> Any:
    layout.addRow(label, widget)
    return widget


@dataclass
class Product:
    topic: str
    metadata: dict
    arrays: dict[str, np.ndarray]
    received_at: float


class MainWindow(QtWidgets.QMainWindow):
    control_request = QtCore.Signal(str, object)

    def __init__(
        self,
        stream_endpoint: str,
        control_endpoint: str,
        topic: str,
        max_render_hz: float,
    ):
        super().__init__()
        self.setWindowTitle("STEM DAQ Console")
        self.resize(1540, 960)
        self.products: dict[int, Product] = {}
        self.dirty_receivers: set[int] = set()
        self.rendered_views: set[tuple[int, int]] = set()
        self.initialized_from_state = False
        self.restart_after_stage = False
        self.control_poll_pending = False
        self.next_control_poll_at = 0.0
        self._build_ui(stream_endpoint, control_endpoint)
        self._start_workers(stream_endpoint, control_endpoint, topic)
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.timeout.connect(self.render_latest)
        self.render_timer.start(max(20, round(1000.0 / max_render_hz)))
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_control)
        self.poll_timer.start(1000)
        self.poll_control(initial=True)

    def _build_ui(self, stream_endpoint: str, control_endpoint: str) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("STEM DAQ CONSOLE")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Live detector products, controlled bursts, and acquisition lifecycle"
        )
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.stream_badge = Badge("Stream connecting")
        self.control_badge = Badge("Control connecting")
        header.addWidget(self.stream_badge)
        header.addWidget(self.control_badge)
        root.addLayout(header)

        endpoints = QtWidgets.QLabel(
            f"DATA  {stream_endpoint}     CONTROL  {control_endpoint}"
        )
        endpoints.setObjectName("endpoint")
        root.addWidget(endpoints)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_viewer())
        splitter.addWidget(self._build_controls())
        splitter.setSizes([940, 560])
        root.addWidget(splitter, 1)
        self.statusBar().showMessage("Waiting for DAQ state")

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#f4f1e9; color:#172321;
              font-family:'Avenir Next','IBM Plex Sans','Helvetica Neue'; }
            QLabel#title { font-size:25px; font-weight:750; letter-spacing:2px; }
            QLabel#subtitle { color:#52605d; font-size:13px; }
            QLabel#endpoint { color:#5d6866; background:#e8e3d7;
              border-radius:7px; padding:7px 11px; font-family:Menlo; }
            QGroupBox { border:1px solid #cbc5b8; border-radius:8px;
              margin-top:12px; padding:12px 8px 8px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px;
              padding:0 5px; color:#1b6658; }
            QTabWidget::pane { border:1px solid #cbc5b8; border-radius:8px;
              background:#fbfaf5; }
            QTabBar::tab { background:#ddd7ca; padding:8px 13px; margin-right:2px; }
            QTabBar::tab:selected { background:#1b6658; color:white; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
              background:#fffefa; border:1px solid #bbb4a7; border-radius:5px;
              padding:5px; selection-background-color:#e49c54; }
            QPushButton { background:#1b6658; color:white; border:0;
              border-radius:6px; padding:7px 12px; font-weight:650; }
            QPushButton:hover { background:#237d6b; }
            QPushButton#danger { background:#a13c2f; }
            QPushButton#secondary { background:#6a716e; }
            QScrollArea { border:0; }
            """
        )

    def _build_viewer(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        tools = QtWidgets.QHBoxLayout()
        tools.addWidget(QtWidgets.QLabel("Receiver"))
        self.receiver = combo(["0", "1"])
        self.receiver.currentIndexChanged.connect(self.refresh_view)
        tools.addWidget(self.receiver)
        self.product_label = QtWidgets.QLabel("No products received")
        self.product_label.setObjectName("subtitle")
        tools.addWidget(self.product_label, 1)
        self.auto_levels = QtWidgets.QCheckBox("Auto levels")
        self.auto_levels.setChecked(True)
        tools.addWidget(self.auto_levels)
        layout.addLayout(tools)

        self.viewer_tabs = QtWidgets.QTabWidget()
        self.representative_view = DetectorImageView()
        self.sum_view = DetectorImageView()
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setLabel("bottom", "Detector column")
        self.profile_plot.setLabel("left", "Mean signal")
        self.profile_plot.showGrid(x=True, y=True, alpha=0.2)
        self.viewer_tabs.addTab(self.representative_view, "Single frame")
        self.viewer_tabs.addTab(self.sum_view, "128-frame sum")
        self.viewer_tabs.addTab(self.profile_plot, "Column profile")
        self.viewer_tabs.currentChanged.connect(self.refresh_view)
        layout.addWidget(self.viewer_tabs, 1)
        return panel

    def _build_controls(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._runtime_tab(), "Live controls")
        tabs.addTab(self._restart_tab(), "Apply on restart")
        status = QtWidgets.QWidget()
        self.status_text = QtWidgets.QPlainTextEdit()
        self.status_text.setReadOnly(True)
        status_layout = QtWidgets.QVBoxLayout(status)
        status_layout.addWidget(self.status_text)
        tabs.addTab(status, "DAQ state")
        return tabs

    def _runtime_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        thinned = QtWidgets.QGroupBox("Thinned visualization stream")
        form = QtWidgets.QFormLayout(thinned)
        self.thin_publish = add_row(form, "Publishing", QtWidgets.QCheckBox())
        self.thin_stage = add_row(form, "Processing stage", combo(STAGES))
        self.thin_rate = add_row(
            form, "Total refresh rate (Hz)", double_spin(0, 1000, 10)
        )
        self.thin_rep = add_row(form, "Representative frame", spin(0, 4095, 64))
        self.thin_include_rep = add_row(
            form, "Send single frame", QtWidgets.QCheckBox()
        )
        self.thin_include_sum = add_row(
            form, "Send bucket sum", QtWidgets.QCheckBox()
        )
        self.thin_topic = add_row(form, "Topic prefix", QtWidgets.QLineEdit("stem"))
        threshold_row = QtWidgets.QHBoxLayout()
        self.thin_zlp_threshold = double_spin(0, 1e9)
        self.thin_core_threshold = double_spin(0, 1e9)
        threshold_row.addWidget(QtWidgets.QLabel("ZLP"))
        threshold_row.addWidget(self.thin_zlp_threshold)
        threshold_row.addWidget(QtWidgets.QLabel("CoreLoss"))
        threshold_row.addWidget(self.thin_core_threshold)
        form.addRow("Thresholds", threshold_row)
        button = QtWidgets.QPushButton("Apply visualization settings")
        button.clicked.connect(self.apply_thinned)
        form.addRow(button)
        layout.addWidget(thinned)

        burst = QtWidgets.QGroupBox("Controlled burst capture")
        form = QtWidgets.QFormLayout(burst)
        self.burst_state = Badge("Not armed")
        form.addRow("State", self.burst_state)
        self.burst_stage = add_row(form, "Processing stage", combo(STAGES))
        self.burst_path = add_row(
            form,
            "File template",
            QtWidgets.QLineEdit("/data/stem_burst_rx{receiver}_{capture}_{stage}.h5"),
        )
        self.burst_dataset = add_row(
            form, "Dataset", QtWidgets.QLineEdit("/frames")
        )
        self.burst_buckets = add_row(form, "Buckets / capture", spin(1, 4096, 1))
        self.burst_count = add_row(
            form, "Captures / arm (0 unlimited)", spin(0, 1_000_000, 1)
        )
        self.burst_rearm = add_row(form, "Re-arm after write", QtWidgets.QCheckBox())
        self.burst_strict = add_row(
            form, "Require complete buckets", QtWidgets.QCheckBox()
        )
        threshold_row = QtWidgets.QHBoxLayout()
        self.burst_zlp_threshold = double_spin(0, 1e9)
        self.burst_core_threshold = double_spin(0, 1e9)
        threshold_row.addWidget(QtWidgets.QLabel("ZLP"))
        threshold_row.addWidget(self.burst_zlp_threshold)
        threshold_row.addWidget(QtWidgets.QLabel("CoreLoss"))
        threshold_row.addWidget(self.burst_core_threshold)
        form.addRow("Thresholds", threshold_row)
        self.burst_configure = QtWidgets.QPushButton("Apply capture settings")
        self.burst_configure.setToolTip(
            "Update burst settings without starting a capture; accepted only while idle."
        )
        self.burst_configure.clicked.connect(lambda: self.apply_burst(""))
        self.burst_arm = QtWidgets.QPushButton("Apply settings and arm")
        self.burst_arm.setToolTip(
            "Update the settings, then capture the next eligible bucket sequence."
        )
        self.burst_arm.clicked.connect(lambda: self.apply_burst("arm"))
        self.burst_disarm = QtWidgets.QPushButton("Disarm after current burst")
        self.burst_disarm.setObjectName("secondary")
        self.burst_disarm.clicked.connect(lambda: self.burst_action("disarm"))
        self.burst_abort = QtWidgets.QPushButton("Abort current burst")
        self.burst_abort.setObjectName("danger")
        self.burst_abort.clicked.connect(lambda: self.burst_action("abort"))
        self.burst_live_buttons = [
            self.burst_configure,
            self.burst_arm,
            self.burst_disarm,
            self.burst_abort,
        ]
        buttons = QtWidgets.QGridLayout()
        buttons.addWidget(self.burst_configure, 0, 0)
        buttons.addWidget(self.burst_arm, 0, 1)
        buttons.addWidget(self.burst_disarm, 1, 0)
        buttons.addWidget(self.burst_abort, 1, 1)
        form.addRow(buttons)
        layout.addWidget(burst)
        layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _restart_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        capabilities = QtWidgets.QGroupBox("Output allocations and endpoints")
        form = QtWidgets.QFormLayout(capabilities)
        self.restart_burst_enabled = add_row(
            form, "Allocate burst writer", QtWidgets.QCheckBox()
        )
        self.restart_burst_capacity = add_row(
            form, "Maximum burst buckets", spin(1, 4096, 1)
        )
        self.restart_thin_enabled = add_row(
            form, "Allocate thinned stream", QtWidgets.QCheckBox()
        )
        self.restart_thin_endpoint = add_row(
            form, "PUB bind endpoint", QtWidgets.QLineEdit("tcp://*:5556")
        )
        self.restart_thin_depth = add_row(form, "PUB queue depth", spin(1, 128, 2))
        self.restart_control_endpoint = add_row(
            form, "Control bind endpoint", QtWidgets.QLineEdit("tcp://*:5557")
        )
        layout.addWidget(capabilities)

        processor = QtWidgets.QGroupBox("GPU processor")
        form = QtWidgets.QFormLayout(processor)
        self.restart_noop = add_row(form, "No reduction", QtWidgets.QCheckBox())
        self.restart_dark = add_row(
            form, "Subtract dark frame", QtWidgets.QCheckBox()
        )
        self.restart_dark_path = add_row(
            form, "Dark HDF5 path", QtWidgets.QLineEdit("/calibration/dark.h5")
        )
        self.restart_dark_dataset = add_row(
            form, "Dark dataset", QtWidgets.QLineEdit("/processed")
        )
        self.restart_valid = add_row(
            form, "Apply valid-pixel mask", QtWidgets.QCheckBox()
        )
        self.restart_valid_dataset = add_row(
            form, "Mask dataset", QtWidgets.QLineEdit("/valid_pixel_mask")
        )
        self.restart_blr = add_row(form, "Apply BLR", QtWidgets.QCheckBox())
        blr_row = QtWidgets.QHBoxLayout()
        self.restart_blr_rows = spin(1, 512, 30)
        self.restart_blr_zlp = spin(1, 3840, 768)
        self.restart_blr_zgroup = spin(1, 3840, 4)
        self.restart_blr_cgroup = spin(1, 3840, 16)
        for label, widget in [
            ("Rows", self.restart_blr_rows),
            ("ZLP width", self.restart_blr_zlp),
            ("ZLP group", self.restart_blr_zgroup),
            ("Core group", self.restart_blr_cgroup),
        ]:
            blr_row.addWidget(QtWidgets.QLabel(label))
            blr_row.addWidget(widget)
        form.addRow("BLR geometry", blr_row)
        self.restart_dynamic = add_row(
            form, "Dynamic half-column mask", QtWidgets.QCheckBox()
        )
        dynamic_row = QtWidgets.QHBoxLayout()
        self.restart_dynamic_window = spin(1, 129, 31)
        self.restart_dynamic_ratio = double_spin(0, 1e6, 1)
        self.restart_dynamic_offset = double_spin(0, 1e9, 500)
        self.restart_dynamic_edge = spin(0, 511, 32)
        for label, widget in [
            ("Window", self.restart_dynamic_window),
            ("Ratio", self.restart_dynamic_ratio),
            ("Offset", self.restart_dynamic_offset),
            ("Edge rows", self.restart_dynamic_edge),
        ]:
            dynamic_row.addWidget(QtWidgets.QLabel(label))
            dynamic_row.addWidget(widget)
        form.addRow("Dynamic-mask parameters", dynamic_row)
        self.restart_two_sided = add_row(
            form, "Two-sided outliers", QtWidgets.QCheckBox()
        )
        layout.addWidget(processor)

        acquisition = QtWidgets.QGroupBox("Acquisition geometry and continuous writer")
        form = QtWidgets.QFormLayout(acquisition)
        receiver_row = QtWidgets.QHBoxLayout()
        self.restart_receivers = spin(1, 8, 2)
        self.restart_frames = spin(1, 4096, 128)
        self.restart_source_mask = QtWidgets.QLineEdit("0x0f")
        receiver_row.addWidget(QtWidgets.QLabel("Receivers"))
        receiver_row.addWidget(self.restart_receivers)
        receiver_row.addWidget(QtWidgets.QLabel("Frames/bucket"))
        receiver_row.addWidget(self.restart_frames)
        receiver_row.addWidget(QtWidgets.QLabel("Source mask"))
        receiver_row.addWidget(self.restart_source_mask)
        form.addRow("Stream assembly", receiver_row)
        packet_row = QtWidgets.QHBoxLayout()
        self.restart_header = spin(0, 65535, 42)
        self.restart_payload = spin(0, 65535, 7680)
        self.restart_slack = spin(0, 10_000_000, 512)
        packet_row.addWidget(QtWidgets.QLabel("Header"))
        packet_row.addWidget(self.restart_header)
        packet_row.addWidget(QtWidgets.QLabel("Payload"))
        packet_row.addWidget(self.restart_payload)
        packet_row.addWidget(QtWidgets.QLabel("Close slack"))
        packet_row.addWidget(self.restart_slack)
        form.addRow("Packet geometry (bytes)", packet_row)
        flags = QtWidgets.QHBoxLayout()
        self.restart_gpu_headers = QtWidgets.QCheckBox("GPU header extraction")
        self.restart_hds = QtWidgets.QCheckBox("Header/data split")
        self.restart_duplicate = QtWidgets.QCheckBox("Duplicate tile prefix")
        self.restart_latency = QtWidgets.QCheckBox("Capture latency")
        for widget in [
            self.restart_gpu_headers,
            self.restart_hds,
            self.restart_duplicate,
            self.restart_latency,
        ]:
            flags.addWidget(widget)
        form.addRow("Receiver flags", flags)
        self.restart_writer = add_row(
            form, "Continuous HDF5 writer", QtWidgets.QCheckBox()
        )
        self.restart_writer_path = add_row(
            form, "Continuous file", QtWidgets.QLineEdit("/data/stem_continuous.h5")
        )
        self.restart_writer_dataset = add_row(
            form, "Continuous dataset", QtWidgets.QLineEdit("/processed")
        )
        self.restart_writer_concurrent = add_row(
            form, "Writer buffers", spin(1, 64, 3)
        )
        layout.addWidget(acquisition)

        advanced = QtWidgets.QGroupBox("Advanced restart override")
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        advanced_layout.addWidget(
            QtWidgets.QLabel(
                "Optional JSON object merged into the YAML configuration. "
                "Use this for DAQIRI NIC, queue, memory-region, or flow changes."
            )
        )
        self.restart_override = QtWidgets.QPlainTextEdit()
        self.restart_override.setPlaceholderText(
            '{"daqiri":{"cfg":{"rx_meta_buffers":2048}}}'
        )
        self.restart_override.setMaximumHeight(95)
        advanced_layout.addWidget(self.restart_override)
        layout.addWidget(advanced)

        actions = QtWidgets.QHBoxLayout()
        stage = QtWidgets.QPushButton("Stage restart settings")
        stage.clicked.connect(lambda: self.stage_restart(False))
        discard = QtWidgets.QPushButton("Discard staged changes")
        discard.setObjectName("secondary")
        discard.clicked.connect(
            lambda: self.send_control("discard", {"command": "discard_restart"})
        )
        restart = QtWidgets.QPushButton("Restart acquisition")
        restart.setObjectName("danger")
        restart.clicked.connect(lambda: self.stage_restart(True))
        actions.addWidget(stage)
        actions.addWidget(discard)
        actions.addWidget(restart)
        layout.addLayout(actions)
        layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _start_workers(
        self, stream_endpoint: str, control_endpoint: str, topic: str
    ) -> None:
        self.stream_thread = QtCore.QThread(self)
        self.stream_worker = StreamWorker(stream_endpoint, topic)
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_thread.started.connect(self.stream_worker.start)
        self.stream_worker.product.connect(self.on_product)
        self.stream_worker.status.connect(self.on_stream_status)
        self.stream_thread.start()

        self.control_thread = QtCore.QThread(self)
        self.control_worker = ControlWorker(control_endpoint)
        self.control_worker.moveToThread(self.control_thread)
        self.control_thread.started.connect(self.control_worker.start)
        self.control_request.connect(self.control_worker.enqueue)
        self.control_worker.response.connect(self.on_control_response)
        self.control_worker.status.connect(self.on_control_status)
        self.control_thread.start()

    def send_control(self, tag: str, request: dict) -> None:
        self.control_request.emit(tag, request)

    def poll_control(self, initial: bool = False) -> None:
        if self.control_poll_pending or time.monotonic() < self.next_control_poll_at:
            return
        self.control_poll_pending = True
        self.send_control("initial" if initial else "poll", {"command": "get_state"})

    @QtCore.Slot(str, bool)
    def on_stream_status(self, text: str, good: bool) -> None:
        self.stream_badge.set_state("DATA ONLINE" if good else "DATA OFFLINE", good)
        self.statusBar().showMessage(text, 5000)

    @QtCore.Slot(str, bool)
    def on_control_status(self, text: str, good: bool) -> None:
        self.control_badge.set_state(
            "CONTROL ONLINE" if good else "CONTROL OFFLINE", good
        )
        if not good:
            self.statusBar().showMessage(text, 5000)

    @QtCore.Slot(str, object, object)
    def on_product(self, topic: str, metadata: dict, arrays: dict) -> None:
        receiver = int(metadata["receiver_id"])
        self.products[receiver] = Product(topic, metadata, arrays, time.time())
        self.dirty_receivers.add(receiver)
        self.stream_badge.set_state("DATA ONLINE", True)
        if self.receiver.findText(str(receiver)) < 0:
            self.receiver.addItem(str(receiver))

    def render_latest(self) -> None:
        if not self.receiver.currentText():
            return
        receiver = int(self.receiver.currentText())
        if receiver not in self.dirty_receivers:
            return
        self.dirty_receivers.discard(receiver)
        self.refresh_view()

    @staticmethod
    def display_levels(image: np.ndarray) -> tuple[float, float] | None:
        sample = image[::8, ::8]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return None
        low, high = np.percentile(finite, (1.0, 99.5))
        if low == high:
            high = low + 1.0
        return float(low), float(high)

    def refresh_view(self) -> None:
        if not self.receiver.currentText():
            return
        product = self.products.get(int(self.receiver.currentText()))
        if product is None:
            return
        metadata, arrays = product.metadata, product.arrays
        complete = "complete" if metadata["complete"] else "INCOMPLETE"
        self.product_label.setText(
            f"{metadata['processing_stage']}  |  batch {metadata['batch_index']}  |  "
            f"first frame {metadata['first_frame']}  |  {complete}"
        )
        tab = self.viewer_tabs.currentIndex()
        receiver = int(self.receiver.currentText())
        view_key = (receiver, tab)
        auto_range = view_key not in self.rendered_views
        if tab == 0 and "representative" in arrays:
            levels = (
                self.display_levels(arrays["representative"])
                if self.auto_levels.isChecked()
                else None
            )
            self.representative_view.set_image(
                arrays["representative"],
                levels=levels,
                auto_range=auto_range,
            )
        elif tab == 1 and "sum" in arrays:
            levels = (
                self.display_levels(arrays["sum"])
                if self.auto_levels.isChecked()
                else None
            )
            self.sum_view.set_image(
                arrays["sum"], levels=levels, auto_range=auto_range
            )
        elif tab == 2:
            profile_source = arrays.get("sum", arrays.get("representative"))
            if profile_source is None:
                return
            profile = np.nanmean(profile_source, axis=0)
            self.profile_plot.clear()
            self.profile_plot.plot(
                profile,
                pen=pg.mkPen("#e1763f", width=1.5),
            )
        self.rendered_views.add(view_key)

    def apply_thinned(self) -> None:
        request = {
            "command": "set_runtime",
            "thinned_stream": {
                "publishing": self.thin_publish.isChecked(),
                "processing_stage": self.thin_stage.currentText(),
                "topic_prefix": self.thin_topic.text(),
                "total_refresh_hz": self.thin_rate.value(),
                "representative_frame_index": self.thin_rep.value(),
                "include_representative_frame": self.thin_include_rep.isChecked(),
                "include_bucket_sum": self.thin_include_sum.isChecked(),
                "threshold": {
                    "zlp": self.thin_zlp_threshold.value(),
                    "core_loss": self.thin_core_threshold.value(),
                },
            },
        }
        self.send_control("apply_thinned", request)

    def burst_payload(self, action: str) -> dict:
        return {
            "command": "set_runtime",
            "burst_writer": {
                "action": action,
                "processing_stage": self.burst_stage.currentText(),
                "filepath_template": self.burst_path.text(),
                "dataset_name": self.burst_dataset.text(),
                "buckets_per_capture": self.burst_buckets.value(),
                "capture_count": self.burst_count.value(),
                "rearm_after_write": self.burst_rearm.isChecked(),
                "strict_complete": self.burst_strict.isChecked(),
                "threshold": {
                    "zlp": self.burst_zlp_threshold.value(),
                    "core_loss": self.burst_core_threshold.value(),
                },
            },
        }

    def apply_burst(self, action: str) -> None:
        self.send_control("apply_burst", self.burst_payload(action))

    def burst_action(self, action: str) -> None:
        self.send_control(
            "burst_action",
            {"command": "set_runtime", "burst_writer": {"action": action}},
        )

    def restart_updates(self) -> dict:
        try:
            source_mask = int(self.restart_source_mask.text(), 0)
        except ValueError as error:
            raise ValueError("Source mask must be an integer such as 0x0f") from error
        updates: dict[str, Any] = {
            "num_receivers": self.restart_receivers.value(),
            "burst_writer": {
                "enabled": self.restart_burst_enabled.isChecked(),
                "processing_stage": self.burst_stage.currentText(),
                "filepath_template": self.burst_path.text(),
                "dataset_name": self.burst_dataset.text(),
                "buckets_per_capture": self.restart_burst_capacity.value(),
                "capture_count": self.burst_count.value(),
                "rearm_after_write": self.burst_rearm.isChecked(),
                "strict_complete": self.burst_strict.isChecked(),
                "threshold": {
                    "zlp": self.burst_zlp_threshold.value(),
                    "core_loss": self.burst_core_threshold.value(),
                },
            },
            "thinned_stream": {
                "enabled": self.restart_thin_enabled.isChecked(),
                "start_publishing": self.thin_publish.isChecked(),
                "endpoint": self.restart_thin_endpoint.text(),
                "queue_depth": self.restart_thin_depth.value(),
                "processing_stage": self.thin_stage.currentText(),
                "topic_prefix": self.thin_topic.text(),
                "total_refresh_hz": self.thin_rate.value(),
                "representative_frame_index": self.thin_rep.value(),
                "include_representative_frame": self.thin_include_rep.isChecked(),
                "include_bucket_sum": self.thin_include_sum.isChecked(),
                "threshold": {
                    "zlp": self.thin_zlp_threshold.value(),
                    "core_loss": self.thin_core_threshold.value(),
                },
            },
            "control": {
                "enabled": True,
                "endpoint": self.restart_control_endpoint.text(),
            },
            "processor": {
                "noop": self.restart_noop.isChecked(),
                "subtract_dark_frame": self.restart_dark.isChecked(),
                "dark_frame_path": self.restart_dark_path.text(),
                "dark_frame_dataset": self.restart_dark_dataset.text(),
                "apply_valid_pixel_mask": self.restart_valid.isChecked(),
                "valid_pixel_mask_dataset": self.restart_valid_dataset.text(),
                "apply_blr_correction": self.restart_blr.isChecked(),
                "blr_rows": self.restart_blr_rows.value(),
                "blr_zlp_width": self.restart_blr_zlp.value(),
                "blr_zlp_group_columns": self.restart_blr_zgroup.value(),
                "blr_core_group_columns": self.restart_blr_cgroup.value(),
                "apply_dynamic_half_column_mask": self.restart_dynamic.isChecked(),
                "dynamic_mask_median_window_pixels": self.restart_dynamic_window.value(),
                "dynamic_mask_threshold_ratio": self.restart_dynamic_ratio.value(),
                "dynamic_mask_threshold_offset": self.restart_dynamic_offset.value(),
                "dynamic_mask_excluded_edge_rows": self.restart_dynamic_edge.value(),
                "dynamic_mask_two_sided": self.restart_two_sided.isChecked(),
            },
            "stem_rx": {
                "frames_per_tensor": self.restart_frames.value(),
                "header_size": self.restart_header.value(),
                "payload_size": self.restart_payload.value(),
                "expected_source_mask": source_mask,
                "batch_close_slack_packets": self.restart_slack.value(),
                "gpu_header_extract": self.restart_gpu_headers.isChecked(),
                "hds": self.restart_hds.isChecked(),
                "tile_duplicate_prefix_to_simulate_payload": (
                    self.restart_duplicate.isChecked()
                ),
                "capture_latency": self.restart_latency.isChecked(),
            },
            "writer": {
                "noop": not self.restart_writer.isChecked(),
                "filepath": self.restart_writer_path.text(),
                "dataset_name": self.restart_writer_dataset.text(),
                "num_concurrent": self.restart_writer_concurrent.value(),
            },
        }
        extra = self.restart_override.toPlainText().strip()
        if extra:
            advanced = json.loads(extra)
            if not isinstance(advanced, dict):
                raise ValueError("Advanced override must be a JSON object")
            updates = self.deep_merge(updates, advanced)
        return updates

    @staticmethod
    def deep_merge(target: dict, update: dict) -> dict:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                MainWindow.deep_merge(target[key], value)
            else:
                target[key] = value
        return target

    def stage_restart(self, restart: bool) -> None:
        try:
            updates = self.restart_updates()
        except (ValueError, json.JSONDecodeError) as error:
            QtWidgets.QMessageBox.warning(self, "Invalid restart settings", str(error))
            return
        self.restart_after_stage = restart
        self.send_control(
            "stage_and_restart" if restart else "stage",
            {"command": "stage_restart", "updates": updates},
        )

    @QtCore.Slot(str, object)
    def on_control_response(self, tag: str, response: dict) -> None:
        if tag in {"initial", "poll"}:
            self.control_poll_pending = False
        if not response.get("ok"):
            if tag in {"initial", "poll"}:
                # Avoid opening a fresh SSH forwarding channel every second
                # while the DAQ control endpoint is not listening.
                self.next_control_poll_at = time.monotonic() + 5.0
            if tag != "poll":
                QtWidgets.QMessageBox.warning(
                    self, "DAQ control rejected", response.get("error", "Unknown error")
                )
            return
        if tag in {"initial", "poll"}:
            self.next_control_poll_at = time.monotonic() + 1.0
        self.update_state(response, populate=not self.initialized_from_state or tag != "poll")
        if tag == "stage_and_restart" and self.restart_after_stage:
            self.restart_after_stage = False
            answer = QtWidgets.QMessageBox.question(
                self,
                "Restart acquisition",
                "The current acquisition will stop cleanly and relaunch with "
                "the staged settings. Continue?",
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self.send_control("restart", {"command": "restart"})
        if tag == "restart":
            self.statusBar().showMessage(
                "Restart accepted; reconnecting to the control endpoint", 8000
            )

    def update_state(self, state: dict, populate: bool) -> None:
        burst, thinned = state["burst_writer"], state["thinned_stream"]
        acquisition = state["acquisition"]
        burst_available = bool(burst["capability_enabled"])
        if not burst_available:
            burst_label, burst_good = "NOT ALLOCATED", False
            self.burst_state.setToolTip(
                "Enable 'Allocate burst writer' under Apply on restart, then restart acquisition."
            )
        elif burst["busy"]:
            burst_label, burst_good = "CAPTURING / WRITING", False
            self.burst_state.setToolTip("A burst capture or asynchronous file write is active.")
        elif burst["armed"]:
            burst_label, burst_good = "ARMED / WAITING", True
            self.burst_state.setToolTip("Waiting for the next eligible bucket sequence.")
        else:
            burst_label, burst_good = "IDLE", True
            self.burst_state.setToolTip("Burst capability is allocated but not armed.")
        self.burst_state.set_state(burst_label, burst_good)
        for button in self.burst_live_buttons:
            button.setEnabled(burst_available)
            if not burst_available:
                button.setToolTip(
                    "Burst buffers are not allocated. Enable them under Apply on restart."
                )
        summary = {
            "acquisition": acquisition,
            "burst_writer": {
                key: burst[key]
                for key in ["capability_enabled", "armed", "busy", "capacity_buckets", "stats"]
            },
            "thinned_stream": {
                key: thinned[key]
                for key in ["capability_enabled", "publishing", "endpoint", "stats"]
            },
        }
        self.status_text.setPlainText(json.dumps(summary, indent=2))
        if not populate:
            return
        self.initialized_from_state = True
        self._set_combo(self.thin_stage, thinned["processing_stage"])
        self.thin_publish.setChecked(thinned["publishing"])
        self.thin_rate.setValue(thinned["total_refresh_hz"])
        self.thin_rep.setValue(thinned["representative_frame_index"])
        self.thin_include_rep.setChecked(thinned["include_representative_frame"])
        self.thin_include_sum.setChecked(thinned["include_bucket_sum"])
        self.thin_topic.setText(thinned["topic_prefix"])
        self.thin_zlp_threshold.setValue(thinned["threshold"]["zlp"])
        self.thin_core_threshold.setValue(thinned["threshold"]["core_loss"])
        self._set_combo(self.burst_stage, burst["processing_stage"])
        self.burst_path.setText(burst["filepath_template"])
        self.burst_dataset.setText(burst["dataset_name"])
        self.burst_buckets.setMaximum(max(1, burst["capacity_buckets"]))
        self.burst_buckets.setValue(burst["buckets_per_capture"])
        self.burst_count.setValue(burst["capture_count"])
        self.burst_rearm.setChecked(burst["rearm_after_write"])
        self.burst_strict.setChecked(burst["strict_complete"])
        self.burst_zlp_threshold.setValue(burst["threshold"]["zlp"])
        self.burst_core_threshold.setValue(burst["threshold"]["core_loss"])
        self.populate_restart(state["pending_config"])

    @staticmethod
    def _set_combo(widget: QtWidgets.QComboBox, value: str) -> None:
        index = widget.findText(value)
        if index >= 0:
            widget.setCurrentIndex(index)

    def populate_restart(self, root: dict) -> None:
        burst = root.get("burst_writer", {})
        thin = root.get("thinned_stream", {})
        processor = root.get("processor", {})
        receiver = root.get("stem_rx", {})
        writer = root.get("writer", {})
        control = root.get("control", {})
        self.restart_burst_enabled.setChecked(bool(burst.get("enabled", False)))
        self.restart_burst_capacity.setValue(int(burst.get("buckets_per_capture", 1)))
        self.restart_thin_enabled.setChecked(bool(thin.get("enabled", False)))
        self.restart_thin_endpoint.setText(thin.get("endpoint", "tcp://*:5556"))
        self.restart_thin_depth.setValue(int(thin.get("queue_depth", 2)))
        self.restart_control_endpoint.setText(
            control.get("endpoint", "tcp://*:5557")
        )
        self.restart_noop.setChecked(bool(processor.get("noop", True)))
        self.restart_dark.setChecked(bool(processor.get("subtract_dark_frame", False)))
        self.restart_dark_path.setText(processor.get("dark_frame_path", ""))
        self.restart_dark_dataset.setText(
            processor.get("dark_frame_dataset", "/processed")
        )
        self.restart_valid.setChecked(
            bool(processor.get("apply_valid_pixel_mask", False))
        )
        self.restart_valid_dataset.setText(
            processor.get("valid_pixel_mask_dataset", "/valid_pixel_mask")
        )
        self.restart_blr.setChecked(bool(processor.get("apply_blr_correction", False)))
        self.restart_blr_rows.setValue(int(processor.get("blr_rows", 30)))
        self.restart_blr_zlp.setValue(int(processor.get("blr_zlp_width", 768)))
        self.restart_blr_zgroup.setValue(
            int(processor.get("blr_zlp_group_columns", 4))
        )
        self.restart_blr_cgroup.setValue(
            int(processor.get("blr_core_group_columns", 16))
        )
        self.restart_dynamic.setChecked(
            bool(processor.get("apply_dynamic_half_column_mask", False))
        )
        self.restart_dynamic_window.setValue(
            int(processor.get("dynamic_mask_median_window_pixels", 31))
        )
        self.restart_dynamic_ratio.setValue(
            float(processor.get("dynamic_mask_threshold_ratio", 1.0))
        )
        self.restart_dynamic_offset.setValue(
            float(processor.get("dynamic_mask_threshold_offset", 500.0))
        )
        self.restart_dynamic_edge.setValue(
            int(processor.get("dynamic_mask_excluded_edge_rows", 32))
        )
        self.restart_two_sided.setChecked(
            bool(processor.get("dynamic_mask_two_sided", True))
        )
        self.restart_receivers.setValue(int(root.get("num_receivers", 1)))
        self.restart_frames.setValue(int(receiver.get("frames_per_tensor", 128)))
        self.restart_source_mask.setText(
            f"0x{int(receiver.get('expected_source_mask', 255)):02x}"
        )
        self.restart_header.setValue(int(receiver.get("header_size", 42)))
        self.restart_payload.setValue(int(receiver.get("payload_size", 7680)))
        self.restart_slack.setValue(
            int(receiver.get("batch_close_slack_packets", 512))
        )
        self.restart_gpu_headers.setChecked(
            bool(receiver.get("gpu_header_extract", False))
        )
        self.restart_hds.setChecked(bool(receiver.get("hds", False)))
        self.restart_duplicate.setChecked(
            bool(receiver.get("tile_duplicate_prefix_to_simulate_payload", True))
        )
        self.restart_latency.setChecked(bool(receiver.get("capture_latency", False)))
        self.restart_writer.setChecked(not bool(writer.get("noop", True)))
        self.restart_writer_path.setText(
            writer.get("filepath", "/data/stem_continuous.h5")
        )
        self.restart_writer_dataset.setText(
            writer.get("dataset_name", "/processed")
        )
        self.restart_writer_concurrent.setValue(
            int(writer.get("num_concurrent", 3))
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.poll_timer.stop()
        self.render_timer.stop()
        QtCore.QMetaObject.invokeMethod(
            self.stream_worker,
            "stop",
            QtCore.Qt.ConnectionType.BlockingQueuedConnection,
        )
        self.stream_thread.quit()
        self.stream_thread.wait(1500)
        QtCore.QMetaObject.invokeMethod(
            self.control_worker,
            "stop",
            QtCore.Qt.ConnectionType.BlockingQueuedConnection,
        )
        self.control_thread.quit()
        self.control_thread.wait(1500)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream-endpoint",
        default="tcp://127.0.0.1:15556",
        help="thinned SUB endpoint, usually an SSH-forwarded local port",
    )
    parser.add_argument(
        "--control-endpoint",
        default="tcp://127.0.0.1:15557",
        help="control REQ endpoint, usually an SSH-forwarded local port",
    )
    parser.add_argument("--topic", default="stem/", help="subscription prefix")
    parser.add_argument(
        "--max-render-hz",
        type=float,
        default=5.0,
        help="maximum Qt redraw rate; products are still received newest-only",
    )
    args = parser.parse_args()
    if args.max_render_hz <= 0:
        parser.error("--max-render-hz must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("STEM DAQ Console")
    window = MainWindow(
        args.stream_endpoint,
        args.control_endpoint,
        args.topic,
        args.max_render_hz,
    )
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
