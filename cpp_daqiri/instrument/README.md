# Instrument service

The DAQIRI supervisor routes instrument commands from either the Qt controller
or DigitalMicrograph to a separate local service. Clients keep using the same
supervisor control endpoint, normally port 5557. Neither GUI needs a route to
the camera or FPGA control network.

There are two explicitly selected backends:

- `instrument.mode: mock`: existing simulated camera, detector, and scan actions.
- `instrument.mode: hardware_readonly`: actual temperature, bias, and link reads,
  with all hardware-changing commands rejected. This is the first live adapter,
  not a port of the old server's initialization or acquisition logic.

## Read-only operations

| Control command | Hardware request | Result |
| --- | --- | --- |
| `camera.read_temperature` | SSH, fixed stdin `dsh sensor temp` | Named Q1-Q4, B1-B2, HS1-HS2 probe measurements in C |
| `camera.read_biases` | SSH, fixed stdin `dsh bias adc volts` | Named measured supplies and biases, retaining V/mA/etc. units |
| `detector.read_link_status` | UDP register 420 read on each configured board | Raw register, six lane flags, synchronization masks, and latched errors |

These commands come from the recovered `CameraHead.cxx`, `STEM4dDetector.cxx`,
and `sw_server/lib/NetworkInterface.cxx`. The old server is **not** imported,
constructed, or run. There is no power-up, cooling/setpoint change, bias write,
register write, link reset, alignment, trigger, or scan operation in this backend.
No hardware read happens automatically at service startup or during ordinary
GUI state polling. Reads happen only when requested.

The temperature labels refer to the camera's board/head probes; this adapter
does not claim to measure the TEC setpoint or actual imaging-sensor temperature.
Out-of-range readings remain visible with `[suspect]` and a warning. The
displayed maximum uses only readings within the configured plausibility range,
default -50 to 150 C. That range is **not** an overtemperature protection limit:
this service is not a thermal interlock and does not replace the camera's
existing protection or the legacy server's thermal guard.

Bias readback includes both currents and voltages. For example, `AVDD` can be in
V while `BIASN` is in mA; units are never inferred from a channel name. Missing
channels appear as `not read` rather than invented zeroes.

The FPGA transport sends six little-endian uint32 words to UDP port 42000:

```text
0x203b2d29, 0, 420, 0, 0, 0
```

The final address/value pair is the legacy protocol's required padding read of
register 0. The response must be 24 bytes, echo the first two words, and return
`0x70000000 | 420` in word 2; word 3 is the status register value. No write bit
is set. Each attempt uses a new connected UDP socket, with a configurable
timeout/retry bound; it never opens a detector data port or changes routing.

Register 420 is decoded as six-bit masks at bits 5:0 (JESD sync), 13:8
(not-in-table), 21:16 (disparity error), and 29:24 (characters-were-Ks). Full
sync is `0x3f`. Other nonzero masks are reported separately even when all lanes
are synchronized. These are **latched observations**, not a measurement of
errors per second. Unlike legacy status/reset sequences, this read does not
write register 400 or clear any status. A failed board read cannot produce an
overall healthy/synchronized result. Board module IDs are distinct from DAQIRI
receiver indices and incoming packet source IDs.

## Asynchronous behavior and observations

The service has one serialized worker. A read request immediately returns
`accepted: true` and an `operation_id`; completion or failure appears in
`instrument.operation` through normal `get_state` polling. Concurrent reads are
rejected as busy rather than building an unbounded queue. Slow SSH/UDP activity
runs outside both the supervisor control loop and RX processes.

`instrument.reads.temperature`, `.biases`, and `.links` carry `status`,
`attempted_at`, `sampled_at`, `age_seconds`, `stale`, `error`, and a bounded raw
response. Times are Unix seconds, displayed in UTC. Readbacks are stale before
their first measurement, while reading, after any error, or after the configured
age (default 120 seconds). Failed camera reads retain the last measured values
and their original timestamp, clearly marked stale. Link reads show individual
board failures alongside any boards that answered. Neither a cached value nor
an online service implies that the camera is safe, powered, inserted, or ready.

An optional JSONL audit records the fixed command and completion/error; it
does not record SSH keys or passwords. An optional `transport: helper` can
replace UDP for testing: trusted configuration supplies `reader_argv`, which
is called with `--host HOST --register 420` and returns JSON containing
`register420`. It is not selectable or parameterizable by remote requests.

## First IGX test

1. Rebuild `Dockerfile.daqiri` with the existing RX/HDF5/ZeroMQ build flags. The
   image now installs the read-only Python executable, PyYAML, pyzmq, and the
   SSH client. A non-container installation additionally needs
   `python3 -m pip install -r cpp_daqiri/instrument/requirements.txt` and `ssh`.
2. Copy `cpp_daqiri/configs/stem_instrument_readonly.yaml` to your host run
   directory. Confirm camera/board addresses with the instrument owner and
   edit them there. The included addresses are recovered examples, not
   automatically discovered devices.
3. Prepare an authorized camera SSH identity and a `known_hosts` file with the
   camera's verified host key. This adapter uses noninteractive key/agent
   authentication (`BatchMode=yes`) and strict host-key checking, not a stored
   password or an interactive prompt. Mount a key directory read-only as below.
4. Start the supervisor with acquisition stopped. This minimal test needs no
   GPU access, DPDK privileges, dark frame, or live receiver stream:

```bash
CFG="/home/daquser/jrenner/stem_daqiri_run/stem_instrument_readonly.yaml"
CAMERA_SSH_DIR="/home/daquser/jrenner/stem_daqiri_run/camera_ssh"

sudo docker run --rm -it --network host \
  -v "$CFG:/run/stem_rx.yaml:ro" \
  -v "$CAMERA_SSH_DIR:/run/camera_ssh:ro" \
  stem_daqiri:burst-zmq \
  /opt/stem_daqiri/bin/stem_daqiri_supervisor /run/stem_rx.yaml
```

The mounted directory should contain the configured `id_ed25519` and
`known_hosts` files. The identity must work without a passphrase prompt; an
appropriately mounted SSH agent is another option. Do not bake credentials into
the image or commit them. If SSH or a board is unreachable, a requested read
shows the error while the supervisor stays available.

Connect the existing GUI control SSH forward, then either launch the Qt GUI
with `--controls-only --control-endpoint tcp://127.0.0.1:15557`, or launch the
DM viewer as usual. The Qt **Camera / detector** tab has read buttons and full
measurement tables; hovering over a link row exposes the lane details and raw
register. In DM use **Temps**, **Biases**, **Rails**, and **Links**, then
**Refresh displayed readings** after the operation finishes. The DM palette
has fields for the known eight temperature probes, 24 supply/bias channels,
and module IDs 0-7. New firmware channel names remain available in full JSON
state and the Qt table even if not yet added to the fixed DM palette.

The minimal configuration intentionally cannot start acquisition. After the
readbacks agree with trusted instrument observations, copy its `instrument`
block into your working dual-FPGA configuration, retaining your existing
network/GPU/DARK/output mounts and `control.start_acquisition: false`. This
changes no packet assembly or GPU analysis code. Instrument settings are
startup configuration; restart the supervisor to change hosts or transport.

## Hardware-free verification

From the repository root with the GUI/test Python environment activated:

```bash
python -m unittest discover -s cpp_daqiri/instrument -p 'test_readonly.py' -v
python -m unittest discover -s cpp_daqiri/dm -p 'test_*.py' -v
QT_QPA_PLATFORM=offscreen python -m unittest discover \
  -s cpp_daqiri/gui -p 'test_gui_protocol.py' -v
```

The instrument tests use fixed camera outputs, fake subprocesses, and localhost
UDP responders only. They check the exact register request bytes, malformed
responses/retries, timeouts, stale samples, units, lane masks, and refusal of
hardware-changing commands. No actual camera or FPGA is contacted.

To also exercise a compiled supervisor launching the real Python service and
routing requests over local ZeroMQ, provide a locally built supervisor:

```bash
STEM_TEST_SUPERVISOR_BIN=/path/to/stem_daqiri_supervisor \
  python -m unittest discover -s cpp_daqiri/instrument -p 'test_readonly.py' -v
```

That test replaces SSH/register subprocesses with `test_hardware_process.py` and
does not launch RX. Running tests requires permission to bind localhost sockets;
the native DM palette still needs a separate Windows/DM check. Full CUDA/Docker
and actual instrument readback validation must be done on the IGX.
