# Instrument service

The DAQIRI supervisor routes instrument commands from either the Qt controller
or DigitalMicrograph to a separate local service. Clients keep using the same
supervisor control endpoint, normally port 5557. Neither GUI needs a route to
the camera or FPGA control network.

There are three explicitly selected backends:

- `instrument.mode: mock`: existing simulated camera, detector, and scan actions.
- `instrument.mode: hardware_readonly`: actual temperature, bias, and link reads,
  with all hardware-changing commands rejected. This is the first live adapter,
  not a port of the old server's initialization or acquisition logic.
- `instrument.mode: emulated_readonly`: the real read-only adapter and parsers,
  connected to persistent local camera-text and FPGA-UDP responders instead of
  hardware. No physical camera connection or credentials are needed.

## Persistent camera/FPGA emulation

Use [`../configs/stem_instrument_emulated.yaml`](../configs/stem_instrument_emulated.yaml)
to test the newly added readbacks interactively from either Qt or DM. The
supervisor launches the usual Python read-only service, which owns the emulator
listeners and threads for as long as that service runs. Stopping acquisition or
disconnecting a GUI does not stop the emulator. Its state resets when the
supervisor/service exits; "persistent" does not mean saved across program restarts.

```text
DM or Qt -> existing public ZMQ control port -> supervisor
  -> read-only instrument service / serialized worker
     -> camera adapter -> loopback TCP -> emulated dsh text -> real parsers
     -> FPGA adapter -> loopback UDP -> emulated register replies -> real decoder
```

This tests command routing, the same fixed camera command selection, camera text
parsing, all 24 supply/bias measurements with units, named temperature probes,
the actual 24-byte FPGA register protocol/retries, asynchronous operation state,
errors, stale observations, and both GUI readouts. It **does not** implement an
SSH server or test SSH handshakes, keys, host verification, physical network
routing, hardware behavior, or a thermal/scan model. Only the camera transport
is replaced by a private TCP text exchange; FPGA reads still use the actual UDP
register-read implementation. This deliberately avoids setting up an SSH daemon
and test credentials just to exercise the camera's text readbacks.

All internal listeners bind only `127.0.0.1`, using automatically selected ports.
Each emulated board has its own UDP port. There is no external-target setting or
real-hardware fallback in this mode: any leftover `readonly.camera` or
`readonly.detector` hardware targets/helpers are ignored. Ordinary `get_state`
polls remain cached; they generate no camera or FPGA read requests. GUI read
status includes **SIMULATED**, and full state reports
`instrument.mode: emulated_readonly` and `instrument.simulated: true`.

After rebuilding the existing image with RX/ZeroMQ support, stop any other
supervisor using port 5557, then run this minimal test on the IGX:

```bash
sudo docker run --rm -it --name stem-instrument-demo --network host \
  stem_daqiri:burst-zmq \
  /opt/stem_daqiri/bin/stem_daqiri_supervisor \
  /opt/stem_daqiri/bin/configs/stem_instrument_emulated.yaml
```

No privileged mode, GPU allocation, hugepages, dark file, FPGA stream, camera
credentials, or additional SSH forwards are required. Existing port-5557
forwarding works unchanged. Launch the Qt controls-only GUI or the DM viewer,
then use their **Read temperatures**, **Read biases**, and **Read link status**
buttons. DM's **Refresh displayed readings** copies the completed samples into
the palette. There are no detector images from this instrument-only config.
Its Start acquisition button is not usable because no RX source is configured.

From a second IGX terminal, validate all readbacks through the supervisor:

```bash
sudo docker exec stem-instrument-demo python3 \
  /opt/stem_daqiri/bin/validate_emulated_instrument.py

# Exercise every fault and recovery path, then return to fresh healthy samples.
sudo docker exec stem-instrument-demo python3 \
  /opt/stem_daqiri/bin/validate_emulated_instrument.py --exercise-failures

# Leave an individual scenario active so its results can be examined in the GUI.
sudo docker exec stem-instrument-demo python3 \
  /opt/stem_daqiri/bin/validate_emulated_instrument.py \
  --scenario link_errors --module-id 1
```

The client can also run from a host checkout or over the forwarded control port;
it needs only `pyzmq` and accepts `--endpoint tcp://127.0.0.1:15557`. It refuses
to run unless the reported backend is `emulated_readonly`. Fault commands are
separately rejected by both the supervisor and real-hardware service.

Available scenarios are `healthy`, `suspect_temperature` (Q3=198.86 C),
`camera_error`, `camera_timeout`, `malformed_temperature`, `fpga_timeout`,
`malformed_fpga`, `link_errors`, and `unsynchronized`. FPGA faults affect only
the selected module, leaving other modules readable. Camera errors/timeouts
affect both camera commands; malformed temperature data affects temperature
only. A suspect thermistor is retained and warned about, not a transport error.
Likewise, successfully reading latched FPGA errors is a completed read; it does
not mean the link is healthy.

The supplied config sets four modules, small response delays, a slow +/-0.2 C
temperature oscillation, and a one-second camera read timeout. These are demo
signals, not measured detector dynamics. `module_count` supports 1-8 modules;
`camera_delay_ms`, `fpga_delay_ms`, `temperature_drift_c`, and
`read_timeout_seconds` are startup options. `emulator.set_scenario` changes
only the scenario and fault module at runtime; it refuses changes during a read
and marks old cached samples stale until they are measured again.

This mode remains read-only: power, insert/retract, resync, and scan actions are
rejected. Use the original `mock` backend to test those simulated control flows.
The emulator creates **no detector frame packets** and does not replace
DAQIRI's network, synthetic, or HDF5 sources. To combine image acquisition with
emulated camera readouts, copy only the `instrument` block into your complete
working acquisition config and retain its actual data-source settings.

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

Each module can optionally override the global UDP port with a `port` field;
normal hardware configurations continue to use port 42000 by default.

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
3. Prepare an authorized camera SSH identity (or the YAML password option below)
   and a `known_hosts` file with the camera's verified host key. Authentication
   is noninteractive and host-key checking remains strict. Mount the SSH files
   read-only as below.
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
`known_hosts` files. For key authentication the identity must work without a
passphrase prompt; an appropriately mounted SSH agent is another option.
With a YAML password only the `known_hosts` file is needed, not a private key.
Do not bake credentials into the image or commit them. If SSH or a board is
unreachable, a requested read shows the error while the supervisor stays available.

### YAML passwords and a mothership2 jump

When the camera can only be reached from `mothership2`, add `jump_host` under
`instrument.readonly.camera`. The IGX connects to the jump host and forwards a
TCP connection from there to the camera's SSH port. Camera commands still run
on the camera, not on mothership2; no camera-control installation on the jump
host is required. The jump SSH server must allow TCP forwarding and be able to
reach `192.168.111.90:22`.

For example, replace the camera settings in your LOCAL host-mounted YAML with:

```yaml
instrument:
  enabled: true
  mode: "hardware_readonly"
  launch_service: true
  service_binary: "/opt/stem_daqiri/bin/stem_daqiri_instrument_readonly"
  readonly:
    timeout_seconds: 20.0  # Total time for both SSH hops and the camera read.
    camera:
      host: "192.168.111.90"
      user: "root"
      port: 22
      password: "YOUR_CAMERA_PASSWORD"
      known_hosts_file: "/run/camera_ssh/known_hosts"
      jump_host:
        host: "mothership2"  # Must resolve/be reachable from the IGX container.
        user: "YOUR_MOTHERSHIP_USER"
        port: 22
        password: "YOUR_MOTHERSHIP_PASSWORD"
```

Keep your other `instrument.readonly` settings as needed. Do not create a second
`instrument` block. Changing these settings requires restarting the supervisor.
The image must be rebuilt once to install the new `stem_camera_ssh.py` helper;
no `sshpass`, GUI package, or separate tunnel process is needed. The existing
Docker command and SSH-directory mount otherwise stay the same. No DM/Qt-side
connection changes are necessary.

Both `password` fields must be quoted YAML strings. If a hop should use a key
instead, omit its `password` and supply its own `identity_file` (or an available
agent). Camera and jump authentication are independent; password takes priority
over an identity-file setting on the same hop. A jump defaults to user `root`
and port 22 if omitted; specify your actual mothership username explicitly.
The jump inherits the camera's `known_hosts_file`, or can provide its own file.
The file must contain verified host keys for both hosts. Password auth does not
disable host verification, and MFA/keyboard-interactive logins or encrypted-key
passphrase prompts are not handled by this adapter.

Password delivery uses OpenSSH's `SSH_ASKPASS_REQUIRE=force` with the bundled
helper. Only exact password prompts for the configured `user@host` are answered.
Passwords are passed in the SSH subprocess environment, never in command-line
arguments. They are redacted from control/config status JSON and camera-read
errors; the supervisor omits the instrument block from the RX child config.
The original local YAML necessarily remains plaintext: restrict its access and
keep it outside Git. Make authentication changes in the local YAML, not through
the GUI; this is a supervisor startup setting.

The implementation uses an SSH `ProxyCommand` with `ssh -W` (the same forwarding
mechanism as ProxyJump), so known-host and authentication options can be applied
explicitly to each hop. The OpenSSH manuals document
[password prompt helpers](https://man.openbsd.org/ssh.1#SSH_ASKPASS_REQUIRE) and
[per-hop proxy configuration](https://man.openbsd.org/ssh_config#ProxyJump).
This jump carries **camera SSH only**. FPGA register reads use UDP and do not
automatically traverse the jump; configure direct routing for those separately.

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
python -m unittest discover -s cpp_daqiri/instrument -p 'test_emulator.py' -v
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
  python -m unittest discover -s cpp_daqiri/instrument -p 'test_*.py' -v
```

The hardware-read test replaces SSH/register subprocesses with
`test_hardware_process.py`. The emulator integration test launches the persistent
backend and runs the complete scenario-validation client, including recovery,
through a compiled supervisor. Neither test launches RX or contacts hardware.
Running tests requires permission to bind localhost sockets;
the native DM palette still needs a separate Windows/DM check. Full CUDA/Docker
and actual instrument readback validation must be done on the IGX.
