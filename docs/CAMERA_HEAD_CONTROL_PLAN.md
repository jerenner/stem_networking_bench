# Camera-head and scan control integration plan

## Purpose

The recovered `sw_server` application is useful as a behavioral reference, but
it should not be restored as the production controller. It combines hardware
control, FPGA data receipt, file writing, and monitoring in one process and
contains several historical source variants whose hardware sequences differ.

The new implementation should preserve the validated instrument operations and
their ordering while keeping the existing DAQIRI data path unchanged:

```text
detector FPGA UDP streams -> DAQIRI RX -> GPU processing -> HDF5 / thin PUB
                                      ^
DigitalMicrograph -> supervisor REP --+-> instrument control service
                                            |-> camera head (SSH + dsh)
                                            `-> detector-board control links
```

DigitalMicrograph should communicate only with the persistent DAQ supervisor.
It should not connect directly to the camera head, detector boards, or a raw
shell-command endpoint.

## What the recovered application does

The current recovered snapshot exposes a plain TCP command server on port
`42003`. It parses whitespace-delimited commands and invokes methods on one
`STEM4dDaq` object. That object owns four detector-board connections, camera
head commands, receive threads, a binary file writer, and a monitoring socket.

### Camera-head transport

`CameraHead.cxx` reaches the camera head over SSH and executes either a script
or a `dsh` command. The recovered active file targets a `192.168` camera-head
address. Representative operations are:

- run camera startup and shutdown scripts;
- insert or retract the camera;
- read or set sensor temperature;
- read or set detector biases;
- set ADC sampling phase and offset;
- configure ADCs; and
- assert/deassert synchronization and enable/disable ADC ramp mode.

The recovered implementation constructs shell strings and invokes `system()`.
It also contains a plaintext credential. The credential must be rotated and
must not be copied into this repository or the replacement service. Production
access should use key-based SSH, strict host-key checking, an allowlist of
commands, and no caller-supplied shell text.

### Detector-board control

`STEM4dDetector.cxx` opens control/data connections to up to four FPGA boards.
The active snapshot uses `10.20.20.32` through `10.20.23.32`; an older
`192.168.20-23.32` form remains commented. This must be checked against the
current instrument network before live testing.

The board-control code performs:

- GTX/JESD reset, synchronization, and status reads;
- ADC data-flow shifting and automatic alignment;
- trigger delay and pre-scan trigger configuration;
- scan dimensions, read/pause counts, flyback, and memory-flush configuration;
- coordinated future-frame scan start and scan stop; and
- legacy FPGA-side dark subtraction, counting, summation, division,
  compression, and test-pattern controls.

The active scan defaults are `pause=200`, `read=1`, `x=1`, `rows=1`,
`flyback=100`, with memory flush enabled. A scan start writes a future frame
number to each selected board so they begin together.

### Compound legacy sequences

The old power-up sequence stops its receive threads, runs the camera startup
and bias/thermal/ADC configuration, performs a resync, restarts receipt, starts
a scan, and runs data auto-alignment.

The old resync sequence is long-running:

1. Stop data-receive, writer, and monitoring threads.
2. Program ADC startup delay.
3. Assert camera-head synchronization.
4. Reset/check GTX and JESD links, with retries and multi-second waits.
5. Reset data flow and deassert synchronization.
6. Check ADC status.
7. Restart data receipt.
8. Start a scan and wait for data.
9. Auto-align ADC data.

This sequence can take tens of seconds. Its success check is hard-coded for a
four-board mask, and the recovered logs show both successful and failed link
checks. It must be converted into explicit steps with per-board results rather
than retained as one opaque Boolean function.

### Legacy data and monitoring paths

The old application receives four board streams into a shared ring of
`576x576` `uint16` buffers. A file thread writes approximately 1 GB raw binary
files. A ZeroMQ `REP` socket on port `6003` responds to GUI requests with a JSON
header and one raw image.

These paths should not be migrated:

- DAQIRI already owns high-rate packet receipt and frame assembly.
- The DAQIRI burst writer already owns controlled HDF5 output.
- The DAQIRI thinned `PUB` stream already provides representative frames and
  bucket sums without request-driven backpressure.
- Dark subtraction, BLR, masking, thresholding, and future counting belong in
  the current GPU pipeline, not the legacy FPGA-processing commands.

The collaborator's reference to separate receiver servers is not represented
in this recovered snapshot; here, four receive threads connect directly to the
boards. Any additional receiver-coordination code or `ktest` source should be
located before claiming that behavior has been fully recovered.

## Responsibility mapping

| Legacy responsibility | New owner | DM exposure |
| --- | --- | --- |
| Start/stop receiver threads | Persistent DAQIRI supervisor | Existing Start/Stop acquisition |
| Raw detector receipt/assembly | DAQIRI RX child | Status only |
| Raw/processed disk writing | DAQIRI burst writer | Existing Burst tab |
| Monitoring image | DAQIRI thinned PUB | Existing DM image windows |
| GPU corrections and reduction | DAQIRI processor | Existing visualization/burst stage selectors |
| Camera SSH/`dsh` commands | New instrument control service | Camera tab, intent-level actions only |
| FPGA control registers | New detector-board adapter in the instrument service | Scan and service/diagnostic actions |
| Multi-step power/resync/scan sequencing | Persistent supervisor orchestration | Async operation status and abort where safe |
| Scan numbering/metadata | Supervisor plus HDF5 metadata | Scan setup/status |

The instrument control service should be a separate local process or tightly
isolated component. The supervisor remains the only public control endpoint and
calls the service over a loopback/IPC transport. This keeps hardware access
serialized and prevents an instrument-command failure from corrupting the
DAQIRI receive process.

## Proposed control protocol

Retain JSON over the existing supervisor `REP` endpoint, but add namespaced,
intent-level commands. Do not expose `run_script`, `raw_dsh`, arbitrary register
writes, or arbitrary SSH commands.

Initial command surface:

```text
instrument.get_state
camera.read_temperature
camera.read_biases
camera.power_up
camera.power_down
camera.insert
camera.retract
detector.read_link_status
detector.resync
detector.auto_align
scan.configure
scan.start
scan.stop
scan.abort
operation.get
```

Long-running actions must be asynchronous. A request should return promptly:

```json
{
  "command": "detector.resync",
  "request_id": "dm-1042"
}
```

```json
{
  "ok": true,
  "accepted": true,
  "operation_id": "op-00017"
}
```

Normal `get_state` polling should then include:

```json
{
  "instrument": {
    "service_online": true,
    "camera": {
      "power_state": "ready",
      "insertion_state": "inserted",
      "temperature_c": -20.1
    },
    "detector": {
      "synchronized": true,
      "links": ["ready", "ready", "ready", "ready"]
    },
    "operation": {
      "id": "op-00017",
      "name": "detector.resync",
      "state": "running",
      "step": "checking JESD links",
      "completed_steps": 4,
      "total_steps": 9,
      "error": ""
    }
  }
}
```

Reported state must distinguish a commanded state from an observed/verified
state. The legacy implementation often returned success without checking the
SSH process exit status or parsing camera output; the replacement must not mark
hardware `ready` solely because a command was sent.

## Scan orchestration

A user-level **Acquire scan** action should coordinate the instrument and data
planes in a deterministic order:

1. Validate camera, detector-link, DAQ, and output readiness.
2. Start DAQIRI acquisition if it is stopped and wait for receiver readiness.
3. Configure and arm the requested burst capture, if recording is requested.
4. Configure scan timing and dimensions on every selected board.
5. Start the boards on a shared future trigger/frame boundary.
6. Report scan progress while DAQIRI assembles and processes frames.
7. Stop or confirm completion, disarm the burst writer, and persist scan
   parameters and instrument status as HDF5 metadata.
8. On failure, stop the scan, abort/disarm output safely, and leave the reason
   visible in supervisor state.

This replaces the legacy implicit coupling between scan start, receiver
threads, and file writing. It also ensures the first saved bucket begins at a
known scan boundary rather than merely capturing the next available stream
bucket.

## DigitalMicrograph layout

Keep the current Python networking loop plus modeless DM-Script palette. Add
two tabs without performing network or hardware work on the DM UI thread:

### Camera

- service connection and camera readiness;
- observed temperature and target temperature;
- Power up / Power down;
- Insert / Retract;
- Read temperature / Read biases; and
- latest operation step, elapsed time, and error.

Potentially disruptive controls should require confirmation and should be
disabled while their preconditions are not met.

### Scan

- pause/read counts, X/Y positions, flyback, and flush policy;
- output stage and burst recipe selection;
- detector-link status;
- Resync and Auto-align service actions;
- Acquire scan, Stop, and Abort; and
- scan number, state, expected frames, received frames, and output files.

The existing DM Python loop can continue to poll the supervisor and update DM
tags. Async operations prevent a resync or power-up sequence from blocking image
receipt for tens of seconds.

## Safety and security requirements

- Rotate the credential exposed in the recovered source before placing that
  source anywhere public.
- Use SSH keys or another managed credential source; never configuration-file
  passwords or `sshpass -p`.
- Enforce host-key verification and command timeouts.
- Pass command arguments without a shell and validate every numeric range.
- Serialize instrument operations; only one hardware-changing operation may run
  at a time.
- Enforce state preconditions in the supervisor, not only in the GUI.
- Record request ID, operator, command, parameters, start/end time, per-step
  results, and final status in an audit log.
- Keep the instrument service on loopback/IPC. Continue exposing the supervisor
  only through the protected management network or SSH tunnel.
- Make stop/abort paths available even when another operation has failed.
- Start all development with mock and dry-run transports. No recovered startup,
  bias, synchronization, insertion, or temperature command should be run on
  hardware until its sequence is approved by the instrument owners.

## Phased implementation

### Phase 0: confirm the reference

- Identify which `CameraHead.cxx` variant is authoritative for DOEELS.
- Recover `ktest` and any separate receiver-server coordination code.
- Confirm camera-head and board addresses, routing, module count, source mask,
  and current SSH policy.
- Review the exact power, temperature, bias, sync, and auto-align sequences with
  the hardware owners.

### Phase 1: mock instrument service

Implemented in the DAQIRI path as the current hardware-free integration:

- Implemented the namespaced protocol, serialized operation worker, operation
  progress, audit log, and dry-run command plan.
- Added a mock camera/board transport and deterministic failure injection.
- Extended supervisor state and added Camera/Scan DM tabs.
- Added hardware-free bridge tests and a supervisor/service validation client.

### Phase 2: read-only hardware probes

- Verify SSH reachability and host identity.
- Read camera temperature and biases.
- Read per-board firmware and link status.
- Compare parsed results against the legacy tools and operator observations.

### Phase 3: isolated maintenance actions

- Add insert/retract and reviewed power sequences.
- Add resync and auto-align one step at a time, with per-step timeouts and
  rollback/stop behavior.
- Test while DAQIRI is stopped, then test supervisor-controlled drain/restart.

### Phase 4: coordinated scans

- Add scan configuration/start/stop.
- Arm DAQIRI burst output before the shared scan boundary.
- Store scan parameters and instrument state in HDF5 metadata.
- Validate frame counts, first-frame alignment, incomplete-bucket behavior, and
  repeatability before enabling normal DM use.

## Questions requiring collaborator confirmation

1. Which recovered camera-head synchronization sequence is current? The active,
   `4dstem`, `doeels`, `orig`, and `v2025` files do not all agree.
2. Is the camera head still at the recovered `192.168` address, and can the IGX
   host route to it outside the DAQ container?
3. Are the detector control addresses currently `10.20.20-23.32`, the older
   `192.168.20-23.32`, or something else?
4. Does one scan use four detector boards, eight sources, eight receiver
   interfaces, or another mapping? The recovered sync success mask is fixed at
   four boards.
5. What are the units and intended ranges for pause/read counts, flyback,
   pre-scan triggers, and the future-frame start offset?
6. Which state must be established before insert, power-up, resync, auto-align,
   and scan start, and which failures require automatic retract or power-down?
7. What output from `dsh` constitutes verified success for each operation?
8. Should a scan always imply a finite burst capture, or may monitoring-only
   scans run without disk output?
9. Where is `ktest`, and is there additional receiver-server protocol code not
   present in this recovered directory?
