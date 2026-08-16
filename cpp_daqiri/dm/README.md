# DigitalMicrograph live viewer and DAQ controls

This directory is a self-contained first integration between the DAQIRI STEM
thinned ZeroMQ stream and Gatan DigitalMicrograph (DM). It displays persistent
DM images for each receiver's representative frame and 128-frame sum. The
same background Python loop services a modeless, native DM control palette for
acquisition lifecycle, thinned visualization, and controlled burst capture.

## Install pyzmq into GMS

DigitalMicrograph supplies a specific NumPy build. Do not upgrade or replace
it. From an Administrator Anaconda Prompt, activate the environment selected by
DM and install only the compatible ZeroMQ binding:

```bat
activate GMS_VENV_PYTHON
python -m pip install -r C:\path\to\stem_networking_bench\cpp_daqiri\dm\requirements_dm.txt
cd C:\path\to\stem_networking_bench
python .\cpp_daqiri\dm\install_dm_module_path.py
```

The installer writes one `stem_daqiri_dm.pth` path file into the active Python
environment. It does not copy modules or modify NumPy, and it follows future
`git pull` updates in the same clone. If the GMS environment under
`C:\ProgramData` is not writable, run the command from an Administrator shell.
Restart DM afterward so its embedded Python processes the new path file.

Run `check_dm_environment.py` once in DM. Its Results output should list the
Python, NumPy, and pyzmq versions and report a successful image update.

## Test with the synthetic DAQ

On a machine reachable from Windows, start the existing mock server:

```bash
python cpp_daqiri/gui/mock_stem_daq.py \
  --pub-endpoint tcp://0.0.0.0:5556 \
  --control-endpoint tcp://0.0.0.0:5557 \
  --height 1024 --width 3840
```

Set `STREAM_ENDPOINT` and `CONTROL_ENDPOINT` near the top of
`stem_dm_viewer.py` to that machine's addresses. If the SSH tunnel terminates
on Windows, retain the defaults `tcp://127.0.0.1:15556` and
`tcp://127.0.0.1:15557`.

Open `stem_dm_viewer.py` from DM's Python script editor, ensure **Execute in
background** is selected, and execute it. DM should open the **STEM DAQ
Control** palette and create up to four live images when two receivers publish
both products. Stop the Python engine with `Ctrl+Shift+Q`. For a bounded first
test, set `MAX_DISPLAYED_PRODUCTS = 10`.

The viewer drains queued messages and renders only the newest product at up to
`MAX_DISPLAY_HZ`. Slow DM display updates therefore do not backpressure DAQ
acquisition. The ZeroMQ publisher remains live-only, so start the viewer before
the products that need to be observed.

## Integrated controls

`stem_dm_controls.s` is a modeless native DM-script palette launched by the
Python viewer. It does not perform networking on DM's UI thread. Button
callbacks commit typed settings to persistent DM tags, and the Python loop
handles the requested ZeroMQ command between image polls.

Phase one provides:

- acquisition start, stop, and status;
- thinned-stream publishing, processing stage, refresh rate, representative
  frame, product selection, and thresholds;
- burst stage, destination, bucket/capture counts, completeness policy,
  thresholds, configure, arm, disarm, and abort; and
- cached control, acquisition, visualization, burst, and response status.

The Python loop polls DAQ state automatically. Press **Refresh** in the palette
to copy the newest cached state and accepted settings into the displayed
fields. Control requests temporarily serialize with image receipt; they do not
pause acquisition or processing on the IGX.

If automatic palette launch reports an error, open
`cpp_daqiri\dm\stem_dm_controls.s` in DM's scripting editor and execute it once
while `stem_dm_viewer.py` remains active. The two components communicate only
through the `STEM DAQ` persistent-tag subtree.

## Optional Qt fallback

Start the compact controller from PowerShell in the environment containing the
Qt GUI dependencies if the DM palette is unavailable or deeper restart controls
are needed:

```powershell
python .\cpp_daqiri\gui\stem_daq_gui.py `
  --controls-only `
  --control-endpoint tcp://127.0.0.1:15557
```

This process connects only to the control endpoint and does not duplicate DM's
image traffic. Keep both SSH forwards active: port `15556` supplies images to
DM and port `15557` supplies state and commands to either control client.

## Local tests

These tests use a fake DigitalMicrograph image object and require no Gatan
installation:

```bash
python cpp_daqiri/dm/test_dm_viewer.py
python cpp_daqiri/dm/test_dm_control.py
```
