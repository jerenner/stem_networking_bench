# DigitalMicrograph live-view prototype

This directory is a self-contained first integration between the DAQIRI STEM
thinned ZeroMQ stream and Gatan DigitalMicrograph (DM). It displays persistent
DM images for each receiver's representative frame and 128-frame sum. The
standalone Qt console remains responsible for DAQ controls during this phase.

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

Set `STREAM_ENDPOINT` near the top of `stem_dm_viewer.py` to that machine's
address. If the SSH tunnel terminates on Windows, retain the default
`tcp://127.0.0.1:15556`.

Open `stem_dm_viewer.py` from DM's Python script editor, ensure **Execute in
background** is selected, and execute it. DM should create up to four live
images when two receivers publish both products. Stop it with `Ctrl+Shift+Q`.
For a bounded first test, set `MAX_DISPLAYED_PRODUCTS = 10`.

The viewer drains queued messages and renders only the newest product at up to
`MAX_DISPLAY_HZ`. Slow DM display updates therefore do not backpressure DAQ
acquisition. The ZeroMQ publisher remains live-only, so start the viewer before
the products that need to be observed.

## Local tests

These tests use a fake DigitalMicrograph image object and require no Gatan
installation:

```bash
python cpp_daqiri/dm/test_dm_viewer.py
```
