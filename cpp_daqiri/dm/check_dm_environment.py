"""One-shot DigitalMicrograph Python and image-update environment check."""

import platform
import sys

import DigitalMicrograph as DM
import numpy as np


print("Python: {}".format(sys.version.replace("\n", " ")))
print("Platform: {}".format(platform.platform()))
print("NumPy: {}".format(np.__version__))
try:
    import zmq

    print("pyzmq: {}".format(zmq.__version__))
except ImportError:
    print("pyzmq: NOT INSTALLED")

# DM.CreateImage rejects NumPy views even when they are C-contiguous. Reshape
# returns a view, so make an explicit owning copy before crossing the DM API.
values = np.arange(96 * 64, dtype=np.float32).reshape(64, 96).copy(order="C")
image = DM.CreateImage(values)
if image is None:
    raise RuntimeError("DigitalMicrograph CreateImage returned no image")
image.SetName("STEM DigitalMicrograph environment check")
image.ShowImage()
mapped = image.GetNumArray()
np.copyto(mapped, np.flipud(values))
image.UpdateImage()
print("DigitalMicrograph image create/map/update: OK")

# DM requires explicit release of every Py_Image reference before script exit.
del image
