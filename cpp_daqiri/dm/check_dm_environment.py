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

values = np.arange(96 * 64, dtype=np.float32).reshape(64, 96)
image = DM.CreateImage(values)
image.SetName("STEM DigitalMicrograph environment check")
image.ShowImage()
mapped = image.GetNumArray()
np.copyto(mapped, np.flipud(values))
image.UpdateImage()
print("DigitalMicrograph image create/map/update: OK")

# DM requires explicit release of every Py_Image reference before script exit.
del image

