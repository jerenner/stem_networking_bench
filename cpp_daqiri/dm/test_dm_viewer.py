"""Hardware- and DigitalMicrograph-free tests for the DM stream adapter."""

import json
import unittest

import numpy as np

from stem_dm_viewer import DigitalMicrographViewer
from stem_stream_protocol import decode_product


class FakeImage(object):
    def __init__(self, array):
        self.array = np.array(array, copy=True)
        self.name = ""
        self.shown = False
        self.updates = 0

    def SetName(self, name):
        self.name = name

    def ShowImage(self):
        self.shown = True

    def GetNumArray(self):
        return self.array

    def UpdateImage(self):
        self.updates += 1


class FakeDM(object):
    def __init__(self):
        self.images = []

    def CreateImage(self, array):
        if not array.flags.owndata:
            raise ValueError("DigitalMicrograph requires an owning NumPy array")
        image = FakeImage(array)
        self.images.append(image)
        return image


def product_parts(array, receiver=0, name="representative"):
    metadata = {
        "schema": "stem.thinned.v1",
        "receiver_id": receiver,
        "processing_stage": "corrected",
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": "float32",
        "byte_order": "little",
        "parts": ["metadata", name],
    }
    return [
        "stem/rx/{}/corrected".format(receiver).encode("utf-8"),
        json.dumps(metadata).encode("utf-8"),
        np.asarray(array, dtype="<f4").tobytes(),
    ]


class StreamProtocolTest(unittest.TestCase):
    def test_decode_is_zero_copy_by_default(self):
        source = np.arange(12, dtype="<f4").reshape(3, 4)
        _topic, _metadata, arrays = decode_product(product_parts(source))
        np.testing.assert_array_equal(arrays["representative"], source)
        self.assertFalse(arrays["representative"].flags.owndata)
        self.assertFalse(arrays["representative"].flags.writeable)

    def test_wrong_payload_size_is_rejected(self):
        parts = product_parts(np.zeros((3, 4), dtype=np.float32))
        parts[-1] = parts[-1][:-4]
        with self.assertRaises(ValueError):
            decode_product(parts)


class DigitalMicrographAdapterTest(unittest.TestCase):
    def test_passes_an_owning_array_to_create_image(self):
        dm = FakeDM()
        viewer = DigitalMicrographViewer(dm, ("representative",))
        source = np.arange(12, dtype="<f4").reshape(3, 4)
        _topic, metadata, arrays = decode_product(product_parts(source))
        self.assertFalse(arrays["representative"].flags.owndata)
        viewer.display(metadata, arrays)
        self.assertEqual(len(dm.images), 1)
        viewer.release()

    def test_reuses_window_and_updates_mapped_storage(self):
        dm = FakeDM()
        viewer = DigitalMicrographViewer(dm, ("representative",))
        first = np.arange(12, dtype=np.float32).reshape(3, 4)
        second = first + 100
        _topic, metadata, arrays = decode_product(product_parts(first))
        viewer.display(metadata, arrays)
        _topic, metadata, arrays = decode_product(product_parts(second))
        viewer.display(metadata, arrays)

        self.assertEqual(len(dm.images), 1)
        self.assertTrue(dm.images[0].shown)
        self.assertEqual(dm.images[0].updates, 1)
        np.testing.assert_array_equal(dm.images[0].array, second)
        viewer.release()

    def test_creates_independent_receiver_and_product_windows(self):
        dm = FakeDM()
        viewer = DigitalMicrographViewer(dm)
        source = np.ones((2, 3), dtype=np.float32)
        for receiver, name in ((0, "representative"), (0, "sum"), (1, "sum")):
            _topic, metadata, arrays = decode_product(
                product_parts(source, receiver=receiver, name=name)
            )
            viewer.display(metadata, arrays)
        self.assertEqual(len(dm.images), 3)
        viewer.release()


if __name__ == "__main__":
    unittest.main()
