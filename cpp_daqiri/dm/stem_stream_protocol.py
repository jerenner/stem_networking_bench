"""Qt- and DigitalMicrograph-independent decoder for STEM thinned products.

This module intentionally supports the conservative Python environment shipped
with GMS 3. It can be copied beside ``stem_dm_viewer.py`` or imported by the
standalone DAQ tools in this repository.
"""

import json

import numpy as np


SCHEMA = "stem.thinned.v1"
FLOAT32_LE = np.dtype("<f4")


def decode_product(parts, copy_arrays=False):
    """Decode one ZeroMQ multipart STEM product.

    Args:
        parts: Topic, JSON metadata, and one or more image payload byte strings.
        copy_arrays: Copy payloads into owned writable arrays when true. The
            default creates zero-copy, read-only views backed by ``parts``.

    Returns:
        ``(topic, metadata, arrays)`` where arrays is keyed by product name.
    """
    if len(parts) < 3:
        raise ValueError(
            "expected at least 3 message parts, received {}".format(len(parts))
        )

    topic = parts[0].decode("utf-8")
    metadata = json.loads(parts[1].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("schema") != SCHEMA:
        raise ValueError(
            "unsupported stream schema {!r}".format(metadata.get("schema"))
        )

    height = int(metadata["height"])
    width = int(metadata["width"])
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if metadata.get("dtype", "float32") != "float32":
        raise ValueError("only float32 stream payloads are supported")
    if metadata.get("byte_order", "little") != "little":
        raise ValueError("only little-endian stream payloads are supported")

    declared_parts = metadata.get("parts")
    if not isinstance(declared_parts, list):
        raise ValueError("metadata parts must be a list")
    names = [name for name in declared_parts if name != "metadata"]
    if len(names) != len(parts) - 2:
        raise ValueError("metadata parts do not match multipart payload")
    if len(names) != len(set(names)):
        raise ValueError("metadata contains duplicate payload names")

    expected_bytes = height * width * FLOAT32_LE.itemsize
    arrays = {}
    for name, payload in zip(names, parts[2:]):
        if not isinstance(name, str) or not name:
            raise ValueError("payload names must be non-empty strings")
        if len(payload) != expected_bytes:
            raise ValueError(
                "{} payload has {} bytes; expected {}".format(
                    name, len(payload), expected_bytes
                )
            )
        array = np.frombuffer(payload, dtype=FLOAT32_LE).reshape(height, width)
        arrays[name] = array.copy() if copy_arrays else array

    return topic, metadata, arrays

