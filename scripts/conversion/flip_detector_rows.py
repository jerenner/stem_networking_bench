#!/usr/bin/env python3
"""Convert detector row layouts in a new HDF5 copy, frame by frame.

Every image-shaped dataset is transformed, including dark means, standard
deviations and masks. No ADC decoding, calibration, or averaging is applied.
Use --to-native-layout to migrate unflipped or version-1 captures/calibrations
to the current sensor coordinates. Untagged files are assumed unflipped.
The individual flags toggle layout states, undoing the source layout first;
the bottom stream mapping itself is not self-inverse.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


FLAGS = ("flip_top_imaging_rows", "flip_bottom_reference_rows")
VERSION_KEY = "detector_row_layout_version"
LAYOUT_VERSION = 2


def read_orientation(dataset):
    present = [key in dataset.attrs for key in FLAGS]
    if (any(present) or VERSION_KEY in dataset.attrs) and not all(present):
        raise ValueError(f"{dataset.name}: both row-orientation attributes are required")
    result = []
    for key in FLAGS:
        value = np.asarray(dataset.attrs.get(key, 0))
        if value.size != 1 or value.item() not in (0, 1):
            raise ValueError(f"{dataset.name}: {key} must be a scalar 0 or 1")
        result.append(bool(value.item()))
    return tuple(result)


def read_layout_version(dataset):
    read_orientation(dataset)
    version = np.asarray(dataset.attrs.get(VERSION_KEY, 1))
    if version.size != 1 or version.dtype.kind not in "iu" or version.item() not in (1, 2):
        raise ValueError(f"{dataset.name}: unsupported {VERSION_KEY}")
    return int(version.item())


def flip_rows(frame, top_imaging=False, bottom_reference=False, *, inverse=False,
              layout_version=LAYOUT_VERSION):
    if frame.ndim != 2 or frame.shape[0] != 1024:
        raise ValueError("detector row flips require a 2D 1024-row frame")
    if layout_version not in (1, 2):
        raise ValueError("unsupported detector row layout version")
    destination = np.arange(1024)
    if top_imaging:
        destination[32:512] = np.arange(511, 31, -1)
    if bottom_reference:
        if layout_version == 1:
            destination[992:1024] = np.arange(1023, 991, -1)
        else:
            destination[512:544] = np.arange(1023, 991, -1)
            destination[544:1024] = np.arange(512, 992)
    if inverse:
        return frame[destination].copy()
    result = np.empty_like(frame)
    result[destination] = frame
    return result


def convert(input_path, output_path, top_imaging=False, bottom_reference=False, *, to_native=False):
    input_path, output_path = Path(input_path), Path(output_path)
    if to_native and (top_imaging or bottom_reference):
        raise ValueError("--to-native-layout cannot be combined with individual flip flags")
    if not (to_native or top_imaging or bottom_reference):
        raise ValueError("select at least one row block to reverse")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output files must differ")
    with h5py.File(input_path, "r") as source:
        images = []
        def inspect(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3) and obj.shape[-2:] == (1024, 3840):
                images.append(name)
        source.visititems(inspect)
        if not images:
            raise ValueError("no [1024,3840] images or [frames,1024,3840] datasets found")
        # Validate all tags before creating any output. Refuse to overwrite an
        # existing file, including a calibration accidentally named as output.
        orientations = {name: read_orientation(source[name]) for name in images}
        versions = {name: read_layout_version(source[name]) for name in images}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x") as target:
            target.attrs.update(source.attrs)
            target.attrs["row_flip_source_file"] = str(input_path)
            def copy_object(name, obj):
                if isinstance(obj, h5py.Group):
                    target.require_group(name).attrs.update(obj.attrs)
                    return
                if name not in orientations:
                    source.copy(obj, target, name=name)
                    return
                options = {"chunks": obj.chunks, "compression": obj.compression,
                           "shuffle": obj.shuffle, "fletcher32": obj.fletcher32}
                if obj.fillvalue is not None:
                    options["fillvalue"] = obj.fillvalue
                if obj.compression is not None:
                    options["compression_opts"] = obj.compression_opts
                if obj.scaleoffset is not None:
                    options["scaleoffset"] = obj.scaleoffset
                if obj.chunks is not None:
                    options["maxshape"] = obj.maxshape
                dataset = target.create_dataset(name, shape=obj.shape, dtype=obj.dtype, **options)
                dataset.attrs.update(obj.attrs)
                before = orientations[name]
                after = (True, True) if to_native else (before[0] ^ top_imaging,
                                                       before[1] ^ bottom_reference)
                def transform(frame):
                    # Undo the *stored* mapping before applying the requested
                    # current mapping. This also repairs cd5d807 bottom halves.
                    stream = flip_rows(frame, *before, inverse=True,
                                       layout_version=versions[name])
                    return flip_rows(stream, *after)
                if obj.ndim == 2:
                    dataset[...] = transform(obj[...])
                else:
                    for frame in range(len(obj)):
                        dataset[frame] = transform(obj[frame])
                for key, value in zip(FLAGS, after):
                    dataset.attrs[key] = np.uint32(value)
                dataset.attrs[VERSION_KEY] = np.uint32(LAYOUT_VERSION)
            source.visititems(copy_object)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--to-native-layout", action="store_true",
                        help="Migrate all images/masks to the current native FPGA sensor layout.")
    parser.add_argument("--flip-top-imaging-rows", action="store_true")
    parser.add_argument("--flip-bottom-reference-rows", action="store_true")
    args = parser.parse_args()
    images = convert(args.input, args.output, args.flip_top_imaging_rows,
                     args.flip_bottom_reference_rows, to_native=args.to_native_layout)
    print(f"Wrote {args.output}; transformed {len(images)} image dataset(s), preserving dtype.")


if __name__ == "__main__":
    main()
