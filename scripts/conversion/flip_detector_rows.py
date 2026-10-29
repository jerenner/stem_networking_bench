#!/usr/bin/env python3
"""Reverse selected detector row blocks in a new HDF5 copy, frame by frame.

Every image-shaped dataset is transformed, including dark means, standard
deviations and masks. No ADC decoding, calibration, or averaging is applied.
The two flags toggle their blocks: applying the same operation twice restores
the original data and orientation tags. Untagged files are assumed unflipped.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


FLAGS = ("flip_top_imaging_rows", "flip_bottom_reference_rows")


def read_orientation(dataset):
    present = [key in dataset.attrs for key in FLAGS]
    if any(present) and not all(present):
        raise ValueError(f"{dataset.name}: both row-orientation attributes are required")
    result = []
    for key in FLAGS:
        value = np.asarray(dataset.attrs.get(key, 0))
        if value.size != 1 or value.item() not in (0, 1):
            raise ValueError(f"{dataset.name}: {key} must be a scalar 0 or 1")
        result.append(bool(value.item()))
    return tuple(result)


def flip_rows(frame, top_imaging=False, bottom_reference=False):
    if frame.ndim != 2 or frame.shape[0] != 1024:
        raise ValueError("detector row flips require a 2D 1024-row frame")
    result = frame.copy()
    if top_imaging:
        result[32:512] = frame[32:512][::-1]
    if bottom_reference:
        result[992:1024] = frame[992:1024][::-1]
    return result


def convert(input_path, output_path, top_imaging=False, bottom_reference=False):
    input_path, output_path = Path(input_path), Path(output_path)
    if not (top_imaging or bottom_reference):
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
                if obj.ndim == 2:
                    dataset[...] = flip_rows(obj[...], top_imaging, bottom_reference)
                else:
                    for frame in range(len(obj)):
                        dataset[frame] = flip_rows(obj[frame], top_imaging, bottom_reference)
                before = orientations[name]
                after = (before[0] ^ top_imaging, before[1] ^ bottom_reference)
                for key, value in zip(FLAGS, after):
                    dataset.attrs[key] = np.uint32(value)
            source.visititems(copy_object)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flip-top-imaging-rows", action="store_true")
    parser.add_argument("--flip-bottom-reference-rows", action="store_true")
    args = parser.parse_args()
    images = convert(args.input, args.output, args.flip_top_imaging_rows,
                     args.flip_bottom_reference_rows)
    print(f"Wrote {args.output}; transformed {len(images)} image dataset(s), preserving dtype.")


if __name__ == "__main__":
    main()
