#!/usr/bin/env python3
"""Pixel-level HDF5 parity check for STEM RX outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np


def _dataset_key(dataset: str) -> str:
    return dataset[1:] if dataset.startswith("/") else dataset


def _checked_dataset(h5_file: h5py.File, path: Path, dataset: str) -> h5py.Dataset:
    key = _dataset_key(dataset)
    if key not in h5_file:
        raise KeyError(f"{dataset!r} not found in {path}")
    return h5_file[key]


def _compared_shape(
    ref_shape: tuple[int, ...],
    cand_shape: tuple[int, ...],
    max_frames: int | None,
) -> tuple[int, ...]:
    if ref_shape == cand_shape and (max_frames is None or len(ref_shape) == 0):
        return ref_shape

    if len(ref_shape) != len(cand_shape):
        raise ValueError(f"rank mismatch: {ref_shape} != {cand_shape}")

    if max_frames is None:
        if ref_shape != cand_shape:
            raise ValueError(f"shape mismatch: {ref_shape} != {cand_shape}")
        return ref_shape

    if len(ref_shape) == 0:
        return ref_shape
    if ref_shape[1:] != cand_shape[1:]:
        raise ValueError(f"non-frame dimensions mismatch: {ref_shape} != {cand_shape}")

    frames = min(ref_shape[0], cand_shape[0], max_frames)
    if frames <= 0:
        raise ValueError(f"no frames available for prefix comparison: {ref_shape} vs {cand_shape}")
    return (frames,) + ref_shape[1:]


def compare_outputs(args: argparse.Namespace) -> int:
    if not args.reference.exists():
        raise FileNotFoundError(args.reference)
    if not args.candidate.exists():
        raise FileNotFoundError(args.candidate)

    with h5py.File(args.reference, "r") as ref_file, h5py.File(args.candidate, "r") as cand_file:
        ref = _checked_dataset(ref_file, args.reference, args.dataset)
        candidate = _checked_dataset(cand_file, args.candidate, args.dataset)

        print(f"reference : {args.reference} {ref.shape} {ref.dtype}")
        print(f"candidate : {args.candidate} {candidate.shape} {candidate.dtype}")

        if ref.dtype != candidate.dtype:
            print(f"FAIL dtype mismatch: {ref.dtype} != {candidate.dtype}", file=sys.stderr)
            return 1

        try:
            compared_shape = _compared_shape(
                tuple(ref.shape), tuple(candidate.shape), args.max_frames
            )
        except ValueError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            return 1
        print(f"compared_shape: {compared_shape}")

        exact_integer = (
            np.issubdtype(ref.dtype, np.integer)
            and np.issubdtype(candidate.dtype, np.integer)
        )
        equal = True
        max_abs = 0.0
        nonzero = 0
        first_mismatch: tuple[int, ...] | None = None
        first_ref = None
        first_candidate = None

        if len(compared_shape) == 0:
            ranges = [(None, None)]
        else:
            frames = compared_shape[0]
            ranges = [
                (start, min(frames, start + args.chunk_frames))
                for start in range(0, frames, args.chunk_frames)
            ]

        for start, end in ranges:
            if start is None:
                ref_chunk = np.asarray(ref[()])
                cand_chunk = np.asarray(candidate[()])
            else:
                ref_chunk = np.asarray(ref[start:end])
                cand_chunk = np.asarray(candidate[start:end])

            diff = cand_chunk.astype(np.float64) - ref_chunk.astype(np.float64)
            if diff.size:
                max_abs = max(max_abs, float(np.max(np.abs(diff))))
                nonzero += int(np.count_nonzero(diff))

            if exact_integer:
                mismatch = ref_chunk != cand_chunk
            else:
                mismatch = ~np.isclose(
                    ref_chunk,
                    cand_chunk,
                    rtol=args.rtol,
                    atol=args.atol,
                    equal_nan=True,
                )

            if np.any(mismatch):
                equal = False
                if first_mismatch is None:
                    local = tuple(int(i) for i in np.argwhere(mismatch)[0])
                    first_mismatch = (
                        (start + local[0],) + local[1:]
                        if start is not None
                        else local
                    )
                    first_ref = ref_chunk[local]
                    first_candidate = cand_chunk[local]

        print(f"max_abs_diff: {max_abs:g}")
        print(f"nonzero_diff_pixels: {nonzero}")

        if equal:
            print("PASS pixel parity")
            return 0

        if first_mismatch is not None:
            print(
                "first_mismatch: "
                f"index={first_mismatch} reference={first_ref!r} "
                f"candidate={first_candidate!r}",
                file=sys.stderr,
            )
        print("FAIL pixel parity", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two STEM HDF5 output datasets pixel by pixel.",
    )
    parser.add_argument("reference", type=Path, help="Reference HDF5 output.")
    parser.add_argument("candidate", type=Path, help="Candidate HDF5 output.")
    parser.add_argument(
        "--dataset",
        default="/processed",
        help="Dataset path to compare. Default: /processed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Compare only the first N aligned prefix frames. Intended for "
            "deterministic replay outputs, not unrelated live captures."
        ),
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=4,
        help="Number of frames to read per chunk. Default: 4.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for non-integer datasets. Use e.g. 1e-5 for reduced float parity.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for non-integer datasets.",
    )
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    return compare_outputs(args)


if __name__ == "__main__":
    raise SystemExit(main())
