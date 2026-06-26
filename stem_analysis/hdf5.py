"""Small HDF5 helpers shared by offline analysis scripts."""

from __future__ import annotations


def normalize_dataset_path(dataset_path: str, default: str = "/processed") -> str:
    if not dataset_path:
        return default
    return dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"


def create_dataset_with_groups(h5_file, dataset_path: str, **kwargs):
    dataset_path = normalize_dataset_path(dataset_path)
    group_path, dataset_name = dataset_path.rsplit("/", 1)
    group = h5_file if not group_path else h5_file.require_group(group_path)
    return group.create_dataset(dataset_name, **kwargs)


def read_single_image(h5_file, dataset_path: str, np, default: str = "/processed"):
    data = h5_file[normalize_dataset_path(dataset_path, default)][...]
    if data.ndim == 2:
        return data.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    raise ValueError(
        f"{dataset_path} must have shape [rows, cols] or [1, rows, cols], got {data.shape}"
    )

