import argparse
from pathlib import Path
import zipfile


def get_plot_modules():
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    return h5py, plt, np


def compute_scale(frame):
    """Compute robust display limits for a frame."""
    _, _, np = get_plot_modules()
    vmin, vmax = np.percentile(frame, [1, 99])
    if vmin == vmax:
        vmin, vmax = np.min(frame), np.max(frame)
    if vmin == vmax:
        vmin, vmax = 0, 1
    return vmin, vmax


def validate_input(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    return file_path


def frame_figure_size(frame):
    """Pick a wide figure size that keeps single-pixel features visible."""
    fig_width = 18.0
    aspect = frame.shape[0] / frame.shape[1]
    fig_height = max(5.0, fig_width * aspect + 1.0)
    return fig_width, fig_height


def plot_single_frame(frame, frame_index, output_path, dpi=150):
    """Save one frame to a standalone image file."""
    _, plt, _ = get_plot_modules()
    vmin, vmax = compute_scale(frame)
    fig_width, fig_height = frame_figure_size(frame)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        frame,
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Frame {frame_index}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_composite_plot(file_path, output_plot, start_frame=0, frames_to_plot=4, dpi=150):
    """Read frames from an HDF5 file and generate a composite plot."""
    h5py, plt, np = get_plot_modules()
    file_path = validate_input(file_path)
    output_plot = Path(output_plot)

    with h5py.File(file_path, "r") as f:
        if "processed" not in f:
            raise KeyError("'processed' dataset not found in HDF5")

        data = f["processed"]
        num_frames = data.shape[0]
        start_frame = max(0, min(start_frame, num_frames))
        num_to_plot = min(frames_to_plot, max(0, num_frames - start_frame))

        if num_to_plot == 0:
            raise ValueError("No frames available for the requested start/count range")

        cols = 1
        rows = num_to_plot
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5.5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for local_idx in range(num_to_plot):
            frame_index = start_frame + local_idx
            frame = data[frame_index][:]
            vmin, vmax = compute_scale(frame)
            im = axes[local_idx].imshow(
                frame,
                cmap="magma",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            axes[local_idx].set_title(f"Frame {frame_index}")
            axes[local_idx].set_xlabel("Column")
            axes[local_idx].set_ylabel("Row")
            fig.colorbar(im, ax=axes[local_idx], shrink=0.8)

        for idx in range(num_to_plot, len(axes)):
            axes[idx].axis("off")

        fig.tight_layout()
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved composite plot of {num_to_plot} frames to {output_plot}")


def export_individual_frames(
    file_path,
    output_dir,
    start_frame=0,
    frames_to_plot=4,
    image_format="png",
    dpi=150,
    prefix="frame",
):
    """Export one image per frame into a single directory."""
    h5py, _, _ = get_plot_modules()
    file_path = validate_input(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    with h5py.File(file_path, "r") as f:
        if "processed" not in f:
            raise KeyError("'processed' dataset not found in HDF5")

        data = f["processed"]
        num_frames = data.shape[0]
        start_frame = max(0, min(start_frame, num_frames))
        num_to_plot = min(frames_to_plot, max(0, num_frames - start_frame))

        if num_to_plot == 0:
            raise ValueError("No frames available for the requested start/count range")

        for local_idx in range(num_to_plot):
            frame_index = start_frame + local_idx
            frame = data[frame_index][:]
            output_path = output_dir / f"{prefix}_{frame_index:04d}.{image_format}"
            plot_single_frame(frame, frame_index, output_path, dpi=dpi)
            exported_paths.append(output_path)
            print(f"Saved frame {frame_index} to {output_path}")

    print(f"Exported {len(exported_paths)} frame images to {output_dir}")
    return exported_paths


def dump_frames_to_text(file_path, output_text, start_frame=0, frames_to_dump=4):
    """Write selected frames to a text file with simple metadata and row data."""
    h5py, _, np = get_plot_modules()
    file_path = validate_input(file_path)
    output_text = Path(output_text)
    output_text.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        if "processed" not in f:
            raise KeyError("'processed' dataset not found in HDF5")

        data = f["processed"]
        num_frames = data.shape[0]
        start_frame = max(0, min(start_frame, num_frames))
        num_to_dump = min(frames_to_dump, max(0, num_frames - start_frame))

        if num_to_dump == 0:
            raise ValueError("No frames available for the requested start/count range")

        with output_text.open("w", encoding="utf-8") as out:
            out.write(f"source_file: {file_path}\n")
            out.write("dataset: processed\n")
            out.write(f"dataset_shape: {tuple(data.shape)}\n")
            out.write(f"start_frame: {start_frame}\n")
            out.write(f"frames_dumped: {num_to_dump}\n")
            out.write("\n")

            for local_idx in range(num_to_dump):
                frame_index = start_frame + local_idx
                frame = data[frame_index][:]
                out.write(f"frame_index: {frame_index}\n")
                out.write(f"frame_shape: {tuple(frame.shape)}\n")
                out.write(
                    "frame_stats: "
                    f"min={np.min(frame)}, max={np.max(frame)}, mean={float(np.mean(frame)):.6f}\n"
                )
                for row_idx, row in enumerate(frame):
                    row_text = " ".join(str(value) for value in row.tolist())
                    out.write(f"row{row_idx}: {row_text}\n")
                if local_idx != num_to_dump - 1:
                    out.write("\n")

    print(f"Dumped {num_to_dump} frames to text file {output_text}")


def create_zip_archive(files, zip_path):
    """Pack exported frame images into a zip file."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            file_path = Path(file_path)
            zf.write(file_path, arcname=file_path.name)

    print(f"Created archive {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot HDF5 frames from STEM networking bench.")
    parser.add_argument(
        "--file",
        type=str,
        default="/home/jrenner/local/lbl/holoscan/sparse_frames_out_net.h5",
        help="Path to the input HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="frames_plot.png",
        help="Path to the composite output image file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="If set, export one file per frame into this directory",
    )
    parser.add_argument(
        "--text-output",
        type=str,
        help="If set, dump selected frames as numeric rows into this text file",
    )
    parser.add_argument(
        "--zip-output",
        type=str,
        help="Optional path to a zip archive containing the exported individual frame images",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Number of frames to plot/export",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start plotting/exporting from",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Image format for individual frame export",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved images",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for individual frame export",
    )

    args = parser.parse_args()

    if args.text_output:
        dump_frames_to_text(
            file_path=args.file,
            output_text=args.text_output,
            start_frame=args.start_frame,
            frames_to_dump=args.frames,
        )
    elif args.output_dir:
        exported = export_individual_frames(
            file_path=args.file,
            output_dir=args.output_dir,
            start_frame=args.start_frame,
            frames_to_plot=args.frames,
            image_format=args.format,
            dpi=args.dpi,
            prefix=args.prefix,
        )
        if args.zip_output:
            create_zip_archive(exported, args.zip_output)
    else:
        generate_composite_plot(
            file_path=args.file,
            output_plot=args.output,
            start_frame=args.start_frame,
            frames_to_plot=args.frames,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
