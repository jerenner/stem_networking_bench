#!/usr/bin/env python3
"""Generate compact, scientifically traceable assets for the LMTO DOEELS demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import tomllib
from pathlib import Path

SIMULATION_ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#07131b"
PANEL = "#102630"
CREAM = "#f5f0e8"
MUTED = "#9bb0ba"
CYAN = "#35d0ba"
ORANGE = "#ffb000"
BLUE = "#4da3ff"
RED = "#ef6262"
ELEMENT_COLORS = {
    "Li": "#3c96ff",
    "Mn": "#41d764",
    "Ti": "#ff4b41",
    "O": "#ffdc28",
}
ELEMENT_ORDER = ("Li", "Mn", "Ti", "O")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("demo_config.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    cache = Path(tempfile.gettempdir()) / "eels-lmto-demo-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (SIMULATION_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = (
        "output_directory",
        "simulation_config",
        "spatial_response",
        "spectrum_image",
        "spectral_library",
        "physical_raw_scan",
        "exposure_sweep",
        "detector_response",
        "dark_calibration",
        "sparse_calibration",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing demo configuration keys: {', '.join(missing)}")
    return config


def _style_axis(axis, title: str | None = None) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#35505d")
    if title:
        axis.set_title(title, color=CREAM, fontsize=13, pad=8)


def _save(figure, path: Path, plt, dpi: int = 170) -> None:
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)


def _application_workflow_asset(output_dir: Path, plt, np) -> None:
    """Draw two clearly schematic end-use examples for a general audience."""
    from matplotlib.patches import Circle, Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)

    chip = axes[0]
    chip.set_xlim(0, 10)
    chip.set_ylim(0, 7)
    chip.set_aspect("equal")
    chip.add_patch(Rectangle((0.3, 0.4), 9.4, 2.0, color="#52616b", alpha=0.95))
    chip.add_patch(Rectangle((0.3, 2.4), 9.4, 0.55, color="#d8c843", alpha=0.85))
    chip.add_patch(Rectangle((3.65, 2.95), 2.7, 1.7, color="#e65a52", alpha=0.9))
    chip.add_patch(Rectangle((2.75, 2.95), 0.65, 2.6, color="#35d0ba", alpha=0.85))
    chip.add_patch(Rectangle((6.6, 2.95), 0.65, 2.6, color="#35d0ba", alpha=0.85))
    chip.add_patch(Rectangle((1.0, 5.55), 8.0, 0.65, color="#8a99a3", alpha=0.65))
    chip.add_patch(Rectangle((2.45, 2.65), 5.1, 3.2, fill=False, edgecolor=CREAM, lw=1.6, ls="--"))
    gx, gy = np.meshgrid(np.linspace(2.65, 7.35, 17), np.linspace(2.82, 5.65, 11))
    chip.scatter(gx, gy, s=3, color=CREAM, alpha=0.32)
    chip.text(
        5.0,
        6.55,
        "CHIP CROSS-SECTION",
        color=CREAM,
        ha="center",
        weight="bold",
        fontsize=14,
    )
    chip.text(5.0, 0.95, "Si substrate", color=CREAM, ha="center", fontsize=11)
    chip.text(5.0, 2.57, "oxygen-containing layer", color="#fff07d", ha="center", fontsize=9)
    chip.text(5.0, 3.75, "metal gate", color="#ffe0dc", ha="center", fontsize=10)
    chip.text(
        5.0,
        6.05,
        "scan a selected device region",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    chip.axis("off")

    battery = axes[1]
    battery.set_xlim(0, 10)
    battery.set_ylim(0, 7)
    battery.set_aspect("equal")
    particle = Circle((5.0, 3.45), 2.55, facecolor="#25343c", edgecolor=CREAM, lw=1.4)
    battery.add_patch(particle)
    rng = np.random.default_rng(20260903)
    for _ in range(34):
        radius = 2.15 * np.sqrt(rng.random())
        angle = 2 * np.pi * rng.random()
        x = 5.0 + radius * np.cos(angle)
        y = 3.45 + radius * np.sin(angle)
        color = rng.choice(
            [ELEMENT_COLORS["Mn"], ELEMENT_COLORS["Ti"], ELEMENT_COLORS["O"]],
            p=[0.27, 0.23, 0.50],
        )
        battery.add_patch(Circle((x, y), rng.uniform(0.12, 0.3), color=color, alpha=0.76))
    battery.add_patch(
        Rectangle((2.8, 1.35), 4.4, 4.2, fill=False, edgecolor=CREAM, lw=1.6, ls="--")
    )
    gx, gy = np.meshgrid(np.linspace(3.0, 7.0, 15), np.linspace(1.55, 5.35, 14))
    mask = (gx - 5.0) ** 2 + (gy - 3.45) ** 2 <= 2.3**2
    battery.scatter(gx[mask], gy[mask], s=3, color=CREAM, alpha=0.32)
    battery.text(
        5.0,
        6.55,
        "BATTERY PARTICLE",
        color=CREAM,
        ha="center",
        weight="bold",
        fontsize=14,
    )
    battery.text(
        5.0,
        0.45,
        "map Mn / Ti / O across a particle or interface",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    battery.axis("off")

    figure.suptitle(
        "End application: choose a region, scan it, and ask where each element is",
        color=CREAM,
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Workflow concept — not simulated chip or full-particle data",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    _save(figure, output_dir / "application_workflows.png", plt)


def _robust(image, np, lower: float = 0.02, upper: float = 0.995):
    values = np.asarray(image, dtype=np.float64)
    lo, hi = np.quantile(values[np.isfinite(values)], [lower, upper])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _periodic_smooth(image, sigma_pixels: float, np):
    values = np.asarray(image, dtype=np.float64)
    if sigma_pixels <= 0.0:
        return values
    fy = np.fft.fftfreq(values.shape[-2])
    fx = np.fft.fftfreq(values.shape[-1])
    transfer = np.exp(-2.0 * np.pi**2 * sigma_pixels**2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    return np.fft.ifft2(np.fft.fft2(values) * transfer).real


def _correlation(left, right, np) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def select_probe_points(element_maps: dict[str, object], np) -> dict[str, tuple[int, int]]:
    """Choose unique positions enriched for Mn, O, and Ti relative to peers."""
    names = ("Mn", "O", "Ti")
    standardized = {}
    for name in names:
        values = np.asarray(element_maps[name], dtype=np.float64)
        standardized[name] = (values - values.mean()) / max(values.std(), 1.0e-12)
    chosen: dict[str, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for name in names:
        others = sum(standardized[other] for other in names if other != name)
        score = standardized[name] - 0.35 * others
        for flat in np.argsort(score.ravel())[::-1]:
            point = tuple(int(value) for value in np.unravel_index(flat, score.shape))
            if point not in occupied:
                chosen[name] = point
                occupied.add(point)
                break
    return chosen


def _structure_assets(spatial, output_dir: Path, plt, np) -> dict:
    atoms = spatial["structure/atoms"]
    x = atoms["x_A"][:]
    y = atoms["y_A"][:]
    z = atoms["z_A"][:]
    element_id = atoms["element_id"][:]
    planes = atoms["plane_index"][:]
    summary = json.loads(spatial.attrs["summary_json"])
    extent = float(summary["field_of_view_A"])
    thickness = float(summary["sample_thickness_A"])

    figure = plt.figure(figsize=(11.8, 7.3), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor(BACKGROUND)
    sizes = {"Li": 70, "Mn": 105, "Ti": 100, "O": 62}
    for index, element in enumerate(ELEMENT_ORDER):
        mask = element_id == index
        axis.scatter(
            x[mask],
            y[mask],
            z[mask],
            s=sizes[element],
            c=ELEMENT_COLORS[element],
            label=element,
            depthshade=True,
            alpha=0.9,
            edgecolors="#eef2f7",
            linewidths=0.25,
        )
    axis.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="depth z (Å)")
    axis.set_xlim(0, extent)
    axis.set_ylim(0, extent)
    axis.set_zlim(0, thickness)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.zaxis.label.set_color(MUTED)
    axis.set_title(
        "All 128 atoms in the simulated periodic LMTO cell",
        color=CREAM,
        fontsize=18,
        pad=16,
    )
    axis.legend(facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM, loc="upper left")
    _save(figure, output_dir / "structure_3d.png", plt)

    unique_planes = np.unique(planes)
    figure, axes = plt.subplots(2, 4, figsize=(12.4, 6.5), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, plane in zip(axes.ravel(), unique_planes):
        plane_mask = planes == plane
        for index, element in enumerate(ELEMENT_ORDER):
            mask = plane_mask & (element_id == index)
            axis.scatter(
                x[mask],
                y[mask],
                s=sizes[element] * 0.72,
                c=ELEMENT_COLORS[element],
                edgecolors="#eef2f7",
                linewidths=0.25,
            )
        axis.set_xlim(-0.25, extent - 0.25)
        axis.set_ylim(-0.25, extent - 0.25)
        axis.set_aspect("equal")
        _style_axis(axis, f"plane {int(plane)} · z={float(z[plane_mask][0]):.3g} Å")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Eight alternating disordered-rocksalt planes · every simulated atom shown",
        color=CREAM,
        fontsize=17,
    )
    _save(figure, output_dir / "structure_planes.png", plt)
    counts = {
        element: int(np.count_nonzero(element_id == index))
        for index, element in enumerate(ELEMENT_ORDER)
    }
    return {
        "atom_count": len(x),
        "element_counts": counts,
        "extent_A": extent,
        "thickness_A": thickness,
    }


def _abtem_assets(spatial, output_dir: Path, plt, np, depth_index: int) -> dict:
    edge_names = json.loads(spatial.attrs["edge_names_json"])
    haadf = spatial["spatial/haadf_fraction"][depth_index]
    edges = spatial["spatial/edge_response_per_A2"][depth_index]
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.7), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    panels = [("HAADF 50–80 mrad", haadf, "gray")]
    edge_colors = {
        "Ti_M23": RED,
        "Mn_M23": "#53d878",
        "Li_K": BLUE,
        "Ti_L23": RED,
        "O_K": "#ffe04f",
        "Mn_L23": "#53d878",
    }
    for index, name in enumerate(edge_names):
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(name, [BACKGROUND, edge_colors[name]])
        panels.append((name.replace("_", " "), edges[index], cmap))
    panels.append(("", np.zeros_like(haadf), "gray"))
    for axis, (title, values, cmap) in zip(axes.ravel(), panels):
        if not title:
            axis.axis("off")
            continue
        axis.imshow(
            _robust(values, np),
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="bicubic",
        )
        _style_axis(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "abTEM supplies probe channeling and relative spatial contrast",
        color=CREAM,
        fontsize=18,
    )
    _save(figure, output_dir / "abtem_spatial_channels.png", plt)
    metadata = json.loads(spatial.attrs["metadata_json"])
    return {
        "edge_names": edge_names,
        "abtem_version": metadata.get("abtem_version"),
        "probe_semiangle_mrad": metadata.get("probe_semiangle_mrad"),
        "haadf_detector_mrad": metadata.get("haadf_detector_mrad"),
        "eels_detector_mrad": metadata.get("eels_detector_mrad"),
        "potential_sampling_A": metadata.get("potential_sampling_A"),
        "haadf_sampling_A": metadata.get("haadf_refinement", {}).get("potential_sampling_actual_A"),
        "double_channel_edges": json.loads(spatial.attrs["summary_json"])["double_channel_edges"],
    }


def _spectral_assets(library, spectrum, output_dir: Path, plt, np, depth_index: int) -> dict:
    energy = spectrum["axes/energy_centers_eV"][:]
    profiles = spectrum["spectrum/component_profiles"][:]
    labels = json.loads(spectrum["spectrum"].attrs["component_labels_json"])
    expected = spectrum["spectrum/expected_component_counts"][depth_index].mean(axis=(0, 1))
    contributions = expected[:, None] * profiles
    colors = {
        "zero_loss": CREAM,
        "low_loss": ORANGE,
        "Ti_M23": RED,
        "Mn_M23": "#53d878",
        "Li_K": BLUE,
        "Ti_L23": "#ff7d74",
        "O_K": "#ffe04f",
        "Mn_L23": "#86ec9c",
    }

    figure, axes = plt.subplots(2, 1, figsize=(12.4, 7.0), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, limits, title in (
        (axes[0], (28, 90), "Overlapping shallow edges"),
        (
            axes[1],
            (420, 700),
            "Localized edges selected for the final Mn / O / Ti maps",
        ),
    ):
        for index, label in enumerate(labels[2:], start=2):
            axis.plot(
                energy,
                contributions[index],
                color=colors[label],
                lw=1.7,
                label=label.replace("_", " "),
            )
        axis.set_xlim(*limits)
        axis.set_yscale("log")
        axis.set_ylim(
            bottom=max(
                1e-6,
                min(v[v > 0].min() for v in contributions[2:] if np.any(v > 0)) * 0.4,
            )
        )
        _style_axis(axis, title)
        axis.set_ylabel("expected electrons / eV / probe", color=MUTED)
        axis.grid(color="#35505d", alpha=0.22)
    axes[1].set_xlabel("energy loss ΔE (eV)", color=MUTED)
    axes[0].legend(ncol=3, fontsize=8, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    figure.suptitle(
        "GOSH supplies energy-dependent atomic cross sections", color=CREAM, fontsize=18
    )
    _save(figure, output_dir / "gosh_edge_components.png", plt)

    figure, axis = plt.subplots(figsize=(12.4, 5.6), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    total = contributions.sum(axis=0)
    floor = max(float(total[total > 0].min()), 1e-6)
    axis.semilogy(
        energy,
        np.maximum(total, floor),
        color=CREAM,
        lw=1.8,
        label="total expected spectrum",
    )
    for index, label in enumerate(labels):
        if label == "zero_loss":
            continue
        axis.semilogy(
            energy,
            np.maximum(contributions[index], floor * 0.15),
            color=colors[label],
            lw=1.0,
            alpha=0.82,
            label=label.replace("_", " "),
        )
    for value, label, color in (
        (35, "Ti M", RED),
        (51, "Mn M", "#53d878"),
        (55, "Li K", BLUE),
        (456, "Ti L", RED),
        (532, "O K", "#ffe04f"),
        (640, "Mn L", "#53d878"),
    ):
        axis.axvline(value, color=color, alpha=0.45, lw=0.9)
    axis.set_xlim(-2, 790)
    axis.set_xlabel("energy loss ΔE (eV)", color=MUTED)
    axis.set_ylabel("expected electrons / eV / probe", color=MUTED)
    _style_axis(
        axis,
        "Average theoretical spectrum assembled from ZLP + low loss + six core edges",
    )
    axis.grid(color="#35505d", alpha=0.2)
    axis.legend(ncol=4, fontsize=7, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    _save(figure, output_dir / "theoretical_spectrum_components.png", plt)

    library_metadata = json.loads(library.attrs["metadata_json"])
    edges = {}
    for name, group in library["core_loss"].items():
        edge_metadata = json.loads(group.attrs["metadata_json"])
        edges[name] = {
            "element": str(group.attrs["element"]),
            "onset_energy_eV": float(group.attrs["onset_energy_eV"]),
            "integrated_cross_section_barn_per_atom": edge_metadata[
                "integrated_cross_section_barn_per_atom"
            ],
            "material_specific_fine_structure": edge_metadata["material_specific_fine_structure"],
        }
    return {
        "component_labels": labels,
        "edges": edges,
        "gosh_doi": library_metadata["gosh_database"]["doi"],
        "core_loss_physics": library_metadata["core_loss_physics"],
        "low_loss_physics": library_metadata["low_loss_physics"],
    }


def _probe_grid_asset(
    sweep, expected_maps, points, output_dir: Path, plt, np, exposure_index: int
) -> None:
    x = sweep["axes/scan_x_A"][:]
    y = sweep["axes/scan_y_A"][:]
    haadf = sweep["reconstruction/haadf_expected_counts"][exposure_index, 0]
    step_x = float(np.median(np.diff(x)))
    step_y = float(np.median(np.diff(y)))
    extent = [
        x[0] - step_x / 2,
        x[-1] + step_x / 2,
        y[0] - step_y / 2,
        y[-1] + step_y / 2,
    ]
    figure, axis = plt.subplots(figsize=(8.7, 7.2), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    axis.imshow(
        _robust(haadf, np),
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bicubic",
    )
    grid_x, grid_y = np.meshgrid(x, y)
    axis.scatter(
        grid_x.ravel(),
        grid_y.ravel(),
        s=7,
        facecolors="none",
        edgecolors=CYAN,
        linewidths=0.45,
        alpha=0.72,
    )
    for element, (row, column) in points.items():
        axis.scatter(
            x[column],
            y[row],
            s=145,
            facecolors="none",
            edgecolors=ELEMENT_COLORS[element],
            linewidths=2.2,
        )
        axis.text(
            x[column] + 0.15,
            y[row] + 0.15,
            f"{element}-rich probe",
            color=ELEMENT_COLORS[element],
            fontsize=10,
            weight="bold",
        )
    axis.set_xlabel("scan x (Å)", color=MUTED)
    axis.set_ylabel("scan y (Å)", color=MUTED)
    _style_axis(axis, "16 × 16 probe raster over the independently calculated 0.83 nm field")
    _save(figure, output_dir / "probe_raster.png", plt)


def _haadf_asset(sweep, output_dir: Path, plt, np, exposure_index: int) -> dict:
    expected = sweep["reconstruction/haadf_expected_counts"][exposure_index, 0]
    sampled = sweep["reconstruction/haadf_counts"][exposure_index, 0]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.7), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image, title in (
        (axes[0], expected, "abTEM annular expectation"),
        (axes[1], sampled, "Poisson-sampled 35 ms HAADF"),
    ):
        axis.imshow(_robust(image, np), origin="lower", cmap="gray", interpolation="bicubic")
        _style_axis(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("HAADF is a separate 50–80 mrad detector branch", color=CREAM, fontsize=17)
    _save(figure, output_dir / "haadf_model.png", plt)
    return {"expected_sampled_correlation": _correlation(expected, sampled, np)}


def _detector_response_asset(response, output_dir: Path, plt, np, beam_energy_keV: float) -> dict:
    templates = response["event_spatial_pairs"][:]
    totals = response["event_total_pairs"][:]
    mean = response["mean_spatial_fraction"][:]
    chosen = [
        int(np.argmin(totals)),
        int(np.argsort(totals)[len(totals) // 2]),
        int(np.argmax(totals)),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(13.0, 3.2), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, index, label in zip(
        axes[:3], chosen, ("low deposit", "median deposit", "high deposit")
    ):
        axis.imshow(templates[index], origin="lower", cmap="magma", interpolation="nearest")
        _style_axis(axis, f"{label}\n{int(totals[index])} e–h pairs")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[3].imshow(mean, origin="lower", cmap="magma", interpolation="nearest")
    _style_axis(axes[3], "mean 7 × 7 charge cloud")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[4].hist(totals, bins=55, color=ORANGE, alpha=0.88)
    axes[4].axvline(np.mean(totals), color=CREAM, lw=1.2, label=f"mean {np.mean(totals):.0f}")
    _style_axis(axes[4], "event-to-event deposited charge")
    axes[4].set_xlabel("e–h pairs", color=MUTED)
    axes[4].set_ylabel("events", color=MUTED)
    axes[4].legend(fontsize=8, facecolor=PANEL, labelcolor=CREAM)
    figure.suptitle(
        f"Geant4 {beam_energy_keV:g} keV electrons in 5 µm silicon → collected charge templates",
        color=CREAM,
        fontsize=17,
    )
    _save(figure, output_dir / "geant4_detector_response.png", plt)
    summary = json.loads(response.attrs["summary_json"])
    return summary


def _calibration_assets(dark, sparse, config: dict, output_dir: Path, plt, np) -> dict:
    dark_summary = json.loads(dark.attrs["summary_json"])
    if "input" in dark_summary:
        dark_summary["input"] = Path(str(dark_summary["input"])).name
    sparse_summary = json.loads(sparse.attrs["summary_json"])
    pedestal = dark["maps/pedestal_adu"][:]
    noise = dark["maps/read_noise_adu"][:]
    bias = dark["maps/validation_bias_adu"][:]
    measured_patch = sparse["measured/empirical_signal_mean_patch_adu"][:]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.5), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    panels = (
        (axes[0, 0], pedestal, "NiO dark: pedestal map", "viridis", None),
        (axes[0, 1], noise, "NiO dark: temporal-noise RMS", "viridis", None),
        (axes[1, 0], bias, "held-out mean − pedestal", "coolwarm", "sym"),
        (axes[1, 1], measured_patch, "mean isolated high-loss event", "magma", None),
    )
    for axis, data, title, cmap, mode in panels:
        if mode == "sym":
            limit = float(np.quantile(np.abs(data[np.isfinite(data)]), 0.995))
            axis.imshow(data, cmap=cmap, aspect="auto", vmin=-limit, vmax=limit)
        else:
            lo, hi = np.quantile(data[np.isfinite(data)], [0.01, 0.99])
            axis.imshow(data, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        _style_axis(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Measured NiO data constrain electronics and effective single-electron response",
        color=CREAM,
        fontsize=17,
    )
    _save(figure, output_dir / "nio_calibration_summary.png", plt)

    copies = (
        (
            resolve_path("calibration/eels_15pa_dark_calibration.png"),
            "eels_15pa_dark_calibration.png",
        ),
        (
            resolve_path(config["sparse_calibration"]).with_suffix(".png"),
            "nio_sparse_detector_calibration.png",
        ),
    )
    for source, output_name in copies:
        if source.exists():
            shutil.copy2(source, output_dir / output_name)
    return {"dark": dark_summary, "sparse": sparse_summary}


def _raw_frame_assets(raw_scan, dark, points, output_dir: Path, plt, np) -> dict:
    row, column = points["Mn"]
    frame_y = raw_scan["scan/frame_y_index"][:]
    frame_x = raw_scan["scan/frame_x_index"][:]
    candidates = np.flatnonzero((frame_y == row) & (frame_x == column))
    if len(candidates) == 0:
        raise ValueError("raw scan does not contain the selected Mn-rich point")
    frame_index = int(candidates[0])
    raw = raw_scan["frames/raw"][frame_index].astype(np.float64)
    pedestal = dark["maps/pedestal_adu"][:]
    residual = raw - pedestal
    residual_scale = max(float(np.quantile(np.abs(residual), 0.995)), 1.0)
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 6.3), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    lo, hi = np.quantile(raw, [0.005, 0.995])
    axes[0].imshow(raw, cmap="viridis", aspect="auto", vmin=lo, vmax=hi)
    _style_axis(axes[0], "one simulated physical 11.49 µs DOEELS frame · raw 12-bit ADC codes")
    axes[1].imshow(
        np.arcsinh(residual / max(np.std(residual), 1.0)),
        cmap="coolwarm",
        aspect="auto",
        vmin=-3,
        vmax=3,
    )
    _style_axis(
        axes[1],
        "same frame after subtracting the measured NiO pedestal map · asinh display",
    )
    for axis in axes:
        axis.set_xlabel("raw detector column", color=MUTED)
        axis.set_ylabel("row", color=MUTED)
        axis.axvline(768, color=ORANGE, lw=1.0, alpha=0.9)
    _save(figure, output_dir / "raw_detector_frame.png", plt)

    hit_counts = raw_scan["truth/raw_column_hit_counts"][frame_index]
    collapsed = raw_scan["reconstruction/collapsed_signal_adu"][frame_index]
    figure, axis = plt.subplots(figsize=(12.3, 4.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    x = np.arange(len(hit_counts))
    axis.plot(x, hit_counts, color=CYAN, lw=1.0, label="incident electron hits by raw column")
    scale = np.nanmax(hit_counts) / max(np.nanmax(np.abs(collapsed)), 1.0)
    axis.plot(
        x,
        np.maximum(collapsed * scale, 0),
        color=ORANGE,
        lw=0.8,
        alpha=0.72,
        label="row-profile estimate from ADC frame (scaled)",
    )
    axis.axvspan(0, 768, color=CYAN, alpha=0.08)
    axis.axvline(768, color=CREAM, lw=1.0)
    for center in (52, 244, 436, 628):
        axis.axvline(center, color=BLUE, lw=0.65, alpha=0.55)
    axis.text(30, axis.get_ylim()[1] * 0.88, "four ZLP lanes", color=CYAN)
    axis.text(805, axis.get_ylim()[1] * 0.88, "CoreLoss", color=ORANGE)
    axis.set_xlim(0, 3840)
    axis.set_xlabel("raw detector column", color=MUTED)
    axis.set_ylabel("relative response", color=MUTED)
    _style_axis(axis, "Spectrometer layout visible in one raw frame")
    axis.legend(fontsize=8, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    _save(figure, output_dir / "raw_column_layout.png", plt)
    return {
        "frame_index": frame_index,
        "probe": [row, column],
        "residual_display_scale_adu": residual_scale,
    }


def _processing_assets(
    sweep, spectrum, points, output_dir: Path, plt, np, exposure_index: int
) -> dict:
    energy = sweep["axes/energy_centers_eV"][:]
    integrations = int(sweep["axes/integrations_per_position"][exposure_index])
    labels = json.loads(spectrum["spectrum"].attrs["component_labels_json"])
    profiles = spectrum["spectrum/component_profiles"][:]

    element, (example_row, example_column) = "Mn", points["Mn"]
    pre = sweep["reconstruction/pre_readout_energy_counts"][
        exposure_index, 0, example_row, example_column
    ]
    analog = sweep["reconstruction/energy_counts"][exposure_index, 0, example_row, example_column]
    counted = sweep["reconstruction/counted_energy_counts"][
        exposure_index, 0, example_row, example_column
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    low = energy < 36.08
    axes[0].plot(energy[low], pre[low], color=CYAN, lw=1.2, label="pre-readout electrons")
    axes[0].plot(
        energy[low],
        analog[low],
        color=ORANGE,
        lw=0.9,
        alpha=0.8,
        label="integrating ADC reconstruction",
    )
    axes[0].axvline(36.08, color=CREAM, ls="--", lw=1.0)
    axes[0].set_xlim(-2, 38)
    axes[0].set_yscale("symlog", linthresh=10)
    _style_axis(axes[0], "Zero-loss and low-loss: integrated signal")
    core = (energy >= 420) & (energy <= 700)
    core_energy = energy[core]
    rebin_channels = 4
    usable_channels = core_energy.size - core_energy.size % rebin_channels
    rebinned_energy = core_energy[:usable_channels].reshape(-1, rebin_channels).mean(axis=1)
    rebinned_counted = counted[core][:usable_channels].reshape(-1, rebin_channels).sum(axis=1)
    rebin_width_eV = float(np.median(np.diff(energy)) * rebin_channels)
    axes[1].step(
        rebinned_energy,
        rebinned_counted,
        where="mid",
        color=CREAM,
        lw=1.2,
        alpha=0.88,
        label=f"counted data ({rebin_width_eV:.1f} eV bins)",
    )
    component_counts = sweep["reconstruction/counted_component_counts"][
        exposure_index, 0, example_row, example_column
    ]
    example_component_counts = component_counts.copy()
    focus = (
        ("Ti_L23", "Ti L₂,₃", RED),
        ("O_K", "O K", "#ffe04f"),
        ("Mn_L23", "Mn L₂,₃", "#53d878"),
    )
    fitted_total = np.zeros_like(energy, dtype=np.float64)
    for component, label, color in focus:
        index = labels.index(component)
        curve = component_counts[index] * profiles[index]
        fitted_total += curve
        rebinned_curve = curve[core][:usable_channels].reshape(-1, rebin_channels).sum(axis=1)
        axes[1].plot(rebinned_energy, rebinned_curve, color=color, lw=1.8, label=f"{label} fit")
    rebinned_fitted_total = (
        fitted_total[core][:usable_channels].reshape(-1, rebin_channels).sum(axis=1)
    )
    axes[1].plot(
        rebinned_energy,
        rebinned_fitted_total,
        color=CYAN,
        lw=1.3,
        ls="--",
        label="combined fit",
    )
    ymax = max(float(np.max(rebinned_counted)), float(np.max(rebinned_fitted_total)), 1.0)
    ymin = min(float(np.min(rebinned_counted)), -1.0)
    axes[1].set_ylim(ymin * 1.12, ymax * 1.25)
    for value, label, color in (
        (456, "Ti begins", RED),
        (532, "O begins", "#ffe04f"),
        (640, "Mn begins", "#53d878"),
    ):
        axes[1].axvline(value, color=color, lw=0.9, alpha=0.65)
        axes[1].text(value + 3, ymax * 1.04, label, color=color, fontsize=8)
    _style_axis(axes[1], "Core-loss: count-preserving energy rebin for readability")
    for axis in axes:
        axis.set_xlabel("energy loss ΔE (eV)", color=MUTED)
        axis.grid(color="#35505d", alpha=0.18)
        axis.legend(fontsize=8, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    axes[0].set_ylabel("reconstructed counts", color=MUTED)
    axes[1].set_ylabel(f"counts per {rebin_width_eV:.1f} eV bin", color=MUTED)
    figure.suptitle(
        f"Reconstructed spectrum at one {element}-rich probe after {integrations:,} detector frames",
        color=CREAM,
        fontsize=17,
    )
    _save(figure, output_dir / "analog_vs_counted_processing.png", plt)

    figure, axes = plt.subplots(3, 1, figsize=(12.5, 8.2), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    probe_colors = {"Mn": "#53d878", "O": "#ffe04f", "Ti": RED}
    focus_components = {"Ti": "Ti_L23", "O": "O_K", "Mn": "Mn_L23"}
    for axis, focus_element in zip(axes, ("Ti", "O", "Mn")):
        component_index = labels.index(focus_components[focus_element])
        for probe_element, (row, column) in points.items():
            component_counts = sweep["reconstruction/counted_component_counts"][
                exposure_index, 0, row, column
            ]
            core_model = component_counts[component_index] * profiles[component_index]
            axis.plot(
                energy,
                core_model,
                color=probe_colors[probe_element],
                lw=1.2,
                label=f"{probe_element}-rich probe",
            )
        onset = {"Ti": 456, "O": 532, "Mn": 640}[focus_element]
        axis.axvline(onset, color=probe_colors[focus_element], lw=1.0, ls="--")
        axis.set_xlim(max(420, onset - 35), min(700, onset + 90))
        _style_axis(axis, f"{focus_element} selection from fitted transferred edge profiles")
        axis.set_ylabel("fitted edge counts/eV", color=MUTED)
        axis.grid(color="#35505d", alpha=0.18)
    axes[-1].set_xlabel("energy loss ΔE (eV)", color=MUTED)
    axes[0].legend(ncol=3, fontsize=8, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    figure.suptitle(
        "The same spectrum contains different edge amplitudes at different probe positions",
        color=CREAM,
        fontsize=17,
    )
    _save(figure, output_dir / "probe_point_coreloss_spectra.png", plt)
    return {
        "probe": element,
        "row": int(example_row),
        "column": int(example_column),
        "fitted_edge_counts": {
            label: float(example_component_counts[labels.index(component)])
            for component, label, _ in focus
        },
        "coreloss_rebin_channels": rebin_channels,
        "coreloss_rebin_eV": rebin_width_eV,
    }


def _exposure_asset(sweep, output_dir: Path, plt, np) -> dict:
    dwell = sweep["axes/dwell_ms"][:]
    figure, axis = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    colors = {"Mn": "#53d878", "O": "#ffe04f", "Ti": RED, "Li": BLUE}
    metrics = {}
    for element in ("Li", "Mn", "O", "Ti"):
        values = sweep[f"reconstruction/elements/{element}/counted_correlation"][:, 0]
        metrics[element] = values.tolist()
        axis.plot(dwell, values, "o-", color=colors[element], lw=1.5, ms=4, label=element)
    axis.axvline(34.48275862068966, color=CREAM, ls="--", lw=1.1)
    axis.text(38, 0.06, "chosen demo dwell", color=CREAM, fontsize=9)
    axis.set_xscale("log")
    axis.set_ylim(-0.12, 1.04)
    axis.set_xlabel("dwell per probe position (ms)", color=MUTED)
    axis.set_ylabel("correlation with expected map", color=MUTED)
    _style_axis(axis, "Exposure sweep: when do element maps become recognizable?")
    axis.grid(color="#35505d", alpha=0.22)
    axis.legend(ncol=4, facecolor=PANEL, edgecolor="#35505d", labelcolor=CREAM)
    _save(figure, output_dir / "exposure_sweep_correlations.png", plt)
    return metrics


def _final_channel_assets(
    sweep,
    output_dir: Path,
    plt,
    np,
    exposure_index: int,
    sigma_pixels: float,
    elements: list[str],
) -> dict:
    from matplotlib.colors import LinearSegmentedColormap

    channels = {"HAADF": sweep["reconstruction/haadf_counts"][exposure_index, 0]}
    correlations = {}
    for element in elements:
        channels[element] = sweep[f"reconstruction/elements/{element}/counted_counts"][
            exposure_index, 0
        ]
        correlations[element] = float(
            sweep[f"reconstruction/elements/{element}/counted_correlation"][exposure_index, 0]
        )
    display_colors = {"HAADF": "#ffffff", **ELEMENT_COLORS}
    for name, values in channels.items():
        shown = _robust(_periodic_smooth(values, sigma_pixels, np), np) ** 0.75
        cmap = LinearSegmentedColormap.from_list(name, ["#000000", display_colors[name]])
        figure, axis = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
        figure.patch.set_facecolor(BACKGROUND)
        axis.imshow(shown, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="bicubic")
        title = (
            name
            if name == "HAADF"
            else f"{name} counted reconstruction · r={correlations[name]:.3f}"
        )
        _style_axis(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
        _save(figure, output_dir / f"final_{name.lower()}.png", plt, dpi=190)
    return correlations


def generate_assets(config_path: Path, force: bool = False) -> Path:
    configure_matplotlib()
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    import matplotlib.pyplot as plt
    import numpy as np

    config_path = config_path.resolve()
    config = load_config(config_path)
    output_dir = resolve_path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "demo_metadata.json"
    if manifest.exists() and not force:
        print(f"Assets already exist at {output_dir} (use --force to rebuild)")
        return output_dir

    paths = {
        name: resolve_path(value)
        for name, value in config.items()
        if name
        in {
            "simulation_config",
            "spatial_response",
            "spectrum_image",
            "spectral_library",
            "physical_raw_scan",
            "exposure_sweep",
            "detector_response",
            "dark_calibration",
            "sparse_calibration",
        }
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing demo input(s): " + ", ".join(missing))
    with paths["simulation_config"].open("rb") as stream:
        simulation_config = tomllib.load(stream)

    with (
        h5py.File(paths["spatial_response"], "r") as spatial,
        h5py.File(paths["spectrum_image"], "r") as spectrum,
        h5py.File(paths["spectral_library"], "r") as library,
        h5py.File(paths["physical_raw_scan"], "r") as raw_scan,
        h5py.File(paths["exposure_sweep"], "r") as sweep,
        h5py.File(paths["detector_response"], "r") as response,
        h5py.File(paths["dark_calibration"], "r") as dark,
        h5py.File(paths["sparse_calibration"], "r") as sparse,
    ):
        expected_schemas = {
            spatial: "eels-sim-abtem-spatial-v1",
            spectrum: "eels-sim-spectrum-image-v1",
            library: "eels-sim-spectral-library-v1",
            raw_scan: "eels-sim-raw-doeels-scan-v1",
            sweep: "eels-sim-raw-doeels-exposure-sweep-v1",
            response: "eels-sim-monte-carlo-response-kernel-v1",
            dark: "eels-sim-readout-calibration-v2",
            sparse: "eels-sim-nio-detector-calibration-v1",
        }
        for handle, schema in expected_schemas.items():
            if handle.attrs.get("schema") != schema:
                raise ValueError(
                    f"{handle.filename} has schema {handle.attrs.get('schema')!r}, expected {schema!r}"
                )
        requested = int(config.get("exposure_integrations", 3000))
        exposures = sweep["axes/integrations_per_position"][:]
        matches = np.flatnonzero(exposures == requested)
        if not len(matches):
            raise ValueError(f"exposure sweep has no {requested}-integration checkpoint")
        exposure_index = int(matches[0])
        depth_index = int(sweep["axes/depth_index"][0])
        expected_maps = {
            element: sweep[f"reconstruction/elements/{element}/expected_counts"][exposure_index, 0]
            for element in ("Mn", "O", "Ti")
        }
        points = select_probe_points(expected_maps, np)

        _application_workflow_asset(output_dir, plt, np)
        structure_summary = _structure_assets(spatial, output_dir, plt, np)
        abtem_summary = _abtem_assets(spatial, output_dir, plt, np, depth_index)
        spectral_summary = _spectral_assets(library, spectrum, output_dir, plt, np, depth_index)
        _probe_grid_asset(sweep, expected_maps, points, output_dir, plt, np, exposure_index)
        haadf_summary = _haadf_asset(sweep, output_dir, plt, np, exposure_index)
        beam_energy_keV = float(simulation_config["experiment"]["beam_energy_keV"])
        detector_summary = _detector_response_asset(response, output_dir, plt, np, beam_energy_keV)
        calibration_summary = _calibration_assets(dark, sparse, config, output_dir, plt, np)
        raw_summary = _raw_frame_assets(raw_scan, dark, points, output_dir, plt, np)
        spectrum_example = _processing_assets(
            sweep, spectrum, points, output_dir, plt, np, exposure_index
        )
        exposure_metrics = _exposure_asset(sweep, output_dir, plt, np)
        final_correlations = _final_channel_assets(
            sweep,
            output_dir,
            plt,
            np,
            exposure_index,
            float(config.get("display_smoothing_pixels", 0.6)),
            list(config.get("final_elements", ["Mn", "O", "Ti"])),
        )

        scan_x = sweep["axes/scan_x_A"][:]
        scan_y = sweep["axes/scan_y_A"][:]
        dwell_ms = float(sweep["axes/dwell_ms"][exposure_index])
        frame_rate = float(simulation_config["experiment"]["frame_rate_hz"])
        electrons_per_integration = int(
            simulation_config["raw_doeels"]["electrons_per_integration"]
        )
        probe_metadata = {}
        for element, (row, column) in points.items():
            probe_metadata[element] = {
                "row": row,
                "column": column,
                "x_A": float(scan_x[column]),
                "y_A": float(scan_y[row]),
                "expected_element_counts": {
                    name: float(values[row, column]) for name, values in expected_maps.items()
                },
            }
        metadata = {
            "schema": "eels-sim-lmto-demo-v1",
            "title": config.get("title"),
            "source_files": {name: str(config[name]) for name in paths},
            "specimen": {
                "formula": "Li1.2Mn0.4Ti0.4O2",
                "model": "synthetic unrelaxed random-cation disordered rocksalt",
                **structure_summary,
            },
            "scan": {
                "shape": [len(scan_y), len(scan_x)],
                "step_A": float(np.median(np.diff(scan_x))),
                "field_of_view_A": float(scan_x[-1] - scan_x[0] + np.median(np.diff(scan_x))),
                "beam_energy_keV": float(simulation_config["experiment"]["beam_energy_keV"]),
                "beam_current_pA": float(simulation_config["experiment"]["beam_current_pA"]),
                "frame_rate_hz": frame_rate,
                "integrations_per_position": requested,
                "dwell_ms": dwell_ms,
                "electrons_per_integration": electrons_per_integration,
                "electrons_per_position": requested * electrons_per_integration,
                "ideal_scan_time_s": len(scan_x) * len(scan_y) * dwell_ms / 1000.0,
                "raw_frames_if_retained": len(scan_x) * len(scan_y) * requested,
                "focal_depth_A": float(sweep["axes/focal_depth_A"][0]),
            },
            "abtem": abtem_summary,
            "spectral_model": spectral_summary,
            "spectrometer": {
                "model": "first-order paraxial transfer, not a field solve",
                "dispersion_eV_per_column": float(
                    simulation_config["spectrometer"]["dispersion_eV_per_column"]
                ),
                "zlp_stitched_column": float(
                    simulation_config["spectrometer"]["zero_loss_stitched_column"]
                ),
                "zlp_lanes": int(simulation_config["spectrometer"]["zlp_repeats"]),
                "zlp_lane_width_columns": int(
                    simulation_config["spectrometer"]["zlp_lane_width_columns"]
                ),
                "collection_semiangle_mrad": float(
                    simulation_config["spectrometer"]["collection_semiangle_mrad"]
                ),
                "energy_blur_sigma_eV": float(
                    simulation_config["spectrometer"]["energy_blur_sigma_eV"]
                ),
            },
            "haadf": haadf_summary,
            "detector": detector_summary,
            "calibration": calibration_summary,
            "selected_probes": probe_metadata,
            "raw_frame": raw_summary,
            "spectrum_example": spectrum_example,
            "exposure_metrics": exposure_metrics,
            "final_correlations": final_correlations,
            "limitations": [
                "The LMTO cell is synthetic, periodic, unrelaxed, and only 0.83 nm wide by 1.66 nm thick.",
                "GOSH supplies independent-atom cross sections, not LMTO-specific ELNES, oxidation state, or bonding fine structure.",
                "The spectrometer is a provisional first-order transfer with provisional dispersion and row-angle mapping.",
                "The 5 um silicon thickness and effective gain remain partly degenerate in the NiO fit.",
                "The 35 ms streaming product retains sufficient statistics and spectra, not all 768,000 raw frames.",
                "Focal sections are separate through-focus simulations, not depth recovered from one conventional scan.",
                "Li K is not used in the final slider because shallow-edge overlap, delocalization, and low spatial modulation make it poorly resolved at this dwell.",
            ],
            "assets": sorted(path.name for path in output_dir.glob("*.png")),
        }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return output_dir


def main() -> None:
    args = parse_args()
    generate_assets(args.config, force=args.force)


if __name__ == "__main__":
    main()
