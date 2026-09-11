from __future__ import annotations

import json
from base64 import b64encode
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import ScanConfig, load_config
from .spectral_library import SpectralDistribution, read_spectral_library

SPECTRUM_IMAGE_SCHEMA = "eels-sim-spectrum-image-v1"
ELEMENTS = ("Li", "Mn", "Ti", "O")
ATOMIC_NUMBERS = {"Li": 3, "Mn": 25, "Ti": 22, "O": 8}
ELEMENT_COLORS = {
    "Li": (60, 150, 255),
    "Mn": (65, 215, 100),
    "Ti": (255, 75, 65),
    "O": (255, 220, 40),
}
BARN_TO_A2 = 1.0e-8


def _validate_scan_config(config: ScanConfig) -> None:
    positive_integer_fields = (
        "scan_pixels",
        "lateral_cells",
        "depth_atomic_planes",
        "depth_sections",
        "electrons_per_probe",
    )
    for name in positive_integer_fields:
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"scan.{name} must be positive")
    positive_float_fields = (
        "lattice_constant_A",
        "probe_sigma_A",
        "axial_sigma_A",
        "energy_step_eV",
        "haadf_peak_counts",
    )
    for name in positive_float_fields:
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"scan.{name} must be positive")
    if config.energy_max_eV <= config.energy_min_eV:
        raise ValueError("scan energy range is invalid")
    if config.model != "synthetic_disordered_rocksalt":
        raise ValueError(f"unsupported scan model {config.model!r}")
    if config.spatial_provider not in {"analytic", "abtem_cache"}:
        raise ValueError(f"unsupported scan.spatial_provider {config.spatial_provider!r}")
    if config.spatial_provider == "abtem_cache" and not config.spatial_response_file:
        raise ValueError("scan.spatial_response_file is required for the abtem_cache provider")


def build_disordered_rocksalt(
    config: ScanConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Build alternating oxygen/cation planes with random LMTO cations."""
    _validate_scan_config(config)
    half_lattice_A = 0.5 * config.lattice_constant_A
    sites_per_axis = 2 * config.lateral_cells
    extent_A = sites_per_axis * half_lattice_A
    records: dict[str, list[float | int]] = {
        "x_A": [],
        "y_A": [],
        "z_A": [],
        "element_id": [],
        "atomic_number": [],
        "plane_index": [],
    }
    element_to_id = {element: index for index, element in enumerate(ELEMENTS)}
    occupancy = np.zeros(
        (
            config.depth_atomic_planes,
            len(ELEMENTS),
            config.scan_pixels,
            config.scan_pixels,
        ),
        dtype=np.float64,
    )
    pixel_size_A = extent_A / config.scan_pixels

    for plane in range(config.depth_atomic_planes):
        cation_sites = [
            (ix, iy)
            for iy in range(sites_per_axis)
            for ix in range(sites_per_axis)
            if (ix + iy + plane) % 2 == 0
        ]
        cation_count = len(cation_sites)
        li_count = int(round(0.6 * cation_count))
        mn_count = int(round(0.2 * cation_count))
        ti_count = cation_count - li_count - mn_count
        cations = np.asarray(["Li"] * li_count + ["Mn"] * mn_count + ["Ti"] * ti_count)
        rng.shuffle(cations)
        cation_lookup = dict(zip(cation_sites, cations))

        for iy in range(sites_per_axis):
            for ix in range(sites_per_axis):
                element = cation_lookup.get((ix, iy), "O")
                x_A = ix * half_lattice_A
                y_A = iy * half_lattice_A
                z_A = plane * half_lattice_A
                records["x_A"].append(x_A)
                records["y_A"].append(y_A)
                records["z_A"].append(z_A)
                records["element_id"].append(element_to_id[element])
                records["atomic_number"].append(ATOMIC_NUMBERS[element])
                records["plane_index"].append(plane)
                pixel_x = int(round(x_A / pixel_size_A)) % config.scan_pixels
                pixel_y = int(round(y_A / pixel_size_A)) % config.scan_pixels
                occupancy[plane, element_to_id[element], pixel_y, pixel_x] += 1.0

    atoms = {
        "x_A": np.asarray(records["x_A"], dtype=np.float32),
        "y_A": np.asarray(records["y_A"], dtype=np.float32),
        "z_A": np.asarray(records["z_A"], dtype=np.float32),
        "element_id": np.asarray(records["element_id"], dtype=np.uint8),
        "atomic_number": np.asarray(records["atomic_number"], dtype=np.uint8),
        "plane_index": np.asarray(records["plane_index"], dtype=np.uint16),
    }
    return atoms, occupancy, extent_A


def _periodic_gaussian_density(
    image: np.ndarray,
    sigma_A: float,
    pixel_size_A: float,
) -> np.ndarray:
    rows, columns = image.shape[-2:]
    frequency_y = np.fft.fftfreq(rows, d=pixel_size_A)
    frequency_x = np.fft.fftfreq(columns, d=pixel_size_A)
    squared_frequency = frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2
    transfer = np.exp(-2.0 * np.pi**2 * sigma_A**2 * squared_frequency)
    blurred = np.fft.ifft2(np.fft.fft2(image, axes=(-2, -1)) * transfer).real
    blurred = np.clip(blurred, 0.0, None)
    source_total = image.sum(axis=(-2, -1), keepdims=True)
    blurred_total = blurred.sum(axis=(-2, -1), keepdims=True)
    blurred *= np.divide(
        source_total,
        blurred_total,
        out=np.zeros_like(blurred_total),
        where=blurred_total > 0.0,
    )
    return blurred / pixel_size_A**2


def _periodic_gaussian_blur_pixels(
    image: np.ndarray,
    sigma_pixels: float,
) -> np.ndarray:
    if sigma_pixels <= 0.0:
        return image.astype(np.float64, copy=True)
    return _periodic_gaussian_density(image, sigma_pixels, 1.0)


def calculate_element_responses(
    occupancy: np.ndarray,
    config: ScanConfig,
    extent_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    plane_depth_A = np.arange(config.depth_atomic_planes) * (0.5 * config.lattice_constant_A)
    focal_depth_A = np.linspace(plane_depth_A[0], plane_depth_A[-1], config.depth_sections)
    axial_weight = np.exp(
        -0.5 * ((focal_depth_A[:, None] - plane_depth_A[None, :]) / config.axial_sigma_A) ** 2
    )
    weighted_occupancy = np.einsum("dp,peyx->deyx", axial_weight, occupancy)
    response = _periodic_gaussian_density(
        weighted_occupancy,
        config.probe_sigma_A,
        extent_A / config.scan_pixels,
    )
    return focal_depth_A, response


def _profile_from_distribution(
    distribution: SpectralDistribution,
    energy_centers_eV: np.ndarray,
    energy_step_eV: float,
    blur_sigma_eV: float,
) -> np.ndarray:
    profile = np.interp(
        energy_centers_eV,
        distribution.energy_loss_eV,
        distribution.probability_density_per_eV,
        left=0.0,
        right=0.0,
    )
    profile *= energy_step_eV
    if blur_sigma_eV > 0.0:
        radius = max(1, int(np.ceil(4.0 * blur_sigma_eV / energy_step_eV)))
        offsets = np.arange(-radius, radius + 1) * energy_step_eV
        kernel = np.exp(-0.5 * (offsets / blur_sigma_eV) ** 2)
        kernel /= kernel.sum()
        profile = np.convolve(profile, kernel, mode="same")
    total = profile.sum()
    if total <= 0.0:
        raise ValueError("spectral component has no support in scan energy window")
    return profile / total


def _zlp_profile(
    energy_centers_eV: np.ndarray,
    energy_step_eV: float,
    blur_sigma_eV: float,
) -> np.ndarray:
    sigma = max(blur_sigma_eV, 0.25 * energy_step_eV)
    profile = np.exp(-0.5 * (energy_centers_eV / sigma) ** 2)
    return profile / profile.sum()


def _quantized_map_stack(maps: np.ndarray) -> list[str]:
    upper = float(np.quantile(maps, 0.995))
    if upper <= 0.0:
        upper = 1.0
    normalized = np.clip(maps / upper, 0.0, 1.0) ** 0.65
    quantized = np.rint(255.0 * normalized).astype(np.uint8)
    return [b64encode(layer.tobytes()).decode("ascii") for layer in quantized]


def write_scan_slider_html(
    path: str | Path,
    focal_depth_A: np.ndarray,
    haadf: np.ndarray,
    element_maps: dict[str, np.ndarray],
    extent_A: float,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = {"All elements (HAADF)": _quantized_map_stack(haadf)}
    channels.update({element: _quantized_map_stack(element_maps[element]) for element in ELEMENTS})
    colors = {"All elements (HAADF)": [255, 255, 255]}
    colors.update({element: list(ELEMENT_COLORS[element]) for element in ELEMENTS})
    payload = {
        "channels": channels,
        "colors": colors,
        "depth_A": [float(value) for value in focal_depth_A],
        "size": int(haadf.shape[-1]),
        "extent_A": float(extent_A),
    }
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LMTO simulated element scan</title>
<style>
body{margin:0;background:#10131a;color:#eef2f7;font:16px system-ui,sans-serif}
.app{max-width:900px;margin:24px auto;padding:0 20px}.panel{background:#191e28;
border:1px solid #303847;border-radius:14px;padding:20px;box-shadow:0 12px 35px #0006}
h1{font-size:24px;margin:0 0 6px}.sub{color:#aeb8c7;margin-bottom:18px}
.controls{display:grid;grid-template-columns:1fr 2fr;gap:18px;margin-bottom:16px}
label{display:block;color:#cbd4df;font-size:13px;margin-bottom:6px}select,input{width:100%}
select{background:#242b38;color:#fff;border:1px solid #465064;border-radius:7px;padding:9px}
canvas{display:block;width:min(100%,640px);aspect-ratio:1;margin:auto;background:#000;
border-radius:8px}.meta{display:flex;justify-content:space-between;color:#aeb8c7;margin-top:9px}
.notice{margin-top:16px;padding:12px;background:#222a36;border-left:4px solid #f0b44c;
color:#d9e0e8;font-size:14px}.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;
margin-right:7px}</style></head><body><div class="app"><div class="panel">
<h1>LMTO simulated STEM–EELS scan</h1><div class="sub">Li1.2Mn0.4Ti0.4O2 · synthetic disordered-rocksalt focal sections</div>
<div class="controls"><div><label for="channel">Reconstruction</label><select id="channel"></select></div>
<div><label for="depth">Focal depth: <span id="depthLabel"></span></label>
<input id="depth" type="range" min="0" step="1"></div></div>
<canvas id="view"></canvas><div class="meta"><span id="legend"></span><span id="fov"></span></div>
<div class="notice"><b>Simulation note:</b> the depth slider represents a synthetic focal-section series.
It is not direct depth recovery from one conventional 2D EELS acquisition.</div>
</div></div><script>
const DATA=__PAYLOAD__;const select=document.getElementById('channel');const slider=document.getElementById('depth');
const canvas=document.getElementById('view');const ctx=canvas.getContext('2d');const off=document.createElement('canvas');
off.width=off.height=DATA.size;canvas.width=canvas.height=DATA.size;slider.max=DATA.depth_A.length-1;
Object.keys(DATA.channels).forEach(k=>{const o=document.createElement('option');o.value=o.textContent=k;select.appendChild(o)});
function render(){const name=select.value,d=Number(slider.value),raw=atob(DATA.channels[name][d]);
const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
const rgb=DATA.colors[name],image=ctx.createImageData(DATA.size,DATA.size);
for(let i=0;i<bytes.length;i++){const j=4*i,v=bytes[i]/255;image.data[j]=rgb[0]*v;
image.data[j+1]=rgb[1]*v;image.data[j+2]=rgb[2]*v;image.data[j+3]=255}
off.getContext('2d').putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.clearRect(0,0,canvas.width,canvas.height);
ctx.drawImage(off,0,0,canvas.width,canvas.height);document.getElementById('depthLabel').textContent=DATA.depth_A[d].toFixed(1)+' Å';
document.getElementById('legend').innerHTML='<span class="swatch" style="background:rgb('+rgb.join(',')+')"></span>'+name;
document.getElementById('fov').textContent='field of view '+(DATA.extent_A/10).toFixed(2)+' nm'}
select.onchange=render;slider.oninput=render;document.getElementById('fov').textContent='';render();
</script></body></html>""".replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    output_path.write_text(html, encoding="utf-8")


def write_scan_montage(
    path: str | Path,
    focal_depth_A: np.ndarray,
    haadf: np.ndarray,
    element_maps: dict[str, np.ndarray],
    extent_A: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    depth_index = len(focal_depth_A) // 2
    images = [haadf[depth_index]] + [element_maps[element][depth_index] for element in ELEMENTS]
    titles = ["HAADF", *ELEMENTS]
    colormaps = ["gray"] + [
        LinearSegmentedColormap.from_list(
            f"black_to_{element.lower()}",
            [
                (0.0, 0.0, 0.0),
                tuple(channel / 255.0 for channel in ELEMENT_COLORS[element]),
            ],
        )
        for element in ELEMENTS
    ]
    figure, axes = plt.subplots(1, 5, figsize=(14, 3.1), constrained_layout=True)
    for axis, image, title, cmap in zip(axes, images, titles, colormaps):
        upper = max(float(np.quantile(image, 0.995)), 1.0)
        axis.imshow(
            image,
            cmap=cmap,
            origin="lower",
            vmin=0.0,
            vmax=upper,
            extent=(0.0, extent_A / 10.0, 0.0, extent_A / 10.0),
        )
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("black")
    figure.suptitle(
        f"LMTO simulated focal section at {focal_depth_A[depth_index]:.1f} Å",
        fontsize=15,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def simulate_lmto_scan(
    config_path: str | Path,
    output_hdf5: str | Path,
    output_montage: str | Path | None = None,
    output_html: str | Path | None = None,
) -> dict[str, object]:
    simulation_config = load_config(config_path)
    scan = simulation_config.scan
    _validate_scan_config(scan)
    library_value = simulation_config.specimen.parameters.get("library_file")
    if library_value is None:
        raise ValueError("LMTO scan requires specimen.parameters.library_file")
    library = read_spectral_library(str(library_value))
    if library.metadata.get("formula") != "Li1.2Mn0.4Ti0.4O2":
        raise ValueError("LMTO scan requires the LMTO spectral library")
    rng = np.random.default_rng(scan.random_seed)
    edge_to_element = [component.element for component in library.core_loss]
    edge_names = tuple(component.name for component in library.core_loss)
    sample_thickness_A = 0.5 * scan.lattice_constant_A * scan.depth_atomic_planes
    if scan.spatial_provider == "analytic":
        atoms, occupancy, extent_A = build_disordered_rocksalt(scan, rng)
        focal_depth_A, element_response = calculate_element_responses(occupancy, scan, extent_A)
        edge_response_per_A2 = np.stack(
            [element_response[:, ELEMENTS.index(element)] for element in edge_to_element],
            axis=1,
        )
        z_weight = np.asarray([ATOMIC_NUMBERS[element] ** 1.7 for element in ELEMENTS])
        haadf_signal = np.einsum("deyx,e->dyx", element_response, z_weight)
        haadf_scale = scan.haadf_peak_counts / max(float(np.quantile(haadf_signal, 0.995)), 1.0e-12)
        haadf_expected = 2.0 + haadf_scale * haadf_signal
        haadf_model = "Poisson-sampled incoherent Z^1.7 atomic-column response"
        spatial_metadata: dict[str, object] = {
            "provider": "analytic",
            "probe_model": "periodic lateral Gaussian and axial Gaussian",
        }
    else:
        from .lmto_abtem_scan import read_lmto_abtem_spatial_response

        response = read_lmto_abtem_spatial_response(str(scan.spatial_response_file))
        if response["element_labels"] != ELEMENTS:
            raise ValueError("abTEM spatial response element order does not match")
        cached_edge_names = tuple(response["edge_names"])
        missing_edges = set(edge_names) - set(cached_edge_names)
        if missing_edges:
            raise ValueError(f"abTEM spatial response is missing edge(s): {sorted(missing_edges)}")
        edge_order = [cached_edge_names.index(name) for name in edge_names]
        edge_response_per_A2 = np.asarray(response["edge_response_per_A2"])[:, edge_order]
        haadf_fraction = np.asarray(response["haadf_fraction"], dtype=np.float64)
        focal_depth_A = np.asarray(response["focal_depth_A"], dtype=np.float64)
        atoms = response["atoms"]
        scan_x_A = np.asarray(response["scan_x_A"], dtype=np.float64)
        scan_y_A = np.asarray(response["scan_y_A"], dtype=np.float64)
        cached_shape = (len(focal_depth_A), len(scan_y_A), len(scan_x_A))
        configured_shape = (
            scan.depth_sections,
            scan.scan_pixels,
            scan.scan_pixels,
        )
        if cached_shape != configured_shape:
            raise ValueError(
                f"abTEM spatial response shape {cached_shape} does not match "
                f"configured scan shape {configured_shape}"
            )
        extent_A = float(response["summary"]["field_of_view_A"])
        sample_thickness_A = float(response["summary"]["sample_thickness_A"])
        if haadf_fraction.shape != configured_shape:
            raise ValueError("abTEM HAADF response has the wrong shape")
        element_response = np.zeros(
            (
                scan.depth_sections,
                len(ELEMENTS),
                scan.scan_pixels,
                scan.scan_pixels,
            ),
            dtype=np.float64,
        )
        for element_index, element in enumerate(ELEMENTS):
            indices = [
                index
                for index, edge_element in enumerate(edge_to_element)
                if edge_element == element
            ]
            element_response[:, element_index] = edge_response_per_A2[:, indices].mean(axis=1)
        haadf_expected = 2.0 + scan.electrons_per_probe * haadf_fraction
        haadf_model = "Poisson-sampled abTEM multislice annular-detector fraction"
        spatial_metadata = {
            "provider": "abtem_cache",
            "source_file": str(scan.spatial_response_file),
            "abtem": response["metadata"],
        }

    cross_sections = np.asarray(
        [
            float(component.metadata["integrated_cross_section_barn_per_atom"])
            for component in library.core_loss
        ]
    )
    optical_depth = np.moveaxis(edge_response_per_A2, 1, -1) * (cross_sections * BARN_TO_A2)
    total_core_depth = optical_depth.sum(axis=-1)
    probability_of_core = -np.expm1(-total_core_depth)
    scale = np.divide(
        probability_of_core,
        total_core_depth,
        out=np.ones_like(probability_of_core),
        where=total_core_depth > 0.0,
    )
    core_probability = optical_depth * scale[..., None]
    low_mean = (
        0.0
        if library.low_loss is None
        else library.low_loss.mean_events
        * sample_thickness_A
        / max(library.specimen_thickness_A, 1.0e-12)
    )
    no_low_probability = np.exp(-low_mean)
    noncore_probability = 1.0 - core_probability.sum(axis=-1)
    outcome_probability = np.concatenate(
        (
            (noncore_probability * no_low_probability)[..., None],
            (noncore_probability * (1.0 - no_low_probability))[..., None],
            core_probability,
        ),
        axis=-1,
    )
    outcome_probability /= outcome_probability.sum(axis=-1, keepdims=True)
    component_counts = rng.multinomial(scan.electrons_per_probe, outcome_probability).astype(
        np.uint32
    )

    energy_edges_eV = np.arange(
        scan.energy_min_eV,
        scan.energy_max_eV + scan.energy_step_eV,
        scan.energy_step_eV,
    )
    energy_centers_eV = 0.5 * (energy_edges_eV[:-1] + energy_edges_eV[1:])
    blur_sigma_eV = simulation_config.spectrometer.energy_blur_sigma_eV
    profiles = [_zlp_profile(energy_centers_eV, scan.energy_step_eV, blur_sigma_eV)]
    if library.low_loss is None:
        profiles.append(profiles[0])
    else:
        profiles.append(
            _profile_from_distribution(
                library.low_loss.distribution,
                energy_centers_eV,
                scan.energy_step_eV,
                blur_sigma_eV,
            )
        )
    profiles.extend(
        _profile_from_distribution(
            component.distribution,
            energy_centers_eV,
            scan.energy_step_eV,
            blur_sigma_eV,
        )
        for component in library.core_loss
    )
    component_profiles = np.asarray(profiles, dtype=np.float64)
    spectrum_counts = np.zeros(
        (*component_counts.shape[:-1], len(energy_centers_eV)), dtype=np.uint32
    )
    for depth_index in range(scan.depth_sections):
        for component_index, profile in enumerate(component_profiles):
            distributed = rng.multinomial(
                component_counts[depth_index, ..., component_index], profile
            )
            spectrum_counts[depth_index] += distributed.astype(np.uint32)
    if not np.all(spectrum_counts.sum(axis=-1) == scan.electrons_per_probe):
        raise RuntimeError("spectrum-image electron counts were not conserved")

    element_maps = {}
    expected_element_maps = {}
    selected_element_edges: dict[str, list[str]] = {}
    for element_index, element in enumerate(ELEMENTS):
        available_edges = [
            name
            for name, edge_element in zip(edge_names, edge_to_element)
            if edge_element == element
        ]
        selected_edges = scan.element_edges.get(element, available_edges)
        if not selected_edges:
            raise ValueError(f"No reconstruction edge selected for {element}")
        invalid_edges = set(selected_edges) - set(available_edges)
        if invalid_edges:
            raise ValueError(
                f"Invalid {element} reconstruction edge(s): {sorted(invalid_edges)}; "
                f"available: {available_edges}"
            )
        selected_element_edges[element] = list(selected_edges)
        edge_indices = [edge_names.index(name) for name in selected_edges]
        raw = component_counts[..., np.asarray(edge_indices) + 2].sum(axis=-1)
        expected = scan.electrons_per_probe * core_probability[..., np.asarray(edge_indices)].sum(
            axis=-1
        )
        element_maps[element] = _periodic_gaussian_blur_pixels(
            raw, scan.reconstruction_sigma_pixels
        ).astype(np.float32)
        expected_element_maps[element] = expected.astype(np.float32)

    haadf_counts = rng.poisson(haadf_expected).astype(np.uint32)

    component_labels = [
        "zero_loss",
        "low_loss",
        *[component.name for component in library.core_loss],
    ]
    summary: dict[str, object] = {
        "material": library.material,
        "model": scan.model,
        "spatial_provider": scan.spatial_provider,
        "scan_shape": [scan.depth_sections, scan.scan_pixels, scan.scan_pixels],
        "energy_bins": len(energy_centers_eV),
        "field_of_view_A": extent_A,
        "sample_thickness_A": sample_thickness_A,
        "electrons_per_probe": scan.electrons_per_probe,
        "virtual_incident_electrons": (
            scan.depth_sections * scan.scan_pixels**2 * scan.electrons_per_probe
        ),
        "component_labels": component_labels,
        "selected_element_edges": selected_element_edges,
        "element_reconstruction_method": "ideal model-component unmixing",
        "depth_interpretation": "synthetic focal-section series",
    }
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = SPECTRUM_IMAGE_SCHEMA
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["config_file"] = str(Path(config_path).resolve())
        h5.attrs["scan_config_json"] = json.dumps(asdict(scan), sort_keys=True)
        h5.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5.attrs["spatial_metadata_json"] = json.dumps(spatial_metadata, sort_keys=True)
        h5.attrs["warning"] = (
            "Depth is a synthetic focal-section response, not direct recovery "
            "from one conventional 2D EELS scan"
        )
        axes = h5.create_group("axes")
        axes.create_dataset("focal_depth_A", data=focal_depth_A)
        axes.create_dataset(
            "scan_x_A",
            data=np.linspace(0.0, extent_A, scan.scan_pixels, endpoint=False),
        )
        axes.create_dataset(
            "scan_y_A",
            data=np.linspace(0.0, extent_A, scan.scan_pixels, endpoint=False),
        )
        axes.create_dataset("energy_edges_eV", data=energy_edges_eV)
        axes.create_dataset("energy_centers_eV", data=energy_centers_eV)
        structure = h5.create_group("structure/atoms")
        structure.attrs["element_labels_json"] = json.dumps(ELEMENTS)
        for name, values in atoms.items():
            structure.create_dataset(name, data=values, compression="gzip", shuffle=True)
        truth = h5.create_group("truth")
        truth.create_dataset(
            "edge_response_per_A2",
            data=edge_response_per_A2.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        truth.create_dataset(
            "haadf_expected_counts",
            data=haadf_expected.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        truth.create_dataset(
            "element_response_per_A2",
            data=element_response.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        expected_group = truth.create_group("expected_element_counts")
        for element, values in expected_element_maps.items():
            expected_group.create_dataset(element, data=values, compression="gzip", shuffle=True)
        reconstruction = h5.create_group("reconstruction")
        reconstruction.attrs["haadf_model"] = haadf_model
        reconstruction.create_dataset(
            "haadf_counts", data=haadf_counts, compression="gzip", shuffle=True
        )
        element_group = reconstruction.create_group("elements")
        element_group.attrs["method"] = "ideal model-component unmixing"
        element_group.attrs["warning"] = (
            "Uses known simulated edge labels; experimental processing must fit "
            "overlapping edges and backgrounds"
        )
        element_group.attrs["selected_edges_json"] = json.dumps(
            selected_element_edges, sort_keys=True
        )
        for element, values in element_maps.items():
            element_group.create_dataset(element, data=values, compression="gzip", shuffle=True)
        spectrum = h5.create_group("spectrum")
        spectrum.attrs["component_labels_json"] = json.dumps(component_labels)
        spectrum.create_dataset(
            "component_total_counts",
            data=component_counts,
            compression="gzip",
            shuffle=True,
        )
        spectrum.create_dataset(
            "expected_component_counts",
            data=(scan.electrons_per_probe * outcome_probability).astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        spectrum.create_dataset("component_profiles", data=component_profiles)
        spectrum.create_dataset(
            "counts",
            data=spectrum_counts,
            chunks=(1, 8, 8, min(256, len(energy_centers_eV))),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    if output_montage is not None:
        write_scan_montage(output_montage, focal_depth_A, haadf_counts, element_maps, extent_A)
    if output_html is not None:
        write_scan_slider_html(output_html, focal_depth_A, haadf_counts, element_maps, extent_A)
    return summary
