import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from eels_sim.config import ScanConfig
from eels_sim.lmto_abtem_scan import (
    ABTEM_SPATIAL_SCHEMA,
    EDGE_SHELLS,
    tile_lmto_abtem_spatial_response,
)
from eels_sim.lmto_scan import (
    ELEMENTS,
    SPECTRUM_IMAGE_SCHEMA,
    build_disordered_rocksalt,
    calculate_element_responses,
    simulate_lmto_scan,
    write_scan_slider_html,
)
from eels_sim.spectral_library import (
    CoreLossComponent,
    LowLossComponent,
    SpectralDistribution,
    SpectralLibrary,
    write_spectral_library,
)


def tiny_lmto_library() -> SpectralLibrary:
    edge_definitions = (
        ("Ti_M23", "Ti", "M2,3", 35.0),
        ("Mn_M23", "Mn", "M2,3", 51.0),
        ("Li_K", "Li", "K", 55.0),
        ("Ti_L23", "Ti", "L2,3", 456.0),
        ("O_K", "O", "K", 532.0),
        ("Mn_L23", "Mn", "L2,3", 640.0),
    )
    core_loss = tuple(
        CoreLossComponent(
            name=name,
            element=element,
            edge=edge,
            onset_energy_eV=onset,
            integrated_probability=0.001,
            angular_sigma_mrad=1.0,
            distribution=SpectralDistribution(
                np.array([onset, onset + 10.0]), np.array([1.0, 1.0])
            ),
            metadata={"integrated_cross_section_barn_per_atom": 1.0e6},
        )
        for name, element, edge, onset in edge_definitions
    )
    return SpectralLibrary(
        material="LMTO test",
        beam_energy_eV=300_000.0,
        specimen_thickness_A=50.0,
        elastic_angular_sigma_mrad=0.1,
        low_loss=LowLossComponent(
            distribution=SpectralDistribution(np.array([10.0, 20.0]), np.array([1.0, 1.0])),
            mean_events=0.2,
            angular_sigma_mrad=0.2,
        ),
        core_loss=core_loss,
        metadata={"formula": "Li1.2Mn0.4Ti0.4O2"},
    )


class LMTOScanTests(unittest.TestCase):
    def test_all_lmto_edges_have_abtem_subshell_mappings(self):
        self.assertEqual(
            set(EDGE_SHELLS),
            {"Ti_M23", "Mn_M23", "Li_K", "Ti_L23", "O_K", "Mn_L23"},
        )

    def test_periodic_abtem_tiling_expands_maps_axes_and_atoms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.h5"
            tiled_path = root / "tiled.h5"
            with h5py.File(source_path, "w") as h5:
                h5.attrs["schema"] = ABTEM_SPATIAL_SCHEMA
                h5.attrs["edge_names_json"] = json.dumps(["O_K"])
                h5.attrs["element_labels_json"] = json.dumps(ELEMENTS)
                h5.attrs["metadata_json"] = json.dumps(
                    {
                        "limitations": ["source limitation"],
                        "normalization": {"O_K": {"atom_count": 2}},
                    }
                )
                h5.attrs["summary_json"] = json.dumps(
                    {
                        "field_of_view_A": 2.0,
                        "sample_thickness_A": 4.0,
                        "scan_shape": [1, 2, 2],
                        "atom_count": 2,
                    }
                )
                axes = h5.create_group("axes")
                axes.create_dataset("focal_depth_A", data=[2.0])
                axes.create_dataset("defocus_A", data=[2.0])
                axes.create_dataset("scan_x_A", data=[0.0, 1.0])
                axes.create_dataset("scan_y_A", data=[0.0, 1.0])
                atoms = h5.create_group("structure/atoms")
                for name, values in {
                    "x_A": [0.0, 1.0],
                    "y_A": [0.0, 1.0],
                    "z_A": [0.0, 2.0],
                    "element_id": [3, 3],
                    "atomic_number": [8, 8],
                    "plane_index": [0, 1],
                }.items():
                    atoms.create_dataset(name, data=values)
                spatial = h5.create_group("spatial")
                pattern = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
                spatial.create_dataset("haadf_fraction", data=pattern)
                spatial.create_dataset("raw_edge_intensity", data=pattern[:, None])
                spatial.create_dataset("edge_response_per_A2", data=pattern[:, None])

            summary = tile_lmto_abtem_spatial_response(source_path, tiled_path, tile_factor=2)
            self.assertEqual(summary["scan_shape"], [1, 4, 4])
            self.assertEqual(summary["field_of_view_A"], 4.0)
            self.assertEqual(summary["atom_count"], 8)
            with h5py.File(tiled_path, "r") as h5:
                expected = np.tile(np.arange(4).reshape(2, 2), (2, 2))
                np.testing.assert_array_equal(h5["spatial/haadf_fraction"][0], expected)
                np.testing.assert_array_equal(h5["axes/scan_x_A"][:], [0.0, 1.0, 2.0, 3.0])
                self.assertEqual(len(h5["structure/atoms/x_A"]), 8)
                metadata = json.loads(h5.attrs["metadata_json"])
                self.assertEqual(metadata["normalization"]["O_K"]["atom_count"], 8)
                self.assertIn("periodic", metadata["limitations"][-1])

    def test_structure_has_lmto_stoichiometry_and_conserves_density(self):
        config = ScanConfig(
            scan_pixels=40,
            lateral_cells=5,
            depth_atomic_planes=4,
            depth_sections=3,
        )
        atoms, occupancy, extent_A = build_disordered_rocksalt(config, np.random.default_rng(7))
        counts = np.bincount(atoms["element_id"], minlength=len(ELEMENTS))
        np.testing.assert_array_equal(counts, [120, 40, 40, 200])

        focal_depth_A, response = calculate_element_responses(occupancy, config, extent_A)
        plane_depth_A = np.arange(config.depth_atomic_planes) * (0.5 * config.lattice_constant_A)
        axial_weight = np.exp(
            -0.5 * ((focal_depth_A[:, None] - plane_depth_A[None, :]) / config.axial_sigma_A) ** 2
        )
        expected_atoms = np.einsum("dp,peyx->de", axial_weight, occupancy)
        pixel_area_A2 = (extent_A / config.scan_pixels) ** 2
        np.testing.assert_allclose(
            response.sum(axis=(-2, -1)) * pixel_area_A2,
            expected_atoms,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_small_scan_writes_count_conserving_spectrum_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library_path = root / "lmto.h5"
            config_path = root / "scan.toml"
            output_path = root / "scan.h5"
            html_path = root / "scan.html"
            write_spectral_library(library_path, tiny_lmto_library())
            config_path.write_text(
                f"""[experiment]
beam_energy_keV = 300.0
frame_rate_hz = 1000.0
beam_current_pA = 1.0

[specimen.parameters]
library_file = "{library_path}"

[scan]
scan_pixels = 8
lateral_cells = 1
depth_atomic_planes = 4
depth_sections = 2
electrons_per_probe = 1000
energy_min_eV = -2.0
energy_max_eV = 660.0
energy_step_eV = 2.0
random_seed = 19
""",
                encoding="utf-8",
            )
            summary = simulate_lmto_scan(config_path, output_path, output_html=html_path)
            with h5py.File(output_path, "r") as h5:
                self.assertEqual(h5.attrs["schema"], SPECTRUM_IMAGE_SCHEMA)
                spectra = h5["spectrum/counts"][:]
                self.assertEqual(spectra.shape, (2, 8, 8, 331))
                self.assertTrue(np.all(spectra.sum(axis=-1) == 1000))
                self.assertEqual(set(h5["reconstruction/elements"].keys()), set(ELEMENTS))
            self.assertEqual(summary["virtual_incident_electrons"], 128_000)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("All elements (HAADF)", html)
            self.assertIn('id="depth"', html)

    def test_slider_html_contains_all_element_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "viewer.html"
            maps = {element: np.ones((2, 4, 4), dtype=np.float32) for element in ELEMENTS}
            write_scan_slider_html(
                path,
                np.array([0.0, 2.0]),
                np.ones((2, 4, 4), dtype=np.float32),
                maps,
                8.0,
            )
            html = path.read_text(encoding="utf-8")
        for element in ELEMENTS:
            self.assertIn(element, html)
        self.assertIn("synthetic focal-section", html)

    def test_abtem_cache_drives_haadf_and_selected_edge_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library_path = root / "lmto.h5"
            spatial_path = root / "spatial.h5"
            config_path = root / "scan.toml"
            output_path = root / "scan.h5"
            library = tiny_lmto_library()
            write_spectral_library(library_path, library)
            edge_names = [component.name for component in library.core_loss]
            with h5py.File(spatial_path, "w") as h5:
                h5.attrs["schema"] = ABTEM_SPATIAL_SCHEMA
                h5.attrs["edge_names_json"] = json.dumps(edge_names)
                h5.attrs["element_labels_json"] = json.dumps(ELEMENTS)
                h5.attrs["metadata_json"] = json.dumps({"engine": "fake-abTEM"})
                h5.attrs["summary_json"] = json.dumps(
                    {
                        "field_of_view_A": 4.15,
                        "sample_thickness_A": 8.3,
                        "scan_shape": [2, 8, 8],
                    }
                )
                axes = h5.create_group("axes")
                axes.create_dataset("focal_depth_A", data=[0.0, 8.3])
                axes.create_dataset("scan_x_A", data=np.arange(8) * 4.15 / 8)
                axes.create_dataset("scan_y_A", data=np.arange(8) * 4.15 / 8)
                atoms = h5.create_group("structure/atoms")
                atoms.create_dataset("x_A", data=[0.0])
                atoms.create_dataset("y_A", data=[0.0])
                atoms.create_dataset("z_A", data=[0.0])
                atoms.create_dataset("element_id", data=[0])
                atoms.create_dataset("atomic_number", data=[3])
                atoms.create_dataset("plane_index", data=[0])
                spatial = h5.create_group("spatial")
                pattern = np.linspace(0.5, 1.5, 64).reshape(8, 8)
                response = np.stack([pattern * (index + 1) for index in range(len(edge_names))])
                response = np.stack((response, response[:, ::-1]))
                spatial.create_dataset("edge_response_per_A2", data=response.astype(np.float32))
                spatial.create_dataset("raw_edge_intensity", data=response.astype(np.float32))
                spatial.create_dataset(
                    "haadf_fraction",
                    data=np.stack((pattern, pattern[::-1])) * 1.0e-3,
                )
            config_path.write_text(
                f"""[experiment]
beam_energy_keV = 300.0
frame_rate_hz = 1000.0
beam_current_pA = 1.0

[specimen.parameters]
library_file = "{library_path}"

[scan]
spatial_provider = "abtem_cache"
spatial_response_file = "{spatial_path}"
scan_pixels = 8
lateral_cells = 1
depth_atomic_planes = 4
depth_sections = 2
electrons_per_probe = 1000
energy_min_eV = -2.0
energy_max_eV = 660.0
energy_step_eV = 2.0
random_seed = 31

[scan.element_edges]
Li = ["Li_K"]
Mn = ["Mn_L23"]
Ti = ["Ti_L23"]
O = ["O_K"]
""",
                encoding="utf-8",
            )
            simulate_lmto_scan(config_path, output_path)
            with h5py.File(output_path, "r") as h5:
                self.assertIn("abTEM multislice", h5["reconstruction"].attrs["haadf_model"])
                selected = json.loads(h5["reconstruction/elements"].attrs["selected_edges_json"])
                self.assertEqual(selected["Ti"], ["Ti_L23"])
                self.assertTrue(np.all(h5["spectrum/counts"][:].sum(axis=-1) == 1000))


if __name__ == "__main__":
    unittest.main()
