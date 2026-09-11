import unittest

import numpy as np
from eels_sim.config import ExperimentConfig, ReadoutConfig, TransportConfig
from eels_sim.readout import digitize, select_calibrated_maps
from eels_sim.transport import (
    SensorGeometry,
    diffusion_sigma_um,
    spread_pairs,
    transport_deposits,
)


class TransportTests(unittest.TestCase):
    def test_paper_source_occupancy(self):
        experiment = ExperimentConfig(300.0, 87000.0, 30.0)
        self.assertAlmostEqual(experiment.integration_time_us, 11.4942529, places=5)
        self.assertAlmostEqual(experiment.expected_primaries_per_frame, 2152.24, places=1)

    def test_deeper_charge_diffuses_more(self):
        config = TransportConfig()
        near = diffusion_sigma_um(2.4, 5.0, config)
        far = diffusion_sigma_um(-2.4, 5.0, config)
        self.assertGreater(far, near)

    def test_multinomial_spreading_conserves_pairs(self):
        geometry = SensorGeometry(16, 16, 10.0, 5.0)
        config = TransportConfig()
        charge = np.zeros((1, 16, 16), dtype=np.uint32)
        collected, lost = spread_pairs(
            charge,
            0,
            0.0,
            0.0,
            0.7,
            10000,
            geometry,
            config,
            np.random.default_rng(1),
        )
        self.assertEqual(collected + lost, 10000)
        self.assertEqual(int(charge.sum()), collected)

    def test_digitization_is_seeded_and_clipped(self):
        charge = np.array([[[0, 100_000_000]]], dtype=np.uint32)
        config = ReadoutConfig(gain_adu_per_electron=1.0, read_noise_adu=0.0)
        analog_a, raw_a = digitize(charge, config, 7)
        analog_b, raw_b = digitize(charge, config, 7)
        np.testing.assert_array_equal(analog_a, analog_b)
        np.testing.assert_array_equal(raw_a, raw_b)
        self.assertEqual(raw_a[0, 0, 1], 65535)

    def test_digitization_uses_calibration_maps(self):
        charge = np.zeros((2, 2, 2), dtype=np.uint32)
        config = ReadoutConfig(read_noise_adu=0.0, adc_bits=12)
        pedestal = np.array([[100.0, 200.0], [300.0, 400.0]])
        noise = np.zeros((2, 2))
        _, raw = digitize(charge, config, 9, pedestal, noise)
        np.testing.assert_array_equal(raw[0], pedestal.astype(np.uint16))

    def test_calibration_effects_are_independent(self):
        calibration = {
            "pedestal_adu": np.full((2, 2), 200.0),
            "healthy_pedestal_adu": np.full((2, 2), 100.0),
            "read_noise_adu": np.full((2, 2), 20.0),
            "healthy_read_noise_adu": np.full((2, 2), 2.0),
            "signal_efficiency": np.array([[0.0, 1.0], [1.0, 1.0]]),
        }
        healthy = ReadoutConfig(
            use_calibrated_pedestal=True,
            use_calibrated_noise=False,
            simulate_known_defects=False,
        )
        pedestal, noise, efficiency = select_calibrated_maps(calibration, healthy)
        np.testing.assert_array_equal(pedestal, calibration["healthy_pedestal_adu"])
        self.assertIsNone(noise)
        self.assertIsNone(efficiency)

        defects = ReadoutConfig(
            use_calibrated_pedestal=False,
            use_calibrated_noise=True,
            simulate_known_defects=True,
        )
        pedestal, noise, efficiency = select_calibrated_maps(calibration, defects)
        self.assertIsNone(pedestal)
        np.testing.assert_array_equal(noise, calibration["read_noise_adu"])
        np.testing.assert_array_equal(efficiency, calibration["signal_efficiency"])

    def test_known_defect_can_suppress_signal(self):
        charge = np.full((1, 1, 2), 100, dtype=np.uint32)
        config = ReadoutConfig(
            gain_adu_per_electron=1.0,
            pedestal_adu=0.0,
            read_noise_adu=0.0,
            use_calibrated_correlated_noise=False,
        )
        efficiency = np.array([[0.0, 1.0]])
        _, raw = digitize(charge, config, 3, signal_efficiency_map=efficiency)
        np.testing.assert_array_equal(raw[0, 0], np.array([0, 100]))

    def test_transport_uses_upstream_frame_ids(self):
        deposits = {
            "event_id": np.array([0, 1], dtype=np.int32),
            "x_um": np.array([0.0, 0.0]),
            "y_um": np.array([0.0, 0.0]),
            "z_um": np.array([0.0, 0.0]),
            "edep_eV": np.array([36.4, 36.4]),
        }
        charge, summary = transport_deposits(
            deposits,
            SensorGeometry(4, 4, 10.0, 5.0),
            TransportConfig(fano_factor=0.0),
            primaries_per_frame=10,
            random_seed=4,
            frame_ids_by_event=np.array([0, 2]),
        )
        self.assertEqual(summary["frame_count"], 3)
        self.assertGreater(charge[0].sum(), 0)
        self.assertEqual(charge[1].sum(), 0)
        self.assertGreater(charge[2].sum(), 0)


if __name__ == "__main__":
    unittest.main()
