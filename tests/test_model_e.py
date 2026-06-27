import unittest

import numpy as np

from sidf_lab.guides import cross
from sidf_lab.model_e import (
    ModelEParams,
    QuantizationSpec,
    bilinear_resize,
    decode_model_e,
    deserialize_model_e_params,
    estimate_model_e_bits,
    example_model_e_params,
    model_e_features,
    model_e_residual,
    serialize_model_e_params,
)


class ModelETests(unittest.TestCase):
    def test_bilinear_resize_preserves_corners(self) -> None:
        image = np.array([[0.0, 1.0], [0.5, 0.75]])
        resized = bilinear_resize(image, (5, 7))
        self.assertEqual(resized.shape, (5, 7))
        self.assertAlmostEqual(resized[0, 0], 0.0)
        self.assertAlmostEqual(resized[0, -1], 1.0)
        self.assertAlmostEqual(resized[-1, 0], 0.5)
        self.assertAlmostEqual(resized[-1, -1], 0.75)

    def test_single_and_coupled_models_are_deterministic_with_same_interface(self) -> None:
        guide = cross(4, width=1)
        for kind in ["single_state", "coupled_state"]:
            params = example_model_e_params(kind)  # type: ignore[arg-type]
            first = decode_model_e(guide, (12, 10), params)
            second = decode_model_e(guide, (12, 10), params)
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(first.shape, (12, 10))
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)

    def test_features_and_residual_shapes(self) -> None:
        guide = cross(4, width=1)
        params = example_model_e_params("coupled_state")
        features = model_e_features(guide, (6, 7))
        residual = model_e_residual(features, params)
        self.assertEqual(features.shape, (6, 7, 5))
        self.assertEqual(residual.shape, (6, 7))
        self.assertLessEqual(float(np.max(np.abs(residual))), params.residual_scale)

    def test_invalid_parameter_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelEParams(
                kind="single_state",
                layers=np.zeros((2, 2, 5)),
                readout=np.ones(2),
            )
        with self.assertRaises(ValueError):
            ModelEParams(
                kind="coupled_state",
                layers=np.zeros((2, 1, 5)),
                readout=np.ones(1),
            )
        with self.assertRaises(ValueError):
            QuantizationSpec(bits_per_value=0)

    def test_serialization_round_trip_and_bit_estimate(self) -> None:
        params = example_model_e_params("coupled_state")
        spec = QuantizationSpec(bits_per_value=12, min_value=-1.0, max_value=1.0)
        serialized = serialize_model_e_params(params, spec)
        restored = deserialize_model_e_params(serialized)
        bits = estimate_model_e_bits(params, spec)
        self.assertEqual(restored.kind, params.kind)
        self.assertEqual(restored.layers.shape, params.layers.shape)
        self.assertEqual(restored.readout.shape, params.readout.shape)
        self.assertEqual(
            bits["incremental_side_bits"],
            spec.header_bits + 32 + params.parameter_count * spec.bits_per_value,
        )

        guide = cross(4, width=1)
        decoded = decode_model_e(guide, (8, 8), restored)
        self.assertEqual(decoded.shape, (8, 8))
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        self.assertLessEqual(float(decoded.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
