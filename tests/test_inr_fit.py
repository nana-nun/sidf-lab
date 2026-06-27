import unittest

import numpy as np

from sidf_lab.guides import cross
from sidf_lab.inr_fit import (
    INRSpec,
    decode_inr,
    dequantize_vector,
    estimate_inr_bits,
    fit_inr,
    flatten_parameters,
    initialize_parameters,
    make_parameter_layout,
    quantize_vector,
    unflatten_parameters,
)
from sidf_lab.model_e import QuantizationSpec, bilinear_resize


class INRFitTests(unittest.TestCase):
    def test_flatten_round_trip_for_supported_families(self) -> None:
        specs = [
            INRSpec("fourier", order=2),
            INRSpec("rff", feature_count=4),
            INRSpec("siren", feature_count=4),
            INRSpec("model_e_single", depth=2, states=1),
            INRSpec("model_e_coupled", depth=2, states=3),
        ]
        for spec in specs:
            with self.subTest(spec=spec.family):
                layout = make_parameter_layout(spec)
                vector = initialize_parameters(spec, seed=17)
                restored = unflatten_parameters(vector, layout)
                self.assertEqual(vector.size, layout.parameter_count)
                self.assertTrue(np.array_equal(vector, flatten_parameters(restored, layout)))

    def test_decode_shape_and_range(self) -> None:
        guide = cross(4, width=1)
        for spec in [INRSpec("rff", feature_count=3), INRSpec("model_e_coupled", states=3)]:
            with self.subTest(spec=spec.family):
                params = initialize_parameters(spec, seed=3)
                decoded = decode_inr(spec, guide, (9, 11), params)
                self.assertEqual(decoded.shape, (9, 11))
                self.assertGreaterEqual(float(decoded.min()), 0.0)
                self.assertLessEqual(float(decoded.max()), 1.0)

    def test_fit_is_deterministic_with_same_seed(self) -> None:
        high = cross(12, width=2)
        low = bilinear_resize(high, (4, 4))
        spec = INRSpec("rff", feature_count=3, residual_limit=0.25)
        first = fit_inr(spec, low, high, seed=99, steps=12)
        second = fit_inr(spec, low, high, seed=99, steps=12)
        self.assertTrue(np.array_equal(first.parameters, second.parameters))
        self.assertTrue(np.array_equal(first.quantized_parameters, second.quantized_parameters))
        self.assertAlmostEqual(first.float_mse, second.float_mse)
        self.assertAlmostEqual(first.quantized_mse, second.quantized_mse)
        self.assertIn("incremental_side_bits", first.bits)

    def test_quantization_round_trip_and_bits(self) -> None:
        spec = INRSpec("model_e_single", depth=2)
        params = initialize_parameters(spec, seed=5)
        quantization = QuantizationSpec(bits_per_value=10, min_value=-1.0, max_value=1.0)
        quantized = quantize_vector(params, quantization)
        restored = dequantize_vector(quantized, quantization)
        self.assertEqual(quantized.shape, params.shape)
        self.assertEqual(restored.shape, params.shape)
        bits = estimate_inr_bits(spec, quantization)
        self.assertEqual(
            bits["incremental_side_bits"],
            quantization.header_bits + 32 + bits["parameter_count"] * quantization.bits_per_value,
        )

    def test_invalid_specs_and_vector_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            INRSpec("model_e_coupled", states=1)
        spec = INRSpec("fourier", order=1)
        with self.assertRaises(ValueError):
            unflatten_parameters(np.zeros(2), make_parameter_layout(spec))


if __name__ == "__main__":
    unittest.main()
