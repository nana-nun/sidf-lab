import unittest

import numpy as np

from sidf_lab.texture import fractal_value_noise, smoothed_noise


class TextureTests(unittest.TestCase):
    def test_smoothed_noise_is_deterministic_zero_mean_and_shape_preserving(self) -> None:
        first = smoothed_noise((8, 6), seed=123, sigma=0.25, radius=2)
        second = smoothed_noise((8, 6), seed=123, sigma=0.25, radius=2)

        self.assertEqual(first.shape, (8, 6))
        self.assertEqual(first.dtype, np.float64)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(first.mean()), 0.0)
        self.assertAlmostEqual(float(first.std()), 0.25)

    def test_fractal_value_noise_is_deterministic_zero_mean_and_shape_preserving(self) -> None:
        first = fractal_value_noise((7, 9), seed=456, octaves=3, sigma=0.1)
        second = fractal_value_noise((7, 9), seed=456, octaves=3, sigma=0.1)

        self.assertEqual(first.shape, (7, 9))
        self.assertEqual(first.dtype, np.float64)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(first.mean()), 0.0)
        self.assertAlmostEqual(float(first.std()), 0.1)

    def test_zero_sigma_returns_zero_field(self) -> None:
        self.assertTrue(np.array_equal(smoothed_noise((3, 4), seed=1, sigma=0.0), np.zeros((3, 4))))
        self.assertTrue(np.array_equal(fractal_value_noise((3, 4), seed=1, sigma=0.0), np.zeros((3, 4))))

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            smoothed_noise((4, 4), seed=1, sigma=-0.1)
        with self.assertRaises(ValueError):
            smoothed_noise((4, 4), seed=1, radius=-1)
        with self.assertRaises(ValueError):
            fractal_value_noise((4, 4), seed=1, octaves=0)
        with self.assertRaises(ValueError):
            fractal_value_noise((4, 4), seed=1, base_frequency=0.0)
        with self.assertRaises(ValueError):
            fractal_value_noise((4, 4), seed=1, lacunarity=0.0)
        with self.assertRaises(ValueError):
            fractal_value_noise((4, 4), seed=1, gain=-0.1)
        with self.assertRaises(ValueError):
            fractal_value_noise((0, 4), seed=1)


if __name__ == "__main__":
    unittest.main()
