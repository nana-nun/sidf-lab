import unittest

import numpy as np

from sidf_lab.metrics import comparison_summary, edge_leakage, edge_width, mad, psnr, region_summary, ssim_global


class MetricsTests(unittest.TestCase):
    def test_mad(self) -> None:
        a = np.array([[0.0, 1.0]])
        b = np.array([[0.5, 0.5]])
        self.assertAlmostEqual(mad(a, b), 0.5)

    def test_region_summary(self) -> None:
        image = np.array([[0.0, 1.0], [0.5, 0.5]])
        mask = np.array([[False, True], [True, False]])
        summary = region_summary(image, mask)
        self.assertAlmostEqual(summary["mean"], 0.75)
        self.assertAlmostEqual(summary["variance"], 0.0625)

    def test_edge_leakage(self) -> None:
        image = np.zeros((5, 5))
        image[1:4, 1:4] = 0.5
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        self.assertGreater(edge_leakage(image, mask, radius=1), 0.0)

    def test_psnr_identical_is_infinite(self) -> None:
        image = np.array([[0.0, 0.5], [0.5, 1.0]])
        self.assertTrue(np.isinf(psnr(image, image)))

    def test_psnr_known_value(self) -> None:
        reference = np.array([[0.0, 0.0]])
        candidate = np.array([[0.5, 0.5]])
        self.assertAlmostEqual(psnr(reference, candidate), 6.020599913279624)

    def test_ssim_global_identical(self) -> None:
        image = np.array([[0.0, 0.5], [0.75, 1.0]])
        self.assertAlmostEqual(ssim_global(image, image), 1.0)

    def test_edge_width_for_vertical_transition(self) -> None:
        row = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        image = np.tile(row, (4, 1))
        mask = np.tile(np.array([False, False, False, True, True, True, True]), (4, 1))
        self.assertAlmostEqual(edge_width(image, mask, max_radius=4), 2.0)

    def test_edge_width_without_regions_returns_none(self) -> None:
        image = np.ones((3, 3))
        mask = np.ones((3, 3), dtype=bool)
        self.assertIsNone(edge_width(image, mask))

    def test_comparison_summary_for_model_d_experiment(self) -> None:
        reference = np.array([[0.0, 0.0], [1.0, 1.0]])
        candidate = np.array([[0.0, 0.25], [0.75, 1.0]])
        mask = np.array([[False, False], [True, True]])
        summary = comparison_summary(candidate, reference=reference, foreground_mask=mask)
        self.assertIn("mad_vs_reference", summary)
        self.assertIn("psnr_vs_reference", summary)
        self.assertIn("ssim_global_vs_reference", summary)
        self.assertIn("edge_width_pixels", summary)
        self.assertAlmostEqual(summary["foreground_mean"], 0.875)
        self.assertAlmostEqual(summary["background_mean"], 0.125)


if __name__ == "__main__":
    unittest.main()

