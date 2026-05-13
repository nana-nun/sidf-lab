import unittest

import numpy as np

from sidf_lab.metrics import edge_leakage, mad, region_summary


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


if __name__ == "__main__":
    unittest.main()

