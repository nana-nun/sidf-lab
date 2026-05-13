import unittest

import numpy as np

from sidf_lab.energy import ModelCParams, model_c_local_energy, valid_neighbors


class EnergyTests(unittest.TestCase):
    def test_valid_neighbors_corner(self) -> None:
        self.assertEqual(valid_neighbors(0, 0, 3, 3), [(1, 0), (0, 1)])

    def test_model_c_prefers_guide_value_when_neighbors_match(self) -> None:
        guide = np.full((3, 3), 0.5)
        state = np.full((3, 3), 0.5)
        params = ModelCParams(j_base=1.0, lambda_data=5.0, gamma=40.0)
        at_guide = model_c_local_energy(0.5, state, guide, 1, 1, params)
        away = model_c_local_energy(0.0, state, guide, 1, 1, params)
        self.assertLess(at_guide, away)


if __name__ == "__main__":
    unittest.main()

