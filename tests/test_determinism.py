import unittest

import numpy as np

from sidf_lab.anneal import (
    AnnealConfig,
    model_c_decode,
    quadratic_coordinate_descent,
    quadratic_objective,
)
from sidf_lab.energy import ModelCParams
from sidf_lab.guides import cross


class DeterminismTests(unittest.TestCase):
    def test_model_c_decode_is_deterministic_for_same_seed(self) -> None:
        guide = cross(8, width=2)
        params = ModelCParams(j_base=0.5, lambda_data=2.0, gamma=20.0)
        config = AnnealConfig(decoder_seed=123, sweeps=2, temp_start=0.5, temp_end=0.1)
        first = model_c_decode(guide, params, config)
        second = model_c_decode(guide, params, config)
        self.assertTrue(np.array_equal(first, second))

    def test_quadratic_coordinate_descent_is_deterministic_and_decreases_objective(
        self,
    ) -> None:
        guide = cross(8, width=2)
        initial = np.clip(guide + 0.1, 0.0, 1.0)
        confidence = np.ones_like(guide)
        kwargs = {
            "j_base": 1.8,
            "lambda_data": 6.0,
            "gamma": 35.0,
            "max_sweeps": 20,
        }
        initial_objective = quadratic_objective(
            initial,
            guide,
            confidence,
            j_base=kwargs["j_base"],
            lambda_data=kwargs["lambda_data"],
            gamma=kwargs["gamma"],
        )
        first, first_diagnostics = quadratic_coordinate_descent(
            guide, initial, confidence, **kwargs
        )
        second, second_diagnostics = quadratic_coordinate_descent(
            guide, initial, confidence, **kwargs
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertLessEqual(first_diagnostics["final_objective"], initial_objective)
        self.assertGreater(first_diagnostics["updates"], 0)


if __name__ == "__main__":
    unittest.main()

