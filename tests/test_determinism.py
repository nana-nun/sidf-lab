import unittest

import numpy as np

from sidf_lab.anneal import AnnealConfig, model_c_decode
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


if __name__ == "__main__":
    unittest.main()

