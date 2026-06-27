"""Command-line entrypoint for small SIDF helper checks."""

from __future__ import annotations

import argparse

import numpy as np

from sidf_lab.guides import cross
from sidf_lab.model_e import (
    decode_model_e,
    estimate_model_e_bits,
    example_model_e_params,
)


def main() -> int:
    """Run a minimal SIDF helper command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-e-example",
        action="store_true",
        help="run a tiny deterministic Model E decode example",
    )
    args = parser.parse_args()
    if args.model_e_example:
        guide = cross(4, width=1, value=0.5)
        params = example_model_e_params("coupled_state")
        decoded = decode_model_e(guide, (8, 8), params)
        bits = estimate_model_e_bits(params)
        print(f"model_e_example_shape={decoded.shape}")
        print(f"model_e_example_range=({float(np.min(decoded)):.6f}, {float(np.max(decoded)):.6f})")
        print(f"model_e_incremental_side_bits={bits['incremental_side_bits']}")
        return 0
    print("sidf-lab Python research helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

