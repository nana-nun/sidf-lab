# sidf-core

This crate is an experimental Rust core spike for SIDF reproducibility work.

The first implemented component is `Philox4x32-10`, matching the saved tiny vectors in:

```text
../../references/prng-test-vectors/philox4x32-10.json
```

The crate now also includes a minimal Model C annealing loop spike:

- `energy.rs`: grayscale `f64` image container and Model C local energy.
- `anneal.rs`: deterministic row-major update loop using the Philox helper for initialization, proposal deltas, and accept/reject thresholds.

This is not yet a bit-perfect SIDF decoder. It intentionally does not implement fixed-point arithmetic, serialized guide parsing, output serialization, or Model D.

Python comparison boundary:

- Matching part: the local Model C energy formula uses the same data fidelity and edge-aware pairwise terms as `src/sidf_lab/energy.py`.
- Non-matching parts: Python currently uses NumPy `default_rng`, per-sweep random permutations, Gaussian proposals, and NumPy/`math.exp` floating-point behavior.
- Rust spike choice: row-major traversal and uniform proposal deltas are fixed so that the first Rust update loop can be tested deterministically.
- Next fixed-point work: value scale, rounding, overflow policy, proposal table, and acceptance table are still open.

The implementation does not provide evidence for Model C/D image quality, compression, or super-resolution behavior.

When Rust tooling is available, verify with:

```powershell
cargo test --manifest-path rust/sidf-core/Cargo.toml
```
