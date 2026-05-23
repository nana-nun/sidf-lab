# sidf-core

This crate is an experimental Rust core spike for SIDF reproducibility work.

The first implemented component is `Philox4x32-10`, matching the saved tiny vectors in:

```text
../../references/prng-test-vectors/philox4x32-10.json
```

This is only a PRNG reproducibility check. It does not implement the SIDF decoder, fixed-point arithmetic, update order, proposal distribution, accept/reject rule, or output serialization. It also does not provide evidence for Model C/D image quality, compression, or super-resolution behavior.

When Rust tooling is available, verify with:

```powershell
cargo test --manifest-path rust/sidf-core/Cargo.toml
```
