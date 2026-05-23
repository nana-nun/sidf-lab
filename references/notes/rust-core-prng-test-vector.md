# Rust core PRNG test vector spike

## Source

Related Issue: [#50](https://github.com/nana-nun/sidf-lab/issues/50)

Background:

- [Deterministic PRNG と Bit-perfect 再現性](deterministic-prng-bit-perfect.md)
- Random123: https://random123.com/
- John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011. DOI: `10.1145/2063384.2063405`

## Decision

Rust core spike の最初の候補は `Philox4x32-10` とする。

理由:

- counter と key が分かれており、SIDF decoder の `stage / sweep / pixel_index / purpose` を counter に割り当てやすい。
- 4つの32-bit wordを返すため、proposal、accept threshold、texture などの用途へ分けやすい。
- Random123 系の代表候補であり、将来の並列化や pixel 単位の独立評価と相性がよい。

これは正式仕様の採用決定ではない。現時点では、Rust core 移植前に同じ seed / counter mapping から同じ値を得られるかを確認するための spike 候補である。

## Variant

- Algorithm: Philox
- Variant: `Philox4x32`
- Rounds: `10`
- Word width: unsigned 32-bit
- Arithmetic: wrapping modulo `2^32`
- Counter words: four little-indexed `u32` values
- Key words: two `u32` values
- Output word order: `[word0, word1, word2, word3]` as produced after the final round

Round constants:

```text
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
```

One round is defined as:

```text
hi0, lo0 = mulhilo(M0, counter[0])
hi1, lo1 = mulhilo(M1, counter[2])

next[0] = hi1 ^ counter[1] ^ key[0]
next[1] = lo1
next[2] = hi0 ^ counter[3] ^ key[1]
next[3] = lo0

key[0] = key[0] + W0 mod 2^32
key[1] = key[1] + W1 mod 2^32
```

The test vectors in `references/prng-test-vectors/philox4x32-10.json` use this exact mapping.

## Minimal SIDF counter layout

For the first Rust core spike, use the following four-word counter layout:

| Word | Field | Meaning |
| --- | --- | --- |
| `counter[0]` | `stage` | Decoder phase, such as initialization, annealing, or texture sampling |
| `counter[1]` | `sweep` | Sweep number within the stage |
| `counter[2]` | `pixel_index` | Row-major pixel index in the output image |
| `counter[3]` | `purpose` | Random-use selector |

Initial purpose values:

| Value | Purpose |
| --- | --- |
| `0` | proposal value |
| `1` | accept/reject threshold |
| `2` | texture prior sample |
| `3` | reserved |

For this spike, the `decoder_seed` is split into two `u32` key words:

```text
key[0] = high 32 bits of decoder_seed
key[1] = low 32 bits of decoder_seed
```

This split is a test-vector convention, not a final seed expansion design. A future specification still needs to define multi-field seed expansion, endian handling for serialized seeds, and domain separation between decoder, texture, and experiment seeds.

## Test vector

The machine-readable test vector is:

```text
references/prng-test-vectors/philox4x32-10.json
```

It records:

- PRNG variant and constants
- counter layout
- output word order
- tiny seed / counter cases
- expected four-word output for each case

The repository test `tests/test_prng_vectors.py` validates the JSON with a small independent Python reference implementation of the round function. The Python helper is only a test oracle for the saved vector; it is not a decoder implementation and should not become the production SIDF PRNG path.

## Boundary

Python remains responsible for:

- experiment orchestration
- metrics and visualization
- non-bit-perfect candidate experiments

Rust core should eventually own:

- PRNG and seed expansion
- fixed-point arithmetic
- update order
- proposal generation
- accept/reject loop
- deterministic output serialization

## Limitation

This test vector does not show that Model C or Model D improves image quality, compression, or super-resolution behavior. It only fixes one candidate PRNG mapping enough for future Rust code to verify exact integer output.
