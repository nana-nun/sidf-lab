# Deterministic PRNG と Bit-perfect 再現性

## Source

Related Issue: [#16](https://github.com/nana-nun/sidf-lab/issues/16)

Primary sources:

- John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011. DOI: `10.1145/2063384.2063405`
- Random123 documentation: https://random123.com/
- NumPy random Generator documentation: https://numpy.org/doc/stable/reference/random/generator.html
- Rust `rand_pcg` documentation: https://docs.rs/rand_pcg/
- Rust `fixed` crate documentation: https://docs.rs/fixed/

## Summary

SIDFの現行Python実装は、同一環境のNumPy上で同じseedから同じ結果を得ることを目標にしている。これは研究プロトタイプとしては十分に有用だが、Rust移植後に「同じserialized inputから実装非依存に同じ画像を得る」こととは別の要件である。

Rust coreで bit-perfect 再現性を狙う場合、少なくとも PRNG、seed expansion、pixel update order、proposal distribution、accept/reject判定、energy計算、丸め、clip、境界条件を仕様として固定する必要がある。

Random123系の counter-based RNG は、乱数状態を逐次更新するのではなく、counter と key から乱数を計算する。Philox と Threefry はその代表的な方式であり、pixel座標、sweep番号、乱数用途をcounterに含める設計にすれば、乱数列を「何回消費したか」に依存しにくい。

## Stateful RNG と Counter-based RNG

| Axis | Stateful RNG | Counter-based RNG |
| --- | --- | --- |
| 基本モデル | RNG stateを更新しながら次の値を得る | `counter` と `key` から値を直接得る |
| 現行Pythonとの関係 | `np.random.default_rng(seed)` は扱いやすいが、NumPy実装・version・呼び出し順に依存する | Rust coreでcounter mappingを仕様化すれば、呼び出し順依存を減らせる |
| update order変更への強さ | 乱数消費順が変わると結果が変わりやすい | counterに `sweep`, `pixel`, `purpose` を入れれば局所的に固定しやすい |
| 並列化 | state分割やjump/stream管理が必要 | pixelごと、stageごとに独立評価しやすい |
| SIDFでのリスク | 実験スクリプト変更で結果がずれる可能性が高い | counter layoutを後から変えると互換性が壊れる |

## Random123 / Philox / Threefry

Random123 は counter-based RNG の実装群で、代表候補として Philox と Threefry がある。SIDFにとって重要なのは、どちらが一般に速いかだけではなく、decoder仕様として固定しやすいかである。

| Candidate | 方向性 | SIDFでの利点 | 注意点 |
| --- | --- | --- | --- |
| Philox | 乗算を含むcounter-based RNG | GPU / parallel RNG文脈で使われる代表候補。counterとkeyの分離が分かりやすい | Rust実装を採用する場合、round数、word幅、endian、出力wordの使い方を固定する必要がある |
| Threefry | Threefish由来の演算を使うcounter-based RNG | 加算・xor・rotation中心の設計で、整数演算として仕様化しやすい | rotation定数、round数、word幅を固定しないと互換性がない |
| PCG系 | stateful RNG候補 | Rust側のportable RNGとして実装しやすい候補がある | counter-basedではないため、pixel単位の独立乱数設計には追加ルールが必要 |

現段階では、Rust coreのbit-perfect targetには Philox または Threefry のような counter-based RNG を第一候補にする。ただし、正式採用はまだ決めない。まずは小さいRust spikeで、同じcounter mappingから固定test vectorを出せるか確認する。

## Fixed-point Arithmetic

浮動小数点は、同じ式でも演算順序、丸め、標準ライブラリ関数、最適化、CPU機能によって結果の微小差が出る可能性がある。現行Model C/Dは `float64`、`math.exp`、Gaussian proposal、`np.clip` を使っているため、そのままRustへ移して実装非依存のbit-perfect結果を保証するのは危険である。

Rust coreで固定すべき候補:

- pixel value: 例として unsigned fixed point `Q0.16` または `Q0.24`
- energy: overflow余裕を持つ signed fixed point または integer accumulator
- guide value: serialized guideを整数gridとして保存し、decode時のfloat変換を避ける
- interaction weight: `exp(-gamma * diff^2)` を直接float計算せず、固定tableまたは整数近似にする
- acceptance rule: `math.exp(-delta / temp)` を使わず、固定tableまたは整数比較にする
- proposal: Gaussian float samplingではなく、固定された整数delta tableまたは整数分布にする

固定小数点化は「精度を上げる」ためではなく、仕様にできる丸め規則を持つための候補である。画質やmetricsが改善することは、この文献メモからは言えない。

## SIDF Decoder Bit-perfect 要件案

Rust移植前に固定すべき要件:

1. Serialized guide format: grayscale値のbit幅、正規化範囲、endian、shape order。
2. Seed fields: `experiment_seed`、`decoder_seed`、texture seed、stream/keyの分離規則。
3. Counter layout: `stage`, `sweep`, `pixel_index`, `proposal_index`, `purpose` をどのwordに入れるか。
4. PRNG algorithm: PhiloxまたはThreefryなど、variant、round数、word幅、出力word順。
5. Pixel traversal: row-major固定、counter-derived permutation、または deterministic checkerboard などのどれを採用するか。
6. Proposal distribution: integer delta、lookup table、または固定近似Gaussianのいずれか。
7. Energy arithmetic: value scale、weight scale、multiplication後のrounding、saturating / wrapping の扱い。
8. Temperature schedule: sweepごとの値をtable化するか、固定小数点式で計算するか。
9. Accept/reject rule: `delta < 0` の扱い、等号、確率比較のthreshold変換。
10. Boundary handling: neighbor order、edge/cornerの扱い、confidence/texture参照範囲。
11. Output serialization: final valueのrounding、PNG保存時とSIDF decoder outputの分離。
12. Test vectors: tiny guide、seed、config、数sweep後のstate hash、最終metricsを保存する。

## Python と Rust Core の境界

Pythonに残す部分:

- experiment orchestration
- parameter search
- plotting and PNG comparison
- metrics aggregation
- exploratory NumPy prototype
- non-bit-perfect candidate model experiments

Rust coreへ移す候補:

- PRNG and seed expansion
- fixed-point value representation
- guide / confidence / texture field decoding when fixed
- energy calculation
- update order
- proposal generation
- accept/reject loop
- deterministic output serialization

PythonはRust coreの出力を読み、metricsや可視化を担当するのがよい。Python側でdecoderの乱数消費順やenergy計算を再実装すると、bit-perfect targetが二重化してずれやすい。

## Relevance to SIDF

SIDFの研究目的では、seedと物理パラメータから再構成過程を再実行できることが重要である。ただし、現段階のModel C/Dの結果は「同一Python環境での再現性」であり、Rust移植後のformalなdecoder互換性ではない。

仕様案では、次の2段階を分けて書くのが安全である。

1. Prototype reproducibility: Python / NumPy / dependency versionを含む同一環境再現性。
2. Decoder reproducibility: Rust core仕様で、PRNG、固定小数点、更新順序まで固定したbit-perfect再現性。

## Limitations

- このメモは文献整理と要件案であり、Rust実装やtest vectorはまだ作っていない。
- Random123、Philox、ThreefryのどれをSIDFで採用するかは未決定である。
- fixed-point scaleやenergy近似を変えると、Python float版Model C/Dと同じ画像にはならない可能性がある。
- bit-perfect化は再現性のための制約であり、画質や圧縮性能を改善する根拠ではない。
- PythonのNumPy再現性を否定するものではない。研究プロトタイプと正式decoder targetを分けるための整理である。

## Follow-up

- Issue #50 で、PRNG候補を1つに絞り、tiny counter test vectorを保存する。初期spikeとして `Philox4x32-10` の設計メモとtest vectorを `references/notes/rust-core-prng-test-vector.md` と `references/prng-test-vectors/philox4x32-10.json` に追加した。
- Model C fixed-point spikeでは、16x16または32x32の小さいguideで、Python referenceと比較可能なenergy / update traceを保存する。
- `specs/sidf-v0.2.1.md` または次のdraft仕様では、Prototype reproducibility と Decoder reproducibility を別項目にする。
