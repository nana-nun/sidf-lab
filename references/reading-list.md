# SIDF Reading List

SIDF研究で読む候補の文献・資料を、調査テーマごとに管理するための一覧です。

このファイルでは「読む予定」「確認中」「読了」を分け、内容の要約やSIDFへの解釈は必要に応じて `references/notes/` に別ファイルとして残します。

## 優先テーマ

- MRF / CRF による画像復元
- simulated annealing / stochastic relaxation
- edge-preserving smoothing / anisotropic diffusion
- guided filter / guided upsampling
- procedural texture synthesis
- deterministic PRNG / fixed-point reproducibility
- quantum-inspired implicit image representation

## 読む予定

| Topic | Title / Resource | Reason | Status |
| --- | --- | --- | --- |
| MRF | Stuart Geman and Donald Geman, "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images" | Model C/D のエネルギーモデルとの関係を整理する | Note added: `references/notes/geman-geman-stochastic-relaxation.md` |
| Edge-preserving smoothing | Pietro Perona and Jitendra Malik, "Scale-Space and Edge Detection Using Anisotropic Diffusion" | Model C の edge-aware interaction と異方性拡散の関係を整理する | Note added: `references/notes/perona-malik-anisotropic-diffusion.md` |
| Guided reconstruction | Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering" | Model D を guided upsampling と比較する | Note added: `references/notes/model-d-guided-upsampling.md` |
| Guided reconstruction | Johannes Kopf et al., "Joint Bilateral Upsampling" | 低解像度solutionを高解像度guidanceで戻すbaseline候補 | Note added: `references/notes/model-d-guided-upsampling.md` |
| Edge-preserving smoothing | Frédo Durand and Julie Dorsey, "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images" | bilateral / guided 系比較の基礎 | Note added: `references/notes/model-d-guided-upsampling.md` |
| Interpolation baseline | Robert G. Keys, "Cubic Convolution Interpolation for Digital Image Processing" | bicubic baseline の古典的背景として確認する | Added to BibTeX |
| Super-resolution baseline | Chao Dong et al., "Learning a Deep Convolutional Network for Image Super-Resolution" | SIDFを超解像と混同しないため、SR代表例を背景として置く | Added to BibTeX |
| Texture | Ken Perlin, "An Image Synthesizer" | deterministic / procedural texture prior の背景候補 | Note added: `references/notes/structured-texture-prior.md` |
| Texture | Robert L. Cook and Tony DeRose, "Wavelet Noise" | Perlin系noiseのaliasing/detail loss回避の参考 | Note added: `references/notes/structured-texture-prior.md` |
| Texture | Ares Lagae et al., "A Survey of Procedural Noise Functions" | procedural noise候補の分類と比較の入口 | Note added: `references/notes/structured-texture-prior.md` |
| Texture | Ares Lagae et al., "Procedural Noise Using Sparse Gabor Convolution" | spectral / directional controlを持つnoise候補 | Note added: `references/notes/structured-texture-prior.md` |
| Texture | Bruno Galerne et al., "Gabor Noise by Example" | exemplarからnoise parameterを推定する方向性の参考 | Note added: `references/notes/structured-texture-prior.md` |
| Determinism | John K. Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" | Rust移植時のcounter-based PRNG候補を整理する | Note added: `references/notes/deterministic-prng-bit-perfect.md` |
| Determinism | Random123 documentation | Philox / Threefry の実装候補とcounter-based RNGの入口 | Note added: `references/notes/deterministic-prng-bit-perfect.md` |
| Determinism | NumPy random Generator documentation | 現行Python実装の `default_rng` とRust移植時の境界を確認する | Note added: `references/notes/deterministic-prng-bit-perfect.md` |
| Determinism | Rust `rand_pcg` / `fixed` crate documentation | Rust core候補のportable RNGとfixed-point実装の調査入口 | Note added: `references/notes/deterministic-prng-bit-perfect.md` |
| Quantum-inspired INR | Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier" | data re-uploadingの基本構成を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Quantum-inspired INR | Schuld et al., "The effect of data encoding on the expressive power of variational quantum-machine-learning models" | encodingと利用可能なFourier周波数の関係を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Quantum-inspired INR | Yu et al., "Power and limitations of single-qubit native quantum neural networks" | single-qubitの1変数表現能力と多変数制限を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Quantum-inspired INR | Zhao et al., "Quantum Implicit Neural Representations" | QIRENの画像表現・super-resolution実験と構造を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Quantum-inspired INR | Eren, "Implementation of Quantum Implicit Neural Representation in Deterministic and Probabilistic Autoencoders" | 2026年のQINR decoder応用と制限を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Classical INR baseline | Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions" | SIRENをModel Eの周期関数baselineとして使う | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| Classical INR baseline | Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" | Fourier feature baselineとspectral bias対策を確認する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |
| INR compression | Dupont et al., "COIN: COmpression with Implicit Neural representations" | parameter quantizationと画像ごとのINR符号化を比較する | Note added: `references/notes/quantum-inspired-implicit-image-representation.md` |

## 横断メモ

- `references/notes/sidf-reference-map.md`: SIDF研究テーマと代表文献の対応表
- `references/notes/quantum-inspired-implicit-image-representation.md`: 量子回路由来の関数表現とModel E候補

## 読書メモの保存方針

- 論文や技術資料の要約は `references/notes/<short-name>.md` に保存する。
- BibTeXで管理できる文献は `references/papers.bib` に追加する。
- Web記事、仕様ページ、実装例などは `references/links.md` に追加する。
- SIDFへの関係は、論文の主張そのものと分けて「SIDFへの関連」として書く。
