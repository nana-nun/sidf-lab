# SIDF Reference Links

SIDF研究に関係するWeb資料、実装例、ドキュメントへのリンクを管理します。

論文としてBibTeX管理できるものは、可能な範囲で `references/papers.bib` にも追加します。読書メモや解釈は `references/notes/` に分けて保存します。

## Links

| Topic | Title | URL | Note |
| --- | --- | --- | --- |
| MRF | Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images | https://www.dam.brown.edu/people/geman/Homepage/Image%20processing%2C%20image%20analysis%2C%20Markov%20random%20fields%2C%20and%20MCMC/stochastic%20relaxation.pdf | Issue #12 の調査候補 |
| Edge-preserving smoothing | Scale-Space and Edge Detection Using Anisotropic Diffusion | https://www.sci.utah.edu/~gerig/CS7960-S2010/materials/Perona-Malik/PeronaMalik-PAMI-1990.pdf | Issue #13 の調査候補 |
| Edge-preserving smoothing | CaltechAUTHORS record: Scale-space and edge detection using anisotropic diffusion | https://authors.library.caltech.edu/records/1p8h5-5x870 | DOIと書誌情報の確認元 |
| Guided reconstruction | Guided Image Filtering | https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf | Issue #14 の調査候補 |
| Guided reconstruction | Guided Image Filtering mirror | https://mmlab.ie.cuhk.edu.hk/2010/eccv10_Guided.pdf | Issue #14 で参照した論文PDF |
| Guided reconstruction | Joint Bilateral Upsampling | https://johanneskopf.de/publications/jbu/paper/FinalPaper_0185.pdf | Issue #14 で参照した joint upsampling 論文PDF |
| Non-local reconstruction | A Non-Local Algorithm for Image Denoising | https://doi.org/10.1109/CVPR.2005.38 | Issue #120 の非局所自己類似性検討 |
| Non-local reconstruction | A Review of Image Denoising Algorithms, with a New One | https://doi.org/10.1137/040616024 | Non-local Means の拡張説明と比較背景 |
| Edge-preserving smoothing | Fast Bilateral Filtering for the Display of High-Dynamic-Range Images | https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf | bilateral filter の基礎文献 |
| Interpolation baseline | Cubic Convolution Interpolation for Digital Image Processing | https://doi.org/10.1109/TASSP.1981.1163711 | bicubic baseline の古典的文献 |
| Super-resolution baseline | Learning a Deep Convolutional Network for Image Super-Resolution | https://doi.org/10.1007/978-3-319-10593-2_13 | 超解像代表例。SIDFの主張と混同しないための背景 |
| Texture | An Image Synthesizer | https://doi.org/10.1145/325334.325247 | Perlin noise / procedural texture の背景候補 |
| Texture | Wavelet Noise | https://doi.org/10.1145/1186822.1073264 | aliasingやdetail lossを避けるprocedural noise候補 |
| Texture | Procedural Noise Using Sparse Gabor Convolution | https://doi.org/10.1145/1576246.1531360 | spectral / directional controlを持つstructured noise候補 |
| Texture | A Survey of Procedural Noise Functions | https://doi.org/10.1111/j.1467-8659.2010.01827.x | procedural noise候補の分類と比較の入口 |
| Texture | Gabor Noise by Example | https://doi.org/10.1145/2185520.2185569 | exemplarからspectral parametersを推定する方向性の参考 |
| Determinism | Random123 | https://random123.com/ | Issue #16 の調査候補 |
| Determinism | Parallel Random Numbers: As Easy as 1, 2, 3 | https://doi.org/10.1145/2063384.2063405 | counter-based PRNG の代表文献 |
| Determinism | NumPy random Generator documentation | https://numpy.org/doc/stable/reference/random/generator.html | 現行Python実装の `default_rng` と移植時の非互換点確認 |
| Determinism | rand_pcg crate documentation | https://docs.rs/rand_pcg/ | Rustでportable RNGを使う場合の候補確認 |
| Determinism | fixed crate documentation | https://docs.rs/fixed/ | Rust fixed-point 実装候補の調査入口 |
| Determinism | The Rust Reference: Operator expressions / Overflow | https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow | Rust整数演算のoverflow確認。debug/release差を避けるため、SIDF仕様では演算規則を明示する必要がある |
| Determinism | Rust `u32` primitive integer methods | https://doc.rust-lang.org/std/primitive.u32.html | `wrapping_*`、`saturating_*`、`overflowing_*` など、固定小数点候補で使う整数演算の確認元 |
| Quantum-inspired INR | Data re-uploading for a universal quantum classifier | https://doi.org/10.22331/q-2020-02-06-226 | data re-uploadingの基本文献 |
| Quantum-inspired INR | The effect of data encoding on the expressive power of variational quantum-machine-learning models | https://doi.org/10.1103/PhysRevA.103.032430 | parameterized quantum circuitとFourier表現の理論 |
| Quantum-inspired INR | Power and limitations of single-qubit native quantum neural networks | https://arxiv.org/abs/2205.07848 | single-qubit QNNの表現能力と多変数制限 |
| Quantum-inspired INR | Quantum Implicit Neural Representations | https://arxiv.org/abs/2406.03873 | QIRENによるsignal・image representation |
| Quantum-inspired INR | QINR Autoencoder / VAE | https://arxiv.org/abs/2603.06755 | 2026年の画像再構成・生成decoder応用 |
| Classical INR | Implicit Neural Representations with Periodic Activation Functions | https://arxiv.org/abs/2006.09661 | SIREN baseline |
| Classical INR | Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains | https://arxiv.org/abs/2006.10739 | Fourier feature baseline |
| INR compression | COIN: COmpression with Implicit Neural representations | https://arxiv.org/abs/2103.03123 | per-image INRとparameter量子化による画像圧縮 |
| INR compression implementation | EmilienDupont/coin | https://github.com/EmilienDupont/coin | COIN公式実装。SIREN構成、half precision注意、Kodak実験再現コマンドの確認元 |
| INR compression | COIN++: Neural Compression Across Modalities | https://arxiv.org/abs/2201.12904 | 共有base networkとinstance modulationを分け、modulationを量子化・entropy codingする方式 |
| INR compression | Implicit Neural Representations for Image Compression | https://arxiv.org/abs/2112.04267 | INR圧縮での量子化、quantization-aware retraining、entropy codingの整理候補 |
| INR compression | Compression with Bayesian Implicit Neural Representations | https://arxiv.org/abs/2305.19185 | weight posterior sampleをrelative entropy codingするINR圧縮方式 |

## 記録する情報

- URL
- タイトル
- 著者または組織
- 公開年または更新日
- SIDF研究との関係
- 参照した日付
