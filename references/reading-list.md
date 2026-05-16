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

## 読む予定

| Topic | Title / Resource | Reason | Status |
| --- | --- | --- | --- |
| MRF | Stuart Geman and Donald Geman, "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images" | Model C/D のエネルギーモデルとの関係を整理する | Note added: `references/notes/geman-geman-stochastic-relaxation.md` |
| Edge-preserving smoothing | Pietro Perona and Jitendra Malik, "Scale-Space and Edge Detection Using Anisotropic Diffusion" | Model C の edge-aware interaction と異方性拡散の関係を整理する | Note added: `references/notes/perona-malik-anisotropic-diffusion.md` |
| Guided reconstruction | Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering" | Model D を guided upsampling と比較する | Added to BibTeX; detailed note planned in #14 |
| Interpolation baseline | Robert G. Keys, "Cubic Convolution Interpolation for Digital Image Processing" | bicubic baseline の古典的背景として確認する | Added to BibTeX |
| Super-resolution baseline | Chao Dong et al., "Learning a Deep Convolutional Network for Image Super-Resolution" | SIDFを超解像と混同しないため、SR代表例を背景として置く | Added to BibTeX |
| Texture | Ken Perlin, "An Image Synthesizer" | deterministic / procedural texture prior の背景候補 | Added to BibTeX; detailed note planned in #15 |
| Determinism | John K. Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" | Rust移植時のcounter-based PRNG候補を整理する | Added to BibTeX; detailed note planned in #16 |

## 横断メモ

- `references/notes/sidf-reference-map.md`: SIDF研究テーマと代表文献の対応表

## 読書メモの保存方針

- 論文や技術資料の要約は `references/notes/<short-name>.md` に保存する。
- BibTeXで管理できる文献は `references/papers.bib` に追加する。
- Web記事、仕様ページ、実装例などは `references/links.md` に追加する。
- SIDFへの関係は、論文の主張そのものと分けて「SIDFへの関連」として書く。
