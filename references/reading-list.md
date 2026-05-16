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
| MRF | Stuart Geman and Donald Geman, "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images" | Model C/D のエネルギーモデルとの関係を整理する | To read |
| Edge-preserving smoothing | Pietro Perona and Jitendra Malik, "Scale-Space and Edge Detection Using Anisotropic Diffusion" | Model C の edge-aware interaction と異方性拡散の関係を整理する | Note added: `references/notes/perona-malik-anisotropic-diffusion.md` |
| Guided reconstruction | Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering" | Model D を guided upsampling と比較する | To read |
| Texture | Perlin noise / fractal noise / procedural texture synthesis | structured noise prior の候補を探す | To read |
| Determinism | Random123 / Philox / Threefry | Rust移植時のbit-perfect再現性の前提を調べる | To read |

## 読書メモの保存方針

- 論文や技術資料の要約は `references/notes/<short-name>.md` に保存する。
- BibTeXで管理できる文献は `references/papers.bib` に追加する。
- Web記事、仕様ページ、実装例などは `references/links.md` に追加する。
- SIDFへの関係は、論文の主張そのものと分けて「SIDFへの関連」として書く。
