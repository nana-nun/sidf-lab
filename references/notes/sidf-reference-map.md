# SIDF Reference Map

## Source

このメモは、SIDFの位置づけに必要な代表文献をテーマ別に整理する入口である。個別文献の詳細読解は、必要に応じてテーマ別Issueで行う。

Related Issues:

- #7: 画像再構成・MRF・超解像の参考文献を集める
- #12: MRF・Gibbs分布・確率的緩和による画像復元
- #13: Model C と異方性拡散・エッジ保持平滑化
- #14: Model D と Guided Filter / guided upsampling
- #15: structured noise prior と procedural texture synthesis
- #16: deterministic PRNG と bit-perfect 再現性

## Summary

SIDFは、現時点では実用圧縮形式ではなく、低解像度guide、seed、物理パラメータ、決定論的な確率的緩和過程による画像再構成を検証する研究である。

そのため、背景文献は一つの分野に閉じない。少なくとも次の軸を分けて扱う必要がある。

- 確率的画像復元: MRF、Gibbs distribution、stochastic relaxation
- エッジ保持平滑化: anisotropic diffusion、edge-preserving smoothing
- guideを使う再構成: guided filter、guided upsampling
- 単純baseline: bilinear / bicubic interpolation
- 超解像研究: SIDFと混同しないための比較対象
- procedural texture: deterministic texture priorの候補
- 再現性: seed、counter-based PRNG、fixed-point arithmetic

## Reference Table

| Topic | Representative reference | Why it matters for SIDF | Current status |
| --- | --- | --- | --- |
| MRF / stochastic relaxation | Geman and Geman 1984, "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images" | Model Cのenergy、Gibbs分布、確率的緩和を説明する背景 | BibTeX追加。詳細メモは #12 |
| Edge-preserving smoothing | Perona and Malik 1990, "Scale-Space and Edge Detection Using Anisotropic Diffusion" | Model Cの `J_ij` がエッジをまたぐ混合を弱める考え方と比較できる | BibTeX追加。メモあり |
| Guided reconstruction | He, Sun, and Tang 2010, "Guided Image Filtering" | Model Dのconfidence-aware reconstructionを既存のguided filtering系と比較する入口 | BibTeX追加。詳細メモは #14 |
| Interpolation baseline | Keys 1981, "Cubic Convolution Interpolation for Digital Image Processing" | bicubic baselineの古典的背景。Model D評価では単純補間との差分が重要 | BibTeX追加 |
| Super-resolution baseline | Dong et al. 2014, "Learning a Deep Convolutional Network for Image Super-Resolution" | SIDFを超解像モデルと混同しないための代表的背景 | BibTeX追加。SIDFの性能主張には使わない |
| Procedural texture | Perlin 1985, "An Image Synthesizer"; Lagae et al. 2010, "A Survey of Procedural Noise Functions" | white noiseではないdeterministic texture priorの候補整理に関係 | BibTeX追加。詳細メモは `structured-texture-prior.md` |
| Deterministic PRNG | Salmon et al. 2011, "Parallel Random Numbers: As Easy as 1, 2, 3"; Random123 documentation | Rust移植時のcounter-based PRNG候補、更新順序非依存性の検討に関係 | BibTeX追加。詳細メモは `deterministic-prng-bit-perfect.md` |

## Relevance to SIDF

### Model C

Model Cは、guideへのdata fidelityと、guide差に基づくedge-aware interactionを持つ。これはMRF/Gibbs型のenergyモデルや、Perona-Malik型のedge-preserving smoothingと比較できる。

ただし、Model CはPDEそのものでも、既存のBayesian image restorationそのものでもない。SIDFで重要なのは、seedと保存可能な設定を含む再構成条件として扱う点である。

### Model D

Model Dは、低解像度guideを使い、confidence mapで拘束の強さを変えながら再構成する。これはguided filter / guided upsampling系の文献と比較できる。

現時点では、Model Dを「超解像」と断定しない。Ground Truth比較、baseline比較、metrics、limitationsが揃うまでは、confidence-aware multi-resolution reconstruction と呼ぶのが安全である。

### Texture

white noise textureは粒状感に留まりやすい。Perlin noiseやfractal noiseのようなprocedural textureは候補になるが、SIDFで自然な質感を改善するかは未検証である。

### Determinism

Python実装では同一環境のNumPy再現性を確認しているが、Rust移植後のbit-perfect再現性にはPRNG、固定小数点、丸め規則、更新順序の固定が必要になる。Random123系のcounter-based PRNGは候補だが、採用判断はまだしていない。

## Limitations

- このメモは文献地図であり、各論文の詳細な読解メモではない。
- 文献の存在は、SIDFの有効性や優位性を示すものではない。
- super-resolution文献は比較対象の背景であり、SIDFが超解像性能を持つという主張には使わない。
- 直接比較実験は未実施である。特にModel Dとguided filter、Model CとMRF/anisotropic diffusionは、同じ入力・同じmetricsで比較する必要がある。
- `docs/research-state.md` は新しい実験結果が出た場合に更新する。この文献地図だけでは研究結果の現在地は変えない。

## Follow-up

- #12でGeman and Geman 1984を読解し、Model Cのenergyとの対応を表にする。
- #14でguided filter / guided upsamplingとModel Dの違いを整理する。
- #15でstructured noise prior候補を比較する。
- #16でRust移植前のPRNGとbit-perfect要件を整理する。
- #6でModel Dをnearest / bilinear / bicubicと比較する実験を行う。
