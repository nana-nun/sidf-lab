# Model C / D energy の確率モデル上の位置づけ

Date: 2026-06-04
Related Issues:

- [#26](https://github.com/nana-nun/sidf-lab/issues/26)
- [#79](https://github.com/nana-nun/sidf-lab/issues/79)

この文書は、SIDF Model C / D の energy terms を、data likelihood、prior、posterior energy、または deterministic decoder objective のどれとして説明するのが現時点で最も正確かを整理する。

結論から言うと、現段階の Model C / D energy は「MRF / Gibbs 型の画像復元と対応づけて読める energy-based decoder objective」と書くのが最も安全である。厳密な posterior distribution、MAP 推定、または正式な Bayesian restoration としては、まだ必要な確率モデルを定義していない。

この整理は仕様確定ではない。目的は、今後 Model D の confidence map、pairwise term、texture prior を再設計するときに「どの仮定を変えているのか」を説明しやすくすることである。

## 前提

Model C は、同解像度 guide `s` と decoder state `v` に対して、現在の実装では次の局所 energy を使う。

```text
lambda_data * (v_i - s_i)^2
+ sum_j J_ij * (v_i - v_j)^2

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

Model D draft では、低解像度 guide を upscaled guide `s` として扱い、confidence map `c_i` と texture field `t_i` を加えた候補 energy として次のように整理している。

```text
lambda_data * sum_i c_i (v_i - s_i)^2
+ sum_(i,j) J_ij (v_i - v_j)^2
+ texture term

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

ただし、現行 Python 実装の texture path は draft の単純な線形項だけではなく、texture target 二乗項と初期状態混入を含む。そのため、Model D の texture term はまだ正式仕様ではなく、候補項として扱う。

## 用語の安全な使い分け

| Term | 現時点で安全な呼び方 | likelihood / prior と呼べる条件 | 未確定事項 |
| --- | --- | --- | --- |
| `lambda_data * (v_i - s_i)^2` | data fidelity term | Gaussian observation model `p(s_i | v_i)` を別途仮定する場合、negative log likelihood に対応づけられる | guide の生成過程、ノイズ分布、低解像度 guide と高解像度 output の関係 |
| `J_ij * (v_i - v_j)^2` | edge-aware interaction / pairwise smoothness | `J_ij` を固定または条件付きで与えた MRF / CRF の pairwise prior に近い項として読める | `J_ij` が guide `s` に依存するため、単純な固定 prior と断定しない |
| confidence-weighted data term | confidence-weighted data fidelity | `c_i` を観測信頼度または逆分散の候補として定義する場合、heteroscedastic likelihood に近づく | 現行 gradient confidence が統計的信頼度を推定しているとは未確認 |
| texture term / texture target | deterministic texture prior candidate | texture field と `v` の関係を prior として定義すれば候補になる | 現行結果では意味的ディテール生成や baseline 改善は確認されていない |
| annealing decoder | stochastic relaxation / low-energy search | posterior sampling や MAP 近似と呼ぶには、target distribution と収束条件が必要 | 有限 sweep、proposal、temperature、seed の推定上の意味 |

通常の人間向け説明では、`likelihood`、`prior`、`posterior` は仮説として限定し、実装済みのものは `data fidelity`、`edge-aware interaction`、`confidence weighting`、`texture prior candidate`、`decoder objective` と呼ぶ。

## Model C の位置づけ

Model C の data term は、出力 `v_i` を guide `s_i` に近づける二乗誤差項である。Gaussian observation noise を仮定すれば負の対数尤度に似た data likelihood term として読めるが、SIDF ではまだ observation model を formal に定義していない。

Model C の interaction term は、guide 差に応じて近傍相互作用を弱める edge-aware pairwise term である。MRF / Gibbs 型の画像復元との対応では pairwise prior に近い役割を持つが、`J_ij` が観測または guide `s` から決まるため、単純な画像 prior として断定しない。

安全な表現:

- Model C は、guide 差に応じて近傍相互作用を弱める。
- `J_ij` は、エッジをまたぐ平滑化を抑える pairwise interaction として働く。
- MRF / Gibbs 型の energy と同じ形で読める部分がある。
- 現段階では deterministic decoder objective として扱う。

避ける表現:

- Model C は厳密な MRF prior である。
- `J_ij` は画像の真の事前分布を表している。
- Model C は Bayesian restoration を実装している。
- Model C は自然画像の一般復元性能を示した。

## Model D の位置づけ

Model D は Model C の data fidelity / pairwise interaction を多解像度条件へ広げ、upscaled guide、confidence map、texture field を加えた候補である。

### Confidence map

Model D の confidence map は、現状では gradient magnitude などから作る拘束強度である。これは「観測の信頼度」や「場所ごとの data likelihood の逆分散」に似た役割として解釈できる可能性がある。

ただし、現行の gradient confidence が統計的信頼度を推定しているとはまだ言えない。Issue #61 の term isolation では、gradient confidence を使う `data_only_conf` や `data_pairwise_conf` が、対応する uniform confidence 条件より悪かった。この結果は confidence map 一般の否定ではないが、現行の gradient confidence を likelihood の信頼度として採用する根拠にはならない。

### Texture

Model D の texture は、seed から決定論的に生成される texture-like variation である。将来的には prior candidate として扱える可能性があるが、現行 white-noise texture や structured texture 比較では、意味的ディテール生成や単純補間 baseline に対する改善は確認されていない。

そのため、texture は現時点では `texture prior candidate` または `deterministic texture field` と呼び、自然画像の失われた高周波を復元する prior と断定しない。

### Pairwise term

Model D の pairwise term は Model C と同じく、upscaled guide 差から決まる edge-aware interaction として扱う。低解像度 guide だけを使う SIDF 条件では、高解像度 guidance image を使う guided filter / joint bilateral upsampling と同じ情報を持っていない。そのため、pairwise term を「高解像度エッジの事前知識」として説明しない。

## Posterior / MAP と呼ぶために足りない仮定

Model C / D を posterior energy や MAP 推定として書くには、少なくとも次を定義する必要がある。

1. Latent clean image `v` と guide `s` の確率変数としての関係。
2. `p(s | v)` に相当する observation model。
3. `p(v)` に相当する prior、または `p(v | s)` に相当する conditional prior。
4. `J_ij` が guide `s` に依存することを、prior、likelihood、または conditional random field のどれとして扱うか。
5. Confidence map `c_i` を観測分散、重み、または heuristic constraint のどれとして扱うか。
6. Texture field `t_i` と output `v_i` の関係を、prior、proposal bias、target field、または initialization のどれとして扱うか。
7. Continuous value `[0, 1]` に対する measure、normalization、boundary condition。
8. Annealing decoder が posterior sampling なのか、MAP 近似なのか、単なる低 energy 探索なのか。
9. Seed、proposal、temperature schedule、有限 sweep 数が推定結果に与える意味。
10. Rust 固定小数点 decoder へ移した場合の丸め、overflow、acceptance 判定の定義。

これらが未定義の間は、Model C / D を posterior や MAP と断定しない。

## Negative results から見える設計論点

保存済みの Model D 比較では、現行 Model D candidate は nearest / bilinear / bicubic baseline を総合的に上回っていない。

特に Issue #61 の term isolation から見える論点:

- `data_pairwise_uniform` は term conditions 内では最小MADだったが、nearest / bilinear / bicubic baseline は上回らなかった。
- `data_only_conf` と `data_pairwise_conf` は、対応する uniform confidence 条件より悪かった。
- `pairwise_only` は、guide への data fidelity なしの復元条件として不十分だった。
- 現行 gradient confidence の空間重み付けは、この設定では有利に働いていない可能性がある。

この結果から、次の設計変更は「同じ式の重み探索を広げる」よりも、どの項の仮定を変えるかを明示して比較するのがよい。

候補:

- confidence map を、gradient magnitude ではなく、flatter confidence、edge-band only、inverse-flat confidence など別の観測信頼度候補として比較する。
- pairwise term を、weaker smoothing、bilateral-like clamp、edge-band限定など別の interaction として比較する。
- texture を、初期状態混入や target 二乗項ではなく、energy 内の独立した prior candidate として切り分ける。
- data fidelity を低解像度 guide の観測モデルとして扱うのか、upscaled guide への近接制約として扱うのかを分ける。

これらは Issue #67、#75、Rust固定小数点関連Issueの判断材料であり、現時点で Model D の優位性を示すものではない。

## 仕様案に入れる文言

仕様案では、次のように書く。

```text
Model C / D energy は、MRF / Gibbs 型の画像復元と概念的に対応する data fidelity term と edge-aware pairwise interaction term を持つ。ただし、SIDF draft では観測モデル、prior、posterior distribution を formal に定義していないため、この energy は確率モデルそのものではなく、seed つき緩和 decoder が最小化または低減しようとする deterministic decoder objective として扱う。
```

Model D については、追加で次のように書く。

```text
Confidence map と texture field は、likelihood weight や prior candidate として解釈できる可能性があるが、現行結果ではそれらが baseline 改善要因であるとは確認されていない。正式仕様ではなく、再設計候補として扱う。
```

この表現なら、Geman and Geman 1984 との概念対応を使いつつ、Model C / D を確率モデルとして断定しすぎない。

## 研究結果との関係

この整理は、新しい実験結果ではない。Model C / D の既存結果は `docs/research-state.md` と各 `results/*/notes.md` に残っている。

現時点で言えること:

- Model C は、synthetic grayscale guide では data fidelity と edge-aware interaction により背景漏れを抑える結果を示した。
- Model C は、SIDF v0.2.1 の基礎モデルとして有望だが、創発性は弱く、安定化フィルタに近い。
- Model D は、confidence-aware multi-resolution reconstruction の候補として観察価値がある。
- 保存済みの Model D 比較では、現行設定が nearest / bilinear / bicubic baseline に対する総合改善を示したとは解釈しない。
- MRF / Gibbs / stochastic relaxation の文脈は、Model C / D の energy を説明する背景として有用である。

言えないこと:

- SIDF が実用圧縮形式として成立した。
- Model D が単純補間に勝っている。
- Ground Truth なしに超解像性能がある。
- texture prior が意味的ディテールを生成している。
- 現行 energy が formal な posterior distribution を定義している。

## Perona-Malik 型 diffusion との直接比較

Issue #40 では、synthetic vertical edge で Model C と Perona-Malik 型 diffusion の最小比較を保存した。

参照:

- `results/2026-05-23-issue-40-model-c-perona-malik/notes.md`
- `references/notes/perona-malik-anisotropic-diffusion.md`

この比較で確認した主な違い:

- Model C の近傍重み `J_ij` は guide `s` の差から決まり、今回の実験では noisy guide に対して固定された重みとして扱った。
- Perona-Malik 型 diffusion の conductance は、現在の画像状態の局所差から各stepで決まるため、初期画像と拡散後の画像で係数分布が変わる。
- Model C には guide への data fidelity が明示的にあり、Perona-Malik 型 diffusion は今回の最小実装では data fidelity を持たない。

したがって、両者は「局所差が大きい近傍で混合を弱める」という意味では類似するが、係数の決定元、更新過程、目的関数が異なる。現時点では「同等の効果」とは書かず、「guideで制御された edge-aware interaction と、画像勾配で制御される anisotropic diffusion には対応する直観がある」と表現する。

## Limitations

- この文書は確率モデルの完全な定式化ではない。
- `J_ij` を conditional prior として扱うか、decoder objective の重みとして扱うかは未確定である。
- Confidence map を likelihood の逆分散や観測信頼度として扱えるかは未確認である。
- Texture term を prior と呼ぶには、texture field と output の関係を再定義する必要がある。
- 現在の Python 実装は同一環境での再現性を対象としており、Rust 固定小数点実装での bit-perfect 再現性は未確定である。
- Model C と古典的な MRF restoration を同一入力・同一 metrics で比較する実験はまだ行っていない。
- Issue #61 の term isolation は cross と1枚の自然画像patchに限られ、Model D の一般的な劣位や別設計候補の否定を示すものではない。

## Next

- Issue #67 では、confidence map や pairwise term の再設計候補を、どの probabilistic interpretation を変えるのかと対応づけて比較する。
- Issue #75 で structured texture prior を扱う場合は、texture が initialization、target term、prior candidate のどれなのかを切り分ける。
- Rust core 移植では、energy の形式だけでなく、PRNG、固定小数点、丸め、acceptance 判定を decoder objective の一部として記録する。
- 確率モデルとして formal に扱う場合は、別Issueで observation model、prior、posterior、decoder procedure の境界を定義する。
