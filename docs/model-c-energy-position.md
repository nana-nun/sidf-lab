# Model C energy の確率モデル上の位置づけ

Date: 2026-05-16
Related Issue: [#26](https://github.com/nana-nun/sidf-lab/issues/26)

この文書は、SIDF Model C の energy を、MRF prior、data likelihood、posterior energy、または deterministic decoder objective のどれとして説明するのが現時点で最も正確かを整理する。

結論から言うと、Model C の energy は、現段階では「MRF / Gibbs 型の画像復元と対応づけて読める energy-based decoder objective」と書くのが最も安全である。厳密な posterior distribution や MAP 推定としては、まだ必要な仮定を明示していない。

## 前提

Model C は、同解像度 guide `s` と decoder state `v` に対して、現在の実装では次の局所 energy を使う。

```text
lambda_data * (v_i - s_i)^2
+ sum_j J_ij * (v_i - v_j)^2

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

`lambda_data` は guide への拘束を強める係数であり、`J_ij` は guide 上で差が大きい近傍の結合を弱める edge-aware interaction である。

## Data term の位置づけ

`lambda_data * (v_i - s_i)^2` は、仕様案や人間向け文書では `data fidelity term` と呼ぶのがよい。

理由:

- 実装上は、出力 `v_i` を guide `s_i` に近づける二乗誤差項として定義されている。
- Gaussian observation noise を仮定すれば、負の対数尤度に似た data likelihood term として読める。
- しかし SIDF では、まだ guide の生成過程、ノイズ分布、観測モデルを formal に定義していない。

したがって、`data likelihood` と呼ぶ場合は「Gaussian observation model を仮定した場合」と限定する。通常の説明では `data fidelity` を使う。

## Interaction term の位置づけ

`J_ij * (v_i - v_j)^2` は、`edge-aware interaction term` または `pairwise smoothness term` と呼ぶのがよい。

MRF / Gibbs 型の画像復元との対応では、近傍画素の依存関係を表す pairwise prior に近い役割を持つ。ただし、`J_ij` は観測または guide `s` の差から決まるため、単純な固定 prior として断定しない。

安全な表現:

- Model C は、guide 差に応じて近傍相互作用を弱める。
- `J_ij` は、エッジをまたぐ平滑化を抑える pairwise interaction として働く。
- MRF / Gibbs 型の energy と同じ形で読める部分がある。

避ける表現:

- Model C は厳密な MRF prior である。
- `J_ij` は画像の真の事前分布を表している。
- Model C は Bayesian restoration を実装している。

## Posterior / MAP と呼ぶために足りない仮定

Model C を posterior energy や MAP 推定として書くには、少なくとも次を定義する必要がある。

1. Latent clean image `v` と guide `s` の確率変数としての関係。
2. `p(s | v)` に相当する observation model。
3. `p(v)` に相当する prior、または `p(v | s)` に相当する conditional prior。
4. `J_ij` が guide `s` に依存することを、prior、likelihood、または conditional random field のどれとして扱うか。
5. Continuous value `[0, 1]` に対する measure、normalization、boundary condition。
6. Annealing decoder が posterior sampling なのか、MAP 近似なのか、単なる低 energy 探索なのか。
7. Seed、proposal、temperature schedule、有限 sweep 数が推定結果に与える意味。

これらが未定義の間は、Model C を posterior や MAP と断定しない。

## 仕様案に入れる文言

仕様案では、次のように書く。

```text
Model C energy は、MRF / Gibbs 型の画像復元と概念的に対応する data fidelity term と edge-aware pairwise interaction term を持つ。ただし、SIDF draft では観測モデル、prior、posterior distribution を formal に定義していないため、この energy は確率モデルそのものではなく、seed つき緩和 decoder が最小化または低減しようとする deterministic decoder objective として扱う。
```

この表現なら、Geman and Geman 1984 との概念対応を使いつつ、Model C を確率モデルとして断定しすぎない。

## 研究結果との関係

この整理は、新しい実験結果ではない。Model C の既存結果は `docs/research-state.md` と `results/2026-05-16-model-c-*` に残っている。

現時点で言えること:

- Model C は、synthetic grayscale guide では data fidelity と edge-aware interaction により背景漏れを抑える結果を示した。
- MRF / Gibbs / stochastic relaxation の文脈は、Model C の energy を説明する背景として有用である。
- Model C が厳密な Bayesian restoration、実用圧縮形式、または自然画像の一般復元モデルであるとは言えない。

## Perona-Malik 型 diffusion との直接比較

Issue #40 では、synthetic vertical edge で Model C と Perona-Malik 型 diffusion の最小比較を保存した。

参照:

- `results/2026-05-23-model-c-perona-malik/notes.md`
- `references/notes/perona-malik-anisotropic-diffusion.md`

この比較で確認した主な違い:

- Model C の近傍重み `J_ij` は guide `s` の差から決まり、今回の実験では noisy guide に対して固定された重みとして扱った。
- Perona-Malik 型 diffusion の conductance は、現在の画像状態の局所差から各stepで決まるため、初期画像と拡散後の画像で係数分布が変わる。
- Model C には guide への data fidelity が明示的にあり、Perona-Malik 型 diffusion は今回の最小実装では data fidelity を持たない。

したがって、両者は「局所差が大きい近傍で混合を弱める」という意味では類似するが、係数の決定元、更新過程、目的関数が異なる。現時点では「同等の効果」とは書かず、「guideで制御された edge-aware interaction と、画像勾配で制御される anisotropic diffusion には対応する直観がある」と表現する。

## Limitations

- この文書は確率モデルの完全な定式化ではない。
- `J_ij` を conditional prior として扱うか、decoder objective の重みとして扱うかは未確定である。
- 現在の Python 実装は同一環境での再現性を対象としており、Rust 固定小数点実装での bit-perfect 再現性は未整理である。
- Model C と古典的な MRF restoration を同一入力・同一 metrics で比較する実験はまだ行っていない。
- Issue #40 の比較は synthetic vertical edge 1条件だけであり、Perona-Malik 型 diffusion の一般的な評価やModel Cの優位性を示すものではない。

## Next

- `specs/sidf-v0.3.0-draft.md` では、Model C / D の energy を decoder objective として明記する。
- 確率モデルとして formal に扱う場合は、別Issueで observation model、prior、posterior、decoder procedure の境界を定義する。
