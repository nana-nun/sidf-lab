# Perona and Malik 1990: 異方性拡散とModel Cの比較

## Source

- Title: Scale-Space and Edge Detection Using Anisotropic Diffusion
- Authors: Pietro Perona and Jitendra Malik
- Year: 1990
- Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(7), 629-639
- DOI: https://doi.org/10.1109/34.56205
- CaltechAUTHORS: https://authors.library.caltech.edu/records/1p8h5-5x870
- PDF候補: https://www.sci.utah.edu/~gerig/CS7960-S2010/materials/Perona-Malik/PeronaMalik-PAMI-1990.pdf
- Related Issue: https://github.com/nana-nun/sidf-lab/issues/13

## Summary

Perona and Malik 1990 は、通常のGaussian scale-spaceがスケールを上げるほどエッジ位置を曖昧にしやすい問題に対し、画像内容に応じて拡散係数を変える非線形拡散を提案している。

基本形は、画像 `I(x, y, t)` を時間 `t` で変化させる拡散過程として書ける。

```text
∂I / ∂t = div(c(x, y, t) ∇I)
```

ここで `c(x, y, t)` は拡散係数であり、勾配が小さい領域では拡散を強くし、勾配が大きいエッジ付近では拡散を弱くする。これにより、領域内部の平滑化を進めながら、領域境界をぼかしにくくする。

代表的な係数の考え方は、`|∇I|` が大きいほど `c` を小さくすることにある。これは「エッジをまたぐ混合を抑える」ための重み付けと見なせる。

## Model Cとの対応

SIDF Model C は、現在の実装では次の2種類の項を持つ。

```text
lambda_data * (v_i - s_i)^2
J_ij * (v_i - v_j)^2
J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

対応関係:

| 観点 | Perona-Malik anisotropic diffusion | SIDF Model C |
| --- | --- | --- |
| 平滑化の対象 | 画像 `I` を時間発展させる | decoder state `v` を確率的緩和で更新する |
| エッジ保持の重み | 勾配に応じた拡散係数 `c(|∇I|)` | guide差に応じた近傍結合 `J_ij` |
| エッジをまたぐ混合 | 勾配が大きい場所で拡散を弱める | `s_i - s_j` が大きい場所で相互作用を弱める |
| データへの拘束 | 元論文の主眼はscale-space / edge detection | `lambda_data * (v_i - s_i)^2` でguideへの忠実度を持つ |
| 更新の性質 | PDEに基づく反復的な拡散 | seedつき確率的緩和 / annealing |

Model C の `J_ij` は、Perona-Malik の拡散係数と同じく「エッジらしい場所で近傍混合を弱める」役割を持つ。このため、Model C は単なる一様ぼかしではなく、guide上の局所差を使うエッジ保持型の再構成モデルとして説明できる。

## Relevance to SIDF

SIDFで重要なのは、低解像度または簡約されたguideから、seedと物理パラメータで再構成を行う点である。Model CはPerona-Malikそのものではないが、edge-aware interaction の設計意図は異方性拡散と近い。

特に、次の説明に使える。

- `J_ij` は、領域内部では近傍値を揃える方向に働く。
- guide上で差が大きい境界では、近傍結合が弱まり、明部が背景へ漏れにくくなる。
- したがって、Model Cは「一様な平滑化」ではなく「guideで制御されたエッジ保持平滑化」に近い。

## Differences from SIDF Model C

相違点も大きい。

- Perona-Malikは画像 `I` 自体の勾配から拡散係数を決めるが、Model Cは主にguide `s` の差から `J_ij` を決める。
- Perona-MalikはPDE的な時間発展として説明されるが、Model Cは局所energyと確率的緩和で説明される。
- Model Cには `lambda_data` によるguide fidelityが明示的にある。
- SIDFではseed、保存可能な設定、decoder再現性が研究対象になる。
- Model CがPerona-Malikより優れているとは、現時点では言えない。

## Limitations

- このメモはPerona-Malikの詳細な数値解析や安定性条件を網羅していない。
- Model Cと異方性拡散の比較は概念対応であり、同じ入力での直接比較実験はまだ行っていない。
- `J_ij` と `c(|∇I|)` の関数形やパラメータ対応は未検証。
- Model Cの優位性、圧縮性能、超解像性能を示すものではない。
- `docs/research-state.md` は実験結果の現在地を優先するため、この文献メモだけでは更新しない。

## Follow-up

- Model CとPerona-Malik diffusionを同じsynthetic guideで比較する実験Issueを作る。
- `J_ij` と拡散係数 `c` の対応を、数式上どこまで厳密に言えるか整理する。
- edge widthやedge leakageを使い、Model C、単純平滑化、異方性拡散を比較する。
- MRF / Gibbs分布の文脈では Issue #12 と接続して、Model Cのenergy表現を整理する。
