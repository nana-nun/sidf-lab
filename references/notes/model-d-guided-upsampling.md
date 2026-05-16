# Model D と Guided Filter / Guided Upsampling

## Source

Related Issue: [#14](https://github.com/nana-nun/sidf-lab/issues/14)

Primary sources:

- Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering", ECCV 2010. DOI: `10.1007/978-3-642-15549-9_1`
- Johannes Kopf, Michael F. Cohen, Dani Lischinski, Matt Uyttendaele, "Joint Bilateral Upsampling", ACM Transactions on Graphics 2007. DOI: `10.1145/1276377.1276497`
- Frédo Durand, Julie Dorsey, "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images", SIGGRAPH 2002. DOI: `10.1145/566570.566574`

## Summary

Guided Image Filtering は、入力画像または別の guidance image の構造を使って、edge-preserving smoothing や detail transfer に使える明示的なフィルタを提案している。局所線形モデルに基づき、bilateral filter と似たエッジ保持性を持つが、エッジ付近の挙動や計算量の扱いが異なる。

Joint Bilateral Upsampling は、低解像度で計算した solution を高解像度へ戻すとき、高解像度 input image を guidance として使う。単純な upsampling が滑らかさだけを仮定するのに対し、高解像度 guidance の構造を prior として利用する点が重要である。

Bilateral filtering は、空間距離と画素値差の両方で重みを決める edge-preserving filter であり、guided filter / joint upsampling と比較するときの基礎になる。

## Relevance to SIDF

Model D は、16x16 guide を bilinear upscaling し、gradient-based confidence map と edge-aware interaction を使って 64x64 output を生成する。

既存手法との近い点:

- low-resolution 情報を high-resolution output に戻す課題を扱う。
- edge-aware な拘束を使い、エッジをまたぐ平滑化や漏れを抑えようとする。
- nearest / bilinear / bicubic だけでなく、guide を使う再構成系と比較すべきである。

既存手法との違い:

- Joint bilateral upsampling は通常、高解像度 guidance image を持つ。現行 Model D は低解像度 guide をupscaleしたものから confidence map を作るため、外部の高解像度構造情報を持たない。
- Guided filter は明示的な deterministic filter であり、Model D は seeded stochastic relaxation と energy objective を使う。
- Model D の texture term は現状 white noise に近く、guided filter 系の structure transfer とは役割が違う。
- Model D は現時点で実用的な super-resolution model ではなく、confidence-aware multi-resolution reconstruction の候補である。

## Comparison Axes

Model D を guided reconstruction 系と比較するときは、次を分ける。

| Axis | Guided filter / joint upsampling | Current Model D |
| --- | --- | --- |
| Guidance source | 高解像度 input / guidance image を使うことが多い | low-resolution guide をupscaleして confidence を作る |
| Operation | 明示的フィルタまたは joint bilateral interpolation | stochastic relaxation / decoder objective |
| Edge behavior | guidance の構造で edge-aware smoothing | guide difference と confidence map で相互作用を調整 |
| Determinism | 実装と入力が固定なら deterministic | seed、PRNG、update order、proposal に依存 |
| Texture/detail | guidance 由来の構造を使う | white-noise texture term は意味的ディテールではない |
| Evaluation | guidance と target の関係を明確にする必要 | Ground Truth と baseline を分けて保存する必要 |

## Baseline Implications

次に実装または比較すべき baseline:

1. Guided filter baseline: high-resolution reference または downsampled/upscaled guide を guidance として、bilinear result を edge-aware に整える。
2. Joint bilateral upsampling baseline: low-resolution solution を high-resolution guidance で upsample する。自然画像 Ground Truth 実験では特に重要。
3. Bilateral filter smoothing baseline: bilinear output に対する単純な edge-preserving smoothing として比較する。
4. Texture ablation: Model D から texture term を外し、confidence / edge-aware interaction だけの効果を見る。

Issue #6 と #30 の結果では、現行 Model D candidate は nearest / bilinear / bicubic に対して総合的な改善を示していない。したがって、guided filter 系baselineとの比較前に、texture term と confidence weight の ablation が必要である。

## Limitations

- このメモは文献整理であり、SIDF上の新しい実験結果ではない。
- Guided filter / joint bilateral upsampling の実装比較はまだ行っていない。
- Model D が既存手法より優れる、または同等であるとは言えない。
- Joint bilateral upsampling と公平に比べるには、高解像度 guidance image を持つ設定と、SIDFの「低解像度guideしか持たない」設定を分ける必要がある。

## Follow-up

- Issue #36 の自然画像 Ground Truth 実験では、high-resolution image を guidance として使う baseline と、low-resolution guide だけを使う baseline を分ける。
- Issue #37 で texture term ablation を行う。
- Guided filter / joint bilateral upsampling の最小Python baselineを追加する場合は、Model D の claims と混ぜずに別Issueで扱う。
