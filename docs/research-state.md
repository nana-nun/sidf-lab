# SIDF Research State

この文書は、AIエージェントと人間が「ここまで何が分かっているか」を短時間で確認するための現在地メモです。詳細な議論は `docs/sidf-research-notes.md` を一次参照にしてください。

## Scope

sidf-lab は、現段階では実用圧縮形式ではなく、低解像度ガイドと決定論的確率過程による画像再構成モデルの研究リポジトリです。

## Current Findings

### Model A

外部場項 `-h_i v_i` は、STATICが正の値を持つ場所で輝度を上げる片方向の力になった。

結果:

- 十字構造は誘導された。
- 黒背景が灰色化した。
- 明部が周囲へ膨張した。

解釈:

Model A は画像復元モデルとして不安定。

### Model C

data fidelity `lambda_data * (v_i - s_i)^2` と edge-aware interaction を導入した。

結果:

- 背景が暗く保たれた。
- 十字平均が目標値に近づいた。
- edge leakage が大きく低下した。

代表値:

```text
MAD                 : 0.0195
Cross Mean          : 0.5014
Background Mean     : 0.0085
Edge Leakage        : 0.0097
Cross Variance      : 0.0016
Background Variance : 0.0004
```

保存形式つき再実行:

- `results/2026-05-16-model-c-cross-baseline/`
- `config.json`、`metrics.json`、`notes.md`、主要PNGを保存。
- 今回のrunでは `MAD = 0.0117`、`Background Mean = 0.0074`、`Edge Leakage = 0.0089`。

freeze候補benchmark:

- `results/2026-05-16-model-c-freeze-benchmark/`
- cross、diagonal、circle、thin line、soft gradientで `config.json`、`metrics.json`、`notes.md`、主要PNGを保存。
- hard edge shapeでは `Model C MAD <= 0.0108`、`Background Mean <= 0.0063`、`Edge Leakage <= 0.0072`。
- soft gradientでは edge leakage を不適用とし、列平均では大きな逆行や急な段差は見られなかった。

解釈:

Model C は、SIDF v0.2.1 の基礎モデルとして有望。ただし創発性は弱く、安定化フィルタに近い。

### Model D

16x16 guide から 64x64 output を生成する multi-resolution pipeline を導入した。

構成:

- bilinear upscaled guide
- gradient-based confidence map
- edge-aware interaction
- seeded texture term

結果:

- bilinear より十字境界が視覚的に締まった。
- confidence map がエッジ拘束として機能した。
- texture は出たが、現段階では意味的ディテールではなく粒状ノイズに近い。

解釈:

Model D は confidence-aware multi-resolution reconstruction と呼ぶのが正確。Ground Truth比較なしに「超解像性能」とは主張しない。

shape benchmark:

- `results/2026-05-16-model-d-shape-benchmark/`
- diagonal、circle、thin line、soft gradientで nearest / bilinear / bicubic / Model D candidate を保存。
- hard edge shapeでは、今回の設定のModel DはMADでbilinearを上回らなかった。
- Model D MAD: diagonal `0.0448`、circle `0.0351`、thin line `0.0436`。
- Model D edge leakage: diagonal `0.0738`、circle `0.2579`、thin line `0.2756`。
- soft gradientでは、列平均の大きな逆行や急な段差は検出されなかった。ただしbilinearがsynthetic referenceにほぼ一致するため、Model Dの追加ノイズはMADを悪化させた。

解釈:

Model D の confidence map は保存形式つきで確認できるようになったが、現時点のwhite-noise texture termはhard edgeの数値指標を改善していない。特にthin lineやcircle境界では、baselineとの差分とedge leakageを見ながら、texture termとconfidence/data fidelityの重みを再検討する必要がある。

cross baseline comparison:

- `results/2026-05-17-model-d-cross-comparison/`
- nearest、bilinear、bicubic、Model D candidate を64x64 synthetic crossで比較。
- Model D MAD `0.0471` は nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- Model D edge leakage `0.2206` は bilinear `0.2197` とほぼ同程度で、nearest `0.1284`、bicubic `0.2119` より悪かった。
- Model D edge width `2.7456` は bilinear `2.8772` よりわずかに小さいが、MADと背景漏れの悪化を伴う。

解釈:

このcross比較では、現行Model D candidateは単純補間に対する総合的な改善を示していない。confidence mapが境界拘束として働く可能性は残るが、white-noise texture termと現在の重みでは背景漏れやreference差分を悪化させるため、texture ablationや重み再調整が必要である。

## Open Questions

- Model D は斜線や曲線でも境界を守れるか。
- confidence map は柔らかいグラデーションを硬く分断しないか。
- white noise ではなく structured noise prior を使うと質感は改善するか。
- bilinear / bicubic に対する優位性は metrics で確認できるか。
- Rust固定小数点実装に移したとき、同じ結果を再現できるか。
- decode time は小画像以外で実用的か。

### Decode Time

scaling benchmark:

- `results/2026-05-17-decode-time-scaling/`
- synthetic crossで 32x32、64x64、128x128、256x256 を12 sweeps固定で計測。
- Model C decode seconds: 32x32 `0.083`、64x64 `0.349`、128x128 `1.370`、256x256 `5.681`。
- Model D decode seconds: 32x32 `0.152`、64x64 `0.567`、128x128 `2.347`、256x256 `9.441`。
- bilinear baselineは256x256でも `0.002` 秒程度で、現行Pythonの確率的緩和decodeとは桁が違う。

解釈:

現行のMetropolis型Python実装は、少なくともこの設定では画素数に近いスケールで時間が増える。256x256でも短時間runは可能だが、12 sweeps固定であり収束品質は確認していないため、実用性や高品質再構成を示す結果ではない。

## Analysis Checklist for AI Agents

新しい実験や分析を始める前に、AIエージェントは次を確認する。

1. `AGENTS.md`
2. `docs/research-plan.md`
3. `docs/sidf-research-notes.md`
4. 関連する `results/*/notes.md`
5. 関連する `references/notes/`
6. baseline と metrics を明確にする。
7. `Limitations` と `Next` を必ず書く。
