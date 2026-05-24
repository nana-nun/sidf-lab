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

- `results/2026-05-16-issue-4-model-c-cross-baseline/`
- `config.json`、`metrics.json`、`notes.md`、主要PNGを保存。
- 今回のrunでは `MAD = 0.0117`、`Background Mean = 0.0074`、`Edge Leakage = 0.0089`。

freeze候補benchmark:

- `results/2026-05-16-issue-5-model-c-freeze-benchmark/`
- cross、diagonal、circle、thin line、soft gradientで `config.json`、`metrics.json`、`notes.md`、主要PNGを保存。
- hard edge shapeでは `Model C MAD <= 0.0108`、`Background Mean <= 0.0063`、`Edge Leakage <= 0.0072`。
- soft gradientでは edge leakage を不適用とし、列平均では大きな逆行や急な段差は見られなかった。

解釈:

Model C は、SIDF v0.2.1 の基礎モデルとして有望。ただし創発性は弱く、安定化フィルタに近い。

Perona-Malik 型 diffusion との最小比較:

- `results/2026-05-23-issue-40-model-c-perona-malik/`
- synthetic vertical edge で Model C と Perona-Malik 型 diffusion を比較した。
- Model C の近傍重みは guide 差から固定的に決まり、Perona-Malik 型 conductance は現在の画像状態からstepごとに決まることを、weight map と metrics で保存した。
- 今回の条件では、両者は「大きな局所差のある近傍で混合を弱める」という点で類似するが、係数決定元、更新過程、data fidelity の有無が異なるため、同等の方法とは扱わない。

Perona-Malik 型 diffusion との複数shape比較:

- `results/2026-05-24-issue-64-model-c-perona-malik-shapes/`
- diagonal、circle、soft gradient で Model C と Perona-Malik 型 diffusion を比較した。
- Model C の pair weight は noisy guide から固定的に計算し、Perona-Malik 型 conductance は初期状態と diffusion 後の状態で保存した。
- diagonal / circle では edge leakage を保存し、soft gradient では明確な foreground/background 境界がないため edge leakage を不適用として、列平均の逆行数、slope error、二階差分を代替指標にした。
- 今回の条件では Perona-Malik 型 diffusion の MAD は Model C より小さかったが、これは synthetic noisy guide に対する最小diffusion比較であり、Model C の一般的劣位や Perona-Malik の一般的優位を示すものではない。

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

- `results/2026-05-16-issue-30-model-d-shape-benchmark/`
- diagonal、circle、thin line、soft gradientで nearest / bilinear / bicubic / Model D candidate を保存。
- hard edge shapeでは、今回の設定のModel DはMADでbilinearを上回らなかった。
- Model D MAD: diagonal `0.0448`、circle `0.0351`、thin line `0.0436`。
- Model D edge leakage: diagonal `0.0738`、circle `0.2579`、thin line `0.2756`。
- soft gradientでは、列平均の大きな逆行や急な段差は検出されなかった。ただしbilinearがsynthetic referenceにほぼ一致するため、Model Dの追加ノイズはMADを悪化させた。

解釈:

Model D の confidence map は保存形式つきで確認できるようになったが、現時点のwhite-noise texture termはhard edgeの数値指標を改善していない。特にthin lineやcircle境界では、baselineとの差分とedge leakageを見ながら、texture termとconfidence/data fidelityの重みを再検討する必要がある。

cross baseline comparison:

- `results/2026-05-17-issue-6-model-d-cross-comparison/`
- nearest、bilinear、bicubic、Model D candidate を64x64 synthetic crossで比較。
- Model D MAD `0.0471` は nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- Model D edge leakage `0.2206` は bilinear `0.2197` とほぼ同程度で、nearest `0.1284`、bicubic `0.2119` より悪かった。
- Model D edge width `2.7456` は bilinear `2.8772` よりわずかに小さいが、MADと背景漏れの悪化を伴う。

解釈:

このcross比較では、現行Model D candidateは単純補間に対する総合的な改善を示していない。confidence mapが境界拘束として働く可能性は残るが、white-noise texture termと現在の重みでは背景漏れやreference差分を悪化させるため、texture ablationや重み再調整が必要である。

natural patch GT evaluation:

- `results/2026-05-17-issue-36-model-d-natural-patch/`
- Public Domain 画像cropを128x128 Ground Truthとし、32x32 block-average guideから128x128 outputを比較。
- baselineはnearest、bilinear、bicubic upscaling。
- Model D MAD `0.0556` は nearest `0.0454`、bilinear `0.0444`、bicubic `0.0424` より悪かった。
- Model D global SSIM `0.9487` は nearest `0.9534`、bilinear `0.9595`、bicubic `0.9628` より低かった。
- 自然画像では明確なforeground/background境界がないため、edge leakageは使わず、gradient MADとstrong-edge MADを代替指標として保存した。

解釈:

この1枚の自然画像cropでは、現行Model D candidateは単純補間に対する改善を示していない。これはsuper-resolutionやcompressionの否定ではなく、現在のwhite-noise texture termと重み設定では、Ground Truth差分と勾配差分を悪化させる条件があることを示す初期測定である。

texture ablation:

- `results/2026-05-24-issue-37-texture-ablation/`
- synthetic crossで `texture_strength=0.00, 0.10, 0.35, 0.70` を同一low guide / decoder seedで比較。
- `texture_strength=0.00` でも Model D の MAD `0.0483` は nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- 非ゼロtexture_strengthの差は小さく非単調で、MADは `0.0472` から `0.0476`、global SSIMは `0.8789` から `0.8812` の範囲だった。
- mean error は `0.0276` から `0.0287`、background mean は `0.0460` から `0.0471` の範囲で、texture_strengthに比例した単純な一方向biasは確認できなかった。

解釈:

このrunでは、white-noise texture term は synthetic cross に対する意味のある質感生成としては扱えない。むしろ `texture_strength=0` でも背景平均と差分がbaselineより悪いため、現行Model Dのrelaxation経路、confidence/data/interaction重み、texture経路を分けて再評価する必要がある。これは structured texture prior 全体を否定する結果ではなく、現行white-noise texture設定の小規模な切り分け結果である。

weight grid:

- `results/2026-05-24-issue-56-model-d-weight-grid/`
- cross と natural patch で、`texture_strength=0` を中心に `lambda_data`、confidence floor、uniform confidence、現行textureありを小規模grid比較した。
- cross の Model D grid内では `flat_conf_tex0` が最小MAD `0.0403` だったが、nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- natural patch の Model D grid内でも `flat_conf_tex0` が最小MAD `0.0529` だったが、nearest `0.0454`、bilinear `0.0444`、bicubic `0.0424` より悪かった。
- `low_data_tex0` は cross / natural patch の両方でgrid内最悪寄りになり、data fidelityを弱めるだけでは改善しなかった。

解釈:

この小規模gridでは、現行Model D candidateの主要重みを少し振っても単純補間に対する総合改善は確認できなかった。`flat_conf_tex0` がgrid内で相対的に良かったため、現行のgradient-based confidence mapが常に改善方向に働いているとは限らない。ただし、これはconfidence map一般の否定ではなく、現在のconfidence設計、data fidelity、pairwise interaction、relaxation設定の組み合わせに対する負の結果である。次の切り分けは Issue #61 で扱う。

structured texture prior comparison:

- `results/2026-05-24-issue-63-structured-texture-prior/`
- cross と natural patch で、`texture_strength=0`、white noise、smoothed noise、fractal value noise を nearest / bilinear / bicubic baseline と比較した。
- cross の Model D texture 条件内では white noise が最小MAD `0.0468` だったが、nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- natural patch の Model D texture 条件内では `texture_strength=0` が最小MAD `0.0560` だったが、nearest `0.0454`、bilinear `0.0444`、bicubic `0.0424` より悪かった。
- smoothed noise / fractal value noise は white noise と異なるtexture fieldとして保存できたが、このrunでは structured texture prior を改善要因とは解釈しない。

解釈:

この結果は structured texture prior 全体の否定ではなく、現行 Model D の texture target 二乗項と初期状態混入の経路に smoothed / fractal field を入れた小規模比較である。意味的ディテール生成、super-resolution、compression の成立は示していない。

term isolation:

- `results/2026-05-24-issue-61-model-d-term-isolation/`
- `texture_strength=0` に固定し、data fidelity only、pairwise only、data+pairwise、gradient confidenceあり/なしを cross と natural patch で比較した。
- cross の term conditions 内では `data_pairwise_uniform` が最小MAD `0.0404` だったが、nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- natural patch の term conditions 内でも `data_pairwise_uniform` が最小MAD `0.0529` だったが、nearest `0.0454`、bilinear `0.0444`、bicubic `0.0424` より悪かった。
- `pairwise_only` は cross MAD `0.1296`、natural patch MAD `0.0713` で、data fidelityなしの復元条件としては不十分だった。
- `data_only_conf` と `data_pairwise_conf` は、対応する uniform confidence 条件より悪く、この設定では現行gradient confidenceの空間重み付けが有利に働かなかった。

解釈:

この結果は、現行Model D式の単純な重み探索や white-noise texture 調整ではbaseline改善に届きにくいことを示す。次は大きなgridではなく、confidence map の設計、pairwise term の形、またはrelaxation objectiveそのものを別候補として再設計する必要がある。再設計候補の比較は Issue #67 で扱う。

## Open Questions

- Model D の white-noise texture term は、今回のsynthetic cross ablationでは改善要因とは見えなかった。この傾向は自然画像patchや他shapeでも再現するか。
- term isolationでも単純補間を上回らなかったため、Issue #67 で confidence map や pairwise term の再設計候補をどう作るか。
- white noise ではなく structured noise prior を使うと、baseline差分や粒状感は改善するか。
- 現行 Model D が baseline を上回っていない結果を、v0.3 draft仕様へどの範囲で反映するか。
- Rust固定小数点実装に移したとき、同じ結果を再現できるか。
- decode time は小画像以外で実用的か。

### Decode Time

scaling benchmark:

- `results/2026-05-17-issue-35-decode-time-scaling/`
- synthetic crossで 32x32、64x64、128x128、256x256 を12 sweeps固定で計測。
- Model C decode seconds: 32x32 `0.083`、64x64 `0.349`、128x128 `1.370`、256x256 `5.681`。
- Model D decode seconds: 32x32 `0.152`、64x64 `0.567`、128x128 `2.347`、256x256 `9.441`。
- bilinear baselineは256x256でも `0.002` 秒程度で、現行Pythonの確率的緩和decodeとは桁が違う。

解釈:

現行のMetropolis型Python実装は、少なくともこの設定では画素数に近いスケールで時間が増える。256x256でも短時間runは可能だが、12 sweeps固定であり収束品質は確認していないため、実用性や高品質再構成を示す結果ではない。

sweep quality benchmark:

- `results/2026-05-24-issue-65-decode-sweep-quality/`
- synthetic crossで 64x64 と 128x128 を対象に、sweeps `1, 4, 12, 24` の decode time と metrics を保存した。
- 64x64 Model D decode seconds は `0.089, 0.334, 0.997, 2.812`、MAD は `0.0525, 0.0564, 0.0562, 0.0484`。
- 128x128 Model D decode seconds は `0.480, 2.113, 5.582, 11.329`、MAD は `0.0428, 0.0456, 0.0484, 0.0374`。
- 24 sweepsでは Model D のMADが短いsweepより改善したが、64x64 / 128x128 とも bilinear / bicubic baseline のMADよりは悪かった。

解釈:

同一サイズ内ではsweep数の増加に伴ってdecode timeが増える。今回のrunは画素数scalingではなくsweep scalingを切り出した制限整理であり、現行Python実装の速度制限と、sweep数を増やすだけでは単純補間baselineを上回らない条件があることを示す。

## Analysis Checklist for AI Agents

新しい実験や分析を始める前に、AIエージェントは次を確認する。

1. `AGENTS.md`
2. `docs/research-plan.md`
3. `docs/sidf-research-notes.md`
4. 関連する `results/*/notes.md`
5. 関連する `references/notes/`
6. baseline と metrics を明確にする。
7. `Limitations` と `Next` を必ず書く。
