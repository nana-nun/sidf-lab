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

structured texture pair-contrast energy:

- `results/2026-06-14-issue-75-texture-contrast-energy/`
- texture fieldを初期状態やpixel targetへ混ぜず、隣接絶対コントラスト `|t_i - t_j|` を目標統計とする独立energy項を比較した。
- white / smoothed / fractal fieldは平均目標コントラストを `0.02` に揃え、`texture_0` と nearest / bilinear / bicubicをbaselineに含めた。
- crossのdecoder条件内ではsmoothed contrastが最小MAD `0.0488`、natural patchではwhite contrastが最小MAD `0.0557` だったが、どちらも単純補間baselineを上回らなかった。
- natural patchのwhite contrastはtextureなし条件よりMAD、SSIM、gradient magnitude MAD、flat residual stdが改善したが、bicubicのMAD `0.0424` とflat residual std `0.0088` には届かなかった。

解釈:

pair-contrast priorが独立energy項として一部指標を変えることは確認できたが、今回の1統計量・1重みでは改善要因とは解釈しない。自然画像出力には粒状差分が残り、意味的ディテール生成、super-resolution、compressionの成立は示していない。

term isolation:

- `results/2026-05-24-issue-61-model-d-term-isolation/`
- `texture_strength=0` に固定し、data fidelity only、pairwise only、data+pairwise、gradient confidenceあり/なしを cross と natural patch で比較した。
- cross の term conditions 内では `data_pairwise_uniform` が最小MAD `0.0404` だったが、nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` より悪かった。
- natural patch の term conditions 内でも `data_pairwise_uniform` が最小MAD `0.0529` だったが、nearest `0.0454`、bilinear `0.0444`、bicubic `0.0424` より悪かった。
- `pairwise_only` は cross MAD `0.1296`、natural patch MAD `0.0713` で、data fidelityなしの復元条件としては不十分だった。
- `data_only_conf` と `data_pairwise_conf` は、対応する uniform confidence 条件より悪く、この設定では現行gradient confidenceの空間重み付けが有利に働かなかった。

解釈:

この結果は、現行Model D式の単純な重み探索や white-noise texture 調整ではbaseline改善に届きにくいことを示した。confidence map とpairwise termの小規模な再設計候補は Issue #67 で比較した。

confidence / pairwise redesign:

- `results/2026-06-14-issue-67-model-d-redesign-candidates/`
- 現行gradient confidence、uniform対照、flatter confidence、edge-band confidence、uniform confidence + clamped pairwiseをcrossとnatural patchで比較した。
- crossでは `uniform_clamped_pairwise` がModel D候補内の最小MAD `0.0390` で、`uniform_quadratic` の `0.0406` より良かったが、nearest `0.0138`、bilinear `0.0331`、bicubic `0.0351` は上回らなかった。
- natural patchでは `uniform_quadratic` がModel D候補内の最小MAD `0.0525` で、clamped pairwiseは `0.0545` に悪化した。bicubicのMAD `0.0424`、SSIM `0.9628`、gradient magnitude MAD `0.0293` が全Model D候補より良かった。
- flatter / edge-band confidenceは現行gradient confidenceより良かったが、両caseでuniform confidenceを一貫して改善しなかった。

解釈:

clamped pairwiseの改善はcrossに限られ、natural patchでは再現しなかった。今回のconfidence 2案とpairwise 1案をModel D draftへ採用する根拠はなく、negative evidenceとして残す。次にModel Dを進める場合は、小さなconfidence floorやcap調整より、annealingによる確率的driftを含むrelaxation objectiveまたは更新手順を切り分ける必要がある。

acceptance / update order isolation:

- `results/2026-06-14-issue-87-model-d-update-procedure/`
- 現行相当のstochastic + random order、stochastic + fixed order、greedy + random order、greedy + fixed orderをcrossとnatural patchで比較した。
- stochastic条件はproposalの約26〜28%をuphill moveとして受理し、最終objectiveはcrossで初期 `13.64` から約 `27.8`、natural patchで初期 `22.78` から約 `182〜186` へ増加した。crossでは別に約16%がenergy差ゼロのneutral moveだった。
- greedy条件はuphill moveを受理せず、最終objectiveをcross約 `13.11`、natural patch約 `22.05〜22.10` へ低下させた。MADもcross約 `0.0346`、natural patch約 `0.0446` まで改善した。
- random / fixed order間のMAD差は小さく、今回の設定では更新順序よりacceptance modeの影響が大きかった。
- greedy条件もcrossのbilinear MAD `0.0331`、natural patchのbilinear MAD `0.0444` とbicubic MAD `0.0424` は上回らなかった。

解釈:

有限温度Metropolisのuphill acceptanceは、今回の条件でobjectiveとreference差分を増やす主要因だった。一方、greedyによりobjectiveを下げても単純補間を改善しなかったため、objective低下とreference品質改善は同一ではない。次はIssue #88でGaussian proposal依存を外したdeterministic ICMを比較し、proposal samplingの非効率とquadratic objective自体の限界を分ける。

deterministic ICM evaluation:

- `results/2026-06-14-issue-88-model-d-deterministic-icm/`
- uniform confidence、textureなし、quadratic pairwise条件で、Gaussian proposal greedy fixedと解析的な局所最小値を使うfixed row-major ICMを比較した。
- ICMはcrossでobjectiveを `13.6373` から `12.9414`、natural patchで `22.7789` から `21.2437` へ低下させ、greedy fixedの最終objective `13.0968` / `22.0790` より低い値へ到達した。
- 一方、ICMのMADはcross `0.0354`、natural patch `0.0451` で、bilinearの `0.0331` / `0.0444` より悪かった。natural patchではbicubic `0.0424` も上回らなかった。
- crossは26 sweepsで最大画素変化が閾値以下になった。natural patchは18 sweeps終了時の最大画素変化が約 `1.39e-10` で、設定した `1e-12` の収束閾値には未到達だった。

解釈:

Gaussian proposal greedyにはquadratic objectiveを十分に下げきらない探索不足があった。ただし、解析的ICMでobjectiveをさらに下げるほどMAD、PSNR、SSIM、gradient magnitude MADは悪化し、現行quadratic objectiveの最小化をreference品質の改善と同一視できないことが明確になった。これはModel D全体や別objective候補の否定ではないが、現行objectiveを標準decoderへ採用する根拠にはならない。この採否と未確定範囲は `specs/sidf-v0.3.0-draft.md` に反映した。

guided filter系baseline比較:

- `results/2026-06-07-issue-74-guided-filter-baselines/`
- cross と natural patch で nearest / bilinear / bicubic、self-guided filter、joint bilateral refinement、bilateral smoothing、現行 Model D candidate を比較した。
- すべてのedge-aware baselineは低解像度guideをupscaleした画像からguidanceを作り、独立した高解像度guidance imageは使っていない。
- crossでは joint bilateral のMAD `0.0194` がedge-aware条件内で最小だったが、nearestの `0.0138` より悪かった。
- natural patchでは joint bilateral のMAD `0.0435` とgradient MAD `0.0732` がbilinearの `0.0444` と `0.0943` より小さかったが、MADではbicubicの `0.0424` が最小だった。
- 現行Model DのMADはcross `0.0472`、natural patch `0.0572` で、このrunの補間およびedge-aware baselineを上回らなかった。

解釈:

low-guide-only条件でもjoint bilateral refinementは比較対象として有用だが、今回の2ケースで単純補間を一貫して上回ったとは解釈しない。高解像度guidanceを使うguided upsampling手法とは条件が異なるため、SIDFとの比較ではguidanceの情報量を分けて記録する。

知覚・勾配系メトリクス比較:

- `results/2026-06-14-issue-78-perceptual-gradient-metrics/`
- dependency追加なしで raw gradient magnitude MAD、gradient magnitude correlation、strong-edge orientation error、Laplacian MAD を追加した。
- crossではnearestのgradient magnitude MAD `0.0139` とgradient correlation `0.7161` が最良だった。Model Dはgradient magnitude MAD `0.0446`、gradient correlation `0.6058`、Laplacian MAD `0.1176` で、今回のbaselineより勾配強度差と局所高周波差が大きかった。
- natural patchではbicubicがgradient magnitude MAD `0.0293`、gradient correlation `0.6125`、orientation error `33.76` 度、Laplacian MAD `0.0875` で各指標の最良値だった。Model Dはgradient correlation `0.3048`、Laplacian MAD `0.1367` だった。
- crossのnearestはorientation errorではModel Dより悪い一方、MADとgradient magnitude MADでは良く、単一指標だけで品質順位を決められない例になった。

解釈:

追加指標はModel Dの優位性を示すものではなく、画素差、勾配強度、勾配位置、方向、局所高周波差を分けて観察するための補助値である。今回の2ケースではModel Dが単純補間を総合的に上回る結果はなく、LPIPSのような学習済み知覚指標との関係も未確認である。

### Model E

Model E は、量子回路由来の data re-uploading / coupled state 構造を、量子SDKに依存しない古典的なimplicit residual representationとして評価する研究系列である。Issue #95で文献整理、Issue #96で研究設計、Issue #97で最小実装を追加した。Issue #98まではSIDF draft仕様へ採用しない方針とした。

同一bit budget INR比較:

- `results/2026-06-27-issue-98-model-e-bit-budget/`
- development / evaluation を分け、diagonal、circle、Public Domain自然画像patch 2枚で、nearest / bilinear / bicubic、Fourier、RFF、small SIREN、Model E single-state、Model E coupled-stateを比較した。
- 全parameterized候補は fixed feature dictionary + ridge least-squares readout でfitし、12-bit量子化後のparameter side bitsとmetricsを保存した。
- evaluation splitでは `rff_mid` がparameterized候補内の最小MAD `0.034915`、mean serialized side bits `1312` だった。
- Model E候補のevaluation mean MADは `model_e_single_low=0.039588`、`model_e_single_mid=0.039439`、`model_e_coupled_low=0.040270`、`model_e_coupled_mid=0.039951` で、今回のfixed-feature条件では RFF baseline や bicubic baseline `0.036953` を上回らなかった。
- `eval_natural_br` では同じ保存parameterを128x128座標へ外挿評価し、主要PNGとgradient / Laplacian系の簡易artifact統計を保存した。

解釈:

このrunは、Model Eの最小候補を採用する根拠ではない。現時点では「量子回路由来の構造」そのものではなく、同じserialized side bitsでclassical INRを上回る測定結果が必要である。今回の負の結果は fixed feature dictionary + linear readout 条件に限定され、Model E全体や非線形最適化済み候補の否定ではない。

## Open Questions

- Model D の white-noise texture term は、今回のsynthetic cross ablationでは改善要因とは見えなかった。この傾向は自然画像patchや他shapeでも再現するか。
- 現行quadratic objectiveと有限温度Metropolisを不採用とした後、次のModel D objective / decoder procedureでどの仮定を変更するか。
- structured textureのpair-contrast項ではbaseline改善に届かなかった。方向性や周波数統計まで進める価値があるか。
- Model E は fixed feature dictionary + linear readout では classical RFF baseline を上回らなかった。全parameter optimizationや角度parameterizationを導入しても同じ傾向か。
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
