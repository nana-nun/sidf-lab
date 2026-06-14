# SIDF Research Plan

## Research Question

低解像度のSTATICガイド、seed、物理パラメータ、決定論的な確率的緩和過程により、どこまで知覚的に妥当な高解像度画像を再構成できるか。

## MVP Hypotheses

1. Model C の data fidelity と edge-aware interaction は、Model A の明部膨張を抑制できる。
2. Model D の confidence map は、bilinear upscaling より境界を視覚的に締められる可能性がある。ただし保存済みの Model D 比較では、現行設定が nearest / bilinear / bicubic を総合的に上回ったとは解釈しない。
3. white noise texture は粒状感に留まり、自然な質感には structured noise prior が必要になる。
4. 十字では成立しても、斜線、曲線、グラデーション、実画像パッチでは別の破綻が出る。

## Experiments

### 1. Model C Cross Baseline

目的:

同解像度のガイド上で、data fidelity と edge-aware interaction が暗部保持・エッジ漏れ抑制に効くか確認する。

Baseline:

- noisy static guide direct display
- Model A

Metrics:

- MAD
- cross mean
- background mean
- edge leakage
- foreground/background variance
- decode time

Saved images:

- static guide
- Model A output
- Model C output
- difference map when useful

### 2. Model D Multi-Resolution Cross

目的:

16x16 guide から 64x64 output を生成し、bilinear / bicubic と比較する。

Baseline:

- nearest
- bilinear
- bicubic

Metrics:

- edge width
- edge leakage
- foreground/background mean
- variance
- decode time

Saved images:

- low-res guide
- upscaled guide
- bilinear output
- bicubic output
- confidence map
- SIDF rendered output
- difference map

注意:

Model D は guided filter / joint bilateral upsampling と比較できるが、現行Model Dは高解像度guidance imageを持たず、low-resolution guideをupscaleしてconfidence mapを作る。そのため、今後のbaselineでは「高解像度guidanceを使うguided filter系」と「低解像度guideだけを使うSIDF条件」を分けて評価する。

追加baseline候補:

- guided filter baseline
- joint bilateral upsampling baseline
- bilateral smoothing baseline
- texture_strength=0 / white noise / structured texture prior の対照
- confidence map / pairwise term の再設計候補

現在の結果:

- shape benchmark、cross comparison、natural patch GT evaluation では、現行 Model D candidate は nearest / bilinear / bicubic baseline に対する総合的な改善を示していない。
- confidence map には境界拘束の候補として観察価値が残るが、white-noise texture term と現在の重み設定が MAD、SSIM、edge leakage、gradient MAD を悪化させる条件がある。
- texture ablation では、synthetic cross において非ゼロ white-noise texture_strength が baseline 指標を改善する傾向は見えなかった。
- confidence / data / texture の小規模gridでは、cross と natural patch の両方で `flat_conf_tex0` がModel D grid内の最小MADだったが、nearest / bilinear / bicubic baseline は上回らなかった。
- term isolation でも、`data_pairwise_uniform` がterm条件内の最小MADだった一方、nearest / bilinear / bicubic baseline は上回らなかった。
- confidence / pairwise再設計では、clamped pairwiseがcrossだけでuniform quadraticを改善したが、natural patchでは悪化した。flatter / edge-band confidenceもuniform confidenceを一貫して上回らなかった。
- acceptance / update order分離では、greedy化がstochastic条件よりMADとobjectiveを大幅に改善し、random / fixed order差は小さかった。ただしgreedy条件もbilinear / bicubic baselineは上回らなかった。
- 現行 Model D 式の単純な重み探索や white-noise texture 調整では、baseline 改善に届きにくい。
- 今回のconfidence / pairwise候補はdraftへ採用せず、有限温度Metropolisも現設定の標準decoderとして採用しない。次はdeterministic ICMでproposal依存とquadratic objective自体の限界を分ける。

### 3. Shape Benchmark

対象:

- cross
- diagonal line
- circle
- thin line
- soft gradient
- checker edge

目的:

十字以外でもエッジ保持と再構成が安定するか確認する。

### 4. Structured Texture Prior

目的:

white noise texture の寄与を ablation で確認したうえで、structured noise に置き換える候補が粒状感やbaseline差分を改善するか確認する。評価では `texture_strength=0` と white noise baseline を必ず含め、structured texture が意味的ディテールを生成できるとは断定しない。

候補:

- smoothed white noise
- fractal noise
- Perlin-like noise
- edge-aligned directional noise

### 5. Real Patch Test

対象:

- anime eye
- hair strand
- cloth shadow
- soft skin gradient
- natural texture

目的:

confidence map が柔らかい陰影を硬く分断しないか確認する。

## Next Implementation Step

短期の優先順:

1. Issue #87 の結果から、有限温度Metropolisのuphill acceptanceは現設定の標準decoderへ採用しない。
2. Issue #88 で、Gaussian proposal greedyと解析的なdeterministic ICMを比較し、proposal依存とquadratic objective自体の限界を分ける。
3. Rust core 関連の残タスクがある場合は、PRNG、固定小数点、更新順序の切り分けを保ったまま進める。
