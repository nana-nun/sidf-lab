# SIDF Research Plan

## Research Question

低解像度のSTATICガイド、seed、物理パラメータ、決定論的な確率的緩和過程により、どこまで知覚的に妥当な高解像度画像を再構成できるか。

## MVP Hypotheses

1. Model C の data fidelity と edge-aware interaction は、Model A の明部膨張を抑制できる。
2. Model D の confidence map は、bilinear upscaling より境界を視覚的に締められる。
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
- texture term ablation

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

white noise texture を structured noise に置き換え、粒状感を減らせるか確認する。

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

まずはPython package skeletonを作り、Model Cを再現可能な実験として保存する。
