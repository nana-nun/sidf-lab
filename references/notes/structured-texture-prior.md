# Structured Noise Prior と Procedural Texture Synthesis

## Source

Related Issue: [#15](https://github.com/nana-nun/sidf-lab/issues/15)

Primary sources:

- Ken Perlin, "An Image Synthesizer", SIGGRAPH 1985. DOI: `10.1145/325334.325247`
- Robert L. Cook and Tony DeRose, "Wavelet Noise", ACM Transactions on Graphics 2005. DOI: `10.1145/1186822.1073264`
- Ares Lagae, Sylvain Lefebvre, George Drettakis, Philip Dutre, "Procedural Noise Using Sparse Gabor Convolution", ACM Transactions on Graphics 2009. DOI: `10.1145/1576246.1531360`
- Ares Lagae et al., "A Survey of Procedural Noise Functions", Computer Graphics Forum 2010. DOI: `10.1111/j.1467-8659.2010.01827.x`
- Bruno Galerne, Ares Lagae, Sylvain Lefebvre, George Drettakis, "Gabor Noise by Example", ACM Transactions on Graphics 2012. DOI: `10.1145/2185520.2185569`

## Summary

Procedural noise は、画像そのものを保存するのではなく、seed、座標、少数のパラメータから決定論的に評価できる texture field を作る考え方である。Perlin noise は gradient noise の古典的な入口であり、複数octaveを足す fractal Brownian motion (fBm) 的な使い方により、単一スケールのwhite noiseより滑らかで階層的な変動を作れる。

Wavelet noise は、procedural noise のaliasingやdetail lossを問題として扱い、帯域やスケールの制御を重視する候補である。SIDFで高解像度再構成にnoiseを足す場合、単に「見た目が細かい」だけではなく、低解像度guideと出力解像度の関係で不要な高周波を足していないかを確認する必要がある。

Gabor noise 系は、周波数、帯域、方向の制御を明示しやすい。特に sparse Gabor convolution は、white noiseやPerlin-like noiseより、方向性を持つtexture priorの候補として扱いやすい。Gabor Noise by Example は、exemplar texture からスペクトルパラメータを推定する方向性を示しているが、現時点のSIDFはexemplar画像を保存する設計ではないため、まずは手動パラメータで使える小さい候補に限定するのが安全である。

## Relevance to SIDF

Model D の現行 texture term は、seeded white noise に近い。既存結果では、粒状感は出るが意味的ディテールではなく、nearest / bilinear / bicubic baseline に対するMADやSSIMを改善していない。したがって structured noise prior は、SIDFの有効性を示す結果ではなく、次に検証すべき候補である。

SIDFに導入しやすい条件:

- seed、shape、scale、amplitude から決定論的に同じfieldを生成できる。
- 低解像度guideを直接変更せず、decoder内のtexture targetまたはproposal biasとして扱える。
- white noise baseline、textureなしablation、bilinear / bicubic baselineと分けて評価できる。
- Rust移植前に、PRNG、補間、丸め、境界条件を固定しやすい。

## Candidate Priors

| Candidate | Idea | Parameters | SIDFでの利点 | 主なリスク |
| --- | --- | --- | --- | --- |
| Smoothed white noise | white noiseをGaussian / box smoothingして低周波化する | `seed`, `sigma`, `smooth_radius` | 実装が最小で、white noise baselineとの差分を見やすい | ぼけたランダム斑点になりやすく、方向性や材質感は弱い |
| Fractal value / gradient noise | 複数octaveのcoherent noiseを足す | `seed`, `octaves`, `base_frequency`, `lacunarity`, `gain`, `amplitude` | 少数パラメータで階層的な変動を作れる | grid artifact、octave設定依存、自然画像GTに対する差分悪化の可能性 |
| Perlin-like gradient noise | 格子gradientと補間で連続的なnoiseを作る | `seed`, `frequency`, `octaves`, `fade` | deterministic procedural textureの標準的入口 | 実装差で再現性がぶれやすく、Rust移植時に補間関数とgradient tableの固定が必要 |
| Gabor / directional noise | 周波数・帯域・方向を持つkernel noise | `seed`, `frequency`, `bandwidth`, `orientation`, `impulse_density`, `amplitude` | edge-aligned textureや線状textureの仮説を試しやすい | 実装が重く、parameter searchなしでは見た目とmetricsが不安定になりやすい |
| Edge-aligned anisotropic noise | guide gradientに沿って方向性を変えるnoise | `seed`, `amplitude`, `parallel_scale`, `normal_scale`, `confidence_weight` | confidence mapやedge-aware interactionと接続しやすい | high-frequency textureがedge leakageを悪化させる可能性がある |

## Evaluation Against White Noise

最低限、次の比較を分ける。

1. `texture_weight = 0`: textureなしablation
2. current `white_noise`: 現行baseline
3. candidate prior: smoothed / fractal / Perlin-like / directional のいずれか
4. nearest / bilinear / bicubic: decoder外の単純baseline

評価指標候補:

- MAD / PSNR / SSIM: Ground Truth がある synthetic / natural patch での全体差分。
- gradient MAD / strong-edge MAD: 自然画像patchで、textureが勾配を壊していないかを見る。
- edge leakage / edge width: hard edge synthetic shapeで、textureが境界漏れを増やしていないかを見る。
- texture residual variance: `output - bilinear` や `output - no_texture` の分散。粒状感の強さを測る。
- spectrum slope / radial power summary: white noiseとの差、過剰な高周波、octaveの偏りを見る。
- local contrast delta: flat regionとedge regionで、textureが不自然に増幅されていないかを見る。
- decode time: procedural evaluationがModel D decode timeをどれだけ増やすかを見る。

結果解釈では、candidate prior が「自然な質感」を生成したとは書かない。まずは「white noiseより粒状性が減ったか」「baselineとの差分を悪化させない条件があるか」「edge leakageが増えないか」に限定する。

## Draft `texture.py` Specification

次の `[impl]` Issue に落とせる最小仕様案:

```python
def white_noise(shape: tuple[int, int], seed: int, sigma: float = 1.0) -> np.ndarray:
    """Existing baseline: deterministic zero-mean white noise."""

def smoothed_noise(
    shape: tuple[int, int],
    seed: int,
    sigma: float = 1.0,
    radius: int = 2,
) -> np.ndarray:
    """Low-pass white noise with deterministic padding and normalization."""

def fractal_value_noise(
    shape: tuple[int, int],
    seed: int,
    octaves: int = 4,
    base_frequency: float = 2.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Coherent multi-octave value noise normalized to zero mean."""

def directional_noise(
    shape: tuple[int, int],
    seed: int,
    angle: float,
    frequency: float,
    bandwidth: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """Small Gabor-like directional prior for later edge-aligned experiments."""
```

実装上の制約:

- 返り値は `np.float64`、shape一致、平均は0に正規化する。
- `sigma < 0`、`octaves < 1`、`radius < 0` などは `ValueError` にする。
- 同じ引数では同じ配列を返す determinism test を追加する。
- 初期実装ではSciPyに依存しない。smoothingはNumPyの小さい separable kernel か既存依存を確認してから決める。
- `np.random.default_rng(seed)` による同一環境再現性を当面の基準にし、Rust移植時にはPRNG仕様を別Issueで固定する。

## Limitations

- このメモは文献整理と実装候補であり、SIDF上の新しい実験結果ではない。
- procedural texture 文献は主にCGのtexture generation文脈であり、低解像度guideからの画像再構成性能を直接示すものではない。
- structured noise prior が Model D のMAD、SSIM、edge leakageを改善するかは未検証である。
- Perlin-like / Gabor-like noise は実装詳細で見た目と再現性が変わるため、仕様化前にPython実装と保存形式つき実験が必要である。
- exemplar-based Gabor noise は面白いが、exemplar画像の保存や圧縮条件をSIDF仕様に混ぜる可能性があるため、現段階の最小実装候補からは外す。

## Follow-up

- Issue #48 で、`smoothed_noise` と `fractal_value_noise` を `src/sidf_lab/texture.py` に追加し、determinism / shape / zero-mean / parameter validation testsを入れる。
- その後の実験Issueで、Issue #37 の texture ablation と接続し、white noise、textureなし、structured candidateを同じconfigで比較する。
- directional / Gabor-like noise は、単純候補の評価後に、edge-aligned priorとして別Issueに分ける。
