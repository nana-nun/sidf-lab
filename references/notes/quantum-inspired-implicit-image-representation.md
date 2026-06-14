# 量子回路由来の implicit image representation

## Source

Related Issue: [#95](https://github.com/nana-nun/sidf-lab/issues/95)

主な一次文献:

- Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, José I. Latorre, "Data re-uploading for a universal quantum classifier", 2020. DOI: `10.22331/q-2020-02-06-226`
- Maria Schuld, Ryan Sweke, Johannes Jakob Meyer, "The effect of data encoding on the expressive power of variational quantum-machine-learning models", 2021. DOI: `10.1103/PhysRevA.103.032430`
- Zhan Yu, Hongshun Yao, Mujin Li, Xin Wang, "Power and limitations of single-qubit native quantum neural networks", 2022. arXiv: `2205.07848`
- Jiaming Zhao, Wenbo Qiao, Peng Zhang, Hui Gao, "Quantum Implicit Neural Representations", 2024. arXiv: `2406.03873`
- Saadet Müzehher Eren, "Implementation of Quantum Implicit Neural Representation in Deterministic and Probabilistic Autoencoders for Image Reconstruction/Generation Tasks", 2026. arXiv: `2603.06755`

Classical comparison sources:

- Vincent Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", 2020. arXiv: `2006.09661`
- Matthew Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", 2020. arXiv: `2006.10739`
- Emilien Dupont et al., "COIN: COmpression with Implicit Neural representations", 2021. arXiv: `2103.03123`

## Summary

### Data re-uploading

Data re-uploading は、入力値を量子回路へ一度だけ符号化するのではなく、trainable unitary block の間へ繰り返し入力する構成である。Pérez-Salinas et al. は、単一qubitでも古典処理と複数回のdata re-uploadingを組み合わせれば、分類関数を柔軟に表現できることを示した。

Schuld et al. は、parameterized quantum circuit の入力依存性をpartial Fourier seriesとして記述した。利用可能な周波数集合はencoding Hamiltonianの固有値差で決まり、同じencodingを繰り返すとより広い周波数集合へ到達できる。一方、周波数へ到達できることと、必要なFourier係数を独立に制御できることは別の条件である。

Yu et al. はsingle-qubit data re-uploading QNNについて、1変数関数では普遍近似へつながる性質を整理した一方、多変数関数では周波数集合と係数自由度に制限があることを示した。2次元座標から画像値を出すSIDFでは、この多変数制限を無視できない。

### QIREN

QIRENは、座標を入力し信号値を出力するimplicit neural representationへ、multi-layer / multi-qubitのdata re-uploading circuitを組み込むhybrid modelである。各hybrid layerは古典Linear / BatchNorm、quantum circuit、測定値から構成される。回路では回転gate、data encoding、CNOTによる結合を使い、observableの期待値を次のlayerへ渡す。

QIREN論文は、回路出力をFourier seriesとして展開し、data re-uploadingとmulti-qubit構造により豊かな周波数集合を構成できると論じる。実験ではsignal representation、image representation、image superresolution、generationを扱い、ReLU、Tanh、random Fourier features、SIRENなどと比較している。

ただし、QIRENの実験値をそのままSIDFの利点とは扱わない。論文の比較は主にparameter countやモデル内memoryを使い、量子化後のserialized bit数、entropy coding、状態準備、回路simulation cost、実機測定回数を含むend-to-end compression比較ではない。

### 2026 QINR-AE / VAE

Eren 2026は、classical CNN encoderとQINR decoderを組み合わせたAE / VAEをMNIST、E-MNIST、Fashion-MNISTで評価する。latent vectorから周期的・高周波な特徴を作るためdata re-uploadingを使い、最適化上の対策としてlearnable angle scalingを導入している。

これはQINRを生成decoderへ接続する後続例だが、低解像度guideから任意の自然画像patchを忠実復元するSIDFの条件とは異なる。classical encoder、dataset prior、latent regularizationの寄与をQINR単独の寄与と分離する必要がある。

## Reported Evaluation Conditions

| Work | Task | Inputs and outputs | Main comparison | SIDFでの注意 |
| --- | --- | --- | --- | --- |
| Data re-uploading | Classification | feature vector to class | single / multi-qubit classifiers | 画像表現実験ではない |
| Encoding expressivity | Function approximation theory | encoded data to expectation | accessible Fourier spectra | 実画像品質や圧縮性能を保証しない |
| Single-qubit limits | Function approximation | univariate / multivariate inputs | Fourier spectrum and coefficient freedom | 2D画像ではmultivariate limitationが重要 |
| QIREN | Signal / image representation, SR, generation | coordinate to signal value | ReLU, Tanh, RFF, SIREN | bitstream全体や実機costの比較ではない |
| QINR-AE / VAE | Reconstruction / generation | latent vector and coordinates to image | AE / VAE and quantum generative candidates | classical encoderとdataset priorを含む |

## Relevance to SIDF

### 導入する候補

- 座標 `(x, y)` とguide featureを繰り返し入力するdata re-uploading。
- 回転gateの積に由来する周期的な非線形関数。
- 複数state間の結合により、`x` と `y` の交差項を表現する構造。
- observable expectation相当のbounded scalar output。
- bilinear guideを基準にしたdeterministic residual field。
- 保存parameterを量子化し、実際のserialized bit数で比較する方針。

### 現段階では導入しないもの

- 量子実機をdecoder要件にすること。
- shot samplingやdevice noiseを画像textureとして利用すること。
- amplitude encodingされた全画像quantum stateをファイル表現とすること。
- quantum advantage、super-resolution、compressionの成立を前提にすること。
- large classical encoderや学習済みdataset priorを最初のModel Eへ含めること。

## Minimal Model E Candidate

最初の候補は、量子SDKを使わずNumPy上で評価できる決定論的な関数とする。

```text
base(x, y) = bilinear(low_guide, x, y)
features   = [x, y, base(x, y), gradient_x, gradient_y]
residual   = alpha * q_theta(features)
output     = clamp(base + residual, 0, 1)
```

`q_theta`は次の2条件から始める。

1. single-state / single-qubit相当:
   回転とdata re-uploadingを重ね、bounded expectation相当値を返す。
2. coupled multi-state相当:
   少数stateを結合し、`x` と `y` を含む多変数frequencyと交差項を表現する。

single-state候補は最小baselineとして有用だが、2次元画像に対する理論上の制限から、最終候補と仮定しない。

## Fair Classical Baselines

- bilinear / bicubic interpolation
- explicit separable Fourier series
- random Fourier features + linear readout
- small SIREN
- parameter数と量子化bit数を合わせた小型MLP
- 必要に応じてCOIN型のper-image INR

比較では少なくとも次を揃える。

- 同じlow-resolution guideとGround Truth。
- 同じ座標domainと出力range。
- 同程度のoptimization stepまたはfit timeを別々に記録。
- parameter数ではなくheaderを含むserialized bit数。
- float条件とquantized parameter条件の両方。
- development patchとheld-out patchの分離。
- MAD、PSNR、SSIM、gradient metrics、decode time、fit time。

## Interpretation

SIDFが利用すべき中心的な考えは「qubitへ画像を格納すること」ではなく、data encodingと再uploadingにより、少数parameterから制御された周波数集合を生成する関数構造である。

ただし、この構造は古典計算上では三角関数と行列演算として評価できる。したがって、量子回路由来であることだけでは採用理由にならない。同じbit budgetのexplicit Fourier model、RFF、SIRENよりrate-distortionまたは表現安定性で良い場合に限って、Model Eへ残す。

## Limitations

- QIRENの理論上の優位性は最適条件に依存し、SIDFの小型古典実装で再現されるとは限らない。
- accessible frequency数が多くても、学習可能性、係数自由度、量子化耐性、画像に必要なfrequency配置が良いとは限らない。
- QIRENのparameter / memory比較は、SIDFが必要とするserialized bitstream比較と一致しない。
- quantum circuit simulationはqubit数に対して高コストになり得る。Model Eでは量子状態vectorの汎用simulationを避け、比較対象として説明可能な小型関数へ限定する必要がある。
- QINR-AE / VAEの結果は文字・衣類dataset中心で、自然画像の忠実復元や圧縮を示していない。
- 本メモは文献整理であり、SIDF上の新しい実験結果ではない。

## Follow-up

- Issue #96でModel Eの研究質問、non-goals、成功・不採用基準を定義する。
- Issue #97でsingle-stateとcoupled multi-stateの最小古典実装を追加する。
- Issue #98でFourier、RFF、SIRENとserialized bit budgetを揃えて比較する。
- 量子化後に周波数・位相parameterの誤差が画像品質へ与える影響をIssue #98の結果から判断する。
