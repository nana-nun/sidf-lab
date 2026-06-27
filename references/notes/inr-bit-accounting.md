# INR圧縮のbit accountingとparameter量子化方針

## Source

- Related Issue: https://github.com/nana-nun/sidf-lab/issues/107
- COIN: COmpression with Implicit Neural representations, Dupont et al., 2021, https://arxiv.org/abs/2103.03123
- COIN implementation, https://github.com/EmilienDupont/coin
- COIN++: Neural Compression Across Modalities, Dupont et al., 2022, https://arxiv.org/abs/2201.12904
- Implicit Neural Representations for Image Compression, Strumpler et al., 2021, https://arxiv.org/abs/2112.04267
- Compression with Bayesian Implicit Neural Representations, Guo et al., 2023, https://arxiv.org/abs/2305.19185
- SIREN: Implicit Neural Representations with Periodic Activation Functions, Sitzmann et al., 2020, https://arxiv.org/abs/2006.09661
- Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains, Tancik et al., 2020, https://arxiv.org/abs/2006.10739

## Summary

COINは、画像ごとに小さなimplicit neural representationをfitし、そのnetwork parametersを量子化して保存する方向の画像圧縮として位置づけられる。SIDFで参照するときは、低解像度guideや固定decoderを前提にした残差モデルではなく、画像そのものをINRのparameter列として表す比較対象として扱う。

COIN公式実装は、SIREN構成とKodak実験の再現コマンドを確認する実装例として使える。ただし、READMEにはhalf precision版の `torch.sin` はCUDAでのみ動くという注意があり、SIDFのportable deterministic decode方針とは別物として扱う。

COIN++は、全てのnetwork weightsを画像ごとに送るのではなく、共有されたbase networkと入力ごとのmodulationを分ける。paper上のbit accountingでは、receiverが同じbase networkを持つという前提のもとで、modulationを量子化し、entropy codingした長さが主要な送信対象になる。この前提をSIDFに持ち込む場合、base networkを「仕様として固定済み」とみなすのか、「学習済み外部依存」とみなすのかを明示しないと、parameter-onlyのbit数と実際のdescription lengthが混ざる。

Strumpler et al.のINR画像圧縮は、INRのcompression pipelineとして、量子化、quantization-aware retraining、entropy coding、meta-learned initializationを組み合わせる。SIDFのModel Eやtrainable INR baselineで「圧縮」と呼ぶには、少なくとも量子化後のdecode結果と実際または推定の符号長を別に報告する必要がある。

Bayesian INR compressionは、uniform quantizationしたparameter列を保存する方式ではなく、posterior weight sampleをrelative entropy codingで送る方向でrate-distortionを扱う。SIDFの初期比較では直接実装対象にしないが、「float parameter count」だけではcompression accountingにならない例として有用。

SIRENとFourier featuresは表現力のbaselineとして重要だが、それ自体はbitstream仕様ではない。SIREN/MLPならweights、biases、activation constants、architecture、quantization metadataを数える。Fourier featuresやRFFなら、周波数行列を固定仕様にするのか、seedで生成するのか、明示的なmatrixとして保存するのかでbit数が変わる。

## SIDFで分けるべきbit数

`parameter_side_bits` は、残差モデルやINRが持つmodel-specific parametersだけを数える診断値とする。float parameter countや「N parameters x Q bits」のような値はここに入るが、これだけでcompression claimには使わない。

`incremental_side_bits` は、同じguide、同じoutput shape、同じdecoder familyを前提にした比較で、あるmodelを追加するために必要な差分bit数とする。parameter quantization metadata、model id、structure id、seed、frequency table id、residual scaleなど、parameter以外でもmodelを再現するために必要な小さなheaderはここに含める。

`total_description_bits` は、画像をdecodeするために必要な全記述長とする。低解像度guide bits、output shape、colorspaceまたはgrayscale指定、model id、decoder version、quantization rule、entropy model metadata、encoded parameters、container overhead、checksumやmagic bytesを含める。SIDFが「圧縮」と言う場合は、この値を基準にし、PNG/JPEG/WebPなどの実ファイルサイズと比較する場合は同じ条件の入力・出力・metadataを揃える。

## 保存対象候補

共通項目:

- format magic/version
- output width/height/channels
- guide imageまたはguide imageへの参照と、その符号長
- model family id
- architectureまたは固定architecture id
- quantization scheme id, bit depth, scale, zero point, clipping rule
- entropy coding scheme idと必要なmodel metadata
- residual scaleやpostprocess rule
- deterministic seedとPRNG idが必要な場合はその値

SIREN/MLP baseline:

- layer count, hidden width, activation type
- all trainable weights and biases after quantization
- SIRENのomegaなど、decodeに必要なactivation constants
- parameter ordering

Fourier/RFF baseline:

- Fourier feature mode
- fixed frequency table id、seed、または明示的なfrequency matrix
- RFF分布のscaleなど、seedだけでは復元できない設定
- readout weights and biases

Model E / quantum-inspired feature model:

- feature family id
- layer count, state count, depthなどのstructure parameters
- angle parameters, frequency parameters, mixing/readout parameters
- fixedかtrainableかの区別
- residual scaling and clipping rule
- quantization metadata

COIN++型の共有base networkを使う場合:

- shared base networkを仕様に含めるかどうか
- 含めない場合のmodel hash、version、配布前提、amortization前提
- per-image modulation parameters
- modulation quantization and entropy coding metadata

## #103/#104への提案

#103のfitting基盤では、まずfloat diagnosticsとquantized decodeを別の結果として保存する。float-onlyのPSNR/SSIMは表現力の確認であり、compression比較ではない。

metricsには、少なくとも次を分けて保存する。

- `parameter_count`
- `float_parameter_bytes`
- `parameter_side_bits`
- `incremental_side_bits`
- `total_description_bits`
- `entropy_coded_bits_estimate`
- `entropy_coded_bits_actual` が未実装なら `null`

#104の再比較では、同じguideと同じdataset splitのもとで `incremental_side_bits` による残差モデル比較を行い、圧縮形式としての主張は `total_description_bits` が計測できるまで保留する。COIN/COIN++と比較する場合は、shared decoderやshared base networkの前提を表に明記する。

## Limitations

INR圧縮文献の多くは、画像全体をINR parameter列で表す設定を扱う。SIDFは低解像度guideを先に持ち、Model Eなどをguideからの残差・詳細生成として扱うため、文献中のbitrateをそのままSIDFのbitrateと比較できない。

このメモは方針整理であり、SIDF bitstreamの実装やentropy coderの実測結果ではない。実際のcompression claimには、保存形式、decoder再現性、量子化後decode画像、実ファイル長が必要になる。

## Follow-up

- #103: Model E/INRの全parameter fitting基盤を追加する
- #104: Model Eをtrainable INR baselineとsource分割datasetで再比較する
- #108: source分割済みgrayscale patch fixtureを追加する
