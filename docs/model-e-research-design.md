# SIDF Model E Research Design

Status: Draft research plan
Date: 2026-06-14
Related Issue: [#96](https://github.com/nana-nun/sidf-lab/issues/96)

Update 2026-06-27:

Issue #98 の fixed feature dictionary + linear readout 比較では、最小Model E候補は
evaluation splitで RFF / bicubic baselineを上回らなかった。この結果はModel E全体の
否定ではないが、#98で評価した single-state / coupled-state 最小候補をSIDF draft仕様へ
採用する根拠はない。その後の #103 / #104 で全parameter fittingとsource分割datasetを扱った。
採否判断の短い一覧は `docs/model-decision-map.md` を参照する。

Update 2026-06-28:

Issue #104 の trainable INR / source-split比較でも、現行のModel E single-state /
coupled-state候補を採用する根拠は得られなかった。evaluation splitでは
`mlp_small` が best parameterized candidate で mean quantized MAD `0.089009`、
mean serialized side bits `708` だった。best Model E candidate は `model_e_single`
で mean quantized MAD `0.090061`、mean serialized side bits `576` だった。
この結果は #104 のparameterization、optimizer、fixture、incremental side-bit見積もりに
限定され、Model E一般の否定ではない。次にModel Eを続ける場合は、現行候補の
random search調整ではなく、angle / frequency parameterization、coupling設計、
bit accountingを再設計するIssueとして扱う。

Update 2026-06-29:

Issue #117 の有限差分optimizer診断では、L-BFGS相当の探索で一部改善はあったが、
現行Model E候補は classical INR baseline や bicubic baselineを上回らなかった。
Issue #122 のcompact parameterization redesign比較では Candidate A/C が現行Model E
より少し改善したが、best classical INR、nearest、bicubic baselineを上回らなかった。
したがって、#98 / #104 / #117 / #122 で評価したModel E候補をSIDF draft仕様へ
採用する根拠はまだない。

Model Eを継続する場合は、#118のcoupling variation比較済み結果と#119のbit-depth耐性結果を
採用根拠にせず、未測定のautograd optimizer条件などを個別に切り分ける。
継続しない場合は、これらのIssueを実施せず一時保留にする理由を、測定済み結果と
未測定範囲に分けて明文化する。

Update 2026-07-04:

Issue #125 では、autograd optimizer依存をdefault requirementsへ追加しない判断を
文書化した。続ける場合は PyTorch CPU を optional backend として別Issueでspikeし、
Model E と classical INR baseline の両方へ同じoptimizer条件を適用できるかだけを
検証する。詳細は `docs/model-e-autograd-optimizer-decision.md` を参照する。

Update 2026-07-27:

Issue #118 では、#104と同じsource-split fixture、12-bit量子化、incremental side-bit
accountingで、現行coupled-state、controlled-rotation風coupling、gated interaction風couplingを
比較した。evaluation splitでは `model_e_gated_coupled` が現行coupledの mean quantized MADを
`0.090095` から `0.089183` へ改善した。一方、best classical INRの `mlp_small` は
`0.089009` であり、gated候補は現行coupled比で平均 `540` bits のcoupling overheadを要した。

したがって、#118の2種類のcoupling候補をSIDF draft仕様へ採用する根拠はない。これは今回の
fixture、48-step dependency-free random-search fit、比較した2案に限定した判断であり、Model E
一般や別のcoupling式の否定ではない。coupling設計を継続理由にせず、次に進める場合は#134で
未測定のoptional autograd optimizer条件を、classical INRと同じ条件で切り分ける。

## 1. Positioning

Model E は、量子回路由来の関数構造を古典計算上で評価する
quantum-inspired implicit image representation の候補である。

量子実機、量子状態としての画像保存、または量子優位を前提にしない。
data re-uploading、回転、state間結合、expectation valueに由来する周期的な
座標関数が、少ない保存情報で画像の補間残差を表現できるかを検証する。

Model E は Model C / D を置き換える確定仕様ではなく、独立した研究系列として扱う。

| Model | Main input | Main procedure | Current role |
| --- | --- | --- | --- |
| Model C | 同解像度guide | edge-aware stochastic relaxation | synthetic guideの安定化baseline |
| Model D | 低解像度guide | confidence / pairwise / textureを含むrelaxation | 現行objectiveと有限温度Metropolisは不採用 |
| Model E | 低解像度guideと保存parameter | deterministic coordinate function | quantum-inspired residual representation候補 |

## 2. Primary Research Question

低解像度guideを単純補間した画像に対して、data re-uploading量子回路由来の
決定論的な座標関数は、同程度のserialized bit数を持つclassical implicit
representationより、Ground Truthに近い補間残差を表現できるか。

候補出力:

```text
base(x, y) = bilinear(low_guide, x, y)
residual(x, y) = alpha * q_theta(features(x, y))
output(x, y) = clamp(base(x, y) + residual(x, y), 0, 1)
```

## 3. Hypotheses

### H1: Frequency efficiency

data re-uploadingとcoupled state構造は、同程度の保存bit数のseparable Fourier、
random Fourier features、small SIRENより、画像残差に必要な2次元周波数成分を
効率よく表現できる可能性がある。

### H2: Coupling requirement

single-state / single-qubit相当モデルは最小baselineとしては有用だが、
2次元座標の交差構造を表現するには制限がある。coupled multi-state候補は、
diagonal、curve、textureを含む残差でsingle-state候補を改善する可能性がある。

### H3: Quantization tolerance

回転角、周波数、位相に相当するparameterを量子化したときの品質低下が、
classical INRの量子化低下より小さい条件が存在する可能性がある。

これらは検証前の仮説であり、量子回路由来の構造が有利であることを前提にしない。

## 4. Primary Objective and Non-Goals

最初のModel E研究では、Ground Truthに対する忠実復元を主目的とする。

最適化対象:

- low-resolution guideから失われた残差を、画像ごとの保存parameterへ符号化する。
- 同じguide、Ground Truth、bit budgetでclassical baselineと比較する。
- float parameterだけでなく、量子化・serialization後の品質を評価する。

最初の段階では扱わないもの:

- seedだけからGround Truthにないdetailを生成すること。
- 人間評価だけを使った知覚的生成品質。
- 量子実機、shot sampling、device noise。
- dataset全体で学習したlarge encoder / decoder。
- RGB、動画、3D scene。
- 実用圧縮形式としての互換性、entropy coding、既存codecへの優位性。

知覚的生成を将来扱う場合は、忠実復元とは別Issueにし、同一guideからの多様性、
自然さ、Ground Truth非一致を分けて評価する。

## 5. Input, Output, and Stored Information

### Input

- grayscale low-resolution guide
- target output width / height
- normalized coordinate `(x, y)`
- optional guide-derived features:
  - bilinear guide value
  - horizontal / vertical gradient

最初の比較では、全モデルへ同じguide-derived featuresを与える。

### Output

- `[0, 1]` のgrayscale residualまたはreconstructed value
- 初期候補はbilinear guideにbounded residualを加える

### Stored information

- model identifier and structure
- output shape
- quantized trainable parameters
- quantization scale / zero point or equivalent rule
- residual amplitude and required normalization metadata
- low-resolution guide

### Seed

decoder outputの必須情報にはしない。seedはparameter初期化やoptimization runの
再現にのみ使用し、最終parameterを保存したdecoderはseedなしで同じ出力を返す。

## 6. Comparison Baselines

画像baseline:

- nearest
- bilinear
- bicubic

parameterized residual baseline:

- explicit separable Fourier series
- random Fourier features + linear readout
- small SIREN
- parameter / bit budgetを揃えたsmall MLP
- single-state / single-qubit相当Model E
- coupled multi-state相当Model E

COIN型per-image INRは、parameter量子化とbitstream評価方法の参考にする。
初回実験で大規模なmeta-learned modelは導入しない。

## 7. Bit Budget

parameter countだけをcompression proxyとして使わない。

最低限、次の2種類を報告する。

```text
incremental_side_bits =
  model_header_bits
  + structure_bits
  + quantized_parameter_bits
  + quantization_metadata_bits

total_description_bits =
  guide_bits
  + output_shape_bits
  + incremental_side_bits
  + container_overhead_bits
```

同じguideを使うresidual model間では `incremental_side_bits` を主要比較にできる。
compressionについて述べる場合は `total_description_bits` を使い、少なくとも
PNGなどの実ファイルサイズとの比較条件を別途定義する。

初回比較では複数の固定budgetを事前に決め、各モデルを最も近いbudgetへ量子化する。
float条件は表現能力の診断として残すが、採否判断はquantized条件を優先する。

## 8. Dataset and Split Policy

最初のdatasetはgrayscaleに限定する。

- synthetic: cross、diagonal、circle、thin line、soft gradient、checker edge
- natural patches: 複数画像・複数領域から切り出した固定patch集合

分割:

- development set: architecture、frequency range、optimizer、量子化規則の選択に使う
- evaluation set: protocol固定後の最終比較だけに使う

Model Eは画像ごとにparameterをfitするため、evaluation imageでも個別最適化は行う。
ここでのheld-outはparameterのゼロショット汎化ではなく、architectureとfit protocolを
評価画像に合わせて再調整していないことを意味する。

同じsource imageの近接cropがdevelopmentとevaluationへ跨がないように分ける。

## 9. Optimization and Reproducibility

各runで次を保存する。

- initialization seed
- optimizer and learning rate
- optimization steps
- loss definition
- fit time
- final float parameters
- quantization rule and quantized parameters
- decode time
- software and dependency versions

公平性のため、次を分けて報告する。

- fixed-step comparison: 同じoptimization step budget
- fixed-time comparison: 可能なら同程度のfit time
- converged diagnostic: 各モデルの到達可能品質を見る補助条件

optimization costはdecoder bit数に含めないが、実用性の独立指標として必ず記録する。

## 10. Metrics

忠実復元:

- MAD
- PSNR
- SSIM
- gradient magnitude MAD
- gradient correlation
- Laplacian MAD
- hard-edge shapeではedge leakage / edge width

表現と実行:

- serialized bits
- bits per output pixel
- float-to-quantized metric delta
- fit time
- decode time
- parameter count

診断:

- output residual image
- Fourier spectrum or radial power summary when practical
- higher output resolutionでのperiodic artifact / aliasing

単一metricだけで順位を決めず、主要判定は事前に指定したaggregate MADまたはPSNRと
serialized bitsのrate-distortion比較に置く。

## 11. Decision Criteria

### Success

Model Eを有望候補とする条件:

- evaluation setの複数bit budgetで、少なくとも1つのModel E候補が主要classical INRを
  aggregate rate-distortionで一貫して改善する。
- 改善が単一shapeまたは単一patchだけに限定されない。
- quantization後にも改善が残る。
- decode timeとartifactが研究継続可能な範囲にある。

この条件を満たしても、量子優位、実用圧縮、一般的super-resolutionは主張しない。

### Continue With Redesign

次の場合は採用せず、1回の再設計候補として扱う。

- classical baselineと概ね同等だが、特定の周波数構造で再現可能な利点がある。
- float条件では良いが量子化で崩れ、parameterization改善の仮説が明確である。
- coupled候補だけがsingle-state制限を改善するが、bit overheadが大きい。

### Do Not Adopt Current Candidate

次の場合は現行候補を不採用とする。

- evaluation setの全主要budgetでclassical INRに支配される。
- 改善がdevelopment setまたは単一caseに限られる。
- 同等品質に必要なserialized bitsまたはdecode timeが大きい。
- periodic artifactや量子化不安定性が主要caseで残る。

#104 のtrainable / source-split結果は、この区分に入る。現行single-state /
coupled-state候補は、#104のevaluation splitで最良classical INR候補を上回らなかった。

#122 のcompact parameterization redesign Candidate A/B/Cも、この区分に入る。Candidate A/Cは
現行Model Eより少し改善したが、best classical INR、nearest、bicubic baselineを
上回らなかったため、採用候補へ戻す根拠には不足している。

#118 のgated interaction風couplingも、この区分に入る。現行coupledを小さく改善したが、
best classical INRには届かず、平均540 bitsのcoupling overheadを要した。controlled-rotation風
候補も現行coupledを改善しなかった。

不採用はquantum-inspired representation一般ではなく、評価した構造とprotocolに対する判断とする。

### Continue Or Pause Gate

Model E系列を継続する場合は、次のいずれかを事前に満たす小さなIssueとして扱う。

- candidate size、frequency count、depth、statesを増やしても、同じsource-split fixtureと
  `incremental_side_bits` でclassical baselineと比較できる。
- #119 のbit-depth耐性は別途保存済みであり、gated couplingの改善や540-bit overheadとは別軸として [#136](https://github.com/nana-nun/sidf-lab/issues/136) で判断文書へ反映する。
- #125 のautograd optimizer判断に従い、PyTorch CPU optional backendを別Issueでspikeし、
  Model Eだけでなくclassical INR baselineにも同じoptimizer条件を適用できる。
- #118 のcoupling variationは完了し、gated interaction風候補の小さな改善、classical基準未達、
  平均540 bitsのoverheadを分離して記録した。coupling式の追加だけを継続根拠にはしない。
- #134 でPyTorch導入済み環境のoptional autograd optimizerを実測し、Model Eとclassical INRへ
  同じoptimizer条件を適用できるか確認できる。

Model E系列を一時保留する場合は、次を根拠にできる。

- #98、#104、#117、#118、#122 のいずれでも、評価済みModel E候補は採用基準を満たしていない。
- #122 では再設計候補が現行候補を少し改善したが、best classical INR、nearest、
  bicubic baselineを上回らなかった。
- #118 ではgated interaction風候補が現行coupledを小さく改善したが、best classical INRを
  上回らず、追加side bitsも増えた。
- 現時点で未解決の改善仮説は、candidate size、bit-depth、optimizerに分かれており、どれも
  採用済み仕様ではなく追加検証の候補である。coupling variationは今回の条件で比較済みである。
- 実用圧縮、super-resolution、quantum advantageは未測定であり、Model E継続の理由にしない。

## 12. Research Sequence

1. Issue #95: 一次文献とclassical baselineの整理。
2. Issue #96: 本研究設計の固定。
3. Issue #97: single-state / coupled multi-stateの最小実装。
4. Issue #98: serialized bit budgetを揃えた比較実験。
5. Issue #103: trainable INR / Model E fitting基盤の最小実装。
6. Issue #108: source分割済みgrayscale patch fixtureの追加。
7. Issue #104: trainable INR baselineとsource分割datasetによるModel E再比較。
8. Issue #117: 現行Model E fittingのoptimizer / initialization診断。
9. Issue #121: Candidate A/B/C parameterizationの最小実装。
10. Issue #122: Candidate A/B/Cをsource-split条件で比較。
11. Issue #127: Model E系列の継続/保留判断の整理。
12. Issue #118: coupling variationをsource-split条件で比較。
13. Issue #138: #118の結果を継続/保留判断へ反映。

#98、#104、#117、#118、#122 の結果から、評価済みModel E候補はSIDF draft specificationへ採用しない。
Model Eを続ける場合は、採用保留のまま candidate size、bit-depth耐性、optimizer、
coupling設計を分けて再検証する。続けない場合は、Model E系列を一時保留として扱う。

## 13. Limitations

- この文書は研究計画であり、Model Eの画像品質を示す結果ではない。
- bitstream layoutとcontainer overheadの具体値は未定義である。
- natural patch集合、budget、optimizerの具体値はIssue #98開始前に固定する必要がある。
- classical simulation上の利点は量子hardware上の計算優位を意味しない。
- per-image fittingはencoder costを必要とし、decode品質だけでは実用圧縮性を判断できない。

## 14. References

- `references/notes/quantum-inspired-implicit-image-representation.md`
- `docs/research-state.md`
- `specs/sidf-v0.3.0-draft.md`
- `results/2026-06-14-issue-87-model-d-update-procedure/notes.md`
- `results/2026-06-14-issue-88-model-d-deterministic-icm/notes.md`
- [Issue #95](https://github.com/nana-nun/sidf-lab/issues/95)
- [Issue #97](https://github.com/nana-nun/sidf-lab/issues/97)
- [Issue #98](https://github.com/nana-nun/sidf-lab/issues/98)
- [Issue #104](https://github.com/nana-nun/sidf-lab/issues/104)
- [Issue #117](https://github.com/nana-nun/sidf-lab/issues/117)
- [Issue #118](https://github.com/nana-nun/sidf-lab/issues/118)
- [Issue #122](https://github.com/nana-nun/sidf-lab/issues/122)
- [Issue #127](https://github.com/nana-nun/sidf-lab/issues/127)
- [Issue #134](https://github.com/nana-nun/sidf-lab/issues/134)
- [Issue #138](https://github.com/nana-nun/sidf-lab/issues/138)
- `results/2026-06-28-issue-104-trainable-inr-source-split/notes.md`
- `results/2026-06-29-issue-117-model-e-fitting-diagnostics/notes.md`
- `results/2026-06-29-issue-122-model-e-parameterization-redesign/notes.md`
- `results/2026-07-27-issue-118-model-e-coupling-variants/notes.md`
- `docs/model-e-autograd-optimizer-decision.md`
- `references/notes/model-e-parameterization-redesign.md`
