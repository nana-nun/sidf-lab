# Model E angle / frequency parameterization redesign

## Source

- Related Issue: [#116](https://github.com/nana-nun/sidf-lab/issues/116)
- Previous results: [Issue #98](https://github.com/nana-nun/sidf-lab/issues/98), [Issue #104](https://github.com/nana-nun/sidf-lab/issues/104)
- Design docs: `docs/model-e-research-design.md`, `docs/model-decision-map.md`
- Existing reference notes:
  - `references/notes/quantum-inspired-implicit-image-representation.md`
  - `references/notes/inr-bit-accounting.md`
- Relevant implementation:
  - `src/sidf_lab/model_e.py`
  - `src/sidf_lab/inr_fit.py`

## Summary

Issue #98 と #104 では、現行の Model E single-state / coupled-state 候補を SIDF draft 仕様へ採用する根拠は得られなかった。#98 は fixed feature dictionary + linear readout 条件、#104 は trainable / source-split 条件であり、どちらも評価済み候補に対する負の結果である。これは Model E 一般、量子回路由来の座標関数一般、または別の angle / frequency parameterization の否定ではない。

現行実装の Model E は、次の feature をそのまま角度入力として使う。

```text
features = [x, y, bilinear guide, gradient_x, gradient_y]
angles   = features @ layer_weights
states   = tanh(sin(angles + states))
readout  = dot(states, normalized_readout)
```

この形は小さく検証しやすい一方で、周波数設計が `layer_weights` に暗黙化されている。Fourier baseline のような明示的な mode、RFF のような frequency matrix、SIREN のような activation scale と初期化規則が分かれていない。そのため、#104 の結果だけからは、Model E 構造そのものが弱いのか、周波数配置・角度scale・量子化範囲・optimizerの組み合わせが悪いのかを分離しにくい。

次の Model E 再設計では、単なる random search のstep数追加ではなく、周波数集合、角度scale、guide-derived feature の混ぜ方、保存parameterの粒度を明示的に分ける必要がある。

## Current Model E Parameter Handling

現行候補の保存parameterは、`layers`、`readout`、`residual_scale` である。

| Item | Current role | Current storage |
| --- | --- | --- |
| `layers[depth, states, features]` | featureからstateごとの角度を作る係数 | 全要素を量子化parameterとして保存 |
| `readout[states]` | 最終stateからresidual scalarを作る線形重み | 全要素を量子化parameterとして保存 |
| `residual_scale` | bounded residualの振幅 | 量子化parameterとして保存 |
| `kind`, `depth`, `states` | single / coupled と構造 | header / structure bits |
| feature order | `x`, `y`, guide, gradient_x, gradient_y | decoder仕様側の固定規則 |

この設計では、`layers` が角度係数、周波数、feature mixing、layerごとのscaleを同時に担う。座標周波数とguide-derived featureの重みも同じ量子化範囲へ入るため、座標の周期構造と画像値・勾配値のゲート構造が分離されていない。

## Comparison With Classical Baselines

### Fourier feature

既存の Fourier baseline は、`sin(pi * freq * x)`、`cos(pi * freq * x)`、`sin(pi * freq * y)`、`cos(pi * freq * y)`、`sin(pi * freq * (x + y))`、`cos(pi * freq * (x - y))` のように、使う周波数modeが明示されている。保存parameterは主に readout で、周波数表を固定仕様にすれば周波数matrix自体を送らなくてよい。

利点は、周波数集合と保存係数を分けて議論しやすいこと。制限は、固定modeが画像ごとの残差構造に合わない場合、少ないmodeでは表現が粗くなることである。

### RFF

RFF baseline は、入力featureへ random または trainable な weight matrix と bias をかけ、`sin` / `cos` の特徴を作る。frequency matrix を seed から固定生成するか、明示parameterとして保存するかで side bits が変わる。#98 では fixed-feature + linear readout 条件で `rff_mid` が evaluation split の最良parameterized候補だった。

利点は、多方向の周波数成分を少数featureで作れること。制限は、周波数分布のscale、seed、matrix保存の扱いを明示しないと bit accounting が曖昧になることである。

### SIREN

SIREN は周期activationを持つ小型networkで、activation scale や初期化が表現できる周波数と学習安定性に強く影響する。SIDFの現行 small SIREN baseline は最小実装であり、通常の十分にtrainされたmulti-layer SIRENとは別物として扱う必要がある。

利点は、layerを重ねて非線形な高周波表現を作れること。制限は、weights、biases、activation constants、初期化、optimizerが結果に強く影響し、parameter side bitsも増えやすいことである。

### Model E

Model E は data re-uploading 由来の「角度入力を繰り返しstateへ作用させる」構造を使う。理論上は Fourier 的な周期構造と state coupling による交差項を作れる可能性がある。しかし現行実装では、周波数集合と係数自由度が明示的に分かれていないため、Fourier / RFF / SIREN と比べたときに「何を変えたら改善するか」が読み取りにくい。

次の候補では、Model E の独自性を「量子っぽい名前」ではなく、保存bitあたりの周波数配置、state coupling、量子化耐性、decodeの決定性として測る。

## Candidate Parameterizations

### Candidate A: fixed frequency ladder with learnable scale

固定の基底周波数集合を仕様側に持ち、保存parameterでは layer / state ごとの scale、phase、readout を学習する。

```text
base_modes = [x, y, x + y, x - y] x [1, 2, 4, 8]
angle_l,s = scale_l,s * dot(base_modes, fixed_coeff_l,s) + phase_l,s
state_l   = update(state_l-1, angle_l, guide_gate)
```

保存候補:

- `model_id`
- frequency ladder id
- depth, state count
- per-layer / per-state `scale`
- per-layer / per-state `phase`
- readout
- residual scale
- optional guide gate weights

期待する切り分け:

- Fourier baseline と同じ、または近い周波数集合を使い、Model E の state update / coupling が readout-only Fourier より役立つかを見る。
- 周波数表を固定仕様にすることで、side bits は scale / phase / readout 中心になる。

量子化上の注意:

- `scale` は正値に制限し、log-scale または小さな離散候補で保存すると clipping による破綻を減らせる可能性がある。
- `phase` は周期量なので、`[-pi, pi]` の wrap 規則を明記する。
- 高周波ladderを増やすほど量子化誤差が画像位置ずれとして目立つため、8-bit / 12-bit / 16-bit の耐性比較が必要になる。

### Candidate B: learned compact frequency table

RFFに近いが、Model E側のstate updateへ入れる frequency table を少数だけ明示保存する。frequency vector、phase、state mixingを分け、readoutと同じ量子化規則で保存する。

```text
freq_k   = [fx_k, fy_k, fguide_k, fgx_k, fgy_k]
angle_k  = dot(features, freq_k) + phase_k
state_l  = model_e_update(state_l-1, angle_k, coupling_l)
```

保存候補:

- frequency vector table
- phase table
- depth, state count
- state mixing / coupling weights
- readout
- residual scale

期待する切り分け:

- RFFの良さが random frequency placement にあったのか、Model E のstate updateと組み合わせても保存bitあたりの改善が残るのかを調べる。
- fixed ladder より自由度は高いが、frequency tableを保存するため side bits が増える。

量子化上の注意:

- frequency vectorの範囲を座標成分とguide-derived成分で分ける。`x, y` は周期を作るため大きめ、guide / gradient はゲートまたは局所変調として小さめに制限する。
- seed生成RFFと違い、tableを保存する場合は `frequency_parameter_bits` を明示する。
- tableをseed生成に戻す場合は、seed、分布、scale、feature orderをheaderに含める。

### Candidate C: separated coordinate frequency and guide modulation

座標周波数とguide-derived featureを同じ角度線形結合に入れず、座標から作る周期基底と、guide / gradient から作る振幅またはgateを分ける。

```text
coord_angle = dot([x, y, x + y, x - y], coord_frequency) + phase
guide_gate  = sigmoid(a0 + a1 * base + a2 * grad_x + a3 * grad_y)
state       = update(state, coord_angle, guide_gate)
residual    = residual_scale * guide_gate * readout(state)
```

保存候補:

- coordinate frequency id or coordinate frequency parameters
- phase
- guide gate weights
- coupling weights if coupled
- readout
- residual scale

期待する切り分け:

- 現行候補で混ざっていた「位置の周期」と「guideによる局所的な振幅調整」を分離する。
- 低解像度guideだけを使う条件を保ったまま、guideが高周波そのものを作るのではなく residual の場所と強さを制御する形にする。

量子化上の注意:

- guide gate weights は過大になると残差が局所的に飽和しやすい。clip rule と `residual_scale` の関係を保存する。
- gateがほぼ0または1へ潰れる場合、float条件では良く見えても量子化後に段差artifactが出る可能性がある。
- 周波数parameterとgate parameterは異なる範囲・bit depthを使う候補があるが、初回比較では単一12-bit規則と、parameter group別規則を分けて測る。

## Minimal Comparison Conditions

次の実験Issueへ渡す最小条件は、#104 の source-split fixture と bit accounting を再利用する。

対象candidate:

- image baselines: nearest, bilinear, bicubic
- classical INR baselines: Fourier, RFF, SIREN, small MLP
- current Model E single-state / coupled-state
- Candidate A fixed ladder
- Candidate B compact frequency table
- Candidate C coordinate + guide modulation

評価条件:

- development / evaluation split は source image 単位で分ける。
- float output と quantized output を分けて保存する。
- quantization はまず 12-bit uniform を使い、可能なら 8-bit / 16-bit を追加する。
- `incremental_side_bits` は parameter group別に分けて保存する。
- guide bits、container overhead、entropy codingを含まない制限を notes に明記する。

主要metrics:

- MAD
- PSNR
- SSIM
- gradient magnitude MAD
- Laplacian MAD
- serialized side bits
- fit seconds
- decode seconds
- float-to-quantized delta

診断:

- residual image
- quantized parameter clipping ratio
- coordinate frequency table or ladder id
- extrapolated output diagnostic when practical
- per-candidate artifact note

## Recommended Next Issues

### Implementation: [#121](https://github.com/nana-nun/sidf-lab/issues/121)

Title: `[impl] Model E parameterization候補を追加する`

Labels: `t:impl`, `p:1`

Goal:

Candidate A / B / C を `src/sidf_lab/` に最小実装し、既存 `fit_inr` interface から呼べるようにする。

Tasks:

- [ ] Candidate A fixed frequency ladder を追加する
- [ ] Candidate B learned compact frequency table を追加する
- [ ] Candidate C coordinate frequency + guide modulation を追加する
- [ ] parameter layout と side-bit estimate を group別に保存できるようにする
- [ ] 既存 `model_e_single` / `model_e_coupled` と同じ decode interface に揃える
- [ ] deterministic smoke test と quantization roundtrip test を追加する

Acceptance Criteria:

- [ ] 新candidateが同じlow guideとparameterで決定論的にdecodeできる
- [ ] parameter count と incremental side bits が保存できる
- [ ] 既存 Model E 候補を壊していない
- [ ] `python -m unittest discover -s tests` が通る

### Experiment: [#122](https://github.com/nana-nun/sidf-lab/issues/122)

Title: `[exp] Model E parameterization候補をsource-splitで比較する`

Labels: `t:exp`, `p:1`

Goal:

Candidate A / B / C を #104 と同じ source-split fixture、baseline、bit accounting で比較し、現行 Model E より続ける価値があるかを測る。

Tasks:

- [ ] #104 の source-split fixture を再利用する
- [ ] nearest / bilinear / bicubic、Fourier / RFF / SIREN / MLP baselineを含める
- [ ] current Model E single / coupled と Candidate A / B / C を比較する
- [ ] float結果と12-bit量子化結果を分けて保存する
- [ ] side-bit内訳、fit time、decode time、主要PNGを保存する
- [ ] 改善があっても compression / super-resolution / quantum advantage と書かない

Acceptance Criteria:

- [ ] `results/YYYY-MM-DD-issue-<number>-model-e-parameterization-redesign/` に `config.json`、`metrics.json`、`notes.md`、主要PNGがある
- [ ] evaluation splitで最良classical INR、現行Model E、新candidateの差が追跡できる
- [ ] 結果、解釈、限界、次の扱いが分かれている

## Relevance to SIDF

この再設計は、Model Eを採用済み候補へ戻すものではない。目的は、#98 / #104 の負の結果を受けて、どの仮定を変えると測定可能な差が出るかを小さく確認することである。

有望と扱えるのは、同じ guide、同じ source-split evaluation、同程度の incremental side bits、量子化後decodeの条件で、classical INR baselineに対して再現性のある改善が見えた場合に限る。単一caseの改善、floatだけの改善、またはbit overheadを無視した改善は、採用根拠ではなく診断結果として扱う。

## Limitations

- このメモは設計整理であり、新しい実験結果ではない。
- Candidate A / B / C は実装前の仮説であり、画像品質を改善するとは限らない。
- 既存の #104 optimizer は最小random-searchであり、parameterizationとoptimizer不足を完全には分離できない。
- side-bit見積もりは `incremental_side_bits` 中心であり、guide bits、container overhead、entropy codingを含む実用圧縮評価ではない。
- quantum advantage、実用圧縮、一般的super-resolutionは未測定である。

## Follow-up

- Candidate A / B / C は実装Issue [#121](https://github.com/nana-nun/sidf-lab/issues/121) へ分けた。
- 実装後に #104 fixture を使った実験Issue [#122](https://github.com/nana-nun/sidf-lab/issues/122) で比較する。
- 実験後、`docs/model-decision-map.md` に「再設計候補として継続」「現行候補と同じく不採用」「さらに分離実験が必要」のどれに近いかを追記する。
