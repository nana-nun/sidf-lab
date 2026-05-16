# Geman and Geman 1984: MRF・Gibbs分布・確率的緩和とModel C

## Source

- Title: Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images
- Authors: Stuart Geman and Donald Geman
- Year: 1984
- Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(6), 721-741
- DOI: https://doi.org/10.1109/TPAMI.1984.4767596
- PubMed: https://pubmed.ncbi.nlm.nih.gov/22499653/
- PDF候補: https://www.dam.brown.edu/people/geman/Homepage/Image%20processing%2C%20image%20analysis%2C%20Markov%20random%20fields%2C%20and%20MCMC/stochastic%20relaxation.pdf
- BibTeX key: `geman1984stochasticRelaxation`
- Related Issue: https://github.com/nana-nun/sidf-lab/issues/12

## Summary

Geman and Geman 1984 は、画像を格子上の確率場として扱い、局所的な画素状態やエッジ状態を統計力学の状態に対応させる。中心にあるのは、画像の好ましさを直接ローカル条件付き確率で書くよりも、energy function を定義し、その energy に対応する Gibbs distribution として画像モデルを扱う考え方である。

この論文では、Gibbs distribution と Markov random field の同値性を背景に、劣化画像が与えられたときの posterior distribution も MRF として扱えることを示す。復元は posterior の最も確からしい状態、つまり MAP estimate を探す問題として整理される。

探索方法としては、Gibbs sampler 型の局所更新と、temperature を徐々に下げる annealing が使われる。temperature が高い段階では状態空間を探索し、低い段階では低 energy の状態へ寄せる。この流れにより、画像復元を確率的緩和として解く枠組みが作られている。

## Concepts

| Concept | Geman and Geman 1984での位置づけ | SIDFで読むときの注意 |
| --- | --- | --- |
| MRF | 近傍構造によって局所依存を持つ画像モデル | Model Cの近傍相互作用を説明する背景になる |
| Gibbs distribution | energy function から定まる確率分布 | SIDFのenergyは候補モデルであり、正式な確率モデルとして検証済みではない |
| Bayesian restoration | 劣化過程とpriorからposteriorを作り、復元を推定問題にする | SIDFでは低解像度guideとseedつき緩和を扱うため、同一問題ではない |
| MAP estimation | posteriorで最も確からしい画像を探す | Model Cの低energy状態探索と対応づけられるが、厳密なposterior定義は未整理 |
| Stochastic relaxation | 局所的な確率更新で画像状態を改善する | SIDFのannealing / relaxation decoder と概念的に近い |
| Annealing | temperatureを下げて低energy状態に近づける | SIDFでもschedule設計とdecode timeが重要になる |

## Model Cとの対応

SIDF Model C は、現在の実装では次の2種類の項を持つ。

```text
lambda_data * sum_i (v_i - s_i)^2
+ sum_(i,j) J_ij (v_i - v_j)^2

J_ij = J_base * exp(-gamma * (s_i - s_j)^2)
```

対応関係:

| 観点 | Geman and Geman型の画像復元 | SIDF Model C |
| --- | --- | --- |
| 状態 | 格子上の画像状態、必要に応じてエッジ状態 | decoder state `v` |
| 観測 | 劣化画像や観測データ | STATIC guide `s` |
| Data term | 観測との整合性をposteriorに入れる | `lambda_data * (v_i - s_i)^2` |
| Prior / interaction | 近傍関係やエッジを含む画像prior | `J_ij * (v_i - v_j)^2` |
| Edge handling | エッジ状態をモデルに含める構成がある | guide差から `J_ij` を弱め、境界をまたぐ混合を抑える |
| Optimization | stochastic relaxation / annealingでMAPを探す | seedつき確率的緩和で低energy状態を探す |

Model C は、Geman and Geman の枠組みそのものではない。しかし、data fidelity と近傍相互作用をenergyとして足し合わせ、局所更新で低energy状態を探す点は、MRF / Gibbs / MAP restoration の文脈で説明しやすい。

## Relevance to SIDF

この文献は、SIDF Model C を「物理っぽい比喩」ではなく、energy-based image reconstruction の候補として説明するための背景になる。

SIDFにとって特に有用な点:

- `lambda_data` は、観測またはguideへの整合性を表す data fidelity として説明できる。
- `J_ij` は、近傍画素の依存関係を表す interaction / prior に近い役割を持つ。
- annealing は、energy landscape 上で低energy状態を探索する decoder 手続きとして説明できる。
- Model A の外部場項よりも、Model C の二乗 data fidelity の方が Bayesian restoration の data term に近い。

ただし、SIDFの研究目的は、古典的な画像復元問題をそのまま再実装することではない。SIDFでは、低解像度guide、seed、保存可能なdecoder設定、将来的な決定論的実装を組み合わせた再構成条件を扱う。そのため、この文献は背景であり、SIDFの性能を証明する根拠ではない。

## Differences from SIDF Model C

相違点:

- Geman and Geman は Bayesian restoration として posterior / MAP を明示的に扱うが、SIDF Model C の確率モデルはまだ正式には定義していない。
- 論文では劣化画像からの復元を扱うが、SIDFでは低解像度または簡約 guide からの再構成を研究している。
- 論文ではエッジ状態を含むモデル化が重要だが、Model C は guide差から近傍結合を弱める単純な edge-aware interaction にしている。
- SIDFでは seed、保存形式、再現可能なdecoder設定、将来のRust固定小数点実装が研究対象になる。
- 現時点のModel C結果は synthetic grayscale guides に限られ、自然画像の一般復元性能や実用圧縮性能は示していない。

## Research State Decision

`docs/research-state.md` は更新しない。

理由:

- このIssueは文献整理であり、新しい実験結果やModel Cの数値評価を追加していない。
- Model Cの現在地はすでに `docs/research-state.md` に記録されている。
- 今回の成果は、研究状態の更新ではなく、Model Cの解釈を支える参考文献メモとして扱うのが適切である。

## Limitations

- このメモは論文全体の数式証明や収束条件を網羅していない。
- Model Cを厳密なMRF posteriorとして定式化したわけではない。
- `J_ij` と論文内のエッジモデルの対応は概念的な比較に留まる。
- SIDFがGeman and Geman型の方法より優れている、または実用圧縮形式として有効である、という主張ではない。
- MRF / Gibbs / MAP の用語をSIDF仕様へ入れる場合は、確率モデルとdeterministic decoder仕様の境界を別途整理する必要がある。

## Follow-up

- [#26](https://github.com/nana-nun/sidf-lab/issues/26) で、Model Cのenergyを、MRF prior、data likelihood、posterior energy、または deterministic decoder objective のどれとして書くのが最も正確か整理する。
- Model CとGeman and Geman型のrestorationを、同じsynthetic inputで直接比較する必要があるか検討する。
- SIDF仕様案では、energy-based reconstruction と deterministic decoder procedure を混同しない書き方にする。
