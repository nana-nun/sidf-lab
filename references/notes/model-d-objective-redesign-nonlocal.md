# Model D objective再設計と非局所自己類似性

## Source

Related Issue: [#120](https://github.com/nana-nun/sidf-lab/issues/120)

Primary sources:

- Antoni Buades, Bartomeu Coll, Jean-Michel Morel, "A Non-Local Algorithm for Image Denoising", CVPR 2005. DOI: `10.1109/CVPR.2005.38`
- Antoni Buades, Bartomeu Coll, Jean-Michel Morel, "A Review of Image Denoising Algorithms, with a New One", Multiscale Modeling & Simulation 2005. DOI: `10.1137/040616024`
- Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering", ECCV 2010. DOI: `10.1007/978-3-642-15549-9_1`
- Johannes Kopf, Michael F. Cohen, Dani Lischinski, Matt Uyttendaele, "Joint Bilateral Upsampling", ACM Transactions on Graphics 2007. DOI: `10.1145/1276377.1276497`

Repository evidence:

- `results/2026-06-14-issue-87-model-d-update-procedure/notes.md`
- `results/2026-06-14-issue-88-model-d-deterministic-icm/notes.md`
- `results/2026-06-07-issue-74-guided-filter-baselines/notes.md`
- `docs/research-state.md`
- `docs/model-decision-map.md`

## Summary

Non-local Means は、近傍画素だけでなく画像内の別位置にある類似patchを探し、その類似度で画素値を加重平均する denoising 手法である。局所平滑化が「近い画素は似ている」という仮定を使うのに対し、NLM は「離れていてもpatchが似ていれば同じ構造を持つ可能性がある」という自己類似性を使う。

Guided filter や joint bilateral upsampling は、入力画像または別の guidance image の構造を使って edge-aware な補正や upsampling を行う。特に joint bilateral upsampling は、高解像度 guidance がある場合に低解像度 solution を高解像度へ戻す文脈で使われる。

このメモでは、これらの文献を SIDF の結果として扱わず、Model D の次objective候補を設計するときにどの情報を使えるか、どこから情報量条件が変わるかを整理する。

## Existing Negative Evidence

Model D の既存負の結果は、少なくとも次の4要素に分けて扱う。

| Area | Evidence | Current interpretation |
| --- | --- | --- |
| Objective | Issue #88 | 現行quadratic objectiveは、deterministic ICMでより低いobjectiveへ到達しても、cross / natural patchのMAD、PSNR、SSIM、gradient magnitude MADを改善しなかった。 |
| Update | Issue #87 | 有限温度Metropolisはuphill moveを多く受理し、objectiveとreference差分を増やした。標準decoder手順としては採用しない。 |
| Texture | Issues #37, #63, #75 | white noiseや単純なstructured texture経路は、意味的ディテールやbaseline改善として確認できていない。 |
| Confidence | Issues #56, #61, #67 | gradient-based confidence、flatter confidence、edge-band confidenceは、uniform confidenceを一貫して上回らなかった。 |

したがって、次に変えるべき仮定は「quadratic local pairwise smoothingを強く解けばよい」ではない。変更するなら、どの画素またはpatchを結び、どの情報源で重みを決め、どのbaselineと比べるかを先に固定する必要がある。

## Information Conditions

Model D で非局所自己類似性を使う場合、使える情報を次のように分ける。

| Condition | Available information | What can be claimed |
| --- | --- | --- |
| Low guide only | low-resolution guide、seed、保存parameter、upscaled guideから作るpatch | 低解像度で残った自己類似性を使う候補。失われた高周波構造を知っているとは扱わない。 |
| Ground Truth evaluation | high-resolution reference is used only for metrics | objectiveや重みには使わず、MAD / PSNR / SSIM / gradient metricsの評価にだけ使う。 |
| High-resolution guidance | targetとは別の高解像度guidance image | guided upsampling条件。SIDF low-guide-only条件とは別枠で比較する。 |
| Dataset or learned prior | training set, learned model, external prior | 現行SIDF Model Dの範囲外。導入するなら別研究系列として扱う。 |

NLM をそのまま高解像度出力上で使うと、patch類似度を何から計算するかが問題になる。高解像度 Ground Truth や高解像度 guidance からpatch類似度を計算すると、SIDF decoderが本来持たない情報を使う。low-guide-only条件では、類似度は低解像度guide、またはその決定論的upscaleから計算する必要がある。

## Objective Candidates

次に試す候補は、現行Model Dの小調整ではなく、情報条件を明示した次の2案に絞る。

### Candidate 1: Low-guide patch graph regularization

低解像度guideまたはbilinear-upscaled guide上でpatch descriptorを作り、類似patch間に非局所edgeを張る。decoder objectiveは、現行の局所4近傍pairwiseだけでなく、patch類似度で選んだ少数の非局所neighborへも差分penaltyを加える。

仮定:

- low guideに残った繰り返し構造は、局所pairwise smoothingより有用な拘束になる可能性がある。
- 非局所edgeは、現在のquadratic local objectiveの代替ではなく、別の結合グラフとして評価する。

制限:

- low guideで見えない細部は復元できない。
- bilinear-upscaled guideから類似patchを探すだけなら、patch graphが補間結果の滑らかさを再表現するだけになる可能性がある。
- quadratic penaltyのままだと、ICMでobjectiveを下げてもreference品質が改善しない問題を再発する可能性がある。

### Candidate 2: Robust edge-aware data term with nonlocal baseline comparison

現行quadratic objectiveを直接拡張する前に、low-guide-onlyの明示的baselineとして self-guided NLM / joint bilateral refinement / guided filter を同じ入力条件で比較する。Model D側は、pixel値を生成するのではなく、robust data termまたはresidual targetの候補を比較する小実験に留める。

仮定:

- まず明示的フィルタbaselineがどこまで改善するかを測らないと、energy decoderを複雑にする価値を判断しにくい。
- robust penaltyやresidual targetは、現行quadratic objectiveよりreference metricsと一致する可能性がある。

制限:

- self-guided NLM はdenoising寄りの手法であり、低解像度から失われた構造を復元する保証はない。
- 高解像度guidanceを使う比較は別条件として明示する必要がある。

## Experiment Plan

次の `t:exp` Issue に渡すなら、最小比較は次の形にする。

### Question

low-guide-only条件で、patch similarity または非局所edgeを使う候補は、nearest / bilinear / bicubic、および既存 guided filter系baselineより reference metrics を改善するか。

### Setup

- Inputs: synthetic cross、diagonalまたはcircle、Public Domain natural patchを最低1枚。
- Guide: syntheticは既存と同じ低解像度guide、natural patchはblock-average low guide。
- Conditions:
  - nearest
  - bilinear
  - bicubic
  - existing low-guide-only guided filter / joint bilateral baseline
  - self-guided NLM from upscaled low guide
  - Candidate 1 nonlocal patch graph with deterministic ICM or direct solver
  - optional robust data term candidate
- Information rule: Ground Truthはmetrics計算にだけ使い、patch matchingやguidanceには使わない。

### Metrics

- MAD
- PSNR
- global SSIM or existing repository SSIM
- gradient magnitude MAD
- strong-edge MAD or edge leakage when masks are meaningful
- decode time
- graph statistics: average nonlocal degree, mean patch distance, selected-neighbor distance distribution

### Saved Artifacts

- `config.json`
- `metrics.json`
- `notes.md`
- low guide and upscaled guide PNG
- baseline PNGs
- self-guided NLM / candidate output PNGs
- difference maps against Ground Truth
- optional nonlocal neighbor visualization for a small set of probe patches

## Relevance to SIDF

この方向で確認したいのは、SIDFが既存手法より優れるかではなく、現行Model Dで失敗した仮定のうち何を変えると測定可能な差が出るかである。

現時点で比較価値がある仮定:

- local 4-neighbor smoothingではなく、low guideから作るpatch graphを使う。
- finite-temperature stochastic relaxationではなく、deterministic solverまたは明示的filter baselineを使う。
- texture termで意味的ディテールを作るのではなく、まず自己類似性が残っている範囲だけを拘束として使う。

## Limitations

- このメモは文献と既存結果の整理であり、新しい実験結果ではない。
- Non-local Means は主にdenoising手法であり、low-resolution guideからのsuper-resolutionを保証しない。
- 高解像度Ground Truthや高解像度guidanceからpatch類似度を計算すると、SIDF decoderが持つ情報量を超える。
- 既存のModel D負の結果は小規模なcrossと1枚のnatural patchを中心にしており、別datasetでの一般的結論ではない。
- compression、super-resolution、既存形式への優位性は未測定である。

## Follow-up

- `t:exp` Issueとして、low-guide-only self-guided NLM / nonlocal patch graph / guided filter系baselineを同じfixtureで比較する。
- 実験前に、patch matchingに使う画像を `low guide only` と `high-resolution guidance` で明確に分ける。
- 改善が見えない場合は、Model D の objective再設計を広げる前に、Model Eまたは別系列の優先度と比較する。
