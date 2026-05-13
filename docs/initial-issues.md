# Initial GitHub Issues

GitHub repository: `nana-nun/sidf-lab`

このファイルは、GitHub Issuesに登録する初期タスク候補です。実際のIssue作成後は、各Issue URLをこのファイルまたは `docs/research-state.md` に反映します。

## Created Issues

- #1: https://github.com/nana-nun/sidf-lab/issues/1
- #2: https://github.com/nana-nun/sidf-lab/issues/2
- #3: https://github.com/nana-nun/sidf-lab/issues/3
- #4: https://github.com/nana-nun/sidf-lab/issues/4
- #5: https://github.com/nana-nun/sidf-lab/issues/5
- #6: https://github.com/nana-nun/sidf-lab/issues/6
- #7: https://github.com/nana-nun/sidf-lab/issues/7
- #8: https://github.com/nana-nun/sidf-lab/issues/8

## 1. [docs] 研究リポジトリ基盤を整備する

Labels: `t:docs`, `p:1`

### Goal

sidf-lab の研究目的、現在地、AI運用、実験保存ルールを文書化する。

### Tasks

- [ ] `AGENTS.md` を整備する
- [ ] `docs/repository-architecture.md` を整備する
- [ ] `docs/research-plan.md` を整備する
- [ ] `docs/research-state.md` を整備する
- [ ] `docs/experiment-log-template.md` を整備する

### Acceptance Criteria

- [ ] AIが最初に読むべきファイルが明確
- [ ] 画像保存ルールが明記されている
- [ ] Git/GitHub運用が明記されている

## 2. [maint] GitHub labels と Projects 運用を整える

Labels: `t:maint`, `p:1`

### Goal

`nana-nun/sidf-lab` で `hash-lab` と同様にIssue/Project運用できる状態にする。

### Tasks

- [ ] labels `t:exp`, `t:ref`, `t:impl`, `t:docs`, `t:maint` を作る
- [ ] labels `p:0`, `p:1`, `p:2` を作る
- [ ] GitHub Projects を作成または既存Projectに追加する
- [ ] Project status列を決める

### Acceptance Criteria

- [ ] Issueテンプレートが使える
- [ ] Projectで状態管理できる

## 3. [impl] Python package skeletonを追加する

Labels: `t:impl`, `p:1`

### Goal

Model C/D実験を共通モジュール化できるPython構成を作る。

### Tasks

- [ ] `src/sidf_lab/` を作る
- [ ] `guides.py`, `energy.py`, `anneal.py`, `metrics.py`, `visualize.py`, `io.py` を追加する
- [ ] `tests/` を追加する
- [ ] `.venv + requirements.txt` 前提のREADMEを更新する

### Acceptance Criteria

- [ ] `PYTHONPATH=src` でimportできる
- [ ] 最小テストが通る

## 4. [exp] Model C cross baselineを保存形式つきで再実行する

Labels: `t:exp`, `p:1`

### Goal

Model Cを、再現可能な保存形式つき実験として固定する。

### Baseline

- noisy static guide direct display
- Model A if implemented

### Metrics

- MAD
- cross mean
- background mean
- edge leakage
- foreground/background variance
- decode time

### Saved Artifacts

- [ ] `config.json`
- [ ] `metrics.json`
- [ ] `notes.md`
- [ ] static guide PNG
- [ ] rendered PNG
- [ ] baseline PNG
- [ ] difference PNG when useful

### Acceptance Criteria

- [ ] 同じ `experiment_seed` と `decoder_seed` で再現できる
- [ ] 画像が保存されている
- [ ] `docs/research-state.md` が必要に応じて更新されている

## 4b. [exp] Model C freeze benchmarkを作る

Labels: `t:exp`, `p:1`

### Goal

Rust移植前の基準実装として、Model Cが複数の基本形状で安定するか確認する。

### Shapes

- cross
- diagonal line
- circle
- thin line
- soft gradient

### Metrics

- MAD
- foreground mean
- background mean
- foreground variance
- background variance
- edge leakage
- decode time
- edge width, if practical

### Acceptance Criteria

- [ ] 各shapeで `config.json`、`metrics.json`、`notes.md`、主要PNGが保存される
- [ ] 同じ `experiment_seed` と `decoder_seed` で再現できる
- [ ] cross baseline で `Background Mean <= 0.02`、`Edge Leakage <= 0.02`、`MAD <= 0.03` を満たす
- [ ] gradientについては数値だけでなく視覚的な階調破綻を `notes.md` に記録する
- [ ] Rust移植前に残すべき未確定事項がIssue化されている

## 5. [exp] Model D と bilinear/bicubic の比較指標を追加する

Labels: `t:exp`, `p:2`

### Goal

Model Dが単純補間より何を改善しているかを画像と数値で比較する。

### Baseline

- nearest
- bilinear
- bicubic

### Metrics

- edge width
- edge leakage
- foreground/background mean
- variance
- decode time

### Saved Artifacts

- [ ] low-res guide PNG
- [ ] upscaled guide PNG
- [ ] nearest/bilinear/bicubic PNG
- [ ] confidence map PNG
- [ ] SIDF rendered PNG
- [ ] difference PNG
- [ ] `notes.md`

## 6. [ref] 画像再構成・MRF・超解像の参考文献を集める

Labels: `t:ref`, `p:2`

### Goal

SIDFの位置づけに必要な参考文献を集める。

### Topics

- Markov Random Field
- Conditional Random Field
- simulated annealing image restoration
- edge-preserving smoothing
- super-resolution baseline
- deterministic PRNG / fixed-point reproducibility
- Perlin / fractal noise

### Acceptance Criteria

- [ ] `references/reading-list.md` が更新される
- [ ] 重要文献のメモが `references/notes/` にある
- [ ] BibTeXが分かるものは `references/papers.bib` に入る

## 7. [docs] SIDF v0.3 draft仕様を作る

Labels: `t:docs`, `p:2`

### Goal

Model Dまでの知見を、確定仕様ではなくdraftとして整理する。

### Tasks

- [ ] `specs/sidf-v0.3.0-draft.md` を作る
- [ ] 未確定事項をIssue化する
- [ ] 実験結果への参照を書く

### Acceptance Criteria

- [ ] draftであることが明記されている
- [ ] 未検証の主張が断定されていない
