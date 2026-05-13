# sidf-lab Repository Architecture

Date: 2026-05-13

## 1. Goal

`sidf-lab` は、SIDFを研究するためのリポジトリである。

中心目的は、低解像度ガイド画像、seed、物理パラメータ、決定論的な確率的緩和過程によって、どこまで知覚的に妥当な画像再構成ができるかを検証すること。

当面は Python で実験速度を優先し、モデルが固まった部分から Rust へ移植する。

## 2. Design Principles

- 研究メモ、仕様、実験結果、実装を分ける。
- AIエージェントが最初に読む場所を明確にする。
- すべての実験に baseline、config、metrics、notes を残す。
- テストや実験で生成した画像は必ず保存する。
- 実験結果のドキュメントは毎回残す。
- 画像の見た目だけでなく、数値評価と制限を残す。
- PythonとRustの役割を混ぜすぎない。

## 3. Proposed File Tree

```text
sidf-lab/
  README.md
  AGENTS.md
  .gitignore
  .agents/
    issue-workflow-ai.md
    skills/
      sidf-lab-research/
        SKILL.md
      sidf-lab-exp-issue/
        SKILL.md
      sidf-lab-impl-issue/
        SKILL.md
      sidf-lab-docs-issue/
        SKILL.md
      sidf-lab-ref-issue/
        SKILL.md
      sidf-lab-maint-issue/
        SKILL.md
  docs/
    repository-architecture.md
    research-plan.md
    research-state.md
    sidf-research-notes.md
    experiment-log-template.md
    issue-workflow-human.md
  specs/
    sidf-v0.2.1.md
    sidf-v0.3.0-draft.md
  references/
    README.md
    links.md
    papers.bib
    reading-list.md
    notes/
      .gitkeep
  src/
    sidf_lab/
      __init__.py
      cli.py
      guides.py
      confidence.py
      texture.py
      energy.py
      anneal.py
      metrics.py
      visualize.py
      io.py
  experiments/
    exp_001_model_a_cross.py
    exp_002_model_c_cross.py
    exp_003_model_d_multires_cross.py
    exp_004_shape_benchmark.py
    exp_005_texture_prior_compare.py
  results/
    README.md
    .gitkeep
  .github/
    pull_request_template.md
    ISSUE_TEMPLATE/
      experiment.md
      implementation.md
      documentation.md
  tests/
    __init__.py
    test_energy.py
    test_metrics.py
    test_determinism.py
  requirements.txt
  rust/
    sidf-core/
      Cargo.toml
      src/
        lib.rs
        prng.rs
        fixed.rs
        guide.rs
        confidence.rs
        energy.rs
        anneal.rs
```

## 4. Top-Level Responsibilities

### `AGENTS.md`

AIエージェント用の一次参照。

ここには研究範囲、避けるべき主張、実験保存ルール、Python/Rust方針を書く。

### `.agents/`

AIエージェントのIssue運用や作業種別ごとの手順を書く。

`hash-lab` と同じ思想で、以下のラベルを使う想定:

- `t:exp`: 実験
- `t:ref`: 文献・参考資料
- `t:impl`: 実装
- `t:docs`: ドキュメント
- `t:maint`: 環境・整理

### `docs/`

人間向けの研究メモ。

重要ファイル:

- `research-plan.md`: 今後の仮説と実験計画
- `research-state.md`: 現在何が分かっているか
- `sidf-research-notes.md`: これまでの会話とModel A/C/Dの総括
- `experiment-log-template.md`: 実験メモの型

### `specs/`

SIDF仕様案。

実験メモと仕様案を分けることで、「測定で分かったこと」と「ファイル形式として定義すること」を混ぜない。

### `references/`

論文、リンク、読書メモ。

画像圧縮、超解像、MCMC、MRF/CRF、アニーリング、Perlin noise、固定小数点再現性などを集める。

### `src/sidf_lab/`

Python実装本体。

実験スクリプトから import される安定部分をここに置く。

### `experiments/`

単発実験スクリプト。

`src/sidf_lab/` の機能を組み合わせて、結果を `results/` に保存する。

### `results/`

実験結果の保存場所。

各実験は独立ディレクトリにする。

画像出力がある実験では、表示だけでなくPNGとして保存する。

### `tests/`

小さな決定性テストと数値テスト。

特に重要:

- 同じ seed で同じ出力になること
- energy計算が意図通りであること
- metrics が固定入力に対して期待値を返すこと

### `rust/`

将来の高速・決定論的コア。

最初はPythonと完全統合しなくてよい。まずModel Cの同一入力に対して近い結果または同じ固定小数点結果を返す kernel を作る。

### `.github/`

GitHub Issues / PR のテンプレートを置く。

研究作業をAIに渡しやすくするため、Issueには `Goal`、`Context`、`Tasks`、`Acceptance Criteria`、`References` を書く。

## 5. Python Module Boundary

### `guides.py`

- synthetic guide generation
- downscale / upscale
- grayscale-only baseline
- future Y channel helper

### `confidence.py`

- gradient confidence
- edge confidence
- flat-region confidence
- confidence visualization helper

### `texture.py`

- white noise
- smoothed noise
- fractal noise
- future directional texture

### `energy.py`

- Model A energy
- Model C energy
- Model D energy

このファイルはRust移植の中心候補。

### `anneal.py`

- Metropolis update
- greedy / ICM update
- sweep schedule
- deterministic traversal

### `metrics.py`

- MAD
- foreground/background mean
- variance
- edge leakage
- edge width
- PSNR / SSIM later
- decode time summary

### `visualize.py`

- comparison plots
- confidence map plots
- difference maps
- metric overlays

### `io.py`

- experiment result save/load
- JSON config
- PNG output
- future SIDF binary read/write

## 6. Result Directory Format

```text
results/
  2026-05-13-model-d-cross/
    config.json
    metrics.json
    notes.md
    static_low.png
    upscaled_guide.png
    confidence.png
    rendered.png
    diff_bilinear.png
```

画像生成を伴う実験では、最低限以下を保存する。

- `static` または `static_low`
- baseline image
- rendered image
- confidence map when used
- difference map when useful

小さいPNG、JSON、CSV、MarkdownはGit管理する。大量画像、大きい比較出力、長時間runの中間生成物はGit外に置く。その場合も `notes.md` に保存方針を書く。

`notes.md` は必ず以下を分ける。

```markdown
# Experiment Title

## Question

## Hypothesis

## Setup

## Baseline

## Result

## Interpretation

## Limitations

## Next
```

## 7. Recommended Initial Milestones

### Milestone 1: Repository Foundation

- `AGENTS.md`
- `docs/research-plan.md`
- `docs/research-state.md`
- `docs/experiment-log-template.md`
- `references/`
- `results/README.md`

### Milestone 2: Python Baseline Package

- `src/sidf_lab/guides.py`
- `src/sidf_lab/energy.py`
- `src/sidf_lab/anneal.py`
- `src/sidf_lab/metrics.py`
- `experiments/exp_002_model_c_cross.py`
- tests for determinism and metrics

### Milestone 3: Model D Benchmark

- low-res to high-res reconstruction
- bilinear / bicubic comparison
- confidence map output
- edge leakage and edge width metrics

### Milestone 4: Structured Texture

- smoothed noise
- fractal noise
- directional texture
- comparison against white noise

### Milestone 5: Rust Core Spike

- PCG32
- fixed-point value type
- Model C energy
- deterministic update order
- tiny 16x16 or 32x32 test case

## 8. Suggested Issue Labels

Use GitHub Issues as the research task ledger.

Type labels:

- `t:exp`
- `t:ref`
- `t:impl`
- `t:docs`
- `t:maint`

Priority labels:

- `p:0`: blocker or urgent correctness issue
- `p:1`: next useful work
- `p:2`: backlog

Recommended first issues:

1. `[docs] 研究計画と現在地をSIDF向けに整備する`
2. `[impl] Python package skeletonを追加する`
3. `[exp] Model C cross baselineを保存形式つきで再実行する`
4. `[exp] Model Dとbilinear/bicubicの比較指標を追加する`
5. `[ref] 画像再構成・MRF・超解像の参考文献を集める`
6. `[impl] Rust core移植の最小設計を書く`

## 9. GitHub Operation

GitHub repository:

```text
nana-nun/sidf-lab
```

Default branch:

```text
main
```

GitHub Issuesを研究タスク台帳として使う。GitHub Projectsも進行管理に使う。

基本ラベル:

- `t:exp`: 実験
- `t:ref`: 文献・参考資料
- `t:impl`: 実装
- `t:docs`: ドキュメント
- `t:maint`: 環境・整理

優先度:

- `p:0`: 現在の作業を止める blocker
- `p:1`: 次にやるべき作業
- `p:2`: backlog

PRの単位:

- 1 PR = 1 Issue を基本にする。
- 実験PRでは `results/`、`docs/research-state.md`、必要なら `src/` を更新する。
- 仕様PRでは `specs/` を更新し、根拠となる `results/` を参照する。
- 文献PRでは `references/` を更新する。
- 仕様PRでは、未確定の検討事項をIssueにも残す。

ブランチ名:

```text
docs/<short-topic>
impl/<short-topic>
exp/<short-topic>
ref/<short-topic>
maint/<short-topic>
```

## 10. AI Usage Pattern

AIに頼む作業は、次の粒度にするとよい。

- 「このIssueを読んで実験計画を作って」
- 「Model Cの実験を保存形式つきで実装して」
- 「この結果の `notes.md` を研究者目線で整理して」
- 「`research-state.md` を最新結果で更新して」
- 「この文献を `references/notes/` に要約して」
- 「PythonのModel C kernelをRustへ移す設計だけして」

AIが迷いにくいように、各Issueには必ず `Goal`、`Context`、`Acceptance Criteria`、`References` を書く。

## 11. More Decisions To Make

決定済み:

1. GitHub repository name: `nana-nun/sidf-lab`
2. default branch name: `main`
3. GitHub Projectsを使う
4. Python dependency manager: `.venv + requirements.txt`
5. 小さいPNG/JSON/CSV/MarkdownはGit管理
6. 大量画像や大きい比較出力はGit外
7. 実験結果ドキュメントは毎回出す
8. `experiment_seed` と `decoder_seed` は分ける
9. Model CをPythonで保存形式つき再現可能にしてからRustへ移る
10. 仕様はまず draft とし、検討事項はIssueにも残す
11. GitHub Projects の列は `Todo / Ready / In Progress / Review / Done / Blocked` とする
12. `results/` にGit管理で保存する画像・JSON・CSV・Markdownは、1実験あたり数MBまでを目安にする
13. 大きい結果、大量画像、長時間runの中間生成物は `artifacts/` に置き、Git管理しない
14. `artifacts/` は `.gitignore` 対象とする
15. deterministic target は、まずNumPy再現性から始め、Model C固定後にPCG32/fixed-pointへ寄せる
16. grayscale-only は、Model C/Dと評価指標が安定するまで継続する

## 12. Model C Freeze Criteria

Model C は、Rust移植前の基準実装として扱う。そのため「見た目が良い」だけでなく、保存形式、再現性、複数形状での安定性を満たした時点で freeze する。

### 必須条件

- 同じ `experiment_seed` と `decoder_seed` で、同一環境のNumPy実装が同じ `metrics.json` と同じ出力画像を再生成できる。
- `config.json`、`metrics.json`、`notes.md`、主要PNGが保存される。
- `notes.md` に `Question`、`Hypothesis`、`Setup`、`Baseline`、`Result`、`Interpretation`、`Limitations`、`Next` がある。
- baseline として少なくとも static direct display、bilinearまたは単純平滑化を含める。
- 画像出力は `plt.show()` だけで終わらず、PNG保存される。

### 最小ベンチマーク

Model C freeze 前に、以下の synthetic guides で破綻しないことを確認する。

- cross
- diagonal line
- circle
- thin line
- soft gradient

### 評価指標

最低限、以下を保存する。

- `MAD`
- foreground mean
- background mean
- foreground variance
- background variance
- edge leakage
- decode time

可能なら追加する。

- edge width
- PSNR
- SSIM

### 暫定合格目安

cross baseline では、過去のModel C結果を目安にする。

```text
Cross Mean      ~= 0.50
Background Mean <= 0.02
Edge Leakage    <= 0.02
MAD             <= 0.03
```

ただし、この数値はcross専用の暫定基準であり、circle、diagonal、gradientでは別途解釈する。特にsoft gradientではedge leakageより階調の自然さを重視する。

### Freeze後にやること

- `specs/sidf-v0.2.1.md` に Model C を draft 仕様として整理する。
- 未確定事項をIssue化する。
- Rust側では Model C core annealing kernel から移植する。
