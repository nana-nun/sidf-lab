# sidf-lab Agent Guide

このファイルは、sidf-lab でAIエージェントが作業するときの一次参照です。迷ったら、まずこの方針を優先してください。

## Research Goal

このリポジトリの目的は、SIDF (Stochastic Image Description Format) を「すぐに実用圧縮形式として完成させる」ことではありません。

目的は、低解像度ガイド画像、seed、物理パラメータ、決定論的な確率的緩和過程を使って、どこまで知覚的に妥当な画像再構成ができるかを検証することです。

## Research Scope

対象にするもの:

- SIDF仕様案
- grayscale reconstruction
- low-resolution STATIC guide
- annealing / relaxation reconstruction
- edge-preserving energy model
- confidence map
- deterministic texture prior
- bilinear / bicubic baseline
- SSIM / PSNR / edge leakage / decode time
- Python prototype
- future Rust core decoder

対象外にするもの:

- 現段階での「既存画像形式より高圧縮」という断定
- Ground Truth比較なしの「超解像性能」主張
- 環境非依存のビット完全再現性を満たしていない実装を正式仕様として扱うこと
- 実用フォーマットとしての互換性保証
- AI生成画像モデルそのものの開発

## Documentation

- 人間が読む文書は日本語で書く。例: `README.md`、`docs/`、`references/`、`results/*/notes.md`。
- AIエージェントが読む文書は英語で書いてよい。例: `.agents/` 配下のワークフロー、スキル、ポリシー。
- 主張、仮説、結果、解釈を混ぜない。
- 実験メモでは、できるだけ `Question`、`Hypothesis`、`Setup`、`Baseline`、`Result`、`Interpretation`、`Limitations`、`Next` を分ける。
- 仕様の主張は慎重に書く。特に `compression`、`super-resolution`、`emergence` は、測定結果と限界を併記する。
- 参考文献を追加するときは、URL、論文題、著者、年、DOI が分かる範囲で残す。
- BibTeXで管理できる文献は `references/papers.bib` に追加する。
- Web記事やリンク集は `references/links.md` に追加する。
- 読書メモは `references/notes/` に Markdown で作成する。

## Experiment Principle

高度な再構成モデルを評価する前に、必ず単純なベースラインを用意する。

例:

- nearest upscaling
- bilinear upscaling
- bicubic upscaling
- static guide direct display
- deterministic smoothing filter

SIDFモデルの出力は、ベースラインとの差分で評価する。

テストや実験で画像を生成した場合、出力画像は必ず保存する。コンソール表示や `plt.show()` だけで終わらせない。

保存する代表画像:

- input / STATIC guide
- upscaled guide
- confidence map
- rendered output
- baseline output
- difference map when useful

## Reproducibility

実験結果を保存する場合は、可能な範囲で以下を残す。

- 実行コマンド
- experiment_seed
- decoder_seed
- input guide size
- output size
- model name
- model config
- metrics
- decode time
- 実行日時
- Python / dependency version

結果を保存する場合は、`results/` 配下に実験ごとのディレクトリを作る。

```text
results/
  2026-05-13-model-d-cross/
    config.json
    metrics.json
    notes.md
    static.png
    guide.png
    confidence.png
    rendered.png
```

小さいPNG、JSON、CSV、Markdownの結果はGit管理する。大量画像、大きい比較出力、長時間runの中間生成物はGit外に置き、`notes.md` に保存場所または未保存理由を残す。

実験PRでは、結果ドキュメントを毎回出す。最低限 `results/<date>-<short-name>/notes.md` を含める。

Git管理する結果は1実験あたり数MBまでを目安にする。大きい結果は `artifacts/` に保存し、Git管理しない。

## Git and GitHub Workflow

- GitHubリポジトリは `nana-nun/sidf-lab` とする。
- default branch は `main` とする。
- GitHub Issuesを研究タスク台帳として使う。
- GitHub Projectsを進行管理に使う。
- GitHub Projects の列は `Todo / Ready / In Progress / Review / Done / Blocked` とする。
- Issueは `t:exp`、`t:ref`、`t:impl`、`t:docs`、`t:maint` のいずれかを基本分類にする。
- 優先度は `p:0`、`p:1`、`p:2` を使う。
- 変更は原則として小さいPR単位に分ける。
- 実験結果を追加するPRでは、`results/<date>-<short-name>/notes.md` と主要画像を含める。
- 研究解釈が変わる実験を追加した場合は、`docs/research-state.md` も更新する。
- 仕様を変える場合は、実験結果ではなく `specs/` にドラフトとして残す。

推奨ブランチ名:

```text
docs/issue-1-research-foundation
impl/issue-3-python-package-skeleton
exp/issue-4-model-c-cross-baseline
exp/issue-6-model-d-multires-cross
ref/issue-7-image-reconstruction-survey
maint/issue-2-project-workflow
```

Issue対応用のブランチ名には、必ずIssue番号を含める。

形式:

```text
<type>/issue-<number>-<branch-name>
```

例:

```text
impl/issue-3-python-package-skeleton
exp/issue-4-model-c-cross-baseline
docs/issue-8-sidf-v0.3-draft
```

PR本文には最低限これを書く。

```markdown
## Summary

## Verification

## Results

## Limitations

## Related Issue
```

## Python

- Pythonを動かす場合は、プロジェクト直下の `.venv` を使う。
- `.venv` がない場合は作成してから使う。
- 依存管理は当面 `.venv + requirements.txt` とする。必要になったら後で `pyproject.toml` に移行する。
- repository module を import する場合は、PowerShellで `$env:PYTHONPATH="src"` を設定してから実行する。
- 実験コードは、まず小さい画像と少ない sweep で動かす。
- 結果を解釈するときは、画像の印象だけでなく metrics も保存する。

PowerShell例:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Rust

- Rustは最初から全体移植しない。
- Model CをPythonで保存形式つき再現可能にしてからRustへ移る。
- 最初のRust移植対象は Model C の core annealing kernel とする。
- Rust側では PRNG、固定小数点、energy計算、update loop を優先する。
- Pythonは実験 orchestration、plot、metrics、parameter search を担当する。

## Model C Freeze Criteria

Model CはRust移植前の基準実装として扱う。freeze前に以下を満たす。

- 同じ `experiment_seed` と `decoder_seed` で同一環境のNumPy実装が再現できる。
- `config.json`、`metrics.json`、`notes.md`、主要PNGが保存される。
- cross、diagonal line、circle、thin line、soft gradient の最小ベンチで破綻しない。
- `MAD`、foreground/background mean、foreground/background variance、edge leakage、decode time を保存する。
- cross baseline では暫定的に `Background Mean <= 0.02`、`Edge Leakage <= 0.02`、`MAD <= 0.03` を目安にする。
- freeze後に `specs/sidf-v0.2.1.md` をdraftとして作り、未確定事項をIssueに残す。

## Research State

新しい実験や分析を始める前に、AIエージェントは次を確認する。

1. `AGENTS.md`
2. `docs/research-state.md`
3. `docs/repository-architecture.md`
4. 関連する `results/*/notes.md`
5. 関連する `references/notes/`

## Change Policy

- 既存構成を尊重し、小さく検証可能な実験として追加する。
- 実験結果を保存する場合は `config.json` と `notes.md` を必ず含める。
- 実験結果を保存する場合は、生成画像も必ず保存する。
- 仕様変更と実験結果を混ぜない。仕様案は `specs/`、実験結果は `results/` に分ける。
- 仕様はまず draft として `specs/` に置く。検討事項はIssueにも残す。
- 実装後は、該当するテストまたはCLIサンプルを実行する。
- 実験結果を解釈するときは、限界と未確認事項を明記する。
