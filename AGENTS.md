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

## Experiment and Results

高度な再構成モデルを評価する前に、必ず単純なベースラインを用意し、SIDFモデルの出力はベースラインとの差分で評価してください。

実験やテストで画像を生成した場合は、コンソール表示だけで終わらせず、PNGなどの成果物として保存します。実験結果を残す場合は `results/YYYY-MM-DD-issue-<number>-<short-title>/` に `config.json`、`metrics.json`、`notes.md`、主要画像を保存し、実行コマンド、seed、設定、metrics、decode time、限界を記録します。Issue番号がまだない探索実験は、先にIssueを作るか、PR前に正式なIssue番号つきディレクトリへ改名します。

Git管理に含める主要画像は、`notes.md` から Markdown の画像参照 `![説明](image.png)` で表示できるようにします。画像ファイルは `notes.md` と同じ結果ディレクトリに置き、相対パスで参照してください。

詳細な実験Issueの手順は `.agents/skills/sidf-lab-exp-issue/SKILL.md` を参照してください。

## Git and GitHub Workflow

- GitHubリポジトリは `nana-nun/sidf-lab` とする。
- default branch は `main` とする。
- GitHub Issuesを研究タスク台帳として使い、GitHub Projectsで進行管理する。
- Issue分類は `t:exp`、`t:ref`、`t:impl`、`t:docs`、`t:maint`、優先度は `p:0`、`p:1`、`p:2` を使う。
- Issue対応時は `.agents/skills/sidf-issue-runner/SKILL.md` を入口にし、Issueの `t:*` ラベルに応じた `.agents/skills/sidf-lab-*-issue/SKILL.md` も確認する。
- ブランチ名、Projectステータス更新、Issueコメント、PR本文、PR作成、マージしない方針は `sidf-issue-runner` に従う。
- Issue対応の完了時に、未検証事項、残った制限、次に必要な実験・文献調査・実装・文書化が明確になった場合は、既存Issueと重複しない範囲で follow-up Issue を作成または提案する。
- 変更は原則として小さいPR単位に分ける。仕様案は `specs/`、実験結果は `results/` に分ける。

## Python

- Pythonはプロジェクト直下の `.venv + requirements.txt` を使う。
- Codex on Windows では `python` が Microsoft Store stub を指すことがある。`.venv` がない場合は、まず Codex runtime の Python を探して `--system-site-packages --without-pip` で作成する。
- repository module を import する場合は、PowerShellで `$env:PYTHONPATH = "src"` を設定する。
- 実装Issueの詳細な検証手順は `.agents/skills/sidf-lab-impl-issue/SKILL.md` を参照する。

Codex向け `.venv` 作成例:

```powershell
$runtimePython = Get-ChildItem -LiteralPath "$env:USERPROFILE\.cache\codex-runtimes" -Recurse -Filter python.exe |
  Where-Object { $_.FullName -notmatch "WindowsApps" } |
  Select-Object -First 1
& $runtimePython.FullName -m venv --system-site-packages --without-pip .venv
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Rust

- Rustは最初から全体移植しない。
- Model CをPythonで保存形式つき再現可能にしてからRustへ移る。
- 最初のRust移植対象は Model C の core annealing kernel とする。
- Rust側では PRNG、固定小数点、energy計算、update loop を優先する。
- Pythonは実験 orchestration、plot、metrics、parameter search を担当する。

## Model C Freeze Criteria

Model CはRust移植前の基準実装として扱う。freeze前に、再現性、保存形式、複数形状ベンチ、metrics、主要PNG、限界の記録をそろえる。

soft gradient のように明確な foreground/background 境界を持たない guide では、edge leakage を無理に主要指標にしない。`edge_leakage=null` または `Not applicable` とした理由を `notes.md` に書き、代替として gradient monotonicity、slope error、smoothness、region summary などを検討する。詳細は `docs/repository-architecture.md` の freeze criteria を参照してください。

詳細な freeze criteria は `docs/repository-architecture.md` を参照してください。

## Research State

新しい実験や分析を始める前に、AIエージェントは次を確認する。

1. `AGENTS.md`
2. `docs/research-state.md`
3. `docs/repository-architecture.md`
4. 関連する `results/*/notes.md`
5. 関連する `references/notes/`

## Change Policy

- 既存構成を尊重し、小さく検証可能な変更として追加する。
- 仕様変更と実験結果を混ぜない。仕様案は `specs/`、実験結果は `results/` に分ける。
- 仕様はまず draft として `specs/` に置く。検討事項はIssueにも残す。
- 実装後は、該当するテスト、CLIサンプル、またはSkillで指定された検証を実行する。
- 実験結果を解釈するときは、限界と未確認事項を明記する。
