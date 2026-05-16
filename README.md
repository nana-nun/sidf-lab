# SIDF Lab

SIDF Lab は、SIDF (Stochastic Image Description Format) の研究用ワークスペースです。

現在の主な対象:

- 確率的画像再構成の Python プロトタイプ
- エッジ保持型アニーリングモデル
- 信頼度マップつき多解像度再構成
- 将来的な Rust コアによる高速・決定論的デコード

最初に読むもの:

- [研究ノート](docs/sidf-research-notes.md)
- [リポジトリ基盤確認](docs/repository-foundation.md)
- [リポジトリ構成](docs/repository-architecture.md)
- [研究計画](docs/research-plan.md)
- [研究の現在地](docs/research-state.md)
- [参考文献リスト](references/reading-list.md)
- [AIエージェント向けガイド](AGENTS.md)
- [AIエージェント向けIssue処理Skill](.agents/skills/sidf-issue-runner/SKILL.md)

リポジトリ運用:

- 実験やテストで生成した画像は `results/<date>-<short-name>/` に保存する。
- 研究タスクは GitHub Issues で管理し、`t:*` と `p:*` ラベルを使う。
- AIエージェントがIssue対応を行う場合は、`.agents/skills/sidf-issue-runner/SKILL.md` を入口にし、Issueの `t:*` ラベルに応じた `.agents/skills/sidf-lab-*-issue/SKILL.md` も参照する。
- Issue対応時は、GitHub Project のステータス更新、Issueコメント、PR作成までを標準フローに含める。マージは明示依頼がない限り行わない。
- 作業状態は GitHub Projects で管理する。
- 仕様、実験、参考文献、実装は分けて管理する。
- 参考文献は `references/reading-list.md`、`references/papers.bib`、`references/links.md`、`references/notes/` に分けて記録する。
- GitHub repository: `nana-nun/sidf-lab`
- Python 環境: `.venv + requirements.txt`

Python セットアップ:

```powershell
$runtimePython = Get-ChildItem -LiteralPath "$env:USERPROFILE\.cache\codex-runtimes" -Recurse -Filter python.exe |
  Where-Object { $_.FullName -notmatch "WindowsApps" } |
  Select-Object -First 1
& $runtimePython.FullName -m venv --system-site-packages --without-pip .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

通常のローカルPythonが使える環境では、従来どおり `python -m venv .venv` と `python -m pip install -r requirements.txt` でもよい。Codex on Windows では `python` が Microsoft Store stub を指す場合があるため、Codex runtime の Python から `.venv` を作る。

最小CLI確認:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m sidf_lab.cli
```
