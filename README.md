# SIDF Lab

SIDF Lab は、SIDF (Stochastic Image Description Format) の研究用ワークスペースです。

現在の主な対象:

- 確率的画像再構成の Python プロトタイプ
- エッジ保持型アニーリングモデル
- 信頼度マップつき多解像度再構成
- 将来的な Rust コアによる高速・決定論的デコード

最初に読むもの:

- [研究ノート](docs/sidf-research-notes.md)
- [リポジトリ構成](docs/repository-architecture.md)
- [研究計画](docs/research-plan.md)
- [研究の現在地](docs/research-state.md)
- [AIエージェント向けガイド](AGENTS.md)

リポジトリ運用:

- 実験やテストで生成した画像は `results/<date>-<short-name>/` に保存する。
- 研究タスクは GitHub Issues で管理し、`t:*` と `p:*` ラベルを使う。
- 作業状態は GitHub Projects で管理する。
- 仕様、実験、参考文献、実装は分けて管理する。
- GitHub repository: `nana-nun/sidf-lab`
- Python 環境: `.venv + requirements.txt`

Python セットアップ:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

最小CLI確認:

```powershell
$env:PYTHONPATH = "src"
python -m sidf_lab.cli
```
