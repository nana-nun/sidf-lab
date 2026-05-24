# リポジトリ基盤確認

この文書は、sidf-lab の研究リポジトリ基盤がどの文書に分かれているかを短時間で確認するための入口です。

## 目的

sidf-lab は、SIDFを完成済みの実用圧縮形式として扱うのではなく、低解像度ガイド、seed、物理パラメータ、決定論的な確率的緩和過程による画像再構成を検証する研究リポジトリです。

この基盤文書群の役割は、次を混ぜずに管理することです。

- 研究目的と対象範囲
- 現在分かっている結果
- 今後の仮説と実験計画
- 実験結果の保存形式
- AIエージェントのIssue処理手順
- GitHub Issues / Projects / PR の運用

## 最初に読むもの

| 読むもの | 役割 |
| --- | --- |
| `AGENTS.md` | AIエージェント向けの一次方針。研究範囲、禁止したい過大主張、実験保存ルール、GitHub運用を定義する。 |
| `README.md` | 人間向けの短い入口。主要文書、Pythonセットアップ、Issue運用への導線を置く。 |
| `docs/repository-architecture.md` | ディレクトリ構成、Python/Rustの役割、results形式、Model C freeze criteriaを説明する。 |
| `docs/research-state.md` | 現時点の結果、解釈、未解決の問いをまとめる。 |
| `docs/research-plan.md` | これから検証する仮説、baseline、metrics、実験候補を整理する。 |
| `docs/experiment-log-template.md` | 実験メモの型。Question、Hypothesis、Setup、Baseline、Result、Interpretation、Limitations、Nextを分ける。 |
| `.agents/skills/sidf-issue-runner/SKILL.md` | Issue開始からPR作成までのAIエージェント向け標準手順。 |

## 実験結果の保存ルール

実験やテストで画像を生成した場合は、コンソール表示だけで終わらせず、PNGなどの成果物として保存します。

実験結果を残す場合は、原則として `results/YYYY-MM-DD-issue-<number>-<short-title>/` に次を保存します。例: `results/2026-05-24-issue-65-decode-sweep-quality/`。

Issue番号がまだない探索実験は、先にIssueを作るか、PR前に正式なIssue番号つきディレクトリへ改名します。

- `config.json`
- `metrics.json`
- `notes.md`
- 入力またはSTATIC guide画像
- baseline画像
- rendered画像
- confidence mapまたはdifference mapが有用な場合はその画像

Git管理に含める主要PNGは、`notes.md` から Markdown の画像参照で表示できるようにします。

```markdown
![Rendered image](rendered.png)
```

画像ファイルは `notes.md` と同じ結果ディレクトリに置き、相対パスで参照します。

## GitHub運用

研究タスクはGitHub Issuesで管理し、GitHub Projectsで状態を追います。

基本ラベル:

- `t:exp`: 実験
- `t:ref`: 文献・参考資料
- `t:impl`: 実装
- `t:docs`: ドキュメント
- `t:maint`: 環境・整理

優先度ラベル:

- `p:0`: blockerまたは急ぎの正確性問題
- `p:1`: 次に進めるべき作業
- `p:2`: backlog

Issue対応では `.agents/skills/sidf-issue-runner/SKILL.md` を入口にし、Issueの `t:*` ラベルに応じた `.agents/skills/sidf-lab-*-issue/SKILL.md` も確認します。

1 PR = 1 Issue を基本とし、PR作成後はProjectを `Review` に移します。明示依頼がない限り、AIエージェントはPRをマージしません。

## 現在の限界

- この文書はリポジトリ運用の入口であり、SIDF仕様そのものではありません。
- 実験結果はshapeや設定に依存します。単一実験から、実用圧縮性能、超解像性能、一般画像品質を断定しません。
- Rust固定小数点実装や環境非依存のbit-perfect再現性は、まだ確定していません。
