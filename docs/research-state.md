# SIDF Research State

この文書は、AIエージェントと人間が「ここまで何が分かっているか」を短時間で確認するための現在地メモです。詳細な議論は `docs/sidf-research-notes.md` を一次参照にしてください。

## Scope

sidf-lab は、現段階では実用圧縮形式ではなく、低解像度ガイドと決定論的確率過程による画像再構成モデルの研究リポジトリです。

## Current Findings

### Model A

外部場項 `-h_i v_i` は、STATICが正の値を持つ場所で輝度を上げる片方向の力になった。

結果:

- 十字構造は誘導された。
- 黒背景が灰色化した。
- 明部が周囲へ膨張した。

解釈:

Model A は画像復元モデルとして不安定。

### Model C

data fidelity `lambda_data * (v_i - s_i)^2` と edge-aware interaction を導入した。

結果:

- 背景が暗く保たれた。
- 十字平均が目標値に近づいた。
- edge leakage が大きく低下した。

代表値:

```text
MAD                 : 0.0195
Cross Mean          : 0.5014
Background Mean     : 0.0085
Edge Leakage        : 0.0097
Cross Variance      : 0.0016
Background Variance : 0.0004
```

保存形式つき再実行:

- `results/2026-05-16-model-c-cross-baseline/`
- `config.json`、`metrics.json`、`notes.md`、主要PNGを保存。
- 今回のrunでは `MAD = 0.0117`、`Background Mean = 0.0074`、`Edge Leakage = 0.0089`。

解釈:

Model C は、SIDF v0.2.1 の基礎モデルとして有望。ただし創発性は弱く、安定化フィルタに近い。

### Model D

16x16 guide から 64x64 output を生成する multi-resolution pipeline を導入した。

構成:

- bilinear upscaled guide
- gradient-based confidence map
- edge-aware interaction
- seeded texture term

結果:

- bilinear より十字境界が視覚的に締まった。
- confidence map がエッジ拘束として機能した。
- texture は出たが、現段階では意味的ディテールではなく粒状ノイズに近い。

解釈:

Model D は confidence-aware multi-resolution reconstruction と呼ぶのが正確。Ground Truth比較なしに「超解像性能」とは主張しない。

## Open Questions

- Model D は斜線や曲線でも境界を守れるか。
- confidence map は柔らかいグラデーションを硬く分断しないか。
- white noise ではなく structured noise prior を使うと質感は改善するか。
- bilinear / bicubic に対する優位性は metrics で確認できるか。
- Rust固定小数点実装に移したとき、同じ結果を再現できるか。
- decode time は小画像以外で実用的か。

## Analysis Checklist for AI Agents

新しい実験や分析を始める前に、AIエージェントは次を確認する。

1. `AGENTS.md`
2. `docs/research-plan.md`
3. `docs/sidf-research-notes.md`
4. 関連する `results/*/notes.md`
5. 関連する `references/notes/`
6. baseline と metrics を明確にする。
7. `Limitations` と `Next` を必ず書く。
