# Experiment Note Template

```markdown
# Experiment Title

## Question

## Hypothesis

## Setup

- Command:
- Seed:
- Experiment seed:
- Decoder seed:
- Input guide:
- Input size:
- Output size:
- Model:
- Model config:
- Python / dependency version:

## Baseline

## Metrics

## Saved Artifacts

- Config:
- Metrics:
- Notes:
- Input / guide image:
- Baseline image:
- Rendered image:
- Confidence map:
- Difference image:

## Images

![Input or guide image](input-or-guide.png)

![Baseline image](baseline.png)

![Rendered image](rendered.png)

![Difference image](difference.png)

## Result

## Interpretation

## Limitations

## Next
```

Notes:

- `notes.md` は毎回保存する。
- 画像生成がある場合は、表示だけで終わらせずPNGを保存する。
- Git管理に含める主要画像は、`notes.md` から Markdown 画像参照で表示できるようにする。
- `experiment_seed` と `decoder_seed` は分けて記録する。
