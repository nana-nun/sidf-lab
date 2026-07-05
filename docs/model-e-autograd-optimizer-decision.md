# Model E autograd optimizer基盤の導入判断

Status: Draft decision
Date: 2026-07-04
Related Issue: [#125](https://github.com/nana-nun/sidf-lab/issues/125)

この文書は、Model E fitting を本格的に比較するために autograd optimizer 依存を導入するかを判断するためのメモである。

結論として、この時点では PyTorch、JAX、tinygrad のいずれも `requirements.txt` の必須依存には追加しない。続ける場合は、PyTorch CPU を optional backend として小さな実装Issueに分け、Model E と classical INR baseline の両方へ同じ optimizer 条件を適用できるかだけを検証する。

Issue #132 では、この方針に沿って `src/sidf_lab/inr_torch_fit.py` に optional PyTorch backend adapter を追加した。PyTorch は default dependency に追加していない。PyTorch 未導入環境では `TorchBackendUnavailable` を明示的に返し、実験scriptはskip理由、config、metrics、notes、入力baseline PNGを保存する。Codex環境では PyTorch が未導入だったため、実際の autograd fit metrics と loss curve は #134 で追跡する。

## 背景

Issue #117 では、依存追加を避けた有限差分 Adam / L-BFGS-like 診断を行った。結果として、L-BFGS相当の探索で一部の loss と MAD は改善したが、現行 Model E 候補は classical INR baseline や bicubic baseline を上回らなかった。

Issue #122 では、Candidate A/C が現行Model Eより少し改善したが、best classical INR、nearest、bicubic baselineを上回らなかった。

したがって、autograd optimizer は Model E を採用候補へ戻すための根拠ではなく、optimizer不足と構造不足を分けるための診断基盤として扱う。

## Sources

- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- JAX installation guide: <https://docs.jax.dev/en/latest/installation.html>
- tinygrad project: <https://github.com/tinygrad/tinygrad>
- Issue #132 optional backend spike: `results/2026-07-05-issue-132-model-e-autograd-optimizer-spike/notes.md`
- Follow-up Issue #134: <https://github.com/nana-nun/sidf-lab/issues/134>
- Issue #117 result: `results/2026-06-29-issue-117-model-e-fitting-diagnostics/notes.md`
- Issue #122 result: `results/2026-06-29-issue-122-model-e-parameterization-redesign/notes.md`

## 候補比較

| Candidate | Dependency impact | Windows / Codex handling | Fit capability | Current decision |
| --- | --- | --- | --- | --- |
| PyTorch CPU | 大きい。既定依存にすると環境作成とCIが重くなる | 公式にWindows pip install導線があり、CPU backendならGPU前提を避けられる | Adam / LBFGS / autogradを同じAPIで使いやすい | optional backend候補。まず別Issueでspikeする |
| JAX | 大きい。backendとwheel条件の影響が強い | 公式install文書ではWindowsはexperimental扱いで、Codex/Windowsの標準依存にしにくい | `jit` / `grad` は強いが、導入リスクが大きい | 現時点では採用しない |
| tinygrad | 小さめだが、研究コードとしての安定API確認が必要 | pip導入は軽い可能性があるが、SIDF側の長期保守APIとしては未検証 | 小規模autogradには使える可能性がある | 現時点では採用しない。PyTorch spike後に再評価 |
| pure NumPy継続 | 追加依存なし | 既存 `.venv + requirements.txt` と相性がよい | 有限差分診断は可能だが、parameter数が増えると遅い | default pathとして維持する |

## 決定

このIssueでは autograd 依存を追加しない。

理由:

- #98 / #104 / #117 / #122 の結果では、評価済みModel E候補を採用する根拠はまだない。
- `requirements.txt` に重い依存を追加すると、Model C/D の軽量な実験・検証にも影響する。
- autograd導入の価値は、Model Eだけを良くすることではなく、classical INR baselineにも同じoptimizer条件を適用できるかで決まる。
- PyTorch CPU は候補として最も現実的だが、導入範囲、保存artifact、実行時間を測る小さなspikeを先に行うべきである。

## 導入する場合の最小範囲

導入する場合は、次の範囲に限定する。

- `requirements.txt` にはまだ追加しない。
- `src/sidf_lab/inr_fit.py` の既存 `fit_inr` を置き換えない。
- 新規 module または実験内 helper として optional torch backend を追加する。
- `INRSpec`、parameter layout、quantization、side-bit estimate は既存APIを再利用する。
- Model E single / coupled / selected candidate と、Fourier / RFF / SIREN / MLP baseline を同じ optimizer adapter でfitする。
- まず CPU-only、small source-split fixture、短いstep数に限定する。

最小API案:

```python
@dataclass(frozen=True)
class OptimizerSpec:
    backend: str
    method: str
    steps: int
    learning_rate: float
    seed: int


def fit_inr_with_optimizer(
    spec: INRSpec,
    low_guide: np.ndarray,
    reference: np.ndarray,
    optimizer: OptimizerSpec,
    quantization: QuantizationSpec,
) -> FitResult:
    ...
```

実装上の境界:

- `OptimizerSpec.backend == "numpy_random_search"` は既存実装を呼ぶ。
- `OptimizerSpec.backend == "torch"` は optional import とし、import不可の場合は明示的にskipまたはエラーにする。
- `FitResult` の形は既存と揃え、metrics集計や実験保存側を大きく変えない。
- optimizer trace は実験artifactとして保存し、decoderに必要な保存情報とは分ける。

## 検証コマンドと保存成果物

導入spikeで最低限必要な検証:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe experiments/<new_autograd_spike>.py
```

保存成果物:

- `results/YYYY-MM-DD-issue-<number>-model-e-autograd-optimizer-spike/config.json`
- `metrics.json`
- `notes.md`
- per-case comparison PNG
- loss curve CSV
- optimizer summary table
- fit time / decode time
- float result and quantized result

`notes.md` では、optimizer不足、構造不足、量子化後劣化を分けて書く。

## 導入しない場合の代替方針

autograd dependency を導入しない場合は、pure NumPy のまま次を続ける。

- 有限差分診断は小さいparameter数に限定する。
- random search / finite-difference Adam / L-BFGS-like は到達可能品質の証明ではなく、粗い診断として扱う。
- candidate size、frequency count、coupling variationを増やすIssueでは、optimizer不足が残ることを明記する。
- Model Eを一時保留する場合は、#98 / #104 / #117 / #122 の負の結果と、autograd未導入の制限を分けて記録する。

## Issue #117 / #122 との関係

Issue #117 は、依存追加なしの optimizer 診断として完了している。L-BFGS相当で一部改善したが、現行Model E候補は classical INR baseline や bicubic baseline を上回らなかった。

Issue #122 は、parameterization redesign の小規模比較として完了している。Candidate A/C は現行Model Eより少し改善したが、best classical INR、nearest、bicubic baselineを上回らなかった。

この判断は、#117 や #122 の負の結果を取り消さない。autograd導入は、採用へ進むためではなく、未解決要因として残る optimizer 条件を公平に切り分けるための候補である。

## Limitations

- この文書は導入判断であり、新しい実験結果ではない。
- PyTorch / JAX / tinygrad の実測インストール時間や実行時間は、このIssueでは測っていない。
- autograd optimizerを導入しても、Model E が classical INR baseline、bicubic baseline、または実用codecを上回るとは限らない。
- compression、super-resolution、quantum advantage は未測定である。

## Next

- PyTorch CPU optional backend を小さくspikeするIssueを分ける。
- そのspikeで環境負荷、実行時間、同一optimizer条件での Model E / classical INR 比較が成立するかを測る。
- spikeで価値が見えなければ、Model E系列は candidate size / bit-depth / coupling variation より先に一時保留を検討する。
