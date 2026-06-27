# Source-Split Grayscale Patch Fixture

このディレクトリは、Model E / INR 比較で development と evaluation が同一source image内のcropに混ざらないようにするための小さなfixtureです。

## 内容

- `manifest.json`: source、license、split、crop、処理手順、patchファイルを記録するmanifest。
- `hobbema_landscape_128.npy` / `hobbema_landscape_128.png`: development source。既存の `experiments/assets/landscape_pd_128.npy` をsource-split fixtureとして複製したもの。
- `hokusai_wave_128.npy` / `hokusai_wave_128.png`: evaluation source。Wikimedia CommonsのPublic Domain画像をdownloaded previewからcenter-square cropし、128x128 grayscaleへ変換したもの。

## Split Policy

source image単位で分割します。現時点では `hobbema_landscape` を development、`hokusai_wave` を evaluation として固定します。#104 でpatchを増やす場合も、同じsourceの近接cropがdevelopmentとevaluationを跨がないようにします。

## License Notes

両sourceはWikimedia Commons上で、二次元のpublic-domain artworkのfaithful reproductionとしてPublic Domain Mark 1.0 / PD-Art扱いになっています。詳細は `manifest.json` の `source_page` と `license_note` を参照してください。
